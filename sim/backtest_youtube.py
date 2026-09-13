#!/usr/bin/env python3
"""Back-test the app's diarization and transcription on a YouTube video.

There is no test corpus for "ten Singaporeans around a table", but there are
thousands of hours of Singaporeans talking on YouTube. This runs the app's
pipeline, component for component, over an excerpt of one and reports what it
found against what can actually be known about the video:

  1. the headcount — speakers found (with at least MIN_SPEAKER_SECONDS of
     speech) against speakers present, and how the speaking time splits.
  2. the transcript, labelled — one label saying two people's lines, or a
     question and its answer under the same label, is obvious to a reader in
     seconds. Where the headcount re-cluster changed a label, the auto label
     is shown beside it.
  3. a manual RTTM for a few minutes (--rttm), which makes DER real.
  4. optionally (--reference) a second offline diarizer: pyannote segmentation
     3.0 via sherpa-onnx with the same CAM++ embeddings. It is off by default
     because on the first two real videos it was worse than the app — it
     merged a podcast's two hosts into one speaker plus a 7 s blip, and found
     eleven speakers on a four-host show — so a DER against it says more about
     it than about the app. It stays available as a second opinion.

For transcripts, WER against creator-uploaded captions only when they exist
(auto-captions are worse than the model under test), flagged as weak.

    python3 sim/backtest_youtube.py URL --speakers 2 [--start 600 --minutes 10] [--reference]
    python3 sim/backtest_youtube.py --manifest sim/backtests.json     # entries: url, speakers, start, minutes, asr, reference

Audio is fetched with yt-dlp (m4a) and converted with macOS afconvert — no
ffmpeg — into sim/backtest/<id>/, which is gitignored: it stays on this
machine, for testing. The pipeline: Silero VAD with the app's thresholds, the
trailing-silence trim, CAM++ on the turn alone, sim/clustering.py with
SPEAKER_MODEL (the JS is its verified port), the known-headcount re-cluster,
same-speaker coalescing, the Whisper hallucination filter, and the Singlish
Whisper — fp32 PyTorch on CPU here, ~0.12x real time; the browser's q4 and
q8 exports score within a point of it.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from clustering import MIN_SPEAKER_SECONDS, Config, OnlineClusterer  # noqa: E402
from speaker_embed import CamPP, trailing_speech_end  # noqa: E402

SR = 16000
SPEAKER_MODEL = Config(threshold=0.65, min_centroid_seconds=1.5, margin=0.06,
                       defer_under_seconds=1.5, recluster_every=100, recluster_threshold=0.75)
SEG_MODEL = ROOT / "web" / "models" / "diarization" / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = ROOT / "web" / "models" / "wespeaker_en_voxceleb_CAM++.onnx"
WHISPER = "mjwong/whisper-small-singlish"
COALESCE_GAP, COALESCE_MAX = 1.2, 20.0


# ─────────────────────────────────────────────────────────────────── fetch ──

def fetch(url: str, out_root: Path) -> dict:
    import yt_dlp
    probe = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}).extract_info(url, download=False)
    vid = probe["id"]
    d = out_root / vid
    d.mkdir(parents=True, exist_ok=True)
    wav = d / "audio.wav"
    if not wav.exists():
        print(f"  fetching audio for {vid} …", flush=True)
        yt_dlp.YoutubeDL({
            "quiet": True, "no_warnings": True, "noprogress": True,
            "format": "bestaudio[ext=m4a]/bestaudio",
            "outtmpl": str(d / "audio.%(ext)s"),
            "writeinfojson": True, "writesubtitles": True, "subtitleslangs": ["en.*"],
        }).extract_info(url, download=True)
        src = next((p for p in d.glob("audio.*") if p.suffix not in (".wav", ".json", ".vtt")), None)
        if src is None:
            raise SystemExit("download produced no audio file")
        r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(src), str(wav)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"afconvert could not decode {src.name} ({r.stderr.strip()[:120]}). "
                             "Install ffmpeg and convert to 16 kHz mono WAV as audio.wav.")
    subs = sorted(d.glob("*.vtt"))
    return {"id": vid, "title": probe.get("title", vid), "duration": probe.get("duration", 0),
            "channel": probe.get("channel") or probe.get("uploader", ""), "dir": d, "wav": wav,
            "subs": subs[0] if subs else None, "url": url}


def load_excerpt(wav: Path, start: float, minutes: float) -> np.ndarray:
    with wave.open(str(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2), wav
        w.setpos(int(start * SR))
        x = np.frombuffer(w.readframes(int(minutes * 60 * SR)), dtype=np.int16).astype(np.float32) / 32768
    return x


# ────────────────────────────────────────────────────────────── diarization ──

def reference_diarization(audio: np.ndarray, num_clusters: int | None) -> list[tuple[float, float, str]]:
    import sherpa_onnx
    kw = dict(num_clusters=num_clusters) if num_clusters else dict(num_clusters=-1, threshold=0.5)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL))),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL)),
        clustering=sherpa_onnx.FastClusteringConfig(**kw), min_duration_on=0.3, min_duration_off=0.5)
    assert cfg.validate()
    res = sherpa_onnx.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()
    return [(float(r.start), float(r.end), f"ref{r.speaker}") for r in res]


def app_segments(audio: np.ndarray, campp: CamPP) -> tuple[list[dict], list[np.ndarray]]:
    """Silero VAD with the app's thresholds, then the trim; returns turns and their embeddings."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    vad = load_silero_vad(onnx=True)
    stamps = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR,
                                   threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=350,
                                   max_speech_duration_s=12, min_silence_duration_ms=700, speech_pad_ms=250)
    turns, embs = [], []
    for st in stamps:
        raw = audio[st["start"]:st["end"]]
        end = trailing_speech_end(raw)
        secs = end / SR
        if secs < 0.35:
            continue
        e = campp.embed(raw[:end])
        if e is None:
            continue
        turns.append({"index": len(turns), "start": st["start"] / SR, "end": (st["start"] + end) / SR,
                      "seconds": secs, "pcm": raw[:end]})
        embs.append(e)
    return turns, embs


def app_diarization(turns: list[dict], embs: list[np.ndarray], speakers: int | None) -> dict:
    cl = OnlineClusterer(SPEAKER_MODEL)
    for t, e in zip(turns, embs):
        cl.add(t["index"], e, t["seconds"])
    cl.finish()
    out = {"auto": [(t["start"], t["end"], f"S{cl.assignment[t['index']]}") for t in turns]}
    # Who sounds like whom, among the labels that spoke: the number a reader
    # needs to judge an over-split. Real within-speaker cosine ran ~0.80 on a
    # podcast; the assignment threshold is 0.65; two labels above ~0.75 are
    # very likely one person, two below 0.55 very likely two.
    live = [c for c in cl.clusters if c.seconds >= MIN_SPEAKER_SECONDS]
    pairs = sorted(((float(a.centroid @ b.centroid), f"S{a.id}", f"S{b.id}") for i, a in enumerate(live) for b in live[i + 1:]), reverse=True)
    out["similar"] = [(a, b, round(c, 2)) for c, a, b in pairs[:8]]
    if speakers:
        cl2 = OnlineClusterer(SPEAKER_MODEL)
        for t, e in zip(turns, embs):
            cl2.add(t["index"], e, t["seconds"])
        cl2.finish()
        cl2.recluster_to(speakers)
        out["headcount"] = [(t["start"], t["end"], f"S{cl2.assignment[t['index']]}") for t in turns]
        out["suspicious"] = cl2.last_recluster.get("suspicious", [])
    return out


# ─────────────────────────────────────────────────────────────────── scoring ──

def frames_of(segs, total, step=0.01):
    n = int(total / step) + 1
    grid = [set() for _ in range(n)]
    for s, e, spk in segs:
        for k in range(max(0, int(s / step)), min(n, int(e / step) + 1)):
            grid[k].add(spk)
    return grid


def der(ref, hyp, total, collar=0.25, step=0.01):
    """Diarization error of hyp against ref with the optimal speaker mapping and
    a standard collar around reference boundaries. ref may overlap; hyp does not."""
    from scipy.optimize import linear_sum_assignment
    R, H = frames_of(ref, total, step), frames_of(hyp, total, step)
    ignore = np.zeros(len(R), bool)
    c = int(collar / step)
    for s, e, _ in ref:
        for b in (s, e):
            k = int(b / step)
            ignore[max(0, k - c):k + c + 1] = True
    rs, hs = sorted({x for s in R for x in s}), sorted({x for s in H for x in s})
    ri, hi = {x: i for i, x in enumerate(rs)}, {x: i for i, x in enumerate(hs)}
    M = np.zeros((len(hs), len(rs)))
    for k in range(len(R)):
        if ignore[k]:
            continue
        for h in H[k]:
            for r in R[k]:
                M[hi[h], ri[r]] += 1
    mapping = {}
    if M.size:
        rows, cols = linear_sum_assignment(-M)
        mapping = {hs[i]: rs[j] for i, j in zip(rows, cols) if M[i, j] > 0}
    miss = fa = conf = total_ref = 0
    for k in range(len(R)):
        if ignore[k]:
            continue
        rset, hset = R[k], {mapping.get(h, h) for h in H[k]}
        matched = len(rset & hset)
        conf += min(len(rset), len(hset)) - matched
        miss += max(0, len(rset) - len(hset))
        fa += max(0, len(hset) - len(rset))
        total_ref += len(rset)
    d = (miss + fa + conf) / max(1, total_ref)
    return {"der": d, "miss": miss / max(1, total_ref), "fa": fa / max(1, total_ref),
            "confusion": conf / max(1, total_ref), "mapping": mapping, "collar": collar}


def speaking_time(segs):
    t = {}
    for s, e, spk in segs:
        t[spk] = t.get(spk, 0.0) + (e - s)
    return dict(sorted(t.items(), key=lambda x: -x[1]))


def found(segs, min_seconds=MIN_SPEAKER_SECONDS):
    return sum(1 for v in speaking_time(segs).values() if v >= min_seconds)


# ────────────────────────────────────────────────────────────────────── ASR ──

_ALWAYS = [re.compile(p, re.I) for p in (r"subtitles?\s+(by|provided|created|made)", r"amara\.org", r"\bwww\.|\.com\b",
                                          r"\b(please\s+)?(like\s+and\s+)?subscribe\b", r"\bcopyright\b|©",
                                          r"transcri(bed|ption)\s+by", r"^\s*[\[(【][^\])】]*[\])】]\s*$")]
_FILLER = re.compile(r"^(thank\s*you|thanks|thank\s*you\s+for\s+watching|thanks\s+for\s+watching|bye|goodbye|you|okay|ok|so|um|uh|oh)[\s.!?,]*$", re.I)


def looks_hallucinated(text: str, seconds: float) -> bool:
    t = text.strip()
    if not t or re.fullmatch(r"[\W_]+", t):
        return True
    if any(p.search(t) for p in _ALWAYS):
        return True
    words = re.sub(r"[^\w' ]+", " ", t.lower()).split()
    if _FILLER.match(t) and seconds >= 2.0:
        return True
    if seconds > 0 and len(words) / seconds > 6.5:
        return True
    if len(words) >= 8:
        if len(set(words)) / len(words) < 0.3:
            return True
        tri = {}
        for i in range(len(words) - 2):
            k = " ".join(words[i:i + 3]); tri[k] = tri.get(k, 0) + 1
        if max(tri.values()) >= 4:
            return True
    return False


def coalesce(turns, labels):
    runs, cur = [], None
    for t, lab in zip(turns, labels):
        joinable = cur and cur["speaker"] == lab and t["start"] - cur["end"] <= COALESCE_GAP \
            and cur["seconds"] + t["seconds"] <= COALESCE_MAX
        if not joinable:
            if cur:
                runs.append(cur)
            cur = {"speaker": lab, "start": t["start"], "end": t["end"], "seconds": 0.0, "pcm": [], "auto": set()}
        cur["pcm"].append(t["pcm"]); cur["end"] = t["end"]; cur["seconds"] += t["seconds"]
        if t.get("auto"):
            cur["auto"].add(t["auto"])
    if cur:
        runs.append(cur)
    return runs


def transcribe(runs):
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=WHISPER, device="cpu", chunk_length_s=30)
    lines, dropped = [], 0
    for r in runs:
        pcm = np.concatenate(r["pcm"])
        text = asr({"raw": pcm, "sampling_rate": SR}, generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
        if looks_hallucinated(text, r["seconds"]):
            dropped += 1
            continue
        lines.append({"t0": r["start"], "speaker": r["speaker"], "seconds": r["seconds"], "text": text, "auto": sorted(r.get("auto", []))})
    return lines, dropped


def normalise(t):
    t = re.sub(r"<[^>]+>", " ", t.lower())
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)
    return re.sub(r"[^a-z0-9' ]+", " ", t).split()


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def captions_in_window(vtt: Path, start: float, end: float) -> str:
    txt, t = [], None
    for line in vtt.read_text(errors="replace").splitlines():
        m = re.match(r"(\d+):(\d\d):(\d\d)\.(\d+)\s+-->\s+(\d+):(\d\d):(\d\d)\.(\d+)", line)
        if m:
            h, mi, s, ms = (int(x) for x in m.groups()[:4])
            t = h * 3600 + mi * 60 + s + ms / 1000
            continue
        if t is not None and start <= t < end and line.strip() and not line.startswith(("WEBVTT", "Kind:", "Language:")):
            txt.append(line.strip())
    return " ".join(txt)


# ─────────────────────────────────────────────────────────────────── report ──

def hhmm(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def run_one(url, speakers, start, minutes, do_asr, out_root, rttm=None, use_reference=False):
    t_all = time.time()
    info = fetch(url, out_root)
    dur = info["duration"] or 0
    if dur and start + 30 > dur:
        start = max(0, dur - minutes * 60)
    audio = load_excerpt(info["wav"], start, minutes)
    total = len(audio) / SR
    print(f"\n{info['title']}\n  {info['channel']} · {hhmm(dur)} long · excerpt {hhmm(start)}–{hhmm(start + total)} · told {speakers or 'nothing'} about headcount")

    timing = {}
    t0 = time.time(); campp = CamPP(); turns, embs = app_segments(audio, campp); timing["vad_embed_s"] = time.time() - t0
    t0 = time.time(); app = app_diarization(turns, embs, speakers); timing["cluster_s"] = time.time() - t0

    ref, ref_note = None, None
    if rttm:
        ref = [(float(f[3]), float(f[3]) + float(f[4]), f[7]) for f in (l.split() for l in Path(rttm).read_text().splitlines()) if f and f[0] == "SPEAKER"]
        ref = [(s - start, e - start, k) for s, e, k in ref if e > start and s < start + total]
        ref_note = f"manual labels, {Path(rttm).name}"
    elif use_reference:
        t0 = time.time(); ref = reference_diarization(audio, speakers); timing["reference_s"] = time.time() - t0
        ref_note = "second diarizer: pyannote segmentation 3.0 + CAM++ via sherpa-onnx — a second opinion, not ground truth"

    suspicious = app.pop("suspicious", [])
    similar = app.pop("similar", [])
    rows = []
    for name, segs in app.items():
        row = {"variant": name, "speakers_found": found(segs), "clusters": len(speaking_time(segs)), "speaking_time": speaking_time(segs)}
        if ref is not None:
            d = der(ref, segs, total)
            row.update(der_vs_reference=d["der"], confusion=d["confusion"], miss=d["miss"], fa=d["fa"])
        rows.append(row)
    ref_row = {"speakers_found": found(ref), "clusters": len(speaking_time(ref)), "speaking_time": speaking_time(ref), "note": ref_note} if ref is not None else None

    final = app.get("headcount", app["auto"])
    auto_of = {t["index"]: lab for t, (_, _, lab) in zip(turns, app["auto"])}
    final_of = {t["index"]: lab for t, (_, _, lab) in zip(turns, final)}
    merged = {}
    for i, lab in final_of.items():
        merged.setdefault(lab, set()).add(auto_of[i])

    lines, dropped, wer = [], 0, None
    if do_asr:
        for t in turns:
            t["auto"] = auto_of[t["index"]]
        t0 = time.time(); lines, dropped = transcribe(coalesce(turns, [final_of[t["index"]] for t in turns])); timing["asr_s"] = time.time() - t0
        if info["subs"]:
            ref_txt = normalise(captions_in_window(info["subs"], start, start + total))
            hyp_txt = normalise(" ".join(l["text"] for l in lines))
            if ref_txt:
                wer = edits(ref_txt, hyp_txt) / len(ref_txt)

    d = info["dir"]
    tag = f"{int(start)}-{int(start + total)}_" + (f"told{speakers}" if speakers else "auto")   # one report per question asked
    result = {"url": url, "id": info["id"], "title": info["title"], "channel": info["channel"], "excerpt": [start, start + total],
              "told_speakers": speakers, "app": rows, "reference": ref_row, "turns": len(turns),
              "headcount_merged": {k: sorted(v) for k, v in merged.items()} if "headcount" in app else None,
              "transcript_lines": len(lines), "hallucinations_dropped": dropped, "wer_vs_captions": wer,
              "timing_s": {k: round(v, 1) for k, v in timing.items()}, "wall_s": round(time.time() - t_all, 1)}
    (d / f"result_{tag}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    (d / f"segments_{tag}.json").write_text(json.dumps({
        "excerpt": [start, start + total], "reference": ref, "app": app,
        "turns": [{k: t[k] for k in ("index", "start", "end", "seconds")} for t in turns]}, ensure_ascii=False))

    md = [f"# {info['title']}", "", f"{info['channel']} · excerpt {hhmm(start)}–{hhmm(start + total)} of {hhmm(dur)} · {url}", "",
          f"Told {speakers or 'nothing'} about the headcount. {len(turns)} turns gated. Wall time {result['wall_s']} s.", "", "## Speakers", ""]
    if ref is not None:
        md += ["| | speakers (≥5 s) | labels | DER vs reference | confusion | miss | false alarm |", "|---|---|---|---|---|---|---|",
               f"| reference — {ref_note} | {ref_row['speakers_found']} | {ref_row['clusters']} | — | — | — | — |"]
        for r in rows:
            md.append(f"| app, {r['variant']} | {r['speakers_found']} | {r['clusters']} | {r['der_vs_reference']*100:.1f}% | {r['confusion']*100:.1f}% | {r['miss']*100:.1f}% | {r['fa']*100:.1f}% |")
    else:
        md += ["| | speakers (≥5 s) | labels |", "|---|---|---|"]
        for r in rows:
            md.append(f"| app, {r['variant']} | {r['speakers_found']} | {r['clusters']} |")
    md += ["", "Speaking time per label (seconds):", ""]
    if ref_row:
        md.append("- reference: " + ", ".join(f"{k} {v:.0f}" for k, v in ref_row["speaking_time"].items()))
    for r in rows:
        md.append(f"- app, {r['variant']}: " + ", ".join(f"{k} {v:.0f}" for k, v in r["speaking_time"].items()))
    if result["headcount_merged"]:
        md += ["", "Headcount re-cluster: " + "; ".join(f"{k} ← {'+'.join(v)}" for k, v in result["headcount_merged"].items() if len(v) > 1 or v != {k}) ]
    if similar:
        md += ["", "Who sounds like whom, among the live labels (cosine; one real voice ran ~0.80 on a podcast, the assignment threshold is 0.65 — above ~0.75 is very likely one person, below 0.55 very likely two):", "",
               "  " + ", ".join(f"{a}~{b} {c:.2f}" for a, b, c in similar)]
    result["similar_labels"] = similar
    if suspicious:
        m = suspicious[0]
        md += ["", f"⚠ **Told {speakers}, but the audio sounds like {speakers + len(suspicious)}:** the re-cluster had to merge two voices that spoke for "
               f"{m['seconds_a']:.0f} s and {m['seconds_b']:.0f} s and did not sound alike (similarity {m['cosine']:.2f})."]
    result["suspicious_merges"] = suspicious
    if ref is not None:
        md += ["", "DER uses the optimal label mapping and a 0.25 s collar around reference boundaries."]
    if do_asr:
        md += ["", "## Transcript", "", f"{len(lines)} lines · {dropped} hallucination(s) dropped"
               + (f" · WER vs creator captions {wer*100:.1f}% (weak reference)" if wer is not None else " · no creator captions to score against"), "",
               "Labels are the headcount variant when a headcount was given; where that differs from the live auto label, the auto label follows in brackets. "
               "Read for a question and its answer under one label, or one voice split across two.", ""]
        for l in lines:
            auto_note = f" ({'/'.join(sorted(l['auto']))})" if l.get("auto") and set(l["auto"]) != {l["speaker"]} else ""
            md.append(f"- **{hhmm(l['t0'])} {l['speaker']}**{auto_note} {l['text']}")
    (d / f"report_{tag}.md").write_text("\n".join(md) + "\n")

    line = f"  app auto: {rows[0]['speakers_found']} speakers " + ", ".join(f"{v:.0f}s" for v in list(rows[0]['speaking_time'].values())[:6])
    if len(rows) > 1:
        line += f"  |  told {speakers}: {rows[1]['speakers_found']} speakers " + ", ".join(f"{v:.0f}s" for v in list(rows[1]['speaking_time'].values())[:6])
    if ref_row:
        line += f"  |  reference: {ref_row['speakers_found']} speakers; DER auto {rows[0]['der_vs_reference']*100:.1f}%" + (f", told {rows[1]['der_vs_reference']*100:.1f}%" if len(rows) > 1 else "")
    print(line)
    if similar:
        print("  who sounds like whom: " + ", ".join(f"{a}~{b} {c:.2f}" for a, b, c in similar[:6]))
    if suspicious:
        m = suspicious[0]
        print(f"  ⚠ told {speakers}, but the audio sounds like {speakers + len(suspicious)}: merged {m['seconds_a']:.0f} s + {m['seconds_b']:.0f} s at similarity {m['cosine']:.2f}")
    if do_asr:
        print(f"  transcript: {len(lines)} lines, {dropped} dropped" + (f", WER vs captions {wer*100:.1f}%" if wer is not None else ""))
    print(f"  timing: {result['timing_s']}  →  {d / f'report_{tag}.md'}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--manifest", help="JSON list of {url, speakers, start, minutes, note}")
    ap.add_argument("--speakers", type=int, default=None, help="how many people are in the video, if you know")
    ap.add_argument("--start", type=float, default=600, help="excerpt start, seconds (default 600: past the intro)")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--reference", action="store_true", help="also run the second diarizer (slow; see the docstring for why it is off)")
    ap.add_argument("--rttm", help="manual reference labels for this excerpt (absolute times)")
    ap.add_argument("--out", default=str(HERE / "backtest"))
    a = ap.parse_args()
    jobs = json.load(open(a.manifest)) if a.manifest else [{"url": u, "speakers": a.speakers, "start": a.start, "minutes": a.minutes, "asr": not a.no_asr, "reference": a.reference} for u in a.urls]
    if not jobs:
        ap.print_help(); return 2
    results = []
    for j in jobs:
        results.append(run_one(j["url"], j.get("speakers"), j.get("start", a.start), j.get("minutes", a.minutes),
                               j.get("asr", not a.no_asr), Path(a.out), j.get("rttm") or a.rttm, j.get("reference", a.reference)))
    if len(results) > 1:
        print("\n" + f"{'video':<46} {'told':>4} {'auto found':>10} {'told found':>10}  speaking time (told, else auto)")
        for r in results:
            auto = r["app"][0]; told = r["app"][1] if len(r["app"]) > 1 else None
            st = (told or auto)["speaking_time"]
            print(f"{r['title'][:46]:<46} {str(r['told_speakers'] or '-'):>4} {auto['speakers_found']:>10} {(told['speakers_found'] if told else '-'):>10}  "
                  + ", ".join(f"{v:.0f}" for v in list(st.values())[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
