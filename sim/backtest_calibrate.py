#!/usr/bin/env python3
"""Measure the embedding geometry the app actually sees on a real video, and
sweep the clusterer's thresholds over the same turns.

Real voices vary across ten minutes far more than TTS voices do — pitch,
energy, laughter — and thresholds measured on TTS need checking against that.
The catch is labels: a second diarizer turned out to be worse than the app on
the first real videos (it merged a podcast's two hosts), so calibrating
against it would have taught the app its mistakes. The label sources here, in
order of trust:

  --labels rttm       your own labels for the excerpt (--rttm file): the real thing
  --labels app-auto   the app's own live clustering, blips removed — ONLY after you have
                      read the harness's transcript and confirmed the labels follow the
                      dialogue (default, because on the first video it did)
  --labels reference  the second diarizer, for comparison

    python3 sim/backtest_calibrate.py URL --speakers 2 [--labels app-auto] [--start 600 --minutes 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from backtest_youtube import SPEAKER_MODEL, app_segments, der, fetch, found, load_excerpt, reference_diarization, speaking_time  # noqa: E402
from clustering import MIN_SPEAKER_SECONDS, Config, OnlineClusterer  # noqa: E402
from speaker_embed import CamPP  # noqa: E402


def overlap_labels(turns, segs, purity=0.8, coverage=0.5):
    out = []
    for t in turns:
        ov = {}
        for s, e, spk in segs:
            o = min(t["end"], e) - max(t["start"], s)
            if o > 0:
                ov[spk] = ov.get(spk, 0.0) + o
        tot = sum(ov.values())
        best = max(ov, key=ov.get) if ov else None
        out.append(best if best and ov[best] / tot >= purity and tot / t["seconds"] >= coverage else None)
    return out


def q(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url"); ap.add_argument("--speakers", type=int, required=True)
    ap.add_argument("--labels", choices=["app-auto", "rttm", "reference"], default="app-auto")
    ap.add_argument("--rttm"); ap.add_argument("--start", type=float, default=600); ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--out", default=str(HERE / "backtest"))
    a = ap.parse_args()

    info = fetch(a.url, Path(a.out))
    audio = load_excerpt(info["wav"], a.start, a.minutes)
    total = len(audio) / 16000
    turns, embs = app_segments(audio, CamPP())
    print(f"{info['title']}\n  excerpt {a.start:.0f}-{a.start + total:.0f} s · told {a.speakers} · {len(turns)} turns · labels: {a.labels}")

    if a.labels == "rttm":
        segs = [(float(f[3]) - a.start, float(f[3]) + float(f[4]) - a.start, f[7]) for f in (l.split() for l in Path(a.rttm).read_text().splitlines()) if f and f[0] == "SPEAKER"]
        labels = overlap_labels(turns, segs)
    elif a.labels == "reference":
        segs = reference_diarization(audio, a.speakers)
        labels = overlap_labels(turns, segs)
    else:
        cl = OnlineClusterer(SPEAKER_MODEL)
        for t, e in zip(turns, embs):
            cl.add(t["index"], e, t["seconds"])
        cl.finish()
        segs = [(t["start"], t["end"], f"S{cl.assignment[t['index']]}") for t in turns]
        keep = {k for k, v in speaking_time(segs).items() if v >= MIN_SPEAKER_SECONDS}
        labels = [lab if lab in keep else None for _, _, lab in segs]
        print("  ⚠ app-auto labels: valid only if the harness transcript showed them following the dialogue")
    print("  label speaking time:", {k: round(v) for k, v in speaking_time([(t['start'], t['end'], l) for t, l in zip(turns, labels) if l]).items()})

    long = [(e, l) for e, l, t in zip(embs, labels, turns) if l and t["seconds"] >= 1.5]
    short = [(e, l) for e, l, t in zip(embs, labels, turns) if l and t["seconds"] < 1.5]
    within, between, within_short = [], [], []
    for i in range(len(long)):
        for j in range(i + 1, len(long)):
            (within if long[i][1] == long[j][1] else between).append(float(long[i][0] @ long[j][0]))
    for e, l in short:
        within_short += [float(e @ e2) for e2, l2 in long if l2 == l]
    print("\n  embedding geometry on this video (turns >= 1.5 s)")
    print(f"    within-speaker  mean {np.mean(within):.3f}  p05 {q(within, 5):.3f}  p25 {q(within, 25):.3f}   (n={len(within)})")
    print(f"    between-speaker mean {np.mean(between):.3f}  p95 {q(between, 95):.3f}  max {q(between, 100):.3f}   (n={len(between)})" if between else "    between-speaker: only one labelled speaker")
    if within_short:
        print(f"    short clips (<1.5 s) vs their own speaker: mean {np.mean(within_short):.3f}  (n={len(within_short)})")
    print("    TTS corpus, for comparison: within 0.744 (p05 0.434)   between 0.166 (p95 0.401)   short 0.589")

    lab_segs = [(t["start"], t["end"], l) for t, l in zip(turns, labels) if l]
    print(f"\n  sweep — speakers found (>= {MIN_SPEAKER_SECONDS:.0f} s), then agreement with the labels; margin 0.06, recluster every 100")
    print(f"  {'thresh':>6} {'defer':>5} {'recl':>5} | {'auto: spk':>9} {'DER':>7} | told {a.speakers}: {'spk':>3} {'DER':>7}")
    for th in (0.50, 0.55, 0.60, 0.65, 0.70):
        for defer in (1.5, 2.0):
            for recl in (0.0, 0.75, 0.80, 0.85):
                cfg = Config(threshold=th, min_centroid_seconds=1.5, margin=0.06, defer_under_seconds=defer,
                             recluster_every=100 if recl else 0, recluster_threshold=recl)
                cl = OnlineClusterer(cfg)
                for t, e in zip(turns, embs):
                    cl.add(t["index"], e, t["seconds"])
                cl.finish()
                s1 = [(t["start"], t["end"], f"S{cl.assignment[t['index']]}") for t in turns]
                cl.recluster_to(a.speakers)
                s2 = [(t["start"], t["end"], f"S{cl.assignment[t['index']]}") for t in turns]
                cur = " <- SPEAKER_MODEL" if (th, defer, recl) == (SPEAKER_MODEL.threshold, SPEAKER_MODEL.defer_under_seconds, SPEAKER_MODEL.recluster_threshold) else ""
                print(f"  {th:6.2f} {defer:5.1f} {recl or '-':>5} | {found(s1):9d} {der(lab_segs, s1, total)['der']*100:6.1f}% |        {found(s2):3d} {der(lab_segs, s2, total)['der']*100:6.1f}%{cur}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
