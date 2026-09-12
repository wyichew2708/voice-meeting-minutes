#!/usr/bin/env python3
"""Reference WER for the Singlish Whisper on a SASRBench-v1 sample, fp32 PyTorch.

The browser runs the same model quantised to q8 through transformers.js. This
is the number that pipeline should reproduce — the role campp_ref.json plays
for the speaker model — and a real measurement of question two: does the app
understand Singlish. Thirty clips of spontaneous conversation, 7.5 minutes,
1,328 words; not the full 3,747-clip benchmark, so treat it as a sanity check
against the model card's 18.49%, not a replication of it.

    python3 sim/singlish_wer_reference.py [--clips web/models/_test/sasr]
                                          [--model mjwong/whisper-small-singlish]

The clips come from the Hugging Face datasets-server, thirty rows of
mjwong/SASRBench-v1, saved next to an index.json of transcripts. They are not
committed. Transcript conventions: (ppl) (uh) are non-lexical and dropped,
!huh! is an interjection and dropped, [ah] [lah] are the particles that make
Singlish Singlish and are kept, b_a is a spelled acronym.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from pathlib import Path

import numpy as np


def normalise(t: str) -> list[str]:
    t = t.lower()
    t = re.sub(r"\([^)]*\)", " ", t)          # (ppl) (uh) (laugh)
    t = re.sub(r"![^!]*!", " ", t)            # !huh!
    t = t.replace("_", " ")                   # b_a -> b a
    t = re.sub(r"[\[\]]", "", t)              # [ah] -> ah, kept
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def edits(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="web/models/_test/sasr")
    ap.add_argument("--model", default="mjwong/whisper-small-singlish")
    a = ap.parse_args()
    clips = Path(a.clips)
    index = json.load(open(clips / "index.json"))

    from transformers import pipeline
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=a.model, device="cpu", chunk_length_s=30)
    print(f"loaded {a.model} in {time.time() - t0:.0f}s")

    out, err_total, words_total, secs_total, t_infer = {}, 0, 0, 0.0, 0.0
    for m in index:
        with wave.open(str(clips / m["file"])) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
            secs = w.getnframes() / w.getframerate()
        t1 = time.time()
        hyp = asr({"raw": x, "sampling_rate": 16000}, generate_kwargs={"language": "en", "task": "transcribe"})["text"]
        t_infer += time.time() - t1
        ref_w, hyp_w = normalise(m["transcript"]), normalise(hyp)
        e = edits(ref_w, hyp_w)
        err_total += e; words_total += len(ref_w); secs_total += secs
        out[m["file"]] = {"hyp": hyp, "wer": round(e / max(1, len(ref_w)), 3), "ref_words": len(ref_w)}
        print(f"  {m['file']}  {secs:5.1f}s  WER {e / max(1, len(ref_w)) * 100:5.1f}%   {hyp.strip()[:70]}")

    wer = err_total / max(1, words_total)
    print(f"\n{a.model}: WER {wer * 100:.1f}% over {words_total} words, {len(index)} clips, {secs_total / 60:.1f} min")
    print(f"model card, full SASRBench-v1: 18.49%   (vanilla whisper-small: 147.80%)")
    print(f"CPU fp32: {t_infer / secs_total:.2f}x real time")
    json.dump({"model": a.model, "wer": wer, "words": words_total, "clips": out},
              open(clips / "reference_hyps.json", "w"), ensure_ascii=False, indent=1)
    print(f"wrote {clips}/reference_hyps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
