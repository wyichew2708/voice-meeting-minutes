"""Run the design's clustering across speaker counts and report."""
from __future__ import annotations

import sys

import numpy as np

from clustering import Config, OnlineClusterer
from meeting import embed_all, score, turns
from speakers import Room

SEEDS = range(8)
MINUTES = 45.0


def trial(n: int, cfg: Config, seed: int) -> dict:
    room = Room(n, seed)
    segs = turns(n, MINUTES, seed)
    embs = embed_all(room, segs)
    cl = OnlineClusterer(cfg)
    for s, e in zip(segs, embs):
        cl.add(s.index, e, s.seconds)
    cl.finish()
    r = score(segs, cl.assignment)
    r["segments"] = len(segs)
    return r


def sweep(cfg: Config, counts=(2, 4, 6, 8, 10, 12)) -> list[dict]:
    rows = []
    for n in counts:
        rs = [trial(n, cfg, s) for s in SEEDS]
        rows.append({
            "n": n,
            "segments": int(np.mean([r["segments"] for r in rs])),
            "clusters": float(np.mean([r["clusters"] for r in rs])),
            "confusion": float(np.mean([r["confusion"] for r in rs])),
            "confusion_worst": float(np.max([r["confusion"] for r in rs])),
            "missed": float(np.mean([r["speakers_missed"] for r in rs])),
            "over_split": float(np.mean([r["over_split"] for r in rs])),
        })
    return rows


def table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'people':>6} {'segs':>5} {'clusters':>9} {'confusion':>10} "
          f"{'worst':>7} {'missed':>7} {'oversplit':>10}")
    for r in rows:
        print(f"{r['n']:>6} {r['segments']:>5} {r['clusters']:>9.1f} "
              f"{r['confusion']*100:>9.1f}% {r['confusion_worst']*100:>6.1f}% "
              f"{r['missed']:>7.1f} {r['over_split']:>10.1f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "v1"
    if which == "v1":
        table("v1 — design as written (threshold 0.70, no margin, no recluster)",
              sweep(Config(threshold=0.70, min_centroid_seconds=1.5)))


def window(n: int, cfg_fn, lo=0.30, hi=0.80, step=0.01,
           max_confusion=0.02, max_extra_clusters=0.5) -> tuple:
    """The band of thresholds that give a usable transcript at n speakers.

    Usable means both: under `max_confusion` of the meeting attributed to the
    wrong person, and no more than `max_extra_clusters` spurious labels per
    real speaker. Outside the band the transcript is either a smear of merged
    voices or a confetti of one-line speakers.
    """
    ok = []
    t = lo
    while t <= hi + 1e-9:
        r = sweep(cfg_fn(t), counts=(n,))[0]
        if r["confusion"] <= max_confusion and \
           (r["clusters"] - n) <= max_extra_clusters * n:
            ok.append(round(t, 2))
        t += step
    return (min(ok), max(ok), len(ok)) if ok else (None, None, 0)
