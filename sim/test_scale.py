"""Regression tests for the design's scale claims. Run: python3 test_scale.py

Each test asserts a claim that `docs/scale-test.md` makes in prose. If a
threshold in the design is changed, these fail — which is the point: the
numbers in that document are not decoration, they are the reason the design
says what it says.

⚠ Synthetic embeddings (see speakers.py). These test the algorithm and the
arithmetic. They say nothing about accuracy on real speech, and no accuracy
claim about a real meeting follows from a green run here.
"""
from __future__ import annotations

import sys

import numpy as np

from clustering import Config, OnlineClusterer
from meeting import embed_all, score, turns
from speakers import Room, calibration
from stress import HardRoom, quiet_participant, run as stress_run
from windowed import run as windowed_run

SEEDS = range(6)
MINUTES = 30.0

DESIGNED = Config(threshold=0.70, min_centroid_seconds=1.5)
SHIPPING = Config(threshold=0.60, min_centroid_seconds=1.5, margin=0.06,
                  defer_under_seconds=1.5, recluster_every=100,
                  recluster_threshold=0.72)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        _failures.append(name)


def plain(n, cfg, seeds=SEEDS):
    out = []
    for s in seeds:
        room, segs = Room(n, s), turns(n, MINUTES, s)
        cl = OnlineClusterer(cfg)
        for seg, e in zip(segs, embed_all(room, segs)):
            cl.add(seg.index, e, seg.seconds)
        cl.finish()
        out.append(score(segs, cl.assignment))
    return out


def main() -> int:
    print("generator calibration")
    c = calibration(10)
    check("within-speaker cosine ~0.72",
          abs(c["within_long_mean"] - 0.72) < 0.02,
          f"{c['within_long_mean']:.3f}")
    check("between-speaker cosine ~0.32",
          abs(c["between_mean"] - 0.32) < 0.04,
          f"{c['between_mean']:.3f}")

    print("\nthe finding: threshold 0.70 without guards does not scale")
    r = plain(10, DESIGNED)
    n_cl = np.mean([x["clusters"] for x in r])
    check("10 people, designed config, over-splits badly", n_cl > 50,
          f"{n_cl:.0f} clusters for 10 speakers")

    print("\nthe fix: guarded config is exact from 2 to 12 people")
    for n in (2, 4, 8, 10, 12):
        r = plain(n, SHIPPING)
        got = np.mean([x["clusters"] for x in r])
        conf = np.mean([x["confusion"] for x in r])
        check(f"{n:>2} people -> {n} clusters",
              abs(got - n) <= 0.5 and conf < 0.02,
              f"{got:.1f} clusters, {conf*100:.1f}% confusion")

    print("\nquiet participants are still found (10 in the room)")
    for q in (3.0, 10.0):
        found = []
        for s in SEEDS:
            segs, _ = quiet_participant(10, MINUTES, s, q)
            x = stress_run(Room(10, s), segs, SHIPPING)
            found.append(x["true_speakers"] - x["speakers_missed"])
        check(f"speaker with only {q:.0f}s of speech", np.mean(found) >= 9.5,
              f"{np.mean(found):.1f}/10 speakers found")

    print("\nknown limit: two people who sound alike are merged")
    lo = [stress_run(HardRoom(10, s, 0.70), turns(10, MINUTES, s), SHIPPING)
          for s in SEEDS]
    hi = [stress_run(HardRoom(10, s, 0.80), turns(10, MINUTES, s), SHIPPING)
          for s in SEEDS]
    check("pair at cos 0.70 stays separate",
          np.mean([x["confusion"] for x in lo]) < 0.02,
          f"{np.mean([x['confusion'] for x in lo])*100:.1f}% confusion")
    check("pair at cos 0.80 merges — needs the offline pass or manual split",
          np.mean([x["confusion"] for x in hi]) > 0.05,
          f"{np.mean([x['confusion'] for x in hi])*100:.1f}% confusion")

    print("\nrolling embedding window rescues an interrupt-heavy meeting")
    a = [windowed_run(10, MINUTES, s, SHIPPING, cap=2.0, windowed=False)
         for s in SEEDS]
    b = [windowed_run(10, MINUTES, s, SHIPPING, cap=2.0, windowed=True)
         for s in SEEDS]
    check("2 s segments, embedding on the segment: collapses",
          np.mean([x["clusters"] for x in a]) > 50,
          f"{np.mean([x['clusters'] for x in a]):.0f} clusters")
    check("2 s segments, embedding on a rolling 4 s window: holds",
          abs(np.mean([x["clusters"] for x in b]) - 10) <= 1.0,
          f"{np.mean([x['clusters'] for x in b]):.1f} clusters")

    print("\ncapacity on the shared recogniser")
    import load
    one = load.capacity(1, partials=True, live_call=True)
    two = load.capacity(2, partials=True, live_call=True)
    two_off = load.capacity(2, partials=False, live_call=True)
    check("1 meeting + partials + a live call fits",
          one["utilisation"] < 0.7, f"{one['utilisation']*100:.0f}% utilisation")
    check("2 meetings + partials does NOT fit",
          two["utilisation"] >= 1.0, f"{two['utilisation']*100:.0f}% utilisation")
    check("2 meetings finals-only fits",
          two_off["utilisation"] < 0.7,
          f"{two_off['utilisation']*100:.0f}% utilisation")

    print("\nminutes fit the context window at 10 people x 60 min")
    v = load.transcript_volume(10, 60.0)
    b2 = load.minutes_budget(v["tokens"])
    check("map-reduce reduce step fits", b2["fits_context"],
          f"{b2['reduce_total_tokens']} tokens over {b2['chunks']} chunks")

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
