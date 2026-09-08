"""The cases that actually break diarization in a ten-person room.

The clean sweep says the algorithm scales. These say what it costs when the
room is not clean — which is the only condition that matters in a boardroom.
"""
from __future__ import annotations

import numpy as np

from clustering import Config, OnlineClusterer
from meeting import embed_all, score, turns
from speakers import BETWEEN, DIM, WITHIN_LONG, Room, _mix

V2 = dict(threshold=0.60, min_centroid_seconds=1.5, margin=0.06,
          defer_under_seconds=1.5, recluster_every=100, recluster_threshold=0.72)


class HardRoom(Room):
    """A room where two people genuinely sound alike.

    Siblings, or two colleagues with the same accent, register and pitch. The
    generic between-speaker distance says nothing about this pair — they sit
    at `pair_cos`, far closer than the 0.44 the rest of the room sits at.
    """

    def __init__(self, n, seed=0, pair_cos=0.75):
        super().__init__(n, seed)
        # Speaker 1 is moved to sit `pair_cos` from speaker 0.
        self.means[1] = _mix(self.means[0], self.means[1], pair_cos)


class QuietRoom(Room):
    """Same voices; the *participation* is what is skewed, in turns()."""


def run(room, segs, cfg: Config) -> dict:
    embs = embed_all(room, segs)
    cl = OnlineClusterer(cfg)
    for s, e in zip(segs, embs):
        cl.add(s.index, e, s.seconds)
    cl.finish()
    return score(segs, cl.assignment)


def quiet_participant(n, minutes, seed, quiet_seconds):
    """One person says almost nothing — the hardest speaker to ever name."""
    segs = turns(n, minutes, seed)
    # Strip speaker n-1 down to roughly `quiet_seconds` of speech.
    theirs = [s for s in segs if s.speaker == n - 1]
    keep, total = set(), 0.0
    for s in theirs:
        if total >= quiet_seconds:
            break
        keep.add(s.index)
        total += s.seconds
    out = [s for s in segs if s.speaker != n - 1 or s.index in keep]
    for i, s in enumerate(out):
        s.index = i
    return out, total


if __name__ == "__main__":
    cfg = Config(**V2)
    seeds = range(8)

    print("=" * 74)
    print("STRESS 1 — two people who genuinely sound alike (10 in the room)")
    print("=" * 74)
    print(f"{'pair cos':>9} {'clusters':>9} {'confusion':>10} {'worst':>7} {'missed':>7}")
    for pc in (0.44, 0.60, 0.70, 0.75, 0.80, 0.85):
        rs = [run(HardRoom(10, s, pc), turns(10, 45.0, s), cfg) for s in seeds]
        print(f"{pc:>9.2f} {np.mean([r['clusters'] for r in rs]):>9.1f} "
              f"{np.mean([r['confusion'] for r in rs])*100:>9.1f}% "
              f"{np.max([r['confusion'] for r in rs])*100:>6.1f}% "
              f"{np.mean([r['speakers_missed'] for r in rs]):>7.1f}")

    print()
    print("=" * 74)
    print("STRESS 2 — the person who barely speaks (10 in the room)")
    print("=" * 74)
    print(f"{'their speech':>13} {'clusters':>9} {'found?':>8} {'confusion':>10}")
    for q in (3.0, 6.0, 10.0, 20.0, 60.0):
        cl_, found_, conf_ = [], [], []
        for s in seeds:
            segs, actual = quiet_participant(10, 45.0, s, q)
            r = run(Room(10, s), segs, cfg)
            cl_.append(r["clusters"])
            found_.append(r["true_speakers"] - r["speakers_missed"])
            conf_.append(r["confusion"])
        print(f"{q:>11.0f}s {np.mean(cl_):>9.1f} "
              f"{np.mean(found_):>7.1f}/10 {np.mean(conf_)*100:>9.1f}%")

    print()
    print("=" * 74)
    print("STRESS 3 — every segment is short (a fast, interrupt-heavy meeting)")
    print("=" * 74)
    print(f"{'max seg':>8} {'clusters':>9} {'confusion':>10} {'worst':>7}")
    for cap in (2.0, 3.0, 5.0, 12.0):
        rs = []
        for s in seeds:
            segs = turns(10, 45.0, s)
            for x in segs:
                x.seconds = min(x.seconds, cap)
            rs.append(run(Room(10, s), segs, cfg))
        print(f"{cap:>7.0f}s {np.mean([r['clusters'] for r in rs]):>9.1f} "
              f"{np.mean([r['confusion'] for r in rs])*100:>9.1f}% "
              f"{np.max([r['confusion'] for r in rs])*100:>6.1f}%")
