"""Embed on a rolling window, not on the ASR segment.

STRESS 3 showed the clustering collapsing when every segment is short — 129
clusters for ten people once nothing runs past 2 s. That is not a hypothetical
meeting: ten people interrupt each other far more than four do, and the
segmenter's own turn-change cut makes segments shorter still.

The cause is that the design uses one window for two jobs. Transcription wants
a tight segment — cut at the silence, cut at the speaker change. Speaker
identity wants as much of one voice as it can get. Those are different needs
and they do not have to share a window.

So: transcribe the segment, but embed the *speaker's last few seconds of
continuous speech*, spanning as many adjacent segments as the turn contains.
Costs one extra buffer and nothing on the GPU — the sidecar is already being
called once per segment either way.
"""
from __future__ import annotations

import numpy as np

from clustering import Config, OnlineClusterer
from meeting import score, turns
from speakers import Room

EMBED_WINDOW = 4.0


def runs(segs):
    """Cumulative speech in the current uninterrupted turn, per segment.

    A real implementation reads this off the audio buffer: consecutive
    segments with no other speech between them are one run. Here the
    simulator knows who spoke, which is the same information.
    """
    out, acc, prev = [], 0.0, None
    for s in segs:
        acc = s.seconds if s.speaker != prev else acc + s.seconds
        prev = s.speaker
        out.append(min(EMBED_WINDOW, acc))
    return out


def run(n, minutes, seed, cfg, cap=None, windowed=True):
    room = Room(n, seed)
    segs = turns(n, minutes, seed)
    if cap:
        for s in segs:
            s.seconds = min(s.seconds, cap)
    widths = runs(segs) if windowed else [s.seconds for s in segs]
    cl = OnlineClusterer(cfg)
    for s, w in zip(segs, widths):
        # The embedding is drawn at the window's width; the segment's own
        # duration still governs how much it is allowed to teach a centroid.
        cl.add(s.index, room.embed(s.speaker, w), s.seconds)
    cl.finish()
    return score(segs, cl.assignment)


if __name__ == "__main__":
    cfg = Config(threshold=0.60, min_centroid_seconds=1.5, margin=0.06,
                 defer_under_seconds=1.5, recluster_every=100,
                 recluster_threshold=0.72)
    seeds = range(8)

    print("STRESS 3 revisited — 10 people, every segment capped short")
    print(f"{'max seg':>8} {'segment-window':>26} {'rolling 4 s window':>26}")
    print(f"{'':>8} {'clusters':>12} {'confusion':>13} {'clusters':>12} {'confusion':>13}")
    for cap in (1.5, 2.0, 3.0, 5.0, 12.0):
        a = [run(10, 45.0, s, cfg, cap, windowed=False) for s in seeds]
        b = [run(10, 45.0, s, cfg, cap, windowed=True) for s in seeds]
        print(f"{cap:>7.1f}s "
              f"{np.mean([r['clusters'] for r in a]):>12.1f} "
              f"{np.mean([r['confusion'] for r in a])*100:>12.1f}% "
              f"{np.mean([r['clusters'] for r in b]):>12.1f} "
              f"{np.mean([r['confusion'] for r in b])*100:>12.1f}%")

    print()
    print("And across meeting sizes, worst case (2 s cap):")
    print(f"{'people':>6} {'segment-window':>16} {'rolling window':>16}")
    for n in (4, 10, 15, 20):
        a = [run(n, 45.0, s, cfg, 2.0, windowed=False) for s in seeds]
        b = [run(n, 45.0, s, cfg, 2.0, windowed=True) for s in seeds]
        print(f"{n:>6} {np.mean([r['clusters'] for r in a]):>10.1f} clusters"
              f" {np.mean([r['clusters'] for r in b]):>10.1f} clusters")
