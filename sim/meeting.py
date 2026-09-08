"""A meeting's turn structure, and how to score a clustering of it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from speakers import Room


@dataclass
class Segment:
    index: int
    speaker: int
    seconds: float


def turns(n_speakers: int, minutes: float, seed: int = 0) -> list[Segment]:
    """Segments for a meeting, with the shape real meetings actually have.

    Three things matter for clustering and all three are modelled:

    * **Unequal participation.** A ten-person meeting is not ten equal voices.
      A chair and two or three main contributors carry most of it while
      several people say almost nothing — and those quiet people are exactly
      the ones a clusterer has least evidence about.
    * **Backchannels.** "Mm", "yeah", "right" are a third of all segments and
      almost all of them are under 1.5 s.
    * **Stickiness.** Whoever just spoke is the most likely next speaker.
    """
    rng = np.random.default_rng(seed)
    # Zipf-ish speaking share: speaker 0 talks ~4x as much as speaker 9.
    share = 1.0 / (1.0 + np.arange(n_speakers) * 0.35)
    share = share / share.sum()

    out: list[Segment] = []
    total = minutes * 60.0
    t = 0.0
    current = 0
    i = 0
    while t < total:
        if rng.random() < 0.45 and out:
            speaker = current                      # same person continues
        else:
            speaker = int(rng.choice(n_speakers, p=share))
        current = speaker

        r = rng.random()
        if r < 0.32:
            secs = float(rng.uniform(0.4, 1.5))    # backchannel
        elif r < 0.87:
            secs = float(rng.uniform(1.5, 8.0))    # ordinary turn
        else:
            secs = float(rng.uniform(8.0, 12.0))   # monologue, at the cap
        out.append(Segment(i, speaker, secs))
        t += secs + 0.6                            # + the endpointer's silence
        i += 1
    return out


def embed_all(room: Room, segs: list[Segment]) -> list[np.ndarray]:
    return [room.embed(s.speaker, s.seconds) for s in segs]


def score(segs: list[Segment], assignment: dict[int, int]) -> dict:
    """Speaker-confusion, weighted by seconds — DER without VAD or overlap.

    Each cluster is credited to the true speaker holding most of its seconds;
    every other second in it is confused. This is the number a reader of the
    transcript feels: the fraction of the meeting attributed to the wrong name.
    """
    by_cluster: dict[int, dict[int, float]] = {}
    total = 0.0
    for s in segs:
        cid = assignment.get(s.index)
        if cid is None:
            continue
        by_cluster.setdefault(cid, {}).setdefault(s.speaker, 0.0)
        by_cluster[cid][s.speaker] += s.seconds
        total += s.seconds

    correct = sum(max(d.values()) for d in by_cluster.values())
    true_speakers = len({s.speaker for s in segs})
    # A speaker is "found" if some cluster is majority-theirs.
    owned = {max(d, key=d.get) for d in by_cluster.values()}
    return {
        "clusters": len(by_cluster),
        "true_speakers": true_speakers,
        "confusion": max(0.0, 1.0 - correct / total) if total else 0.0,
        "speakers_found": len(owned),
        "speakers_missed": true_speakers - len(owned),
        "over_split": max(0, len(by_cluster) - true_speakers),
    }
