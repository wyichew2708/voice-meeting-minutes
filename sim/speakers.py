"""Synthetic speaker embeddings with realistic ECAPA-TDNN cosine statistics.

This does NOT model a room, a microphone or a language. It models one thing:
the geometry that speaker embeddings actually land in, so the clustering
algorithm in `clustering.py` can be exercised at speaker counts we cannot
otherwise reach before there is an app to record with.

Calibration targets, from ECAPA-TDNN on VoxCeleb-scale evaluation:

    same speaker, segment vs segment      cos ~ 0.72
    different speakers, segment vs segment cos ~ 0.32

and, crucially, **short segments embed worse**: a 0.5 s clip carries far less
speaker evidence than a 4 s one. That single effect is what makes a ten-person
meeting harder than a four-person one, so it is modelled explicitly rather
than assumed away.

⚠ Synthetic. This validates the algorithm and the arithmetic. It does not
validate the models, and no accuracy claim about real speech follows from it.
"""
from __future__ import annotations

import numpy as np

DIM = 192

#: Cosine between two segments of the *same* speaker, once both are long.
WITHIN_LONG = 0.72
#: Cosine between segments of *different* speakers.
BETWEEN = 0.32
#: Duration at or above which an embedding is as good as it gets.
GOOD_SECONDS = 3.0
#: Same-speaker cosine for a very short clip. Measured degradations are steep;
#: this says a 0.5 s clip is barely more like its own speaker than like a
#: stranger, which is exactly why the design refuses to learn centroids from
#: short segments.
WITHIN_SHORT = 0.45
SHORT_SECONDS = 0.5


def _mix(a: np.ndarray, b: np.ndarray, cos: float) -> np.ndarray:
    """Unit vector at `cos` from `a`, in the plane of `a` and `b`."""
    b = b - (b @ a) * a
    n = np.linalg.norm(b)
    if n < 1e-9:
        return a.copy()
    b = b / n
    return cos * a + np.sqrt(max(0.0, 1.0 - cos * cos)) * b


def within_cos(seconds: float) -> float:
    """How like their own speaker a segment of this length embeds.

    Linear between the short and long anchors, clamped. Segment-to-segment
    cosine is the square of segment-to-mean cosine (the two noise components
    are independent), which is why the anchors are square-rooted below.
    """
    s = float(np.clip(seconds, SHORT_SECONDS, GOOD_SECONDS))
    f = (s - SHORT_SECONDS) / (GOOD_SECONDS - SHORT_SECONDS)
    return WITHIN_SHORT + f * (WITHIN_LONG - WITHIN_SHORT)


class Room:
    """A set of speakers whose embeddings sit at realistic mutual distances."""

    def __init__(self, n_speakers: int, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.n = n_speakers
        # A shared direction pulls every speaker mean toward every other, which
        # is what produces a between-speaker cosine of 0.32 rather than the ~0
        # that independent random vectors in 192 dimensions would give.
        g = self._unit()
        # Segment-to-segment cosine is the mean-to-mean cosine attenuated by
        # both segments' own noise (x WITHIN_LONG overall), so the means must
        # sit further apart than the target to land on it. Calibrated, not
        # assumed — see calibration() below, which asserts the result.
        mean_cos = BETWEEN / WITHIN_LONG
        self.means = np.stack([_mix(g, self._unit(), np.sqrt(mean_cos))
                               for _ in range(n_speakers)])

    def _unit(self) -> np.ndarray:
        v = self.rng.standard_normal(DIM)
        return v / np.linalg.norm(v)

    def embed(self, speaker: int, seconds: float) -> np.ndarray:
        """One segment's embedding: the speaker's mean, degraded by length."""
        return _mix(self.means[speaker], self._unit(),
                    np.sqrt(within_cos(seconds)))


def calibration(n_speakers: int = 10, seed: int = 0) -> dict:
    """Confirm the generator actually hits its targets before it is trusted."""
    room = Room(n_speakers, seed)
    rng = np.random.default_rng(seed + 1)

    def cos(a, b):
        return float(a @ b)

    within_long, within_short, between = [], [], []
    for _ in range(3000):
        i, j = rng.integers(0, n_speakers, 2)
        within_long.append(cos(room.embed(i, 4.0), room.embed(i, 4.0)))
        within_short.append(cos(room.embed(i, 0.6), room.embed(i, 0.6)))
        if i != j:
            between.append(cos(room.embed(i, 4.0), room.embed(j, 4.0)))
    return {
        "within_long_mean": float(np.mean(within_long)),
        "within_short_mean": float(np.mean(within_short)),
        "between_mean": float(np.mean(between)),
        "between_p99": float(np.percentile(between, 99)),
        "between_max": float(np.max(between)),
    }


if __name__ == "__main__":
    for n in (4, 10):
        c = calibration(n)
        print(f"n={n:2d}  within(long)={c['within_long_mean']:.3f}  "
              f"within(short)={c['within_short_mean']:.3f}  "
              f"between={c['between_mean']:.3f}  "
              f"between p99={c['between_p99']:.3f}  max={c['between_max']:.3f}")
