"""The online speaker clustering from solution-design.md §4.1, runnable.

Two variants:

  v1 — the design exactly as first written: assign to the nearest centroid
       above a fixed cosine threshold, else open a new cluster. Segments
       shorter than `min_centroid_seconds` are assigned but do not update a
       centroid.

  v2 — the same, plus the three guards §4.1 turned out to need at ten
       speakers: a margin test against the runner-up, deferral of segments
       too short to decide on, and a periodic agglomerative re-clustering
       pass over the embeddings already banked.

Both operate on embeddings, not audio, so they are exactly the code a real
implementation would carry — the only synthetic part is where the embeddings
come from.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    id: int
    centroid: np.ndarray
    seconds: float = 0.0
    n: int = 0
    #: Every embedding banked against this cluster, kept for the re-clustering
    #: pass. A 60-minute meeting is a few thousand 192-float vectors — under
    #: 2 MB, so there is no reason to throw them away.
    bank: list = field(default_factory=list)


@dataclass
class Config:
    threshold: float = 0.70          # design §4.1: same-cluster cosine
    min_centroid_seconds: float = 1.5
    # --- v2 only -----------------------------------------------------------
    margin: float = 0.0              # nearest must beat runner-up by this much
    defer_under_seconds: float = 0.0 # below this, hold rather than guess
    recluster_every: int = 0         # segments between agglomerative passes
    recluster_threshold: float = 0.0 # merge two clusters below this distance


class OnlineClusterer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.clusters: list[Cluster] = []
        self._next = 0
        self._since_recluster = 0
        #: Segments held back because they were too short to decide on. They
        #: are resolved on the next pass, once the centroids they must choose
        #: between have been built out of evidence that could carry the weight.
        self.deferred: list[tuple[int, np.ndarray, float]] = []
        #: seg_index -> cluster id, rewritten by re-clustering and deferral.
        self.assignment: dict[int, int] = {}

    # ------------------------------------------------------------------ core

    def _similarities(self, emb: np.ndarray) -> list[float]:
        return [float(emb @ c.centroid) for c in self.clusters]

    def _new_cluster(self, emb: np.ndarray, seconds: float) -> Cluster:
        c = Cluster(id=self._next, centroid=emb.copy())
        self._next += 1
        c.seconds, c.n = seconds, 1
        c.bank.append((emb, seconds))
        self.clusters.append(c)
        return c

    def _absorb(self, c: Cluster, emb: np.ndarray, seconds: float) -> None:
        c.bank.append((emb, seconds))
        c.n += 1
        c.seconds += seconds
        # Short segments are assigned but never learned from: their embeddings
        # are noisy, and a run of them drags a centroid onto the wrong person.
        if seconds < self.cfg.min_centroid_seconds:
            return
        w = seconds / max(1e-9, c.seconds)
        v = (1 - w) * c.centroid + w * emb
        c.centroid = v / np.linalg.norm(v)

    def add(self, index: int, emb: np.ndarray, seconds: float) -> int | None:
        """Assign one segment. Returns its cluster id, or None if deferred."""
        cfg = self.cfg

        if cfg.defer_under_seconds and seconds < cfg.defer_under_seconds \
                and self.clusters:
            self.deferred.append((index, emb, seconds))
            return None

        sims = self._similarities(emb)
        if not sims:
            c = self._new_cluster(emb, seconds)
            self.assignment[index] = c.id
            self._tick()
            return c.id

        order = np.argsort(sims)[::-1]
        best = int(order[0])
        runner = float(sims[order[1]]) if len(order) > 1 else -1.0

        ok = sims[best] >= cfg.threshold
        if ok and cfg.margin and (sims[best] - runner) < cfg.margin:
            # Above threshold but not decisively nearer than the next cluster.
            # At four speakers this almost never fires; at ten it is the
            # difference between a transcript and a mess.
            ok = False

        if ok:
            self._absorb(self.clusters[best], emb, seconds)
            cid = self.clusters[best].id
        else:
            cid = self._new_cluster(emb, seconds).id
        self.assignment[index] = cid
        self._tick()
        return cid

    # ---------------------------------------------------------- v2 additions

    def _tick(self) -> None:
        self._since_recluster += 1
        if self.cfg.recluster_every and self._since_recluster >= self.cfg.recluster_every:
            self._since_recluster = 0
            self.recluster()

    def recluster(self) -> int:
        """Agglomerative merge over banked embeddings. Returns merges made.

        Online assignment is causal: it decides with the evidence it had at
        the time and cannot revisit. This does revisit — repeatedly merging
        the closest pair of centroids while they are closer than two different
        people should ever be. It is what turns an early over-split back into
        one person.
        """
        if not self.cfg.recluster_threshold or len(self.clusters) < 2:
            return 0
        merges = 0
        while len(self.clusters) >= 2:
            best, pair = -1.0, None
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    s = float(self.clusters[i].centroid @ self.clusters[j].centroid)
                    if s > best:
                        best, pair = s, (i, j)
            if best < self.cfg.recluster_threshold or pair is None:
                break
            i, j = pair
            a, b = self.clusters[i], self.clusters[j]
            keep, drop = (a, b) if a.seconds >= b.seconds else (b, a)
            keep.bank.extend(drop.bank)
            keep.seconds += drop.seconds
            keep.n += drop.n
            self._recentre(keep)
            for k, v in self.assignment.items():
                if v == drop.id:
                    self.assignment[k] = keep.id
            self.clusters.remove(drop)
            merges += 1
        return merges

    def _recentre(self, c: Cluster) -> None:
        long = [(e, s) for e, s in c.bank if s >= self.cfg.min_centroid_seconds]
        use = long or c.bank
        w = np.array([s for _, s in use], dtype=float)
        m = (np.stack([e for e, _ in use]) * w[:, None]).sum(0) / w.sum()
        c.centroid = m / np.linalg.norm(m)

    def resolve_deferred(self) -> None:
        """Place every held-back segment against the finished centroids.

        No threshold here: by now the clusters are built, and a short segment
        has to belong to whoever it is nearest. Holding it back was about
        keeping it out of the *centroids*, not about refusing to transcribe it.
        """
        for index, emb, seconds in self.deferred:
            if not self.clusters:
                self.assignment[index] = self._new_cluster(emb, seconds).id
                continue
            sims = self._similarities(emb)
            best = int(np.argmax(sims))
            self.clusters[best].n += 1
            self.clusters[best].seconds += seconds
            self.assignment[index] = self.clusters[best].id
        self.deferred.clear()

    def finish(self) -> None:
        if self.cfg.recluster_threshold:
            self.recluster()
        self.resolve_deferred()
