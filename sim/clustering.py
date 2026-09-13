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


#: A label with less speech than this is a blip, not a person holding a seat.
MIN_SPEAKER_SECONDS = 5.0


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

    def _new_cluster(self, emb: np.ndarray, seconds: float, index: int = -1) -> Cluster:
        c = Cluster(id=self._next, centroid=emb.copy())
        self._next += 1
        c.seconds, c.n = seconds, 1
        c.bank.append((emb, seconds, index))
        self.clusters.append(c)
        return c

    def _absorb(self, c: Cluster, emb: np.ndarray, seconds: float, index: int = -1) -> None:
        c.bank.append((emb, seconds, index))
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
            c = self._new_cluster(emb, seconds, index)
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
            self._absorb(self.clusters[best], emb, seconds, index)
            cid = self.clusters[best].id
        else:
            cid = self._new_cluster(emb, seconds, index).id
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
        c.centroid = self._mean_of(c.bank)

    def _mean_of(self, bank: list) -> np.ndarray:
        """Duration-weighted mean of a bank, over its long segments when it has any."""
        long = [x for x in bank if x[1] >= self.cfg.min_centroid_seconds]
        use = long or bank
        w = np.array([x[1] for x in use], dtype=float)
        m = (np.stack([x[0] for x in use]) * w[:, None]).sum(0) / w.sum()
        return m / np.linalg.norm(m)

    def resolve_deferred(self) -> None:
        """Place every held-back segment against the finished centroids.

        No threshold here: by now the clusters are built, and a short segment
        has to belong to whoever it is nearest. Holding it back was about
        keeping it out of the *centroids*, not about refusing to transcribe it.
        """
        for index, emb, seconds in self.deferred:
            if not self.clusters:
                self.assignment[index] = self._new_cluster(emb, seconds, index).id
                continue
            sims = self._similarities(emb)
            best = int(np.argmax(sims))
            c = self.clusters[best]
            c.n += 1
            c.seconds += seconds
            c.bank.append((emb, seconds, index))   # banked so a re-cluster can move it; too short to recentre on
            self.assignment[index] = c.id
        self.deferred.clear()

    # ------------------------------------------------------- known headcount

    def nearest(self, emb: np.ndarray) -> int | None:
        """Nearest cluster by cosine, or None if there are none yet."""
        if not self.clusters:
            return None
        return self.clusters[int(np.argmax(self._similarities(emb)))].id

    def recluster_to(self, n: int) -> int:
        """Re-cluster to exactly `n` speakers — the port of web/js/clustering.js
        reclusterTo, for when the user knows how many people are in the room.

        Merge the closest pair while there are too many, split the least
        coherent cluster in two while there are too few, then give every banked
        segment its nearest centroid. Returns how many segments changed label.
        """
        if not n or n < 1 or not self.clusters:
            return 0
        before = dict(self.assignment)
        self.last_recluster = {"forced_merges": []}
        self.resolve_deferred()
        # Blips first: a laugh must not hold a seat while two real speakers merge.
        while len(self.clusters) > n and (self._absorb_tiniest() or self._merge_closest()):
            pass
        while len(self.clusters) < n and self._split_widest():
            pass
        everything = [x for c in self.clusters for x in c.bank]
        for c in self.clusters:
            c.bank, c.n, c.seconds = [], 0, 0.0
        for e, s, i in everything:
            c = self.clusters[int(np.argmax(self._similarities(e)))]
            c.bank.append((e, s, i))
            c.n += 1
            c.seconds += s
            self.assignment[i] = c.id
        self.clusters = [c for c in self.clusters if c.bank]
        for c in self.clusters:
            self._recentre(c)
        changed = sum(1 for k, v in self.assignment.items() if before.get(k) != v)
        self.last_recluster["changed"] = changed
        # A forced merge of two voices that both spoke at length and were not
        # alike is the signature of a headcount that is too low.
        self.last_recluster["suspicious"] = [m for m in self.last_recluster["forced_merges"]
                                             if m["cosine"] < self.cfg.threshold and m["seconds_a"] >= 30 and m["seconds_b"] >= 30]
        return changed

    def _absorb_tiniest(self) -> bool:
        tiny = [c for c in self.clusters if c.seconds < MIN_SPEAKER_SECONDS]
        if not tiny or len(self.clusters) < 2:
            return False
        drop = min(tiny, key=lambda c: c.seconds)
        keep = max((c for c in self.clusters if c is not drop), key=lambda c: float(c.centroid @ drop.centroid))
        keep.bank.extend(drop.bank)
        keep.seconds += drop.seconds
        keep.n += drop.n
        self._recentre(keep)
        for k, v in self.assignment.items():
            if v == drop.id:
                self.assignment[k] = keep.id
        self.clusters.remove(drop)
        return True

    def _merge_closest(self) -> bool:
        best, pair = -np.inf, None
        for i in range(len(self.clusters)):
            for j in range(i + 1, len(self.clusters)):
                s = float(self.clusters[i].centroid @ self.clusters[j].centroid)
                if s > best:
                    best, pair = s, (i, j)
        if pair is None:
            return False
        a, b = self.clusters[pair[0]], self.clusters[pair[1]]
        keep, drop = (a, b) if a.seconds >= b.seconds else (b, a)
        if hasattr(self, "last_recluster"):
            self.last_recluster["forced_merges"].append({"seconds_a": keep.seconds, "seconds_b": drop.seconds, "cosine": best})
        keep.bank.extend(drop.bank)
        keep.seconds += drop.seconds
        keep.n += drop.n
        self._recentre(keep)
        for k, v in self.assignment.items():
            if v == drop.id:
                self.assignment[k] = keep.id
        self.clusters.remove(drop)
        return True

    def _split_widest(self) -> bool:
        """Split the cluster whose long segments agree least with its centroid,
        by 2-means on cosine seeded from its two most distant members."""
        target, worst = None, np.inf
        for c in self.clusters:
            long = [x for x in c.bank if x[1] >= self.cfg.min_centroid_seconds]
            if len(long) < 4:
                continue
            agree = float(np.mean([x[0] @ c.centroid for x in long]))
            if agree < worst:
                worst, target = agree, c
        if target is None:
            return False
        pts = [x for x in target.bank if x[1] >= self.cfg.min_centroid_seconds]
        ia, ib, far = 0, 1, np.inf
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                s = float(pts[i][0] @ pts[j][0])
                if s < far:
                    far, ia, ib = s, i, j
        ca, cb = pts[ia][0], pts[ib][0]
        A, B = [], []
        for _ in range(8):
            A, B = [], []
            for x in target.bank:
                (A if float(x[0] @ ca) >= float(x[0] @ cb) else B).append(x)
            if not A or not B:
                return False
            ca, cb = self._mean_of(A), self._mean_of(B)
        if sum(x[1] for x in A) < MIN_SPEAKER_SECONDS or sum(x[1] for x in B) < MIN_SPEAKER_SECONDS:
            return False        # a blip, not a person

        def mk(bank, centroid):
            c = Cluster(id=self._next, centroid=centroid)
            self._next += 1
            c.bank, c.n, c.seconds = bank, len(bank), float(sum(x[1] for x in bank))
            return c
        a, b = mk(A, ca), mk(B, cb)
        i = self.clusters.index(target)
        self.clusters[i:i + 1] = [a, b]
        for x in a.bank:
            self.assignment[x[2]] = a.id
        for x in b.bank:
            self.assignment[x[2]] = b.id
        return True

    def finish(self) -> None:
        if self.cfg.recluster_threshold:
            self.recluster()
        self.resolve_deferred()
