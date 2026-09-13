#!/usr/bin/env python3
"""The Python port of the known-headcount re-clustering repairs both
directions, like its JS original (sim/verify_js_port.mjs). python3 sim/verify_headcount.py"""
import numpy as np
from clustering import Config, OnlineClusterer
from meeting import score, turns
from speakers import Room

DESIGNED = Config(threshold=0.70, min_centroid_seconds=1.5)
SHIPPING = Config(threshold=0.60, min_centroid_seconds=1.5, margin=0.06, defer_under_seconds=1.5,
                  recluster_every=100, recluster_threshold=0.72)
UNDER = Config(threshold=0.30, min_centroid_seconds=1.5, margin=0.06, defer_under_seconds=1.5,
               recluster_every=100, recluster_threshold=0.72)


def run(n, cfg, target, seed):
    room, segs = Room(n, seed), turns(n, 45.0, seed)
    cl = OnlineClusterer(cfg)
    for s in segs:
        cl.add(s.index, room.embed(s.speaker, s.seconds), s.seconds)
    cl.finish()
    auto = score(segs, cl.assignment)
    cl.recluster_to(target)
    return auto, score(segs, cl.assignment), len(cl.clusters)


bad = 0
def check(name, ok, detail):
    global bad
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

seeds = range(5)
over = [run(10, DESIGNED, 10, s) for s in seeds]
under = [run(10, UNDER, 10, s) for s in seeds]
exact = [run(12, SHIPPING, 12, s) for s in seeds]
fewer = [run(10, SHIPPING, 6, s) for s in seeds]
m = lambda rs, f: float(np.mean([f(r) for r in rs]))  # noqa: E731
check("over-split 10 -> told 10", m(over, lambda r: r[2]) == 10 and m(over, lambda r: r[1]['confusion']) < 0.05,
      f"{m(over, lambda r: r[0]['clusters']):.0f} clusters auto -> {m(over, lambda r: r[2]):.0f}, {m(over, lambda r: r[1]['confusion'])*100:.1f}% confusion")
check("under-split 10 -> told 10", m(under, lambda r: r[2]) == 10 and m(under, lambda r: r[1]['confusion']) < 0.15,
      f"{m(under, lambda r: r[0]['clusters']):.1f} clusters auto ({m(under, lambda r: r[0]['confusion'])*100:.0f}%) -> {m(under, lambda r: r[2]):.0f}, {m(under, lambda r: r[1]['confusion'])*100:.1f}% confusion")
check("exact 12 -> told 12 stays exact", m(exact, lambda r: r[2]) == 12 and m(exact, lambda r: r[1]['confusion']) < 0.02,
      f"{m(exact, lambda r: r[2]):.0f} clusters, {m(exact, lambda r: r[1]['confusion'])*100:.1f}% confusion")
check("10 people -> told 6 gives 6", m(fewer, lambda r: r[2]) == 6, f"{m(fewer, lambda r: r[2]):.0f} clusters")

def blips(seed):
    room, segs = Room(10, seed), turns(10, 45.0, seed)
    rng = np.random.default_rng(seed + 77)
    cl = OnlineClusterer(SHIPPING)
    for s in segs:
        cl.add(s.index, room.embed(s.speaker, s.seconds), s.seconds)
    for b in range(5):                       # laughs, stings: random embeddings, 1.6 s each
        v = rng.standard_normal(192); cl.add(10000 + b, v / np.linalg.norm(v), 1.6)
    cl.finish()
    before = len(cl.clusters)
    cl.recluster_to(10)
    sc = score(segs, cl.assignment)
    return before, len(cl.clusters), sc["speakers_found"], sc["confusion"]
def under(seed):
    room, segs = Room(4, seed), turns(4, 45.0, seed)
    cl = OnlineClusterer(SHIPPING)
    for s in segs:
        cl.add(s.index, room.embed(s.speaker, s.seconds), s.seconds)
    cl.finish(); cl.recluster_to(3)
    return len(cl.last_recluster["suspicious"])
un = [under(s) for s in seeds]
check("4 people told 3 -> the forced merge is flagged", all(n >= 1 for n in un), f"{un} suspicious merges flagged")
bl = [blips(s) for s in seeds]
check("10 people + 5 noise blips -> told 10 keeps all 10 people",
      m(bl, lambda r: r[1]) == 10 and m(bl, lambda r: r[2]) == 10 and m(bl, lambda r: r[3]) < 0.02,
      f"{m(bl, lambda r: r[0]):.1f} clusters before -> {m(bl, lambda r: r[1]):.0f}, {m(bl, lambda r: r[2]):.0f} people found, {m(bl, lambda r: r[3])*100:.1f}% confusion")
print("\n" + ("%d wrong" % bad if bad else "all checks passed"))
raise SystemExit(1 if bad else 0)
