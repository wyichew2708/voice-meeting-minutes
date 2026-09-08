"""Where does the clustering threshold stop being safe as people are added?"""
from clustering import Config
from run_clustering import window, SEEDS
import run_clustering

run_clustering.SEEDS = range(4)          # leaner: the window edges are stable
run_clustering.MINUTES = 30.0

plain = lambda t: Config(threshold=t, min_centroid_seconds=1.5)
guarded = lambda t: Config(threshold=t, min_centroid_seconds=1.5,
                           margin=0.06, defer_under_seconds=1.5,
                           recluster_every=100, recluster_threshold=0.72)

for name, fn in (("design as written", plain), ("with the §4.1 guards", guarded)):
    print(f"\nUsable threshold window — {name}")
    print(f"{'people':>6} {'from':>6} {'to':>6} {'width':>7}")
    for n in (2, 4, 6, 8, 10, 12, 15, 20):
        lo, hi, k = window(n, fn, lo=0.34, hi=0.74, step=0.02)
        w = f"{hi-lo:.2f}" if lo is not None else "NONE"
        print(f"{n:>6} {str(lo):>6} {str(hi):>6} {w:>7}", flush=True)
