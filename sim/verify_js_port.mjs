/* Does web/js/clustering.js still behave like sim/clustering.py?
 *
 * The generator here mirrors speakers.py and meeting.py — same geometry, same
 * turn structure — but not numpy's PCG64, so segment-for-segment identity is
 * not the goal and would not be meaningful. What is checked is that the same
 * claims hold: the guarded config is exact at ten people, and the first
 * draft's config still over-splits badly. If the port drifted, these break.
 *
 *   node sim/verify_js_port.mjs
 */
import { OnlineClusterer, SHIPPING, DESIGNED, normalise } from '../web/js/clustering.js';

const DIM = 192, WITHIN_LONG = 0.72, BETWEEN = 0.32;
const GOOD_SECONDS = 3.0, WITHIN_SHORT = 0.45, SHORT_SECONDS = 0.5;

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const makeRng = (seed) => {
  const u = mulberry32(seed);
  let spare = null;
  return {
    u,
    normal() {                                  // Box-Muller
      if (spare !== null) { const s = spare; spare = null; return s; }
      let a = 0, b = 0;
      while (a === 0) a = u();
      b = u();
      const r = Math.sqrt(-2 * Math.log(a));
      spare = r * Math.sin(2 * Math.PI * b);
      return r * Math.cos(2 * Math.PI * b);
    },
    unit() {
      const v = new Float32Array(DIM);
      for (let i = 0; i < DIM; i++) v[i] = this.normal();
      return normalise(v);
    },
  };
};
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

/** Unit vector at `cos` from `a`, in the plane of `a` and `b`. */
function mix(a, b, cos) {
  const p = dot(b, a);
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = b[i] - p * a[i];
  let n = 0; for (let i = 0; i < o.length; i++) n += o[i] * o[i];
  n = Math.sqrt(n);
  if (n < 1e-9) return new Float32Array(a);
  const s = Math.sqrt(Math.max(0, 1 - cos * cos));
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = cos * a[i] + s * (o[i] / n);
  return out;
}

const withinCos = (seconds) => {
  const s = Math.min(GOOD_SECONDS, Math.max(SHORT_SECONDS, seconds));
  const f = (s - SHORT_SECONDS) / (GOOD_SECONDS - SHORT_SECONDS);
  return WITHIN_SHORT + f * (WITHIN_LONG - WITHIN_SHORT);
};

class Room {
  constructor(n, seed = 0) {
    this.rng = makeRng(seed);
    this.n = n;
    const g = this.rng.unit();
    const meanCos = BETWEEN / WITHIN_LONG;
    this.means = Array.from({ length: n }, () => mix(g, this.rng.unit(), Math.sqrt(meanCos)));
  }
  embed(speaker, seconds) {
    return mix(this.means[speaker], this.rng.unit(), Math.sqrt(withinCos(seconds)));
  }
}

function turns(n, minutes, seed = 0) {
  const rng = makeRng(seed + 991);
  const share = Array.from({ length: n }, (_, i) => 1 / (1 + i * 0.35));
  const tot = share.reduce((a, b) => a + b, 0);
  const cum = []; let acc = 0;
  for (const s of share) { acc += s / tot; cum.push(acc); }
  const pick = () => { const r = rng.u(); return cum.findIndex(c => r <= c); };

  const out = []; let t = 0, current = 0, i = 0;
  const total = minutes * 60;
  while (t < total) {
    const speaker = (rng.u() < 0.45 && out.length) ? current : pick();
    current = speaker;
    const r = rng.u();
    let secs;
    if (r < 0.32) secs = 0.4 + rng.u() * 1.1;
    else if (r < 0.87) secs = 1.5 + rng.u() * 6.5;
    else secs = 8 + rng.u() * 4;
    out.push({ index: i, speaker, seconds: secs });
    t += secs + 0.6; i += 1;
  }
  return out;
}

function score(segs, assignment) {
  const byCluster = new Map(); let total = 0;
  for (const s of segs) {
    const cid = assignment.get(s.index);
    if (cid === undefined || cid === null) continue;
    if (!byCluster.has(cid)) byCluster.set(cid, new Map());
    const d = byCluster.get(cid);
    d.set(s.speaker, (d.get(s.speaker) || 0) + s.seconds);
    total += s.seconds;
  }
  let correct = 0;
  const owned = new Set();
  for (const d of byCluster.values()) {
    let bestK = null, bestV = -1;
    for (const [k, v] of d) if (v > bestV) { bestV = v; bestK = k; }
    correct += bestV; owned.add(bestK);
  }
  const trueSpeakers = new Set(segs.map(s => s.speaker)).size;
  return {
    clusters: byCluster.size,
    confusion: total ? Math.max(0, 1 - correct / total) : 0,
    speakersFound: owned.size,
    trueSpeakers,
  };
}

const run = (n, minutes, seed, cfg) => {
  const room = new Room(n, seed);
  const segs = turns(n, minutes, seed);
  const cl = new OnlineClusterer(cfg);
  for (const s of segs) cl.add(s.index, room.embed(s.speaker, s.seconds), s.seconds);
  cl.finish();
  return score(segs, cl.assignment);
};

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const SEEDS = [0, 1, 2, 3, 4];
let failures = 0;
const check = (name, ok, detail) => {
  if (!ok) failures++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}: ${detail}`);
};

console.log('\ngenerator calibration');
{
  const room = new Room(10, 0);
  const wl = [], bt = [];
  for (let k = 0; k < 3000; k++) {
    const i = k % 10, j = (k * 7 + 3) % 10;
    wl.push(dot(room.embed(i, 4), room.embed(i, 4)));
    if (i !== j) bt.push(dot(room.embed(i, 4), room.embed(j, 4)));
  }
  check('within-speaker cosine ~0.72', Math.abs(mean(wl) - 0.72) < 0.04, mean(wl).toFixed(3));
  check('between-speaker cosine ~0.32', Math.abs(mean(bt) - 0.32) < 0.05, mean(bt).toFixed(3));
}

console.log('\nthe finding: threshold 0.70 without guards does not scale');
{
  const c = mean(SEEDS.map(s => run(10, 45, s, DESIGNED).clusters));
  check('10 people, designed config, over-splits badly', c > 40, `${c.toFixed(1)} clusters for 10 speakers`);
}

console.log('\nthe fix: guarded config is exact from 2 to 12 people');
for (const n of [2, 4, 8, 10, 12]) {
  const rs = SEEDS.map(s => run(n, 45, s, SHIPPING));
  const c = mean(rs.map(r => r.clusters)), conf = mean(rs.map(r => r.confusion));
  check(`${String(n).padStart(2)} people -> ${n} clusters`,
        Math.abs(c - n) <= 0.5 && conf < 0.05,
        `${c.toFixed(1)} clusters, ${(conf * 100).toFixed(1)}% confusion`);
}

console.log('\nknown headcount: reclusterTo(n) repairs both directions');
{
  const runTo = (n, cfg, target, seed) => {
    const room = new Room(n, seed), segs = turns(n, 45, seed);
    const cl = new OnlineClusterer(cfg);
    for (const s of segs) cl.add(s.index, room.embed(s.speaker, s.seconds), s.seconds);
    cl.finish();
    const auto = score(segs, cl.assignment);
    cl.reclusterTo(target);
    return { auto, fixed: score(segs, cl.assignment), clusters: cl.clusters.length };
  };
  const over = SEEDS.map(s => runTo(10, DESIGNED, 10, s));            // ~128 clusters -> merge down
  const under = SEEDS.map(s => runTo(10, { ...SHIPPING, threshold: 0.30 }, 10, s));  // ~1 cluster -> split up
  const exact = SEEDS.map(s => runTo(12, SHIPPING, 12, s));           // already right -> stays right
  const fewer = SEEDS.map(s => runTo(10, SHIPPING, 6, s));            // user says 6 -> 6
  const m = (xs, f) => mean(xs.map(f));
  check('over-split 10 -> told 10', m(over, r => r.clusters) === 10 && m(over, r => r.fixed.confusion) < 0.05,
        `${m(over, r => r.auto.clusters).toFixed(0)} clusters auto -> ${m(over, r => r.clusters)} fixed, ${(m(over, r => r.fixed.confusion) * 100).toFixed(1)}% confusion`);
  check('under-split 10 -> told 10', m(under, r => r.clusters) === 10 && m(under, r => r.fixed.confusion) < 0.15,
        `${m(under, r => r.auto.clusters).toFixed(1)} clusters auto (${(m(under, r => r.auto.confusion) * 100).toFixed(0)}% confusion) -> ${m(under, r => r.clusters)} fixed, ${(m(under, r => r.fixed.confusion) * 100).toFixed(1)}% confusion`);
  check('exact 12 -> told 12 stays exact', m(exact, r => r.clusters) === 12 && m(exact, r => r.fixed.confusion) < 0.02,
        `${m(exact, r => r.clusters)} clusters, ${(m(exact, r => r.fixed.confusion) * 100).toFixed(1)}% confusion`);
  check('10 people -> told 6 gives 6', m(fewer, r => r.clusters) === 6, `${m(fewer, r => r.clusters)} clusters`);
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
