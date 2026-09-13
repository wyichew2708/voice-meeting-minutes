/* Online speaker clustering — a faithful port of sim/clustering.py (design §4.1).
 *
 * The Python version is the one the scale test exercises; this is the same
 * algorithm in the same order, so the guards that made it exact from 2 to 12
 * speakers carry over unchanged. Vectors are plain Float32Array, assumed
 * L2-normalised on the way in.
 *
 * Validated config (sim/test_scale.py SHIPPING) is exported as SHIPPING below.
 * The first draft's 0.70/no-guards config is DESIGNED, kept only so the
 * failure it produces stays reproducible in the browser too.
 */

export const DESIGNED = {
  threshold: 0.70, minCentroidSeconds: 1.5,
  margin: 0, deferUnderSeconds: 0, reclusterEvery: 0, reclusterThreshold: 0,
};

export const SHIPPING = {
  threshold: 0.60,
  minCentroidSeconds: 1.5,
  margin: 0.06,
  deferUnderSeconds: 1.5,
  reclusterEvery: 100,
  reclusterThreshold: 0.72,
};

/* Calibrated for the built-in spectral embedder in embed.js, which does NOT
 * share ECAPA's geometry and must not share its thresholds. Measured in-browser
 * on synthetic voices, after MFCC + running-mean centring:
 *
 *     within-speaker  mean 0.879   p05 0.49
 *     between-speaker mean -0.130  p95 0.854   max 0.926
 *
 * The tails overlap, so no threshold separates cleanly and the scan is a
 * trade rather than an optimum:
 *
 *     4 people   th 0.65 -> 4 clusters, 31% confusion
 *                th 0.88 -> 7 clusters,  0% confusion
 *     10 people  th 0.90 -> 14 clusters, 27% confusion
 *
 * 0.88 is chosen deliberately on the over-splitting side. An over-split is
 * repaired by renaming two labels to the same person, which the UI merges;
 * confusion is misattributed lines scattered through a transcript, which a
 * reader has to catch one at a time. The first is a chore, the second is a
 * document that lies.
 *
 * ⚠ At ten people this config still returns ~27% confusion. The built-in
 *   embedder does not meet the brief's ten-person requirement and is not
 *   claimed to. Use the ECAPA ONNX backend for a real meeting.
 */
export const SPECTRAL_FALLBACK = {
  threshold: 0.88,
  minCentroidSeconds: 1.5,
  margin: 0.06,
  deferUnderSeconds: 1.5,
  reclusterEvery: 100,
  reclusterThreshold: 0.95,
};

/* Calibrated for the CAM++ speaker model (wespeaker_en_voxceleb_CAM++), the
 * ONNX backend's default. Measured through the reference pipeline —
 * torchaudio Kaldi fbank + sim/clustering.py — on 12 TTS voices, 96
 * utterances, per-utterance mean subtraction:
 *
 *     within-speaker  mean 0.744   p05 0.434   short clips (<2 s) 0.589
 *     between-speaker mean 0.166   p95 0.401   max 0.770
 *
 * A wider gap than the ECAPA geometry SHIPPING assumes (0.72 / 0.32), so the
 * design's thresholds nearly fit, and a scan over real embeddings moved two:
 *
 *     threshold 0.60 -> 0.65   10 people: 8.9 -> 9.5 clusters, 10.4% -> 5.8%
 *     recluster 0.72 -> 0.80   the 0.72 pass was merging the closest pair
 *
 * Then real speech moved the recluster pass once more, 0.80 -> 0.75. On a
 * four-host podcast (sim/backtest_youtube.py) one person's voice drifted into
 * two labels whose centroids sat at cosine 0.79 — missed by the 0.80 pass by a
 * hundredth — while two different hosts on a second podcast sat at 0.37 and
 * the TTS sound-alike pair at 0.723. 0.75 heals the drift and still keeps
 * those apart. One real video drove it; the other two sources confirm no harm.
 *
 * The confusion that remains is ONE pair, in six seeds of eight: two Indian-
 * English male voices at centroid cosine 0.723. No threshold under 0.72 separates
 * them and anything over it splits the same person (within p05 0.43). With
 * either of them out of the room, ten people cluster exactly. This is §4.6 —
 * two people who genuinely sound alike are merged — measured rather than
 * predicted, and the reason manual split is in P1 rather than deferred.
 *
 * ⚠ TTS voices, close-talk, no room. Real far-field audio will be worse and
 *   these are starting values to tune on recordings of the actual room.
 */
export const SPEAKER_MODEL = {
  threshold: 0.65,
  minCentroidSeconds: 1.5,
  margin: 0.06,
  deferUnderSeconds: 1.5,
  reclusterEvery: 100,
  reclusterThreshold: 0.75,
};

/** A label with less speech than this is a blip — a laugh, a cough, a music
 *  sting, a "ya" — not a person holding a seat. The design's auto-prompt uses a
 *  similar bar (§4.4: ≥ 6 s before asking who someone is). */
export const MIN_SPEAKER_SECONDS = 5.0;

/** Speakers above which the built-in embedder should not be trusted. */
export const SPECTRAL_RELIABLE_SPEAKERS = 4;

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

function normalise(v) {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1e-9;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export class OnlineClusterer {
  constructor(cfg = SHIPPING) {
    this.cfg = { ...SHIPPING, ...cfg };
    this.clusters = [];       // {id, centroid, seconds, n, bank:[{emb,seconds}]}
    this._next = 0;
    this._sinceRecluster = 0;
    this.deferred = [];       // [{index, emb, seconds}]
    this.assignment = new Map(); // segIndex -> clusterId
  }

  _similarities(emb) { return this.clusters.map(c => dot(emb, c.centroid)); }

  _newCluster(emb, seconds, index = -1) {
    const c = { id: this._next++, centroid: new Float32Array(emb), seconds, n: 1,
                bank: [{ index, emb, seconds }] };
    this.clusters.push(c);
    return c;
  }

  _absorb(c, emb, seconds, index = -1) {
    c.bank.push({ index, emb, seconds });
    c.n += 1;
    c.seconds += seconds;
    // Short segments are assigned but never learned from: their embeddings are
    // noisy, and a run of them drags a centroid onto the wrong person.
    if (seconds < this.cfg.minCentroidSeconds) return;
    const w = seconds / Math.max(1e-9, c.seconds);
    const v = new Float32Array(emb.length);
    for (let i = 0; i < v.length; i++) v[i] = (1 - w) * c.centroid[i] + w * emb[i];
    c.centroid = normalise(v);
  }

  /** Assign one segment. Returns its cluster id, or null if deferred. */
  add(index, emb, seconds) {
    const cfg = this.cfg;

    if (cfg.deferUnderSeconds && seconds < cfg.deferUnderSeconds && this.clusters.length) {
      this.deferred.push({ index, emb, seconds });
      return null;
    }

    const sims = this._similarities(emb);
    if (!sims.length) {
      const c = this._newCluster(emb, seconds, index);
      this.assignment.set(index, c.id);
      this._tick();
      return c.id;
    }

    const order = sims.map((s, i) => i).sort((a, b) => sims[b] - sims[a]);
    const best = order[0];
    const runner = order.length > 1 ? sims[order[1]] : -1.0;

    let ok = sims[best] >= cfg.threshold;
    // Above threshold but not decisively nearer than the next cluster. At four
    // speakers this almost never fires; at ten it is the difference between a
    // transcript and a mess.
    if (ok && cfg.margin && (sims[best] - runner) < cfg.margin) ok = false;

    let cid;
    if (ok) { this._absorb(this.clusters[best], emb, seconds, index); cid = this.clusters[best].id; }
    else { cid = this._newCluster(emb, seconds, index).id; }
    this.assignment.set(index, cid);
    this._tick();
    return cid;
  }

  _tick() {
    this._sinceRecluster += 1;
    if (this.cfg.reclusterEvery && this._sinceRecluster >= this.cfg.reclusterEvery) {
      this._sinceRecluster = 0;
      this.recluster();
    }
  }

  /** Agglomerative merge over banked embeddings. Returns merges made.
   *
   * Online assignment is causal: it decides with the evidence it had at the
   * time and cannot revisit. This does revisit — repeatedly merging the closest
   * pair of centroids while they are closer than two different people should
   * ever be. It is what turns an early over-split back into one person.
   */
  recluster() {
    if (!this.cfg.reclusterThreshold || this.clusters.length < 2) return 0;
    let merges = 0;
    while (this.clusters.length >= 2) {
      let best = -1.0, pair = null;
      for (let i = 0; i < this.clusters.length; i++) {
        for (let j = i + 1; j < this.clusters.length; j++) {
          const s = dot(this.clusters[i].centroid, this.clusters[j].centroid);
          if (s > best) { best = s; pair = [i, j]; }
        }
      }
      if (best < this.cfg.reclusterThreshold || !pair) break;
      const [i, j] = pair;
      const a = this.clusters[i], b = this.clusters[j];
      const [keep, drop] = a.seconds >= b.seconds ? [a, b] : [b, a];
      keep.bank.push(...drop.bank);
      keep.seconds += drop.seconds;
      keep.n += drop.n;
      this._recentre(keep);
      for (const [k, v] of this.assignment) if (v === drop.id) this.assignment.set(k, keep.id);
      this.clusters.splice(this.clusters.indexOf(drop), 1);
      merges += 1;
    }
    return merges;
  }

  _recentre(c) { c.centroid = this._meanOf(c.bank); }

  /** Duration-weighted mean of a bank, over its long segments when it has any. */
  _meanOf(bank) {
    const long = bank.filter(x => x.seconds >= this.cfg.minCentroidSeconds);
    const use = long.length ? long : bank;
    const dim = use[0].emb.length;
    const m = new Float32Array(dim);
    let wsum = 0;
    for (const { emb, seconds } of use) {
      wsum += seconds;
      for (let i = 0; i < dim; i++) m[i] += emb[i] * seconds;
    }
    for (let i = 0; i < dim; i++) m[i] /= (wsum || 1e-9);
    return normalise(m);
  }

  /** Place every held-back segment against the finished centroids.
   *
   * No threshold here: by now the clusters are built, and a short segment has
   * to belong to whoever it is nearest. Holding it back was about keeping it
   * out of the centroids, not about refusing to transcribe it.
   */
  resolveDeferred() {
    for (const { index, emb, seconds } of this.deferred) {
      if (!this.clusters.length) {
        this.assignment.set(index, this._newCluster(emb, seconds, index).id);
        continue;
      }
      const sims = this._similarities(emb);
      let best = 0;
      for (let i = 1; i < sims.length; i++) if (sims[i] > sims[best]) best = i;
      const c = this.clusters[best];
      c.n += 1;
      c.seconds += seconds;
      c.bank.push({ index, emb, seconds });   // banked, so a re-cluster can move it; too short to recentre on
      this.assignment.set(index, c.id);
    }
    this.deferred.length = 0;
  }

  /* ─────────────────────────────────────────── known headcount ── */

  /** Re-cluster to exactly `n` speakers, for when the user knows how many
   *  people are in the room.
   *
   *  Thresholds are a guess about a room the model has never heard; a
   *  headcount is a fact. Merge the closest pair while there are too many,
   *  split the least coherent cluster in two while there are too few, then
   *  give every banked segment — deferred ones included — its nearest
   *  centroid. Banked embeddings make this instant, so it works mid-meeting
   *  and again at End. Returns how many segments changed label. */
  reclusterTo(n) {
    if (!n || n < 1 || !this.clusters.length) return 0;
    const before = new Map(this.assignment);
    // What the headcount made us do. A merge of two voices that both spoke at
    // length and were not alike is the signature of a headcount that is too
    // low — on a real podcast, "told 4" merged two different people because
    // the room had five. The app shows it rather than hiding it.
    this.lastRecluster = { forcedMerges: [] };
    this.resolveDeferred();
    // Blips go first. On a real podcast the closest pair of clusters was the
    // two hosts, and merging them while a 3 s laugh kept its own label gave
    // "one speaker" for a dialogue. A label that has not spoken for
    // MIN_SPEAKER_SECONDS is absorbed into its nearest neighbour before any
    // two real speakers are considered for merging.
    while (this.clusters.length > n && (this._absorbTiniest() || this._mergeClosest())) { /* reduce */ }
    while (this.clusters.length < n && this._splitWidest()) { /* split */ }
    const all = this.clusters.flatMap(c => c.bank);
    for (const c of this.clusters) { c.bank = []; c.n = 0; c.seconds = 0; }
    for (const s of all) {
      const sims = this._similarities(s.emb);
      let b = 0;
      for (let i = 1; i < sims.length; i++) if (sims[i] > sims[b]) b = i;
      const c = this.clusters[b];
      c.bank.push(s); c.n += 1; c.seconds += s.seconds;
      this.assignment.set(s.index, c.id);
    }
    this.clusters = this.clusters.filter(c => c.bank.length);
    for (const c of this.clusters) this._recentre(c);
    let changed = 0;
    for (const [k, v] of this.assignment) if (before.get(k) !== v) changed++;
    this.lastRecluster.changed = changed;
    this.lastRecluster.suspicious = this.lastRecluster.forcedMerges
      .filter(m => m.cosine < this.cfg.threshold && m.secondsA >= 30 && m.secondsB >= 30);
    return changed;
  }

  _absorbTiniest() {
    const tiny = this.clusters.filter(c => c.seconds < MIN_SPEAKER_SECONDS);
    if (!tiny.length || this.clusters.length < 2) return false;
    const drop = tiny.reduce((a, b) => (a.seconds <= b.seconds ? a : b));
    let keep = null, best = -Infinity;
    for (const c of this.clusters) {
      if (c === drop) continue;
      const s = dot(c.centroid, drop.centroid);
      if (s > best) { best = s; keep = c; }
    }
    keep.bank.push(...drop.bank);
    keep.seconds += drop.seconds;
    keep.n += drop.n;
    this._recentre(keep);
    for (const [k, v] of this.assignment) if (v === drop.id) this.assignment.set(k, keep.id);
    this.clusters.splice(this.clusters.indexOf(drop), 1);
    return true;
  }

  _mergeClosest() {
    let best = -Infinity, pair = null;
    for (let i = 0; i < this.clusters.length; i++) {
      for (let j = i + 1; j < this.clusters.length; j++) {
        const s = dot(this.clusters[i].centroid, this.clusters[j].centroid);
        if (s > best) { best = s; pair = [i, j]; }
      }
    }
    if (!pair) return false;
    const a = this.clusters[pair[0]], b = this.clusters[pair[1]];
    const [keep, drop] = a.seconds >= b.seconds ? [a, b] : [b, a];
    this.lastRecluster?.forcedMerges.push({ secondsA: keep.seconds, secondsB: drop.seconds, cosine: best });
    keep.bank.push(...drop.bank);
    keep.seconds += drop.seconds;
    keep.n += drop.n;
    this._recentre(keep);
    for (const [k, v] of this.assignment) if (v === drop.id) this.assignment.set(k, keep.id);
    this.clusters.splice(this.clusters.indexOf(drop), 1);
    return true;
  }

  /** Split the cluster whose long segments agree least with its centroid,
   *  by 2-means on cosine seeded from its two most distant members. */
  _splitWidest() {
    let target = null, worst = Infinity;
    for (const c of this.clusters) {
      const long = c.bank.filter(x => x.seconds >= this.cfg.minCentroidSeconds);
      if (long.length < 4) continue;
      const agree = long.reduce((a, x) => a + dot(x.emb, c.centroid), 0) / long.length;
      if (agree < worst) { worst = agree; target = c; }
    }
    if (!target) return false;
    const pts = target.bank.filter(x => x.seconds >= this.cfg.minCentroidSeconds);
    let ia = 0, ib = 1, far = Infinity;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const s = dot(pts[i].emb, pts[j].emb);
        if (s < far) { far = s; ia = i; ib = j; }
      }
    }
    let ca = pts[ia].emb, cb = pts[ib].emb, A = [], B = [];
    for (let it = 0; it < 8; it++) {
      A = []; B = [];
      for (const p of target.bank) (dot(p.emb, ca) >= dot(p.emb, cb) ? A : B).push(p);
      if (!A.length || !B.length) return false;
      ca = this._meanOf(A); cb = this._meanOf(B);
    }
    // A split that leaves one side with less than a real speaker's worth of
    // speech has found a blip, not a person.
    const secs = (g) => g.reduce((s, x) => s + x.seconds, 0);
    if (secs(A) < MIN_SPEAKER_SECONDS || secs(B) < MIN_SPEAKER_SECONDS) return false;
    const mk = (bank, centroid) => ({ id: this._next++, centroid, bank, n: bank.length,
                                      seconds: bank.reduce((s, x) => s + x.seconds, 0) });
    const a = mk(A, ca), b = mk(B, cb);
    this.clusters.splice(this.clusters.indexOf(target), 1, a, b);
    for (const p of a.bank) this.assignment.set(p.index, a.id);
    for (const p of b.bank) this.assignment.set(p.index, b.id);
    return true;
  }

  /** Nearest cluster by cosine, or null if there are none yet. A provisional
   *  label for a deferred segment, so the transcript never shows a blank —
   *  resolveDeferred() decides for real once the centroids are built. */
  nearest(emb) {
    if (!this.clusters.length) return null;
    const sims = this._similarities(emb);
    let b = 0;
    for (let i = 1; i < sims.length; i++) if (sims[i] > sims[b]) b = i;
    return this.clusters[b].id;
  }

  finish() {
    if (this.cfg.reclusterThreshold) this.recluster();
    this.resolveDeferred();
  }
}

export { normalise };
