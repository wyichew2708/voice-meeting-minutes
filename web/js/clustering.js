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
  reclusterThreshold: 0.80,
};

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

  _newCluster(emb, seconds) {
    const c = { id: this._next++, centroid: new Float32Array(emb), seconds, n: 1,
                bank: [{ emb, seconds }] };
    this.clusters.push(c);
    return c;
  }

  _absorb(c, emb, seconds) {
    c.bank.push({ emb, seconds });
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
      const c = this._newCluster(emb, seconds);
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
    if (ok) { this._absorb(this.clusters[best], emb, seconds); cid = this.clusters[best].id; }
    else { cid = this._newCluster(emb, seconds).id; }
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

  _recentre(c) {
    const long = c.bank.filter(x => x.seconds >= this.cfg.minCentroidSeconds);
    const use = long.length ? long : c.bank;
    const dim = use[0].emb.length;
    const m = new Float32Array(dim);
    let wsum = 0;
    for (const { emb, seconds } of use) {
      wsum += seconds;
      for (let i = 0; i < dim; i++) m[i] += emb[i] * seconds;
    }
    for (let i = 0; i < dim; i++) m[i] /= (wsum || 1e-9);
    c.centroid = normalise(m);
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
        this.assignment.set(index, this._newCluster(emb, seconds).id);
        continue;
      }
      const sims = this._similarities(emb);
      let best = 0;
      for (let i = 1; i < sims.length; i++) if (sims[i] > sims[best]) best = i;
      this.clusters[best].n += 1;
      this.clusters[best].seconds += seconds;
      this.assignment.set(index, this.clusters[best].id);
    }
    this.deferred.length = 0;
  }

  finish() {
    if (this.cfg.reclusterThreshold) this.recluster();
    this.resolveDeferred();
  }
}

export { normalise };
