/* Speaker embeddings, in the browser.
 *
 * ⚠ Read this before trusting a speaker label.
 *
 * The design assumes ECAPA-TDNN (§4.1), and the scale test's geometry is
 * calibrated to ECAPA's cosine statistics — within-speaker 0.72,
 * between-speaker 0.32. The clustering thresholds in clustering.js are tuned
 * for *that* geometry and mean nothing against a different one.
 *
 * Two backends:
 *
 *   'onnx'     — a real ECAPA-TDNN via onnxruntime-web. This is the one the
 *                design specifies and the thresholds are calibrated for. It
 *                needs a model file; see tools/README.md.
 *
 *   'spectral' — a dependency-free fallback: long-term log-mel statistics.
 *                It captures gross vocal-tract timbre and will separate two
 *                clearly different voices. It is NOT a speaker-verification
 *                model, its cosine geometry is not ECAPA's, and it degrades
 *                badly on the case §4.6 already flags as the hard one — two
 *                people who sound alike. Treat labels from it as a draft that
 *                a human renames, which is what the UI is built around anyway.
 *
 * The fallback exists so the app runs with nothing downloaded. It is not a
 * claim that it is good enough.
 */
import { SAMPLE_RATE } from './audio.js';
import { normalise } from './clustering.js';

const N_FFT = 512, HOP = 160, N_MELS = 40;

const hann = (() => {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N_FFT - 1));
  return w;
})();

/** Iterative radix-2 FFT, in place, real+imag split. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);

const melBank = (() => {
  const bins = N_FFT / 2 + 1;
  const lo = hz2mel(80), hi = hz2mel(SAMPLE_RATE / 2);
  const pts = Array.from({ length: N_MELS + 2 }, (_, i) =>
    Math.floor((N_FFT + 1) * mel2hz(lo + (hi - lo) * i / (N_MELS + 1)) / SAMPLE_RATE));
  const bank = [];
  for (let m = 1; m <= N_MELS; m++) {
    const f = new Float32Array(bins);
    for (let k = pts[m - 1]; k < pts[m]; k++) if (k >= 0 && k < bins) f[k] = (k - pts[m - 1]) / Math.max(1, pts[m] - pts[m - 1]);
    for (let k = pts[m]; k < pts[m + 1]; k++) if (k >= 0 && k < bins) f[k] = (pts[m + 1] - k) / Math.max(1, pts[m + 1] - pts[m]);
    bank.push(f);
  }
  return bank;
})();

function logMelFrames(pcm) {
  const frames = [];
  const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
  for (let off = 0; off + N_FFT <= pcm.length; off += HOP) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < N_FFT; i++) re[i] = pcm[off + i] * hann[i];
    fft(re, im);
    const bins = N_FFT / 2 + 1;
    const power = new Float32Array(bins);
    for (let k = 0; k < bins; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    const mel = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const f = melBank[m];
      for (let k = 0; k < bins; k++) s += power[k] * f[k];
      mel[m] = Math.log(s + 1e-10);
    }
    frames.push(mel);
  }
  return frames;
}

const N_CEPS = 20;   // MFCCs kept per frame, c0 excluded (see below)

/** DCT-II over the mel bank — the step that turns log-mel into MFCCs.
 *
 * This matters more than it looks. Raw log-mel dimensions are heavily
 * correlated and share a large common component (overall spectral tilt: room,
 * mic, distance, gain). Cosine over them puts every voice in the same
 * neighbourhood — measured on this very code, between-speaker cosine came out
 * at 0.677 against ECAPA's 0.32, and a whole four-person meeting collapsed
 * into one cluster. Decorrelating and dropping c0, which *is* that tilt,
 * is what separates them.
 */
const dctBasis = (() => {
  const b = [];
  for (let k = 1; k <= N_CEPS; k++) {      // k starts at 1: c0 is dropped
    const row = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) row[m] = Math.cos(Math.PI * k * (m + 0.5) / N_MELS);
    b.push(row);
  }
  return b;
})();

function mfccFrames(pcm) {
  return logMelFrames(pcm).map((mel) => {
    const c = new Float32Array(N_CEPS);
    for (let k = 0; k < N_CEPS; k++) {
      let s = 0;
      const row = dctBasis[k];
      for (let m = 0; m < N_MELS; m++) s += mel[m] * row[m];
      c[k] = s * Math.sqrt(2 / N_MELS);
    }
    return c;
  });
}

/** Mean and standard deviation of MFCCs, plus their deltas. */
function spectralEmbedding(pcm) {
  const frames = mfccFrames(pcm);
  if (frames.length < 3) return normalise(new Float32Array(N_CEPS * 4).fill(1e-6));
  const D = N_CEPS;
  const mean = new Float32Array(D), sd = new Float32Array(D);
  const dMean = new Float32Array(D), dSd = new Float32Array(D);
  for (const f of frames) for (let m = 0; m < D; m++) mean[m] += f[m] / frames.length;
  for (const f of frames) for (let m = 0; m < D; m++) sd[m] += (f[m] - mean[m]) ** 2 / frames.length;
  for (let m = 0; m < D; m++) sd[m] = Math.sqrt(sd[m]);

  const deltas = [];
  for (let i = 1; i < frames.length; i++) {
    const d = new Float32Array(D);
    for (let m = 0; m < D; m++) d[m] = frames[i][m] - frames[i - 1][m];
    deltas.push(d);
  }
  for (const d of deltas) for (let m = 0; m < D; m++) dMean[m] += d[m] / deltas.length;
  for (const d of deltas) for (let m = 0; m < D; m++) dSd[m] += (d[m] - dMean[m]) ** 2 / deltas.length;
  for (let m = 0; m < D; m++) dSd[m] = Math.sqrt(dSd[m]);

  const out = new Float32Array(D * 4);
  out.set(mean, 0); out.set(sd, D); out.set(dMean, D * 2); out.set(dSd, D * 3);
  return out;                                   // centred and normalised by the caller
}

export class Embedder {
  constructor() {
    this.backend = 'spectral';
    this.session = null;
    this.ort = null;
    // Running mean of every raw embedding seen, subtracted before the cosine.
    // Whatever this room, this microphone and this gain do to the spectrum is
    // common to everyone in the meeting, so it carries no speaker information
    // and only inflates the similarity between different people. Removing it
    // is what makes the cosine discriminative; it is the cheap cousin of the
    // channel compensation a real speaker model is trained to do.
    this._mu = null;
    this._seen = 0;
  }

  _centre(v) {
    if (!this._mu) this._mu = new Float32Array(v.length);
    this._seen += 1;
    const a = 1 / this._seen;
    for (let i = 0; i < v.length; i++) this._mu[i] += a * (v[i] - this._mu[i]);
    // Until a few voices have been heard the mean is mostly the first speaker,
    // so centring early would erase them. The recluster pass repairs whatever
    // the first few segments get wrong.
    if (this._seen < 4) return normalise(v);
    const o = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) o[i] = v[i] - this._mu[i];
    return normalise(o);
  }

  get label() {
    return this.backend === 'onnx'
      ? 'ECAPA-TDNN (ONNX)'
      : 'spectral fallback — approximate';
  }

  /** Load a real ECAPA-TDNN. `url` points at an .onnx exported per tools/README.md. */
  async useOnnx(url, ortModule) {
    this.ort = ortModule;
    this.session = await ortModule.InferenceSession.create(url, {
      executionProviders: ['webgpu', 'wasm'],
    });
    this.backend = 'onnx';
  }

  useSpectral() { this.backend = 'spectral'; this.session = null; }

  async embed(pcm) {
    if (this.backend === 'onnx' && this.session) {
      const name = this.session.inputNames[0];
      const t = new this.ort.Tensor('float32', pcm, [1, pcm.length]);
      const out = await this.session.run({ [name]: t });
      const v = out[this.session.outputNames[0]].data;
      return normalise(Float32Array.from(v));
    }
    return this._centre(spectralEmbedding(pcm));
  }
}

export { spectralEmbedding };
