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
 *   'onnx'     — a real speaker-verification model via onnxruntime-web:
 *                CAM++ from sherpa-onnx by default (tools/fetch_models.py).
 *                Kaldi fbank in (fbank.js, verified against torchaudio),
 *                per-utterance mean subtraction, 512-d embedding out.
 *
 *                Measured on 12 TTS voices through the reference pipeline:
 *                with CMN, within-speaker 0.744 / between 0.166 — a wider
 *                gap than the ECAPA numbers the design assumed. WITHOUT CMN
 *                the same model gives between 0.506 and ten people collapse
 *                into 2.5 clusters. The subtraction is not optional.
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
import { kaldiFbank, subtractMean, fft } from './fbank.js';

const N_FFT = 512, HOP = 160, N_MELS = 40;

/* ⚠ The onnxruntime-web pin matters for this model. On 1.22.0 (wasm) the
 * graph optimiser returns deterministic but WRONG embeddings at every level
 * but 'disabled' — cosine to the Python reference 0.17 for a 4.1 s clip,
 * 0.04 for 4.7 s, 0.9997 for 8.0 s, the same wrong answer every time. The
 * fbank into it matches torchaudio to 5e-3, so it is the optimiser. On
 * 1.29.0 every level matches to 1.0000 at the same speed (118-136 ms per
 * 4-5 s clip, wasm, one thread). So: 1.29.0, optimiser on. Do not lower the
 * pin in app.js / audio.js without re-running the in-browser check described
 * in docs/html-version.md — the failure is silent and the labels look fine.
 *
 * WebGPU is not used. The 1.22.0 JSEP build threw an Emscripten exception
 * creating a session for this model even with a Metal adapter, and wasm is
 * already comfortably real-time, so it was not worth a second dependency. */
export const ORT_GRAPH_OPT = 'all';

const hann = (() => {
  const w = new Float32Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N_FFT - 1));
  return w;
})();

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
      ? `${this.modelName || 'speaker model'} (ONNX)`
      : 'spectral fallback — approximate';
  }

  /** Load a sherpa-onnx speaker model. `url` is served from web/models/;
   *  `ortModule` must be the wasm build (see ORT_GRAPH_OPT above). */
  async useOnnx(url, ortModule) {
    this.ort = ortModule;
    this.session = await ortModule.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: ORT_GRAPH_OPT,
    });
    this.modelName = url.split('/').pop().replace(/\.onnx$/, '');
    this.inputName = this.session.inputNames[0];    // 'feats'
    this.outputName = this.session.outputNames[0];  // 'embs'
    this.backend = 'onnx';
  }

  useSpectral() { this.backend = 'spectral'; this.session = null; }

  async embed(pcm) {
    if (this.backend === 'onnx' && this.session) {
      const fb = subtractMean(kaldiFbank(pcm));
      if (fb.frames < 10) return normalise(new Float32Array(512).fill(1e-6)); // < 0.1 s: nothing to embed
      const t = new this.ort.Tensor('float32', fb.data, [1, fb.frames, 80]);
      const out = await this.session.run({ [this.inputName]: t });
      return normalise(Float32Array.from(out[this.outputName].data));
    }
    return this._centre(spectralEmbedding(pcm));
  }
}

export { spectralEmbedding };
