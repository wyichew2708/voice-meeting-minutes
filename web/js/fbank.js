/* Kaldi-compatible log mel filterbank, in JavaScript.
 *
 * The sherpa-onnx speaker models (wespeaker, 3D-Speaker) were trained on
 * Kaldi fbank features, and a model fed features that are *almost* Kaldi's
 * — wrong window, no pre-emphasis, HTK mel instead of Kaldi mel, float scale
 * instead of int16 — produces embeddings that are wrong in ways that are hard
 * to see. So this follows torchaudio.compliance.kaldi.fbank step for step,
 * with the defaults wespeaker uses, and sim/verify_fbank.mjs checks it
 * against torchaudio's output on a real utterance.
 *
 *   25 ms frame, 10 ms shift, snip_edges, remove DC, pre-emphasis 0.97,
 *   povey window, 512-point FFT, 80 Kaldi-mel bins 20 Hz..Nyquist, log
 *   of power floored at float epsilon, no dither, no energy.
 *
 * Input is float PCM in [-1, 1] at 16 kHz; it is scaled to int16 range first
 * because that is what these models saw in training (normalize_samples=0 in
 * the model metadata). Output is [frames x 80], row-major Float32Array.
 */

export const SAMPLE_RATE = 16000;
export const N_MELS = 80;
const FRAME = 400;        // 25 ms
const SHIFT = 160;        // 10 ms
const N_FFT = 512;        // next power of two above 400
const N_BINS = N_FFT / 2; // torchaudio builds the mel bank over 256 bins and pads one zero
const PREEMPH = 0.97;
const LOG_FLOOR = 1.1920929e-7;   // torch.finfo(float32).eps
const INT16_SCALE = 32768;

const kaldiMel = (hz) => 1127.0 * Math.log(1.0 + hz / 700.0);

/** povey = hann(N, periodic=false) ** 0.85 */
const window = (() => {
  const w = new Float32Array(FRAME);
  for (let n = 0; n < FRAME; n++) {
    w[n] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * n / (FRAME - 1)), 0.85);
  }
  return w;
})();

/** Sparse triangular mel bank: for each bin, [firstK, weights[]]. */
const melBank = (() => {
  const binWidth = SAMPLE_RATE / N_FFT;
  const melLow = kaldiMel(20.0), melHigh = kaldiMel(SAMPLE_RATE / 2);
  const delta = (melHigh - melLow) / (N_MELS + 1);
  const bank = [];
  for (let i = 0; i < N_MELS; i++) {
    const left = melLow + i * delta, centre = melLow + (i + 1) * delta, right = melLow + (i + 2) * delta;
    let first = -1;
    const ws = [];
    for (let k = 0; k < N_BINS; k++) {
      const mel = kaldiMel(binWidth * k);
      const up = (mel - left) / (centre - left);
      const down = (right - mel) / (right - centre);
      const w = Math.max(0, Math.min(up, down));
      if (w > 0) { if (first < 0) first = k; ws.push(w); }
      else if (first >= 0) break;
    }
    bank.push([first, Float32Array.from(ws)]);
  }
  return bank;
})();

/** In-place iterative radix-2 FFT. */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

/**
 * @param {Float32Array} pcm  float samples in [-1, 1], 16 kHz mono
 * @returns {{data: Float32Array, frames: number}}  row-major [frames x 80]
 */
export function kaldiFbank(pcm) {
  const n = pcm.length;
  const frames = n < FRAME ? 0 : 1 + Math.floor((n - FRAME) / SHIFT);
  const out = new Float32Array(frames * N_MELS);
  const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
  const frame = new Float32Array(FRAME);
  const power = new Float32Array(N_BINS);

  for (let f = 0; f < frames; f++) {
    const off = f * SHIFT;
    let mean = 0;
    for (let j = 0; j < FRAME; j++) { frame[j] = pcm[off + j] * INT16_SCALE; mean += frame[j]; }
    mean /= FRAME;
    for (let j = 0; j < FRAME; j++) frame[j] -= mean;                 // remove_dc_offset
    // pre-emphasis; the sample before the frame is taken as the frame's own
    // first sample (replicate padding), exactly as torchaudio does it.
    for (let j = FRAME - 1; j >= 1; j--) frame[j] -= PREEMPH * frame[j - 1];
    frame[0] -= PREEMPH * frame[0];
    re.fill(0); im.fill(0);
    for (let j = 0; j < FRAME; j++) re[j] = frame[j] * window[j];
    fft(re, im);
    for (let k = 0; k < N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];

    const row = f * N_MELS;
    for (let m = 0; m < N_MELS; m++) {
      const [first, ws] = melBank[m];
      let e = 0;
      for (let t = 0; t < ws.length; t++) e += power[first + t] * ws[t];
      out[row + m] = Math.log(Math.max(e, LOG_FLOOR));
    }
  }
  return { data: out, frames };
}

/** Per-utterance mean subtraction over time, one mean per mel bin. */
export function subtractMean(fb) {
  const { data, frames } = fb;
  if (!frames) return fb;
  const mu = new Float32Array(N_MELS);
  for (let f = 0; f < frames; f++) for (let m = 0; m < N_MELS; m++) mu[m] += data[f * N_MELS + m];
  for (let m = 0; m < N_MELS; m++) mu[m] /= frames;
  const out = new Float32Array(data.length);
  for (let f = 0; f < frames; f++) for (let m = 0; m < N_MELS; m++) out[f * N_MELS + m] = data[f * N_MELS + m] - mu[m];
  return { data: out, frames };
}
