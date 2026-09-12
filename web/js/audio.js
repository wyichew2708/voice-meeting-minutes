/* Microphone capture, the speech gate, and turn segmentation.
 *
 * Everything the design put in the browser (§2 Browser box) plus the segmenter
 * that used to live in the gateway (§5) — there is no gateway here, so the
 * turn cut happens client-side against the same numbers.
 *
 * Two gates, same contract:
 *
 *   'silero' — Silero VAD v5 (2.3 MB) via @ricky0123/vad-web, a neural gate.
 *              This is the default. An energy gate opens for HVAC, keyboards
 *              and breathing, and every false open is a clip the recogniser
 *              hallucinates words onto and the embedder clusters as a phantom
 *              speaker. Silero does not.
 *   'energy'  — RMS against an adaptive floor. Dependency-free; the fallback
 *              when the CDN is unreachable, and the gate the design's numbers
 *              were first written against.
 *
 * Two windows, deliberately not shared (this is the sim/windowed.py finding):
 *   - the ASR window is the turn: silence to silence, what a person just said.
 *   - the embedding window is a rolling EMBED_WINDOW seconds ending at the cut,
 *     so an interrupt-heavy meeting still gives the embedder enough voice to
 *     work with. Sharing one window collapsed to 104 clusters for 10 people.
 *
 * "Ending at the cut" needs care. Every hangover-based gate decides a turn is
 * over some time *after* the last word — measured at ~1.3 s for Silero with
 * these settings, 0.7 s for the energy gate — so the audio it hands back ends
 * in silence, and a rolling window taken at that moment is silence-padded.
 * For a 2 s turn that is a window that is more silence than speaker. So the
 * clip's trailing silence is measured and both windows are aligned to the
 * last audible sample, not to the moment the gate noticed.
 */

export const SAMPLE_RATE = 16000;
export const EMBED_WINDOW = 4.0;   // sim/windowed.py
const PRE_ROLL = 0.25;             // audio kept from before the gate opened
const HANGOVER = 0.70;             // silence that ends a turn (design §5)
const MIN_TURN = 0.35;             // shorter than this is not a turn
const MAX_TURN = 12.0;             // monologue cap, matches the sim

const VAD_CDN = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/';
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';  // same pin as embed.js: one runtime download

const WORKLET = `
class Pump extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('pump', Pump);
`;

/** Rolling PCM buffer holding the last `seconds` of audio. */
class Ring {
  constructor(seconds, rate = SAMPLE_RATE) {
    this.buf = new Float32Array(Math.ceil(seconds * rate));
    this.w = 0;
    this.filled = false;
  }
  push(frame) {
    for (let i = 0; i < frame.length; i++) {
      this.buf[this.w] = frame[i];
      this.w = (this.w + 1) % this.buf.length;
      if (this.w === 0) this.filled = true;
    }
  }
  /** The most recent `seconds`, oldest sample first. */
  tail(seconds, rate = SAMPLE_RATE) {
    const want = Math.min(Math.ceil(seconds * rate), this.filled ? this.buf.length : this.w);
    const out = new Float32Array(want);
    for (let i = 0; i < want; i++) {
      out[want - 1 - i] = this.buf[(this.w - 1 - i + this.buf.length * 2) % this.buf.length];
    }
    return out;
  }
}

const rms = (f) => { let s = 0; for (let i = 0; i < f.length; i++) s += f[i] * f[i]; return Math.sqrt(s / f.length); };

/**
 * Index just past the last audible sample. Audible is relative to the clip's
 * own loudest 20 ms, so a room with a noise floor still trims to the speech
 * rather than to digital zero; the absolute floor stops a clip that is all
 * hiss from counting as all speech.
 */
export function trailingSpeechEnd(pcm, { chunk = 320, rel = 0.06, abs = 0.0015 } = {}) {
  const n = Math.floor(pcm.length / chunk);
  if (!n) return pcm.length;
  let peak = 0;
  const levels = new Float32Array(n);
  for (let c = 0; c < n; c++) { levels[c] = rms(pcm.subarray(c * chunk, (c + 1) * chunk)); if (levels[c] > peak) peak = levels[c]; }
  const thr = Math.max(abs, rel * peak);
  for (let c = n - 1; c >= 0; c--) if (levels[c] >= thr) return (c + 1) * chunk;
  return 0;
}

const scriptOnce = (src) => new Promise((res, rej) => {
  if ([...document.scripts].some(s => s.src === src)) return res();
  const el = document.createElement('script');
  el.src = src; el.onload = res; el.onerror = () => rej(new Error(`could not load ${src}`));
  document.head.appendChild(el);
});

export class AudioCapture extends EventTarget {
  constructor({ embedWindow = EMBED_WINDOW, gate = 'silero' } = {}) {
    super();
    this.embedWindow = embedWindow;
    this.gateWanted = gate;
    this.gate = null;               // what actually started
    this.ctx = null;
    this.stream = null;
    this.vad = null;
    this.running = false;
    this.paused = false;

    this.speaking = false;
    this.t = 0;                     // seconds of audio seen
    this.ring = new Ring(Math.max(embedWindow, 6) + 1);

    // energy-gate state
    this.floor = 0.005;
    this.silence = 0;
    this.turn = [];
    this.turnStart = 0;
    this.preRoll = [];
  }

  async start() {
    if (this.running) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: SAMPLE_RATE,
               echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    if (this.gateWanted === 'silero') {
      try { await this._startSilero(); }
      catch (e) {
        this.dispatchEvent(new CustomEvent('gatefallback', { detail: { reason: e.message } }));
        await this._startEnergy();
      }
    } else {
      await this._startEnergy();
    }
    this.running = true;
    this.dispatchEvent(new CustomEvent('started', { detail: { gate: this.gate } }));
  }

  /* ─────────────────────────────────────────────────────────────── silero ── */

  async _startSilero() {
    await scriptOnce(`${ORT_CDN}ort.wasm.min.js`);
    await scriptOnce(`${VAD_CDN}bundle.min.js`);
    if (!window.vad?.MicVAD) throw new Error('vad-web did not initialise');

    let speechStart = 0;
    this.vad = await window.vad.MicVAD.new({
      getStream: async () => this.stream,          // reuse ours; do not open a second mic
      model: 'v5',
      baseAssetPath: VAD_CDN,
      onnxWASMBasePath: ORT_CDN,
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      redemptionMs: HANGOVER * 1000,
      preSpeechPadMs: PRE_ROLL * 1000,
      minSpeechMs: MIN_TURN * 1000,
      submitUserSpeechOnPause: true,               // pause() flushes the turn: our MAX_TURN cut
      onFrameProcessed: (probs, frame) => {
        if (!frame) return;
        this.t += frame.length / SAMPLE_RATE;
        this.ring.push(frame);
        this.dispatchEvent(new CustomEvent('level', {
          detail: { level: rms(frame), speech: probs.isSpeech >= 0.5 },
        }));
        if (this.speaking && this.t - speechStart >= MAX_TURN && !this._cutting) {
          // A monologue past the cap is cut here and the gate restarted, so
          // the transcript keeps moving instead of waiting for a breath.
          this._cutting = true;
          Promise.resolve(this.vad.pause()).then(() => { if (this.running && !this.paused) this.vad.start(); })
            .finally(() => { this._cutting = false; });
        }
      },
      onSpeechStart: () => {
        this.speaking = true;
        speechStart = Math.max(0, this.t - PRE_ROLL);
        this.dispatchEvent(new CustomEvent('speechstart'));
      },
      onVADMisfire: () => { this.speaking = false; this.dispatchEvent(new CustomEvent('speechend')); },
      onSpeechEnd: (audio) => {
        this.speaking = false;
        this.dispatchEvent(new CustomEvent('speechend'));
        this._emit(audio, speechStart);
      },
    });
    this.vad.start();
    this.gate = 'silero';
  }

  /* ─────────────────────────────────────────────────────────────── energy ── */

  async _startEnergy() {
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'pump');
    this.node.port.onmessage = (e) => this._energyFrame(e.data);
    src.connect(this.node);
    // A worklet with no downstream connection is not pulled in some browsers;
    // a zero-gain sink keeps the graph alive without making a sound.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);
    this.gate = 'energy';
  }

  _energyFrame(frame) {
    if (!this.running || this.paused) return;
    const dt = frame.length / SAMPLE_RATE;
    this.t += dt;
    this.ring.push(frame);

    const level = rms(frame);
    // Track the floor downward fast and upward slowly: a door slamming should
    // not convince the gate that the room is loud for the next minute.
    this.floor = level < this.floor ? this.floor * 0.9 + level * 0.1
                                    : this.floor * 0.999 + level * 0.001;
    const isSpeech = level > Math.max(this.floor * 3.0, 0.004);
    this.dispatchEvent(new CustomEvent('level', { detail: { level, speech: isSpeech } }));

    if (isSpeech) {
      if (!this.speaking) {
        this.speaking = true;
        this.turnStart = this.t - dt - PRE_ROLL;
        this.turn = this.preRoll.slice();
        this.dispatchEvent(new CustomEvent('speechstart'));
      }
      this.silence = 0;
      this.turn.push(frame);
      if (this._turnSeconds() >= MAX_TURN) this._energyCut();
    } else if (this.speaking) {
      this.silence += dt;
      this.turn.push(frame);                 // trailing silence stays in the clip
      if (this.silence >= HANGOVER) this._energyCut();
    }

    // Pre-roll: the quarter second before the gate opens, so a turn does not
    // start with its own first consonant already clipped off.
    this.preRoll.push(frame);
    let held = this.preRoll.reduce((a, f) => a + f.length, 0) / SAMPLE_RATE;
    while (held > PRE_ROLL && this.preRoll.length > 1) {
      held -= this.preRoll.shift().length / SAMPLE_RATE;
    }
  }

  _turnSeconds() { return this.turn.reduce((a, f) => a + f.length, 0) / SAMPLE_RATE; }

  _energyCut() {
    if (!this.speaking) return;
    const pcm = flatten(this.turn);
    this.speaking = false;
    this.turn = [];
    this.silence = 0;
    this.dispatchEvent(new CustomEvent('speechend'));
    this._emit(pcm, this.turnStart);
  }

  /* ─────────────────────────────────────────────────────────────── shared ── */

  _emit(raw, start) {
    const speechEnd = trailingSpeechEnd(raw);
    const seconds = speechEnd / SAMPLE_RATE;
    if (seconds < MIN_TURN) return;                 // a cough, not a turn
    const lag = raw.length - speechEnd;             // samples the gate took to notice
    // The embedding window ends at the last word, not at the cut: take a
    // window that reaches back far enough, then drop the gate's lag off it.
    const win = this.ring.tail(this.embedWindow + lag / SAMPLE_RATE);
    const embedPcm = win.subarray(0, Math.max(0, win.length - lag));
    this.dispatchEvent(new CustomEvent('segment', {
      detail: {
        // The turn plus 0.2 s of its own silence: Whisper does better with a
        // clean end than with a word cut at the sample.
        pcm: raw.subarray(0, Math.min(raw.length, speechEnd + 0.2 * SAMPLE_RATE)),
        embedPcm,
        seconds,
        start,
        end: start + seconds,
      },
    }));
  }

  pause() {
    this.paused = true;
    if (this.gate === 'silero') this.vad?.pause();   // flushes the open turn (submitUserSpeechOnPause)
    else this._energyCut();
  }

  resume() {
    this.paused = false;
    if (this.gate === 'silero') this.vad?.start();
  }

  async stop() {
    if (this.gate === 'silero') { try { await this.vad?.pause(); } catch {} this.vad?.destroy(); this.vad = null; }
    else this._energyCut();
    this.running = false;
    if (this.node) this.node.port.onmessage = null;
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) await this.ctx.close().catch(() => {});
    this.ctx = null;
    this.dispatchEvent(new CustomEvent('stopped'));
  }
}

export function flatten(frames) {
  const n = frames.reduce((a, f) => a + f.length, 0);
  const out = new Float32Array(n);
  let o = 0;
  for (const f of frames) { out.set(f, o); o += f.length; }
  return out;
}

/** Float32 PCM -> a 16-bit mono WAV blob, which is what every ASR wants. */
export function toWav(pcm, rate = SAMPLE_RATE) {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 2, true);
  str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  let o = 44;
  for (let i = 0; i < pcm.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
