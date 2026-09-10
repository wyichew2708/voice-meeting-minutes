/* Microphone capture, the speech gate, and turn segmentation.
 *
 * Everything the design put in the browser (§2 Browser box) plus the segmenter
 * that used to live in the gateway (§5) — there is no gateway here, so the
 * turn cut happens client-side against the same numbers.
 *
 * Two windows, deliberately not shared (this is the sim/windowed.py finding):
 *   - the ASR window is the turn: silence to silence, what a person just said.
 *   - the embedding window is a rolling EMBED_WINDOW seconds ending at the cut,
 *     so an interrupt-heavy meeting still gives the embedder enough voice to
 *     work with. Sharing one window collapsed to 104 clusters for 10 people.
 */

export const SAMPLE_RATE = 16000;
export const EMBED_WINDOW = 4.0;   // sim/windowed.py
const PRE_ROLL = 0.25;             // audio kept from before the gate opened
const HANGOVER = 0.70;             // silence that ends a turn (design §5)
const MIN_TURN = 0.35;             // shorter than this is not a turn
const MAX_TURN = 12.0;             // monologue cap, matches the sim

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

export class AudioCapture extends EventTarget {
  constructor({ embedWindow = EMBED_WINDOW } = {}) {
    super();
    this.embedWindow = embedWindow;
    this.ctx = null;
    this.stream = null;
    this.running = false;
    this.paused = false;

    // The gate. A fixed threshold fails in every room but the one it was set
    // in, so the floor tracks the quietest recent audio and speech is judged
    // relative to it.
    this.floor = 0.005;
    this.speaking = false;
    this.silence = 0;
    this.turn = [];
    this.turnStart = 0;
    this.t = 0;                     // seconds of audio seen
    this.ring = new Ring(Math.max(embedWindow, 6) + 1);
    this.preRoll = [];
  }

  async start() {
    if (this.running) return;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await this.ctx.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'pump');
    this.node.port.onmessage = (e) => this._frame(e.data);
    src.connect(this.node);
    // A worklet with no downstream connection is not pulled in some browsers;
    // a zero-gain sink keeps the graph alive without making a sound.
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.node.connect(mute).connect(this.ctx.destination);

    this.running = true;
    this.dispatchEvent(new CustomEvent('started', { detail: { rate: this.ctx.sampleRate } }));
  }

  pause() { this.paused = true; this._cut(true); }
  resume() { this.paused = false; }

  async stop() {
    this._cut(true);
    this.running = false;
    if (this.node) this.node.port.onmessage = null;
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) await this.ctx.close().catch(() => {});
    this.ctx = null;
    this.dispatchEvent(new CustomEvent('stopped'));
  }

  _frame(frame) {
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
      if (this._turnSeconds() >= MAX_TURN) this._cut(false);
    } else if (this.speaking) {
      this.silence += dt;
      this.turn.push(frame);                 // trailing silence stays in the clip
      if (this.silence >= HANGOVER) this._cut(false);
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

  _cut(forced) {
    if (!this.speaking) return;
    const seconds = this._turnSeconds() - (forced ? 0 : this.silence);
    const pcm = flatten(this.turn);
    this.speaking = false;
    this.turn = [];
    this.silence = 0;
    this.dispatchEvent(new CustomEvent('speechend'));
    if (seconds < MIN_TURN) return;          // a cough, not a turn

    this.dispatchEvent(new CustomEvent('segment', {
      detail: {
        pcm,                                  // the turn, for the recogniser
        embedPcm: this.ring.tail(this.embedWindow), // rolling window, for the embedder
        seconds,
        start: this.turnStart,
        end: this.turnStart + seconds,
      },
    }));
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
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
  let o = 44;
  for (let i = 0; i < pcm.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
