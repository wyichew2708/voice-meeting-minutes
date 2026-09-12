/* Speech recognition, two ways.
 *
 * The design's recogniser is MERaLiON on the GPU box (§3). Neither it nor the
 * gateway exists here, so this offers the two things that genuinely run in a
 * browser — with the trade-off between them stated rather than buried.
 *
 * ── 'webspeech' ────────────────────────────────────────────────────────────
 * The browser's built-in SpeechRecognition. Nothing to download, interim and
 * final results native, starts instantly.
 *
 *   ⚠ In Chrome this is *not* local. Audio goes to Google's servers for
 *     recognition. For a meeting under NDA that is the wrong default, and it
 *     is the reason §9's consent flow is not optional here.
 *   ⚠ No Singlish tuning. `en-SG` is requested where supported, but what that
 *     maps to server-side is not documented and not guaranteed.
 *   ⚠ It runs on its own clock, not on our segmenter's. Text is matched to the
 *     nearest segment by time, so speaker attribution is approximate —
 *     good enough to read, not exact at a fast interruption.
 *
 * ── 'whisper' ──────────────────────────────────────────────────────────────
 * A Singlish-finetuned Whisper via transformers.js, ONNX, WebGPU. Fully local:
 * once the weights are cached nothing leaves the machine. Transcribes exactly
 * the turns our segmenter cuts, so attribution is exact.
 *
 *   Vanilla Whisper scores 147.8% WER on SASRBench-v1 (spontaneous Singlish) —
 *   worse than useless. The finetune is what makes this viable:
 *     mjwong/whisper-small-singlish            0.2B  Apache-2.0  18.49% WER
 *     mjwong/whisper-large-v3-turbo-singlish   0.8B  MIT         13.35% WER
 *   Neither ships ONNX weights; convert once with tools/export_singlish_onnx.py.
 */

const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';

export const ENGINES = {
  webspeech: { name: 'Browser speech recognition', local: false, setup: false },
  whisper: { name: 'Whisper Singlish (local)', local: true, setup: true },
};

/* ─────────────────────────────────────────────────────── browser built-in ── */

export class WebSpeechASR extends EventTarget {
  constructor({ lang = 'en-SG' } = {}) {
    super();
    this.lang = lang;
    this.rec = null;
    this.wantRunning = false;
  }

  static get available() {
    return typeof window !== 'undefined' &&
      !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  }

  async load() {
    if (!WebSpeechASR.available) {
      throw new Error('This browser has no SpeechRecognition. Chrome or Safari, or switch to the Whisper engine.');
    }
  }

  start() {
    const Ctor = window.SpeechRecognition || window.webkitSpeechRecognition;
    this.rec = new Ctor();
    this.rec.lang = this.lang;
    this.rec.continuous = true;
    this.rec.interimResults = true;
    this.rec.maxAlternatives = 1;

    this.rec.onresult = (e) => {
      let interim = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        const text = r[0].transcript.trim();
        if (!text) continue;
        if (r.isFinal) {
          this.dispatchEvent(new CustomEvent('final', {
            detail: { text, at: performance.now() / 1000, confidence: r[0].confidence },
          }));
        } else {
          interim += text + ' ';
        }
      }
      if (interim.trim()) {
        this.dispatchEvent(new CustomEvent('interim', { detail: { text: interim.trim() } }));
      }
    };

    // Chrome ends the session on its own every so often; restart unless we
    // were the ones who stopped it.
    this.rec.onend = () => { if (this.wantRunning) { try { this.rec.start(); } catch {} } };
    this.rec.onerror = (e) => {
      if (e.error === 'no-speech' || e.error === 'aborted') return;
      this.dispatchEvent(new CustomEvent('asrerror', { detail: { message: e.error } }));
    };

    this.wantRunning = true;
    try { this.rec.start(); } catch {}
  }

  stop() {
    this.wantRunning = false;
    if (this.rec) { try { this.rec.stop(); } catch {} }
    this.rec = null;
  }

  /** Segment-driven transcription is not how this engine works. */
  async transcribe() { return null; }
}

/* ──────────────────────────────────────────────────────── whisper, local ── */

export class WhisperASR extends EventTarget {
  constructor({ model, quantized = 'q8', device = 'webgpu', language = 'en' } = {}) {
    super();
    this.model = model;
    this.quantized = quantized;
    this.device = device;
    this.language = language;
    this.pipe = null;
  }

  async load(onProgress) {
    const { pipeline, env } = await import(/* @vite-ignore */ `${TRANSFORMERS_CDN}`);
    // A model served from this origin (tools/export_singlish_onnx.py drops it
    // in web/models/) should be found there rather than on the Hub.
    if (this.model.startsWith('local/')) {
      env.allowRemoteModels = false;
      env.localModelPath = './models/';
      this.model = this.model.slice('local/'.length);
    }
    this.pipe = await pipeline('automatic-speech-recognition', this.model, {
      dtype: this.quantized,
      device: this.device,
      progress_callback: (p) => {
        if (onProgress && p.status === 'progress' && p.total) {
          onProgress({ file: p.file, loaded: p.loaded, total: p.total });
        }
      },
    });
  }

  start() { /* segment-driven: nothing runs continuously */ }
  stop() { }

  /** Transcribe one cut turn (or a coalesced run of them). Float32 PCM at 16 kHz.
   *  Returns null for nothing usable — including a confident hallucination. */
  async transcribe(pcm, seconds = pcm.length / 16000) {
    if (!this.pipe) throw new Error('Whisper model not loaded');
    const out = await this.pipe(pcm, {
      language: this.language,
      task: 'transcribe',
      chunk_length_s: 30,
      return_timestamps: false,
    });
    const text = (out?.text ?? '').trim();
    if (!text || looksHallucinated(text, seconds)) return null;
    return text;
  }
}

/* ── Whisper hallucination filter ───────────────────────────────────────────
 *
 * Whisper was trained on subtitled video, and on a clip with little or no
 * speech it produces subtitles: "Thank you.", "Subtitles by the Amara.org
 * community", "Please subscribe", or one phrase looping until the token
 * budget runs out. The design's §3.1 gates exist to stop the recogniser
 * inventing sentences over silence; Silero VAD stops most of it upstream, and
 * this catches what gets through.
 *
 * The one thing it must not do is delete backchannels. "Yeah." and "Right."
 * are a third of a meeting's segments (sim/meeting.py) and they are short.
 * So a short generic phrase is only rejected when the clip is long — two
 * words from a 0.6 s clip is a person agreeing; two words from a 5 s clip is
 * the model filling silence.
 */
const ALWAYS = [
  /subtitles?\s+(by|provided|created|made)/i, /amara\.org/i, /\bwww\.|\.com\b/i,
  /\b(please\s+)?(like\s+and\s+)?subscribe\b/i, /\bcopyright\b|©/i,
  /transcri(bed|ption)\s+by/i, /^\s*[\[(【][^\])】]*[\])】]\s*$/,   // [Music]  (applause)
  /字幕|感谢观看|請訂閱|谢谢观看/,
];
const FILLER = /^(thank\s*you|thanks|thank\s*you\s+for\s+watching|thanks\s+for\s+watching|bye|goodbye|you|okay|ok|so|um|uh|oh)[\s.!?,]*$/i;

export function looksHallucinated(text, seconds = 0) {
  const t = String(text || '').trim();
  if (!t || /^[\p{P}\p{S}\s_]+$/u.test(t)) return true;   // punctuation/symbols only
  if (ALWAYS.some(p => p.test(t))) return true;
  const words = t.toLowerCase().replace(/[^\p{L}\p{N}' ]+/gu, ' ').split(/\s+/).filter(Boolean);
  if (FILLER.test(t) && seconds >= 2.0) return true;             // filler over a long clip
  if (seconds > 0 && words.length / seconds > 6.5) return true;  // nobody talks that fast
  if (words.length >= 8) {
    if (new Set(words).size / words.length < 0.3) return true;   // "the the the the…"
    const tri = new Map();
    for (let i = 0; i + 3 <= words.length; i++) {
      const k = words.slice(i, i + 3).join(' ');
      tri.set(k, (tri.get(k) || 0) + 1);
    }
    if (Math.max(...tri.values()) >= 4) return true;             // a phrase looping
  }
  return false;
}

export function makeASR(engine, opts) {
  if (engine === 'whisper') return new WhisperASR(opts);
  return new WebSpeechASR(opts);
}
