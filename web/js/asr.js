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

  /** Transcribe one cut turn. Float32 PCM at 16 kHz. */
  async transcribe(pcm) {
    if (!this.pipe) throw new Error('Whisper model not loaded');
    const out = await this.pipe(pcm, {
      language: this.language,
      task: 'transcribe',
      chunk_length_s: 30,
      return_timestamps: false,
    });
    const text = (out?.text ?? '').trim();
    return text || null;
  }
}

export function makeASR(engine, opts) {
  if (engine === 'whisper') return new WhisperASR(opts);
  return new WebSpeechASR(opts);
}
