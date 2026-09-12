# web — the browser-only version

```
python3 tools/fetch_models.py          # once: the 29 MB speaker model
python3 tools/export_singlish_onnx.py  # once, optional: local Singlish Whisper (~1 GB download)
python3 tools/serve.py                 # http://localhost:8791
```

Then open <http://localhost:8791/selftest.html> first — it checks the pipeline
in your browser against the reference, then your microphone and room — and
<http://localhost:8791> for the app. `serve.py` is `http.server` with caching
off and the page cross-origin isolated (four wasm threads for Whisper);
`localhost` is a secure context, so the microphone works without TLS.

Full write-up, including what this version gives up against the server design
and where its speaker identification breaks: [`../docs/html-version.md`](../docs/html-version.md).

## First run

1. **Settings → Minutes API.** Base URL, key, model. Anything speaking the
   OpenAI `/v1/chat/completions` shape works; Anthropic `/v1/messages` is also
   supported. *Test connection* checks it before a meeting rather than during.
2. **Settings → Speech recognition.** The browser recogniser needs no setup but
   sends audio to Google. Whisper-Singlish is local and understands Singlish
   (19% WER on real spontaneous speech, against 148% for vanilla Whisper);
   export it once with `python3 tools/export_singlish_onnx.py`. On WebGPU it
   runs ten times faster than live; on wasm about four times, on the isolated
   server.
3. **Settings → Speaker identification.** CAM++ is the default and is what the
   ten-person figures were measured with; `fetch_models.py` above is all it
   needs. Without it the app falls back to a built-in embedder that is fine to
   about four voices and says so.
4. **Settings → Speech gate.** Silero VAD by default, from the CDN. The energy
   gate is the offline fallback. Browser audio processing is off by default;
   it is built for calls and hurts speaker labels.
5. **Speakers: auto ▾** in the header. If you know how many people are in the
   room, say so — the clustering is redone to that count, now and at End.

## Files

| | |
|---|---|
| `js/audio.js` | Mic capture, Silero VAD / energy gate, turn segmentation, windows aligned to the last word |
| `js/asr.js` | The two recognisers, the trade-off between them, the Whisper hallucination filter |
| `js/fbank.js` | Kaldi fbank in JS, verified against torchaudio |
| `js/embed.js` | Speaker embeddings — CAM++ via onnxruntime-web, or the MFCC fallback |
| `js/clustering.js` | `sim/clustering.py` ported, guards intact; configs measured for CAM++ and for the fallback |
| `js/minutes.js` | The configurable API client, §6's schema and prompts, the grounding check |
| `js/store.js` | IndexedDB — sessions, audio clips, voiceprints |
| `js/app.js` | Session orchestration: the gateway's job, in the page |

Nothing is uploaded anywhere. The only outbound request the app makes is the
minutes call, to the endpoint you configure, when you press *Write minutes*.
