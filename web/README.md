# web — the browser-only version

```
cd web && python3 -m http.server 8791
```

Then <http://localhost:8791>. A static server is required (ES modules do not
load over `file://`); `localhost` is a secure context, so the microphone works
without TLS.

Full write-up, including what this version gives up against the server design
and where its speaker identification breaks: [`../docs/html-version.md`](../docs/html-version.md).

## First run

1. **Settings → Minutes API.** Base URL, key, model. Anything speaking the
   OpenAI `/v1/chat/completions` shape works; Anthropic `/v1/messages` is also
   supported. *Test connection* checks it before a meeting rather than during.
2. **Settings → Speech recognition.** The browser recogniser needs no setup but
   sends audio to Google. Whisper-Singlish is local; convert it first with
   `python3 ../tools/export_singlish_onnx.py`.
3. **Settings → Speaker identification.** The built-in embedder needs nothing
   and is fine up to about four voices. Past that, convert an ECAPA-TDNN to
   ONNX and point at it.

## Files

| | |
|---|---|
| `js/audio.js` | Mic capture, the speech gate, turn segmentation, the rolling embedding window |
| `js/asr.js` | The two recognisers, and the trade-off between them |
| `js/embed.js` | Speaker embeddings — ECAPA via ONNX, or the MFCC fallback |
| `js/clustering.js` | `sim/clustering.py` ported, guards intact, plus the fallback's own calibration |
| `js/minutes.js` | The configurable API client, §6's schema and prompts |
| `js/store.js` | IndexedDB — sessions, audio clips, voiceprints |
| `js/app.js` | Session orchestration: the gateway's job, in the page |

Nothing is uploaded anywhere. The only outbound request the app makes is the
minutes call, to the endpoint you configure, when you press *Write minutes*.
