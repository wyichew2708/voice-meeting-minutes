# The HTML version

Everything in the browser. No gateway, no GPU box, no deployment — one static
directory, plus an LLM API you supply.

This is **not** a port of [the solution design](solution-design.md); it is a
different set of trade-offs against the same product. The design's central
argument is *reuse*: MERaLiON and Qwen3.6 are already running on the RHEL box
for voicebot, so the whole product costs ~1.5 GB of new VRAM. Nothing here can
reuse them. What follows is what that costs and what it buys.

## What runs where

| Component | Design | Here |
|---|---|---|
| Mic capture, gate, pre-roll | Browser | Browser — unchanged |
| Segmenter, turn cut | Gateway (FastAPI :8790) | Browser, same numbers (§5) |
| ASR | MERaLiON on vLLM :8801, shared with voicebot | Browser recogniser **or** Whisper-Singlish via WebGPU |
| Speaker embeddings | ECAPA sidecar :8803 | ECAPA via ONNX Runtime Web, **or** a built-in fallback |
| Clustering | Gateway | Browser — the same algorithm, ported and re-verified |
| Minutes | Qwen3.6-35B-A3B on :8000 | **Any API you configure** |
| Storage | SQLite + WAV | IndexedDB + WAV blobs |
| Gateway, TLS, deploy scripts | Required | Gone |

## Running it

```
cd web && python3 -m http.server 8791
```

Then <http://localhost:8791>. A static server is needed — ES modules do not
load over `file://` — and `localhost` counts as a secure context, so
`getUserMedia` works without certificates.

## Speech recognition

Two engines, and the choice is a privacy decision before it is a quality one.

**Browser recognition** starts instantly and downloads nothing. In Chrome it is
*not local*: microphone audio goes to Google's servers. For a meeting under NDA
that is the wrong default.

**Whisper-Singlish** is fully local once cached. Vanilla Whisper is not an
option — on SASRBench-v1, spontaneous Singlish:

| Model | Params | Licence | WER |
|---|---|---|---|
| `openai/whisper-small` | 0.2B | MIT | **147.80%** |
| [`mjwong/whisper-small-singlish`](https://huggingface.co/mjwong/whisper-small-singlish) | 0.2B | Apache-2.0 | **18.49%** |
| `openai/whisper-large-v3-turbo` | 0.8B | MIT | 27.58% |
| [`mjwong/whisper-large-v3-turbo-singlish`](https://huggingface.co/mjwong/whisper-large-v3-turbo-singlish) | 0.8B | MIT | **13.35%** |

Above 100% WER means the model inserts more words than the reference contains.
The finetune is the difference between working and not.

None of them publish ONNX weights, and transformers.js needs ONNX. Convert once:

```
pip install "optimum[onnxruntime]" onnx
python3 tools/export_singlish_onnx.py
```

⚠ All of these are trained on IMDA National Speech Corpus close-talk read
speech. A boardroom far-field mic is a different acoustic problem and none of
these numbers predict it — the same caveat [`sim/README.md`](../sim/README.md)
makes about the models it does not test.

## Speaker identification, and where it breaks

The clustering algorithm is [`sim/clustering.py`](../sim/clustering.py) ported
to JavaScript, guards and all. [`sim/verify_js_port.mjs`](../sim/verify_js_port.mjs)
re-runs the scale test's claims against the port; it reproduces the Python to
within a tenth of a cluster at every size from 2 to 12.

The **embedder** is where this version is genuinely weaker. ECAPA via ONNX is
supported and is what the thresholds are calibrated for. The built-in fallback
exists so the app runs with nothing downloaded, and testing it in the browser
turned up the same class of problem the original scale test found:

Its first version scored **between-speaker cosine 0.677** against ECAPA's 0.32 —
every voice in the same neighbourhood — and a four-person meeting collapsed into
**one cluster, 75% confusion**. The cause was using raw log-mel dimensions,
which are correlated and dominated by a common spectral tilt (room, mic, gain).
Taking the DCT to MFCCs, dropping `c0`, and subtracting a running mean moved
between-speaker cosine to **−0.130** and separated four people correctly.

Then the thresholds. `SHIPPING` is calibrated for ECAPA's geometry and means
nothing against a different one, so the fallback got its own scan:

| Speakers | Threshold | Clusters | Confusion |
|---|---|---|---|
| 4 | 0.65 | 4 | 31% |
| 4 | 0.88 | 7 | **0%** |
| 10 | 0.85 | 8 | 47% |
| 10 | 0.90 | 14 | 27% |

0.88 is chosen on the **over-splitting** side deliberately. An over-split is
repaired by naming two labels as the same person, which the UI merges; confusion
is misattributed lines scattered through a transcript that a reader has to catch
one at a time. The first is a chore, the second is a document that lies.

⚠ **The built-in embedder does not meet the ten-person brief.** At ten speakers
it still returns ~27% confusion at its best threshold, and the app says so on
screen when a session ends with more than four. Ten people needs the ECAPA
backend. This is the browser equivalent of the limit
[`docs/scale-test.md`](scale-test.md) already found — two people who sound alike
get merged — only much closer in.

## Minutes

`§6`'s schema, prompts and 3000-token budget are kept intact, so moving back
onto the real Qwen endpoint is a settings change. Any OpenAI-compatible
`/v1/chat/completions` endpoint works — OpenAI, OpenRouter, Groq, Together, or
a vLLM or Ollama server on your own network — and Anthropic's `/v1/messages` is
supported as a second wire format. Single pass by default; map-reduce above
~24k tokens, with the reduce step fed the chunk JSONs rather than the raw
transcript, exactly as §6 specifies.

⚠ The API key is stored in this browser and sent to the base URL you configure,
from the page. **A wrong base URL is a leaked key.**

⚠ §6's warning applies here unchanged and is repeated in the UI: hallucinated
action items are a larger risk than transcription errors. Every decision and
action carries `t0` so a reader can click back and check it, and the output is
an editable draft — never a finished record.

## What this version gives up

- **MERaLiON**, and with it the `[en, zh]` + Singlish handling that open
  question §12.3 is about. A Singlish Whisper finetune is a real substitute for
  English; it is not the same model.
- **The reuse argument.** The design's whole economic case was that inference
  was already paid for. Here the minutes are somebody's API bill.
- **pyannote's offline refine pass** (§4.2), which [scale-test](scale-test.md)
  moved *forward* into P2 because it is the only automatic fix for a sound-alike
  pair. There is no browser equivalent.
- **Ten-person speaker ID**, unless you convert and load ECAPA.

## What it buys

No deployment, no TLS, no auth, no GPU allocation, no shared-endpoint governor,
and no §3.3 risk of a batch minutes run landing on top of a live voicebot call.
Audio can stay on the machine that recorded it. It is the fastest way to put
the recording UI, the live transcript and the clustering in front of a real
user — which is exactly what P0 in the delivery plan is for.
