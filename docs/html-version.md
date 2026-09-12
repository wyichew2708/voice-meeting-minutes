# The HTML version

Everything in the browser. No gateway, no GPU box, no deployment — one static
directory, plus an LLM API you supply.

This is **not** a port of [the solution design](solution-design.md); it is a
different set of trade-offs against the same product. The design's central
argument is *reuse*: MERaLiON and Qwen3.6 are already running on the RHEL box
for voicebot, so the whole product costs ~1.5 GB of new VRAM. Nothing here can
reuse them. What follows is what that costs and what it buys — and, because
this branch's habit is to measure rather than assume, what was measured.

## What runs where

| Component | Design | Here |
|---|---|---|
| Mic capture, gate, pre-roll | Browser | Browser — Silero VAD, energy gate as fallback |
| Segmenter, turn cut | Gateway (FastAPI :8790) | Browser, same numbers (§5), aligned to the last word |
| ASR | MERaLiON on vLLM :8801, shared with voicebot | Browser recogniser **or** Whisper-Singlish via transformers.js |
| Speaker embeddings | ECAPA sidecar :8803 | **CAM++ via onnxruntime-web** (29 MB, `tools/fetch_models.py`); MFCC fallback |
| Clustering | Gateway | Browser — the same algorithm, ported, re-verified, re-calibrated on real speech |
| Minutes | Qwen3.6-35B-A3B on :8000 | **Any API you configure**, with a grounding check on the output |
| Storage | SQLite + WAV | IndexedDB + WAV blobs |
| Gateway, TLS, deploy scripts | Required | Gone |

## Running it

```
python3 tools/fetch_models.py          # once: the 29 MB speaker model
python3 tools/export_singlish_onnx.py  # once, optional: the local Singlish Whisper (~1 GB download)
python3 tools/serve.py                 # http://localhost:8791
```

`tools/serve.py` is `http.server` with two changes. Caching is off — browsers
heuristically cache ES modules from a plain static server, and after pulling an
update you can quietly be running last week's `asr.js`. And the page is served
cross-origin isolated, which unlocks multi-threaded wasm and makes Whisper on
wasm four times faster (below). `localhost` counts as a secure context, so
`getUserMedia` works without certificates. Without the
speaker model the app runs on the built-in embedder and says so; see
[the limit](#the-built-in-fallback) below.

**Start with <http://localhost:8791/selftest.html>.** It proves the pipeline in
your browser matches the reference without a microphone, records you for 14 s
and reports what your microphone and room look like to the speaker model — the
thing no test corpus can tell you — and scores the Singlish export on real
Singlish.

## Speech recognition

Two engines, and the choice is a privacy decision before it is a quality one.

**Browser recognition** starts instantly and downloads nothing. In Chrome it is
*not local*: microphone audio goes to Google's servers. For a meeting under NDA
that is the wrong default.

It also runs on its own clock. Results arrive about a second after speech ends,
and the first build attached each one to *the latest segment without text* —
which by then was often the next speaker's. Text landing on the wrong person
reads exactly like bad diarization, and it was not the embedder at all. Each
result now carries when it was first heard and when it was finalised, and goes
to the segment it overlaps most, after subtracting the recogniser's lags
(~0.6 s at the start, ~0.8 s at the end). It is still approximate — a result
spanning two speakers lands on whichever it overlaps more — which is the
strongest reason to prefer Whisper, which transcribes exactly the clip cut for
each speaker.

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

None of them publish ONNX weights, and transformers.js needs ONNX.
[`tools/export_singlish_onnx.py`](../tools/export_singlish_onnx.py) drives the
official transformers.js conversion script in an isolated venv. It took three
attempts to find the two pins that matter — torch 2.5.1, because newer torch's
dynamo exporter writes a file name optimum does not expect, and `onnxscript` —
and they are in the script. It exports two pairs: q4 (encoder 66 MB + merged
decoder 233 MB) for WebGPU and q8 (92 + 315 MB) for wasm, loaded from
`web/models/` as `local/mjwong/whisper-small-singlish` and cached by the browser
after the first run. One more thing the self-test
caught before a user did: transformers.js disallows *local* models in browsers
by default, and a loader that only turns remote off has turned everything off.

**Measured on real Singlish.** Thirty clips of spontaneous conversation from
[SASRBench-v1](https://huggingface.co/datasets/mjwong/SASRBench-v1) — 7.5
minutes, 1,350 words, fetched row by row from the Hugging Face datasets-server
rather than the 1.6 GB shards — through the fp32 model in PyTorch
([`sim/singlish_wer_reference.py`](../sim/singlish_wer_reference.py)):
**18.5% WER**, against the model card's 18.49% on the full benchmark. The
particles survive — *"so cheem"*, *"wah"*, *"ya"* — rather than being fought.
**And through the browser.** The self-test page runs the same thirty clips
through the local export — on wasm, and on WebGPU at the precision the app
actually loads. Measured on this machine (Apple silicon, Chrome):

| Backend | Precision | Correct | Speed | Download |
|---|---|---|---|---|
| wasm, one thread | q8 | **19.1% WER** | 0.84× real time | 407 MB |
| wasm, four threads — isolated page | q8 | 19.1% WER | **0.24× real time** | 407 MB |
| WebGPU | q8 | **garbage** — `!!!!!!` | 3.2× real time | — |
| WebGPU | fp32 | text identical to PyTorch | 0.08–0.10× real time | 968 MB |
| WebGPU | **q4 encoder + q4 decoder** | **18.6% WER** | **0.09× real time** | **299 MB** |

Three things follow, and all three are in the code. Precision is chosen per
backend (`whisperDtypeFor` in `asr.js`), and q8 is never loaded on WebGPU —
that combination is not slow, it is *wrong*, and it looked fine until the words
came out. WebGPU with q4 is the default: ten times faster than live, a 300 MB
download, and a fallback to wasm when the browser has no WebGPU. And
`tools/serve.py` serves the page cross-origin isolated by default, because that
is what lets onnxruntime use four threads and turns wasm from "just behind
live" into four times ahead of it; the CDNs used here send the
`Cross-Origin-Resource-Policy` header isolation requires, and the minutes call
is an ordinary CORS fetch. q4 on WebGPU lands at 18.6% against the fp32
reference's 18.5% — quantisation's price is a word in a thousand; q8 on wasm
pays 19.1%. The hallucination filter flagged none of the thirty clips on either
backend.

### Getting more out of Whisper

Whisper on a 1.5 s snippet is far worse than Whisper on 10 s of the same voice,
and a 0.7 s hangover cuts a turn at every breath. So consecutive turns by the
same speaker are **coalesced** before recognition — until the speaker changes,
1.2 s of silence passes, or the run reaches 20 s. Cut turns show as a pending
row in the speaker's colour until their text lands, which is the ~1.2 s settle
§1 describes.

Whisper was trained on subtitled video, and on a clip with little speech it
produces subtitles: *"Thank you."*, *"Subtitles by the Amara.org community"*,
one phrase looping to the token limit. A **hallucination filter** drops those.
The rule that matters is the one it must *not* fire on: backchannels — *"Yeah."*,
*"Right."* — are a third of a meeting's segments and they are short, so a filler
phrase is rejected only when the clip is long. Two words in 0.6 s is a person
agreeing; two words in 5 s is the model filling silence.
[`sim/verify_hallucination_filter.mjs`](../sim/verify_hallucination_filter.mjs)
holds 20 cases, and writing it caught a bug on the spot: `\W` without the `u`
flag treats every Chinese character as punctuation, and a whole Chinese sentence
was being dropped as "punctuation only".

One lever is *not* available: Whisper's `initial_prompt`, for attendee names and
jargon. transformers.js does not expose it —
[huggingface/transformers.js#923](https://github.com/xenova/transformers.js/issues/923)
is still open.

⚠ All of these models are trained on IMDA National Speech Corpus close-talk read
speech. A boardroom far-field mic is a different acoustic problem and none of
these numbers predict it — the same caveat [`sim/README.md`](../sim/README.md)
makes about the models it does not test.

## The speech gate

An energy gate opens for air-conditioning, keyboards and breathing, and every
false open is a clip the recogniser hallucinates words onto and the embedder
clusters as a phantom speaker. The default gate is therefore **Silero VAD v5**
(2.3 MB) via `@ricky0123/vad-web`, loaded from the CDN, reusing the app's own
microphone stream. Measured in-browser on three TTS utterances with silences
between: 22.2 s of audio segmented in 95 ms, three segments, starts within
0.1 s of the truth. The energy gate remains as the fallback when the CDN is
unreachable, and the app says which one it is running.

The same measurement showed every segment's **end running ~1.3 s late** — the
gate needs a hangover to decide a turn is over, so the audio it hands back ends
in silence. That matters more than it sounds: the embedding window is the
rolling 4 s ending at the cut, and for a 2 s turn a window taken at the moment
the gate noticed is more silence than speaker. So both windows are now aligned
to the **last audible sample**, found relative to the clip's own loudest 20 ms
so a room with a noise floor still trims to the speech.
[`sim/verify_trim.mjs`](../sim/verify_trim.mjs) covers digital silence, room
hiss at two levels, a backchannel, and a quiet speaker at −20 dB — which failed
at first, because the absolute floor was set for a close mic; it is now −56 dBFS.

**Browser audio processing is off by default.** Noise suppression, auto gain
and echo cancellation are built for calls: the first reshapes the spectrum a
speaker model reads, the second hides the level differences between people, and
there is no far end here for the third to cancel. A setting turns them back on
for a very noisy room, with a note about what it costs.

### One window, not two

[`sim/windowed.py`](../sim/windowed.py) embedded on a rolling 4 s window ending
at the cut, so one speaker chopped into 2 s pieces still gave the embedder
enough voice — and the first browser build did the same. Measured with CAM++ on
real speech in meeting-shaped conversations
([`sim/window_vs_segment.py`](../sim/window_vs_segment.py)), the window is a
net loss: it reaches back into the *previous* speaker's turn, and **61–75% of
short replies then embed nearer to whoever spoke before**.

| People | Segment only | Rolling window |
|---|---|---|
| 4 | 4.0 clusters, 0.0% | 4.6 clusters, 2.6% |
| 8 | 7.8 clusters, 0.1% | 8.8 clusters, 2.2% |
| 10 | 8.8 clusters, 8.2% | 10.6 clusters, 9.9% |

The embedder now gets the turn and nothing else. The case the window protected
against is covered another way: Silero's hangover does not cut a turn at a
breath, and the clusterer's guards keep segments under 1.5 s out of the
centroids and defer them to the end. A scale-test decision reversed by a
measurement is this branch working as intended.

## Speaker identification

The clustering algorithm is [`sim/clustering.py`](../sim/clustering.py) ported
to JavaScript, guards and all. [`sim/verify_js_port.mjs`](../sim/verify_js_port.mjs)
re-runs the scale test's claims against the port and reproduces the Python to
within a tenth of a cluster from 2 to 12 people.

### The model

The design specifies ECAPA-TDNN. sherpa-onnx publishes speaker models as
ready-made ONNX — no conversion — and the default here is
**`wespeaker_en_voxceleb_CAM++`**: 29 MB, 80-dim Kaldi fbank in, 512-d
embedding out, VoxCeleb-trained. GitHub release assets send no CORS header, so
the browser cannot fetch it itself; `tools/fetch_models.py` puts it in
`web/models/`. A zh+en CAM++ is listed too, for open question §12.3.

Kaldi fbank had to be written in JavaScript ([`web/js/fbank.js`](../web/js/fbank.js)),
and a speaker model fed features that are *almost* Kaldi's still returns
numbers and still clusters synthetic tones. So it is checked frame-for-frame
against `torchaudio.compliance.kaldi.fbank` on a real utterance
([`tools/make_fbank_ref.py`](../tools/make_fbank_ref.py) →
[`sim/verify_fbank.mjs`](../sim/verify_fbank.mjs)): 407 of 407 frames, maximum
difference 4.7e-3 in the log domain, 4 s of audio in 10 ms.

### Measured on speech, not tones

The earlier fallback was tested on synthetic harmonic tones. A neural speaker
model wants speech, so the test signal is now twelve macOS TTS voices across
US, UK, Australian, Irish, South African and Indian English, eight
meeting-flavoured utterances each from 1.7 to 8.8 s
([`sim/make_tts_corpus.sh`](../sim/make_tts_corpus.sh)), run through the
*reference* pipeline — torchaudio fbank, onnxruntime, `sim/clustering.py` —
by [`sim/campp_reference.py`](../sim/campp_reference.py). Two things came out.

**Per-utterance mean subtraction is not optional.** The model's metadata does not
mention it, and without it the model is useless:

| | within-speaker | between-speaker | gap | 10 people, `SHIPPING` |
|---|---|---|---|---|
| no CMN | 0.830 | 0.506 (max 0.936) | 0.324 | **2.5 clusters, 75.5% confusion** |
| CMN | 0.744 | 0.166 (max 0.770) | **0.578** | 8.9 clusters, 10.4% |
| ECAPA, assumed by the design | 0.720 | 0.320 | 0.400 | exact (synthetic) |

With CMN the gap is *wider* than the ECAPA geometry the scale test assumed.

**The thresholds moved, a little, and the residual is one pair.** `SHIPPING`
(0.60, recluster at 0.72) is exact through 4 people on CAM++, within 1.5% at 8,
and under-splits at 10–12. A scan over the real embeddings gave `SPEAKER_MODEL`: threshold
**0.65**, recluster **0.80** — the 0.72 recluster pass turned out to be the
thing merging the closest pair. With it, over 8 seeds:

| Room | true | clusters | confusion | found |
|---|---|---|---|---|
| 4 people | 4 | 4.2 | 0.0% | 4.0 |
| 8 people | 8 | 8.4 | 0.0% | 8.0 |
| 10 people, first ten voices | 10 | 9.5 | **5.8%** | 9.4 |
| **10 people, one voice swapped** | 10 | **10.1** | **0.0%** | 10.0 |
| 11 people, all but one | 11 | 11.4 | 0.0% | 11.0 |
| 12 people, everyone | 12 | 11.5 | 5.5% | 11.2 |

Every point of confusion in that table is the same pair: two Indian-English
male voices at centroid cosine **0.723**, merged in six seeds of eight. The next
closest pair is at 0.623; the median pair is 0.203. No threshold under 0.72
separates them, and anything over it splits the same person (within-speaker
p05 is 0.43). With either of them out of the room, ten people cluster exactly.

This is §4.6 — *two people who genuinely sound alike are merged* — measured with
a real model on real speech rather than predicted from a geometry, and it is the
concrete reason manual split belongs in P1. The ten-person brief is met, with
that stated exception.

### In the browser

CAM++ runs on onnxruntime-web's wasm backend at **118–136 ms per 4–5 s clip**,
225 ms for 8 s, single-threaded, model load ~370 ms. That is comfortably
real-time for turns that arrive every few seconds.

Getting there found a runtime bug worth recording. On onnxruntime-web
**1.22.0**, the graph optimiser returns deterministic but wrong embeddings for
some clip lengths — cosine to the Python reference 0.17 for a 4.1 s clip, 0.04
for 4.7 s, 0.9997 for 8.0 s, the same wrong answer every run, at every level
but `disabled`. The fbank going in matched torchaudio, so it was the optimiser.
On **1.29.0** every level matches to 1.0000 at the same speed, so that is the
pin, and the reason it must not be lowered casually: the failure is silent and
the labels still look plausible. WebGPU is not used — the 1.22.0 JSEP build
threw creating a session for this model even with a Metal adapter, and wasm is
already fast enough to not be worth a second path.

The browser check itself is simple and worth repeating after any change to
`fbank.js`, `embed.js` or the pin: embed the three files in `web/models/_test/`
and compare against `campp_ref.json` written by `sim/campp_reference.py`. All
three should be above 0.999.

### When the room is not the corpus

Everything above is twelve close-talk synthetic voices. A real table mic shifts
the geometry, and the thresholds are a guess about a room the model has never
heard. Three things put the user in charge of that instead of the numbers.

**A known headcount.** The header has a *Speakers* control. Set it, and the
banked embeddings are re-clustered to exactly that many — the closest pair
merged while there are too many, the least coherent cluster split in two while
there are too few, then every segment given its nearest centroid — instantly,
mid-meeting or at End. On the synthetic geometry it repairs both directions: an
over-split of 128 clusters and an under-split of 5 both come back to exactly 10
at 0% confusion ([`sim/verify_js_port.mjs`](../sim/verify_js_port.mjs)). A
headcount is a fact; a threshold is not.

**Labels that follow the evidence.** Re-clustering and deferral rewrite
assignments after the fact, and the transcript used to follow them only at End
— so an early over-split sat on screen looking wrong until the meeting was
over, and a deferred segment showed as *Speaker 0*. Labels now track the
clusterer live, and a deferred segment shows its nearest centroid with a dashed
outline until it is settled.

**Your microphone, measured.** The self-test's second check records 14 s of
you, reports what the browser actually applied to the track, and prints the
cosine between your own segments — one voice should read above the 0.65
speaker-change threshold. If it does not, the room is splitting one person into
several, and the fixes are physical before they are numerical: closer to the
mic, processing off, a quieter room.

### The built-in fallback

The app still runs with nothing downloaded, on long-term MFCC statistics. It is
not a speaker-verification model. Testing it found the same class of problem
twice: raw log-mel put every voice in one neighbourhood (between-speaker 0.677;
four people → one cluster), fixed by the DCT to MFCCs, dropping `c0`, and a
running mean; then `SHIPPING`'s thresholds, calibrated for a different
geometry, needed their own scan (`SPECTRAL_FALLBACK`, 0.88, chosen on the
over-splitting side because a merge the user undoes beats misattributed lines
they have to find). **At ten speakers it still returns ~27% confusion.** The
app warns on screen past four voices. It is a way to try the UI, not a way to
run a meeting.

## Minutes

`§6`'s schema, prompts and 3000-token budget are kept intact, so moving back
onto the real Qwen endpoint is a settings change. Any OpenAI-compatible
`/v1/chat/completions` endpoint works — OpenAI, OpenRouter, Groq, Together, or
a vLLM or Ollama server on your own network — and Anthropic's `/v1/messages` is
supported as a second wire format. JSON mode is requested where the server has
it and dropped on a `400` that names it. Single pass by default; map-reduce
above ~24k tokens, with the reduce step fed the chunk JSONs rather than the raw
transcript, exactly as §6 specifies.

**Grounding.** §6 names hallucinated action items as the real risk, more than
transcription errors, and asks for `t0` on every item so a reader can check it.
The app now does the first pass of that checking: for each decision and action,
is there transcript within 20 s of its `t0`, and do its content words — reduced
to a four-letter prefix, so *temps* and *temporary* agree — appear in it? Items
come back `ok`, `weak` (a heavy paraphrase, or an invention; only a reader can
tell) or `no-source`. Nothing is deleted; the flags are rendered, the status
line counts them, and a reader's attention goes to the items that need it.
[`sim/verify_grounding.mjs`](../sim/verify_grounding.mjs) covers a verbatim
item, a paraphrase, an invention with a plausible `t0`, an item with nothing
said near its time, and one with no `t0` at all.

⚠ The API key is stored in this browser and sent to the base URL you configure,
from the page. **A wrong base URL is a leaked key.**

⚠ The output is an editable draft — never a finished record. The grounding pass
narrows where to look; it does not make the model honest.

## Verification, in one place

| Check | What it proves |
|---|---|
| `node sim/verify_js_port.mjs` | The JS clusterer reproduces the Python scale-test claims; a known headcount repairs over- and under-splits |
| `node sim/verify_fbank.mjs` | `fbank.js` is Kaldi fbank, frame for frame |
| `node sim/verify_trim.mjs` | Windows end at the last word, not at the gate's hangover |
| `node sim/verify_hallucination_filter.mjs` | Whisper subtitles dropped, backchannels kept |
| `node sim/verify_grounding.mjs` | Minutes items flagged for the right reasons |
| `python3 sim/campp_reference.py` | CAM++ geometry, CMN, the threshold scan, the near-clone pair; writes the browser reference |
| `python3 sim/window_vs_segment.py` | The rolling embedding window against the turn alone, on real embeddings |
| `python3 sim/singlish_wer_reference.py` | Singlish WER through fp32 PyTorch on 30 SASRBench-v1 clips — 18.5% — the number the browser export should reproduce |
| `web/selftest.html` | In your browser: the live pipeline against the reference without a microphone; your microphone and room; the Singlish export's WER |

`sim/make_tts_corpus.sh` needs macOS (`say`). The corpus is not committed; it
regenerates in under a minute.

## What this version gives up

- **MERaLiON**, and with it the `[en, zh]` + Singlish handling that open
  question §12.3 is about. A Singlish Whisper finetune is a real substitute for
  English; it is not the same model.
- **The reuse argument.** The design's whole economic case was that inference
  was already paid for. Here the minutes are somebody's API bill.
- **pyannote's offline refine pass** (§4.2), which [scale-test](scale-test.md)
  moved *forward* into P2 because it is the only automatic fix for a sound-alike
  pair. There is no browser equivalent — and the measurement above is exactly
  the case it exists for.

## What it buys

No deployment, no TLS, no auth, no GPU allocation, no shared-endpoint governor,
and no §3.3 risk of a batch minutes run landing on top of a live voicebot call.
Audio can stay on the machine that recorded it. And ten people are identified
by a real speaker model, in the page, with the one limit the design predicted
now measured rather than predicted. It is the fastest way to put the recording
UI, the live transcript and the clustering in front of a real user — which is
exactly what P0 in the delivery plan is for.
