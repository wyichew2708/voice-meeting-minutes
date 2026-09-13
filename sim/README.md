# sim — does the design hold at ten people?

Runnable checks behind the numbers in
[`../docs/scale-test.md`](../docs/scale-test.md). There is no app yet, so
these test the two things that can be tested without one: **the clustering
algorithm** and **the capacity arithmetic**.

```
python3 test_scale.py      # the regression suite — every claim the doc makes
python3 run_clustering.py  # cluster counts and confusion, 2..12 speakers
python3 stress.py          # sound-alike voices, quiet participants, short turns
python3 windowed.py        # the rolling-embedding-window fix
python3 load.py            # ASR load, prompt timing, transcript and token volume
python3 window_scan.py     # how the safe threshold band narrows with headcount
```

Only `numpy` is needed.

## ⚠ What these do and do not prove

The embeddings are **synthetic** — drawn from a geometry calibrated to
published ECAPA-TDNN cosine statistics (within-speaker 0.72, between-speaker
0.32, degrading sharply for short clips), not extracted from audio.

That makes them a fair test of **the algorithm**: whether greedy
nearest-centroid assignment holds up as speakers are added, where its
thresholds have to sit, and which guards it needs. It makes them **no test at
all of the models**: nothing here says what MERaLiON's word error rate is in a
boardroom, or what ECAPA does with a far-field microphone and two Singaporean
colleagues who sound alike.

Every threshold these tests recommend is still a starting point to be tuned on
real recordings of the real room. What they buy is knowing the *shape* of the
problem before writing the code — and, in one case, that the shipping value in
the first draft of the design was outside the working range entirely.

## Files

| | |
|---|---|
| `speakers.py` | The synthetic embedding geometry, and `calibration()` which asserts it hits its targets |
| `meeting.py` | Turn structure — unequal participation, backchannels, stickiness — and the confusion metric |
| `clustering.py` | The design's online clustering, in two variants: as first written, and with the guards |
| `windowed.py` | Embedding on a rolling window instead of on the ASR segment |
| `stress.py` | Sound-alike voices, the person who barely speaks, an interrupt-heavy meeting |
| `load.py` | ASR requests and audio-seconds, prompt timing, transcript volume, minutes tokens, shared-endpoint capacity |
| `window_scan.py` | The band of thresholds that works, against meeting size |
| `test_scale.py` | Asserts every claim in `docs/scale-test.md` |

## The browser build's checks

The same habit, applied to [`../web/`](../web/). Each is a plain `node` script
with no dependencies beyond the file it checks; the last two need Python and a
Mac. Full write-up in [`../docs/html-version.md`](../docs/html-version.md).

| | |
|---|---|
| `verify_js_port.mjs` | The JS clusterer reproduces the claims above, 2 to 12 people |
| `verify_fbank.mjs` | `web/js/fbank.js` is Kaldi fbank, frame for frame against torchaudio |
| `verify_trim.mjs` | Segment windows end at the last word, not at the gate's hangover |
| `verify_hallucination_filter.mjs` | Whisper's subtitle hallucinations dropped, backchannels kept |
| `verify_grounding.mjs` | Minutes items flagged against the transcript for the right reasons |
| `make_tts_corpus.sh` | Twelve TTS voices, eight utterances each — real speech for the speaker model (macOS `say`) |
| `campp_reference.py` | CAM++ through the reference pipeline: geometry, CMN, the threshold scan, who merges with whom; writes the browser check's reference |
| `window_vs_segment.py` | The rolling embedding window against the turn alone, on real embeddings — the window lost, and `web/js/audio.js` changed |
| `singlish_wer_reference.py` | Singlish WER for the Whisper finetune on 30 SASRBench-v1 clips through fp32 PyTorch: 18.5%, the model card's 18.49% |
| `verify_headcount.py` | The Python port of the known-headcount re-cluster repairs over- and under-splits, like the JS |
| `speaker_embed.py` | CAM++ the reference way — torchaudio Kaldi fbank, mean subtraction, onnxruntime — and the trailing-silence trim, shared by the scripts below |
| `backtest_youtube.py` | The app's pipeline over an excerpt of a YouTube video: speakers found against the known headcount, a labelled transcript to read, optionally a manual RTTM or a second diarizer; audio stays local |
| `backtest_calibrate.py` | The embedding geometry the app sees on a real video and a threshold sweep, against labels you have validated |
| `verify_backtest_parts.py` | The back-test's DER scoring on hand-built cases, and the Python hallucination filter against the JS test's twenty |

⚠ The TTS corpus is real speech through a real speaker model, which the
synthetic geometry above is not — and it is still twelve close-talk synthetic
voices, not ten colleagues around a table mic. It found a real limit (two
near-identical voices merge) and it does not promise the room.
