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
