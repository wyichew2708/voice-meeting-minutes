# Does it hold at ten people? — test run

The brief says the app must support a meeting of **at least ten people**. There
is no app yet, so this tests the two things that can be tested without one:
the **clustering algorithm** that decides who is speaking, and the **capacity
arithmetic** of the shared GPU services.

Everything below is reproducible: `cd sim && python3 test_scale.py`.

> ⚠ **The embeddings are synthetic.** They are drawn from a geometry calibrated
> to published ECAPA-TDNN cosine statistics — within-speaker 0.72,
> between-speaker 0.32, degrading sharply for short clips — not extracted from
> audio. This is a fair test of the *algorithm* and a fair test of the
> *arithmetic*. It is **no test of the models**: nothing here says what
> MERaLiON's word error rate is in a boardroom, or what ECAPA does with a
> far-field mic. Every threshold recommended below is still a starting point to
> tune on real recordings.

---

## Summary

Four findings, two of which changed the design.

| # | Finding | Severity |
|---|---|---|
| 1 | **The clustering threshold in the first draft was wrong** — 0.70 sits outside the working range at *every* meeting size, and produces ~95 clusters for 10 speakers | 🔴 design changed |
| 2 | **Embedding on the ASR segment collapses in an interrupt-heavy meeting** — 10 people, 2 s turns → 104 clusters | 🔴 design changed |
| 3 | Two people who genuinely sound alike get merged, and 10 people means 45 chances of that instead of 6 | 🟡 scope changed |
| 4 | Partials fit for **one** concurrent meeting, not two | 🟡 limit documented |

And the good news, which is most of it: **ASR load does not grow with
headcount**, the auto-prompt and transcript volumes are comfortable, the
minutes fit the context window with room to spare, and once the algorithm is
fixed it is *exact* from 2 to 12 people.

---

## 1. 🔴 The threshold was wrong

The design specified assignment to the nearest centroid at **cosine ≥ 0.70**,
described honestly as "a starting point from VoxCeleb-scale conventions". It is
worse than a rough starting point — it is the wrong comparison. 0.70 is a
*segment-to-segment* convention, and the algorithm compares a
**segment to a centroid**. Measured on the generator:

| segment length | vs a mature centroid | vs a centroid that is still one segment |
|---|---|---|
| 0.6 s | 0.679 | 0.577 |
| 1.2 s | 0.725 | 0.616 |
| 2.0 s | 0.782 | 0.664 |
| 4.0 s | 0.849 | 0.720 |
| **different speaker, 4 s** | **0.456** (p99 0.534, max 0.548) | |

A threshold of 0.70 is *above* what a short segment scores against its own
speaker and above what any segment scores against a young centroid. So almost
every backchannel — 35% of all segments — opened a new cluster:

```
v1 — design as written (threshold 0.70, no guards), 60-minute meetings
people  segs  clusters  confusion  oversplit
     2   753     116.8       0.0%      114.8
     4   753     129.2       0.0%      125.2
    10   753     154.0       0.0%      144.0
    12   753     162.9       0.0%      150.9
```

Note the confusion column: **0%**. Every cluster is pure. This failure is
invisible to a purity metric and obvious to a user — a transcript with 154
different `Speaker N` labels for 10 people. Two people would have produced 117.
It was never a ten-person bug; ten people are just where it stops being
arguable.

### The fix

Three guards, and a corrected threshold:

- **Threshold 0.60**, not 0.70 — calibrated against segment-to-centroid cosine.
- **Margin test:** the nearest centroid must beat the runner-up by ≥ 0.06.
  Above threshold but not decisively nearer means *don't guess*.
- **Defer segments under 1.5 s.** The design already refused to *learn*
  centroids from short segments; it still let them *create* clusters, which is
  the same bad evidence doing worse damage. They are now held and placed
  against the finished centroids at the end.
- **Re-cluster every 100 segments**, agglomeratively merging centroid pairs
  closer than 0.72. Online assignment is causal and cannot revisit its early
  guesses; this revisits them.

```
v2 — threshold 0.60 + margin + defer + recluster, 60-minute meetings
people  segs  clusters  confusion  oversplit
     2   753       2.1       0.0%        0.1
     4   753       4.1       0.0%        0.1
     8   753       8.1       0.0%        0.1
    10   753      10.0       0.0%        0.0
    12   753      12.1       0.0%        0.1
```

The guards also make the whole thing far less sensitive to the threshold, which
matters more than the exact value — the number still has to be tuned on real
audio, and this is how much room that tuning has:

| people | usable band, as written | usable band, guarded |
|---|---|---|
| 2 | 0.38 – 0.58 (0.20) | 0.38 – 0.74 (**0.36**) |
| 4 | 0.44 – 0.60 (0.16) | 0.44 – 0.74 (**0.30**) |
| 10 | 0.44 – 0.58 (0.14) | 0.38 – 0.74 (**0.36**) |
| 15 | 0.44 – 0.58 (0.14) | 0.38 – 0.74 (**0.36**) |
| 20 | 0.48 – 0.58 (0.10) | 0.42 – 0.74 (**0.32**) |

Two things to read here. The unguarded band **narrows as people are added** —
0.20 wide at two people, 0.10 at twenty — so a threshold tuned in a small
meeting drifts out of range in a large one. And 0.70 is outside it at every
size. Guarded, the band roughly doubles and stops depending on headcount, which
is what makes a single configured value safe from 2 to 20.

---

## 2. 🔴 One window cannot do two jobs

Ten people interrupt each other far more than four do, and the segmenter's own
turn-change cut makes segments shorter still. Capping every segment short:

```
10 people, embedding computed on the ASR segment
 max segment   clusters
       1.5 s      275.8
       2.0 s      129.2
       3.0 s       10.0
      12.0 s       10.0
```

A cliff between 2 s and 3 s. Below it the transcript is unusable, and a fast
meeting is exactly where it lands.

The cause is that the design used **one window for two jobs**. Transcription
wants a tight segment — cut at the silence, cut at the speaker change. Speaker
identity wants as much of one continuous voice as it can get. Those are
different needs and they do not have to share a window.

**Fix:** transcribe the segment; embed the speaker's **last 4 s of continuous
speech**, spanning as many adjacent segments as the turn contains. It costs one
extra ring buffer on the gateway and nothing at all on the GPU — the sidecar is
called once per segment either way.

```
10 people, 2 s segments      clusters
  embedding on the segment      129.2
  embedding on a rolling 4 s     10.1

worst case (2 s cap), by size:  segment   rolling
   4 people                        93.1       4.1
  10 people                       129.2      10.1
  15 people                       150.2      16.6
  20 people                       128.2      23.8
```

---

## 3. 🟡 Sound-alike voices — and why ten people makes this urgent

Two speakers moved deliberately close together, ten in the room:

| their mutual cosine | clusters | confusion |
|---|---|---|
| 0.44 *(typical strangers)* | 10.0 | 0.0% |
| 0.70 | 10.0 | 0.0% |
| **0.75** | **9.1** | **16.2%** |
| 0.85 | 9.1 | 16.2% |

A sharp cliff at 0.75: the two become one cluster and a sixth of the meeting is
attributed to the wrong person. This is **caused by the re-clustering guard** —
merging anything closer than 0.72 is what repairs over-split, and the price is
that two genuinely similar voices are also merged. It is a real trade, not a
bug, and it is worth making: over-split is certain and constant, sound-alike
pairs are occasional.

But the odds change with headcount. Four people is 6 speaker pairs; **ten people
is 45**. If any given pair has even a 1% chance of sitting that close, the
chance of at least one such pair in the room goes from 6% to 36%.

**Scope change:** manual **split** was deferred out of v1 on the grounds that
the offline `pyannote` pass would handle most of it. At ten people that is no
longer a safe deferral. Per-segment reassignment moves into P1, and the offline
refine pass — which uses different, overlap-aware embeddings and so is not
subject to this same 0.72 ceiling — moves from P4 to P2.

---

## 4. 🟡 Capacity: the good news, and the one limit

**ASR load does not grow with headcount.** With one room microphone only one
person talks at a time, so the audio submitted is bounded by wall-clock speech,
not by how many people are in the room. A 10-person meeting and a 4-person
meeting cost the recogniser the same:

```
60-minute meeting, per minute of meeting
              requests/min   audio s/min   x realtime
finals only           12.6          52.5        0.87x
with partials         38.0         142.2        2.37x
```

Against the endpoint's throughput — taking voicebot's measured ~600 ms for a
short utterance as ~5× realtime on one request stream, which is conservative
since vLLM batches:

```
meetings  partials  live call    demand  headroom   util   verdict
       1      True       True     2.75x     2.25x    55%   OK
       2      True       True     5.12x    -0.12x   102%   OVER
       2     False       True     2.12x     2.88x    42%   OK
       3     False       True     3.00x     2.00x    60%   OK
```

**The limit: partials for one concurrent meeting.** One 10-person meeting with
live interim text, running alongside a live voicebot call, uses 55% of the
recogniser — comfortable. A second such meeting takes it over 100%. This is now
a governor rule: the first meeting to start gets partials, later concurrent
meetings get finals-only until it ends, and they are told so in the header.

Finals-only scales to at least three concurrent meetings with a call running.

---

## 5. What was already fine

**Quiet participants are found.** The person in a ten-person meeting who says
almost nothing is the hardest to cluster, and the guarded algorithm still
recovers them:

| their total speech | speakers found |
|---|---|
| 3 s | 10.0 / 10 |
| 10 s | 10.0 / 10 |
| 60 s | 10.0 / 10 |

⚠ Found is not the same as *named*: the auto-prompt needs 6 s across 2 segments,
so somebody with 3 s of speech gets a cluster but never a prompt card. They stay
`Speaker 7` unless renamed by hand. That is the right trade — prompting on 3 s
of audio would ask the user to name a voice the system has barely heard — but
the UI should show the roster entry so it is at least visible.

**The prompt queue needs ordering, not throttling.** Ten unnamed voices do not
all arrive at once:

```
speakers becoming promptable, median minute:
 10 people:  0.2  0.6  0.9  1.4  1.6  2.1  3.2  4.4  5.7  8.1
 15 people:  0.2  0.6  1.0  1.4  1.7  2.9  3.6  3.9  4.5  5.3  6.2  6.7
```

Five cards inside the first two minutes at ten people, the rest trickling in
over eight. Five stacked cards is a wall, so: **one card at a time**, queued by
speaking time so the person who has talked most is named first, with a count of
how many are waiting behind it.

**Transcript volume is comfortable.** ~753 segments in a 60-minute meeting,
~12.6 per minute, independent of headcount. A 2-hour meeting is ~1500 rows —
enough to want a virtualised list in the transcript pane, not enough to need
pagination.

**The minutes fit easily.** 10 people × 60 minutes ≈ 7,900 words ≈ 10,600
tokens. Map-reduce takes 3 chunks and the reduce step sees 3,200 tokens.

Worth noting: at 12,600 tokens, an hour of ten people **fits a single pass**
inside a 32k context. Map-reduce is therefore not needed until roughly 2.5
hours, and a single pass produces better minutes because nothing has to be
stitched across chunk boundaries. Changed: single pass by default, map-reduce
above a configured token count.

One thing does scale with headcount — the **output**. Ten attendees generate
more action items and more names to resolve than four, and the drafted
`max_tokens: 2000` is tight for that. Raised to 3000.

---

## What changed in the design

| Change | Where |
|---|---|
| Threshold 0.70 → **0.60**, calibrated against segment-to-centroid cosine | §4.1 |
| **Margin test** (0.06) against the runner-up centroid | §4.1 |
| Short segments **deferred**, not allowed to open clusters | §4.1 |
| **Periodic re-clustering** every 100 segments at 0.72 | §4.1 |
| **Rolling 4 s embedding window**, decoupled from the ASR segment | §4.1, §5 |
| Manual **split** into P1; offline refine **P4 → P2** | §4.5, §11 |
| Partials limited to **one concurrent meeting** | §3.3 |
| Prompt cards **queued one at a time**, ordered by speaking time | §4.4 |
| Minutes: **single pass** by default, map-reduce above a token limit; `max_tokens` 2000 → 3000 | §6 |
| Transcript pane **virtualised** | §1 |

## What still has to be tested on real audio

None of the above is a substitute for a recording. In rough order of how likely
they are to overturn something here:

1. **Every threshold**, against the real room and the real microphone.
2. **Far-field capture** — the reverberation of a boardroom table mic degrades
   embeddings in a way this model does not attempt to represent.
3. **Overlapping speech**, which is not modelled here at all and is the single
   biggest gap between this and a real ten-person meeting.
4. **The partial governor against a live voicebot call** — the load test §3.3
   already calls for, now with a concrete number to beat: one meeting should
   sit near 55% utilisation, and a second must be refused partials.
