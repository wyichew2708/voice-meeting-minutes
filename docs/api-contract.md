# API contract (draft)

Companion to [`solution-design.md`](solution-design.md). Everything here is copy-pasteable
into an implementation; nothing here is built yet.

Audio format is **16 kHz, mono, 16-bit little-endian PCM** everywhere — the browser, the
WebSocket, the ASR request body and every speaker model. This is voicebot's format
(`config/rhel.yaml: audio.sample_rate: 16000`), which is why its client code and its ASR
endpoint can be reused without a resampling step anywhere in the path.

---

## 1. WebSocket `/ws`

One socket per meeting. Binary frames are audio; text frames are JSON control and events.
The shape follows voicebot's `/ws` (`src/voicebot/server.py:511`) so the same client
plumbing carries over.

### Client → server

Binary frames — raw PCM, ~64 ms per frame (1024 samples at 16 kHz). Ignored unless the
meeting is `recording`.

```jsonc
// Open a meeting. Server replies meeting_started.
{ "type": "start", "title": "Weekly underwriting sync", "lang": "en",
  "expected_participants": ["Jimmy Chew", "Aisha"],   // optional: seeds library matching
  "interim": true }                                    // live partial lines on/off

// Suspend. Audio frames after this are dropped; the clock stops; the roster survives.
{ "type": "pause" }
{ "type": "resume" }

// Close the recording. Triggers the refine pass, then minutes generation.
{ "type": "end" }

// Name a cluster. `enroll` stores the centroid in the voiceprint library.
{ "type": "rename_speaker", "cluster_id": "c2", "name": "Aisha Rahman", "enroll": true }

// Fold one cluster into another. Segments re-point; centroids merge duration-weighted.
{ "type": "merge_speakers", "from": "c5", "into": "c2" }

// Move a single segment to a different speaker (the manual fix for a misfiled line).
{ "type": "reassign_segment", "segment_id": 418, "cluster_id": "c2" }

// Correct transcribed text in place.
{ "type": "edit_segment", "segment_id": 418, "text": "we cleared about sixty percent" }

// Stop offering the prompt for this cluster.
{ "type": "dismiss_prompt", "cluster_id": "c3" }
```

### Server → client

Every event carries `kind`. Unknown kinds must be ignored by the client, so the server can
add events without breaking an older page.

```jsonc
{ "kind": "meeting_started", "meeting_id": "mtg_01K3n", "started_at": "2026-09-08T02:11:00Z" }

// The line currently being spoken. Replaces any previous partial with the same segment_id.
// Rendered grey and italic. Never persisted.
{ "kind": "partial", "segment_id": 419, "cluster_id": "c3",
  "speaker": "Speaker 3", "text": "and the rest is waiting on medical",
  "t0": 838.2, "t1": 841.9 }

// A closed, transcribed, attributed segment. Persisted. Replaces the partial.
{ "kind": "segment", "segment_id": 419, "cluster_id": "c3", "speaker": "Speaker 3",
  "text": "…and the rest is waiting on medical reports.", "lang": "en",
  "t0": 838.2, "t1": 843.4, "asr_ms": 611, "similarity": 0.81 }

// A segment the gates refused. Not shown in the transcript; surfaced only in the audit log.
{ "kind": "dropped", "t0": 851.0, "t1": 851.6, "why": "voiced_ratio 0.07 over 1.4 s" }

// Ask the user who this is. Non-blocking; the transcript keeps running.
{ "kind": "speaker_prompt", "cluster_id": "c3", "label": "Speaker 3",
  "seconds_heard": 51.4, "segments": 9,
  "sample_url": "/api/meetings/mtg_01K3n/clusters/c3/sample.wav",
  "suggestions": [ {"speaker_id": "spk_7f3a", "name": "Wei Ling", "score": 0.62} ] }

// Applies retroactively: the client re-renders every line whose cluster_id matches.
{ "kind": "speaker_renamed", "cluster_id": "c3", "name": "Aisha Rahman", "enrolled": true }
{ "kind": "speakers_merged", "from": "c5", "into": "c2", "name": "Aisha Rahman" }

// Roster for the sidebar, sent on change. `promptable` is false for a voice
// that has been heard but not enough to ask about (under 6 s) — the roster
// still lists them, or a barely-speaking participant would be invisible.
{ "kind": "roster", "speakers": [
    {"cluster_id": "c1", "name": "Jimmy Chew", "known": true,  "seconds": 812.4, "colour": 0},
    {"cluster_id": "c3", "name": null, "known": false, "seconds": 51.4, "colour": 2,
     "promptable": true, "queue_position": 1} ] }

{ "kind": "state", "state": "recording|paused|refining|generating|done", "elapsed": 843.4 }

// The partial governor (design §3.3) turning live text off and on. Finals are
// unaffected and keep arriving, so this is a note in the header, not an error.
{ "kind": "interim_state", "active": false,
  "why": "other_meeting" }
  // voicebot_call | asr_slow | disabled | other_meeting
  // other_meeting: another meeting holds the partial budget (one at a time)
{ "kind": "status", "text": "Mic went quiet — check the input device" }

// Post-End progress and completion.
{ "kind": "refine_progress", "done": 0.42 }
{ "kind": "refine_done", "relabelled": 37 }     // segments the offline pass moved
{ "kind": "minutes_ready", "meeting_id": "mtg_01K3n" }
{ "kind": "error", "text": "ASR unreachable — recording continues, transcript will backfill" }
```

**Failure posture.** If ASR or the speaker sidecar is unreachable, the recording does **not**
stop. Audio keeps being written to disk and segments queue; the client is told once via
`error`, and the transcript backfills when the service returns. Losing the recording because a
summariser was down would be the worst possible failure of this product.

---

## 2. REST

```
GET    /api/health
       -> {"asr": "…", "speaker": "…", "llm": "…", "ready": true, "recording": 1}

GET    /api/meetings?limit=25
GET    /api/meetings/{id}
DELETE /api/meetings/{id}                        audio + transcript + minutes

GET    /api/meetings/{id}/transcript.json
GET    /api/meetings/{id}/transcript.md
GET    /api/meetings/{id}/transcript.vtt         speaker-labelled captions
GET    /api/meetings/{id}/audio.wav

POST   /api/meetings/{id}/minutes                regenerate (optional {"instructions": "…"})
GET    /api/meetings/{id}/minutes.json
GET    /api/meetings/{id}/minutes.md
GET    /api/meetings/{id}/minutes.docx
PATCH  /api/meetings/{id}/minutes                save user edits

GET    /api/meetings/{id}/clusters/{cid}/sample.wav

GET    /api/speakers                             the voiceprint library
PATCH  /api/speakers/{sid}                       rename
DELETE /api/speakers/{sid}                       delete embedding + unlink name
```

---

## 3. Storage

SQLite for a single-team deployment; the schema moves to Postgres unchanged if it needs to be
multi-tenant. Audio lives on disk as one WAV per meeting, not in the database.

```sql
CREATE TABLE meetings (
  id            TEXT PRIMARY KEY,
  title         TEXT,
  started_at    TEXT NOT NULL,
  ended_at      TEXT,
  duration_s    REAL,
  lang          TEXT,
  state         TEXT NOT NULL,          -- recording|paused|refining|generating|done|failed
  audio_path    TEXT,
  audio_expires TEXT                    -- retention TTL; NULL = keep
);

-- A cluster is one voice within one meeting. The name is here and nowhere else,
-- which is what makes a rename retroactive for free.
CREATE TABLE clusters (
  id          TEXT NOT NULL,
  meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
  label       TEXT NOT NULL,            -- "Speaker 3"
  name        TEXT,                     -- NULL until named
  speaker_id  TEXT REFERENCES speakers(id) ON DELETE SET NULL,
  centroid    BLOB NOT NULL,            -- 192 float32
  seconds     REAL NOT NULL DEFAULT 0,
  segments    INTEGER NOT NULL DEFAULT 0,
  prompted    INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (meeting_id, id)
);

CREATE TABLE segments (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  meeting_id  TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
  cluster_id  TEXT NOT NULL,            -- never a name: see clusters.name
  t0          REAL NOT NULL,
  t1          REAL NOT NULL,
  text        TEXT NOT NULL,
  lang        TEXT,
  similarity  REAL,                     -- cosine to the assigned centroid
  edited      INTEGER NOT NULL DEFAULT 0,
  source      TEXT NOT NULL DEFAULT 'online'   -- 'online' | 'refined'
);
CREATE INDEX segments_meeting_t0 ON segments(meeting_id, t0);

-- The cross-meeting voiceprint library. Opt-in, individually deletable.
CREATE TABLE speakers (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  embedding  BLOB NOT NULL,             -- 192 float32, duration-weighted mean
  samples    INTEGER NOT NULL DEFAULT 0,
  seconds    REAL NOT NULL DEFAULT 0,
  meetings   INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE minutes (
  meeting_id  TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
  json        TEXT NOT NULL,
  markdown    TEXT NOT NULL,
  model       TEXT NOT NULL,
  generated_at TEXT NOT NULL,
  edited      INTEGER NOT NULL DEFAULT 0
);
```

---

## 4. Upstream services

### 4.1 ASR — reused from voicebot, unchanged

```
POST http://127.0.0.1:8801/v1/audio/transcriptions
Content-Type: multipart/form-data
  model = MERaLiON/MERaLiON-3-3B-ASR
  file  = audio.wav                     16 kHz mono 16-bit PCM in a WAV container

200 -> {"text": "we cleared about sixty percent of it"}
```

Called exactly as `voicebot/src/voicebot/runtime/cuda_backend.py:74-104` calls it —
hand-rolled multipart, blocking `urllib` on a small thread pool. No change to the service and
no second copy of the model: **this one endpoint serves finals, partials and voicebot's live
calls.** Finals use the 30 s timeout that client already has; partials go through the governor
(`partials:` above) with a 1.2 s timeout, one in flight, and are dropped rather than queued.

### 4.1b voicebot health — read-only, for yielding

```
GET http://127.0.0.1:8788/api/health
-> {"profile": "cuda", "asr": "…", "ready": true, "on_call": true, …}
```

`on_call` is voicebot's existing signal that a customer call is in progress; its own pre-render
warm-up already blocks on it. The gateway polls it every 5 s and suspends partials while it is
true. If the endpoint is unreachable the gateway **assumes a call is live** and keeps partials
suspended — failing toward the customer's latency rather than toward our nice-to-have.

### 4.2 Speaker sidecar — new, `:8803`

```
POST /embed        {"sample_rate": 16000, "audio": "<base64 PCM>"}
              ->   {"embedding": [192 floats], "ms": 19}

POST /embed_batch  {"sample_rate": 16000, "clips": ["<b64>", …]}
              ->   {"embeddings": [[…], …], "ms": 44}

POST /diarize      {"path": "/data/meetings/mtg_01K3n.wav", "num_speakers": null}
              ->   {"turns": [{"speaker": "SPEAKER_00", "t0": 0.5, "t1": 4.2}, …],
                    "ms": 214000}

GET  /health  ->   {"embed_model": "speechbrain/spkrec-ecapa-voxceleb",
                    "diarize_model": "pyannote/speaker-diarization-3.1", "ready": true}
```

`/diarize` is long-running: called once per meeting on End, and polled via the gateway's
`refine_progress` events rather than held open.

### 4.3 LLM — reused from voicebot, larger `max_tokens`

```
POST http://127.0.0.1:8000/v1/chat/completions
{ "model": "Qwen/Qwen3.6-35B-A3B",
  "messages": [{"role": "system", …}, {"role": "user", …}],
  "max_tokens": 2000,           // voicebot uses 220 — it is classifying one reply
  "temperature": 0.2 }
```

Same client shape as `cuda_backend.complete` (`cuda_backend.py:108-129`). Point it at the
replica reserved for voice work, or a long minutes generation will land on top of a live
voicebot call.

---

## 5. Configuration

voicebot's pattern: a YAML profile per deployment, with environment variables overriding
individual values so one image serves every environment (`voicebot/src/voicebot/config.py`).

```yaml
# config/rhel.yaml
profile: rhel
audio:
  sample_rate: 16000
asr:
  # The voicebot ASR, shared. Finals and partials both go here — there is no
  # second replica. See solution-design.md §3.3 for why, and for the governor
  # below that keeps partials off a live call's back.
  base_url: http://127.0.0.1:8801
  model: MERaLiON/MERaLiON-3-3B-ASR
  final_timeout_s: 30                    # finals are the product; never dropped
partials:
  enabled: true
  max_concurrent_meetings: 1             # measured: a 2nd meeting takes the ASR to 102%
  interval_ms: 1500                      # refresh cadence for the open segment
  interval_ms_after_6s: 3000             # widen as the segment grows (cost, not liveness)
  timeout_s: 1.2                         # a late partial is worthless — drop, never retry
  max_in_flight: 1                       # per meeting, never queued
  backoff_p50_ms: 900                    # above this, double the interval; below, halve it
  # voicebot publishes `on_call` for exactly this — its own warm-up stands aside
  # on the same signal. Partials suspend while a call is live; finals do not.
  yield_to:
    health_url: http://127.0.0.1:8788/api/health
    poll_seconds: 5
speaker:
  base_url: http://127.0.0.1:8803
  # Calibrated in sim/, not borrowed: the first draft's 0.70 was a
  # segment-to-segment convention applied to a segment-to-centroid comparison,
  # and it produced 154 clusters for a 10-person meeting. See docs/scale-test.md.
  match_threshold: 0.60                  # same-cluster, within a meeting
  match_margin: 0.06                     # ...and this much clear of the runner-up
  library_threshold: 0.70                # cross-meeting identification — stricter
  turn_change_threshold: 0.60
  min_centroid_seconds: 1.5              # shorter clips never update a centroid
  defer_under_seconds: 1.5               # ...and never open one: they are held to the end
  embed_window_seconds: 4.0              # embed the turn, not the ASR segment
  recluster_every_segments: 100          # agglomerative repair pass
  recluster_threshold: 0.72              # merge centroids closer than this
llm:
  base_url: http://127.0.0.1:8000        # the voicebot LLM, shared
  model: Qwen/Qwen3.6-35B-A3B
  max_tokens: 3000                       # 10 attendees produce more actions than 4
  temperature: 0.2
  # An hour of ten people is ~10,600 tokens and fits one pass, which produces
  # better minutes than stitching chunks. Chunk only above this.
  single_pass_max_tokens: 24000
segmenter:
  silence_ms: 600
  max_segment_s: 12
  overlap_ms: 200
  partial_interval_ms: 1500
prompt:
  min_seconds: 6.0
  min_segments: 2
  redisplay_after_seconds: 30
  max_visible: 1                         # queued, ordered by speaking time
retention:
  audio_days: 30
  transcript_days: 365
```

```
MINUTES_PROFILE=rhel
MINUTES_ASR_URL=http://127.0.0.1:8801
MINUTES_VOICEBOT_HEALTH_URL=http://127.0.0.1:8788/api/health   # unset to disable yielding
MINUTES_SPEAKER_URL=http://127.0.0.1:8803
MINUTES_LLM_URL=http://127.0.0.1:8000
MINUTES_DB=/var/lib/minutes/minutes.db
MINUTES_AUDIO_DIR=/var/lib/minutes/audio
```

The thresholds are in config rather than in code specifically because §4.1 of the design says
they must be tuned against the real room — tuning them should not need a redeploy. The values
above are the ones `sim/` measured; `sim/window_scan.py` reports how much room each has before
the transcript degrades, which is the number to check after re-tuning.
