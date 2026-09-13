/* Session orchestration — the gateway's job (§2 ORCH), running in the page.
 *
 * One audio segment produces, in order: an embedding, a cluster assignment, a
 * speaker label, and a transcript line. The two recognisers reach that last
 * step differently and the difference is visible in the code, because it is
 * visible to the user too (see asr.js).
 */
import { AudioCapture, toWav, flatten } from './audio.js';
import { OnlineClusterer, SPEAKER_MODEL, SPECTRAL_FALLBACK, SPECTRAL_RELIABLE_SPEAKERS } from './clustering.js';
import { Embedder } from './embed.js';
import { makeASR, WebSpeechASR } from './asr.js';
import { MinutesClient, toMarkdown, renderTranscript, groundMinutes } from './minutes.js';
import * as store from './store.js';

const $ = (id) => document.getElementById(id);
const hhmm = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const PALETTE = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899',
                 '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#06b6d4', '#a855f7'];

let settings = store.loadSettings();
let capture = null, asr = null, clusterer = null, embedder = null;
let session = null, segIndex = 0, busy = false;

// Browser-recogniser results waiting to be matched to a segment: {text, s, e}
// in capture seconds. Google hears a start ~0.6 s late and finalises ~0.8 s
// after the end; those offsets are subtracted before overlap is measured.
let unplaced = [], placeTimer = null;
const WS_START_LAG = 0.6, WS_END_LAG = 0.8;

// Whisper only: consecutive turns by the same speaker are transcribed as one
// call. Whisper on a 1.5 s snippet is far worse than Whisper on 10 s of the
// same voice, and a 0.7 s hangover cuts a turn at every breath. The run ends
// on a change of speaker, COALESCE_GAP of silence, or COALESCE_MAX seconds —
// the same ~1.2 s settle §1 describes for the final line.
const COALESCE_GAP = 1.2, COALESCE_MAX = 20;
let run = null, runTimer = null;

/* ───────────────────────────────────────────────────────────── session ── */

function newSession() {
  return {
    id: `s${Date.now()}`,
    startedAt: new Date().toISOString(),
    date: new Date().toISOString().slice(0, 10),
    durationSeconds: 0,
    lines: [],                 // {index, t0, seconds, clusterId, text}
    names: {},                 // clusterId -> human name
    minutes: null,
    engine: settings.asrEngine,
  };
}

const nameFor = (cid) => session?.names?.[cid] ?? `Speaker ${cid + 1}`;
const colourFor = (cid) => PALETTE[cid % PALETTE.length];

/** Speaking time per person, keyed by *name* rather than by cluster.
 *
 * Over-splitting is the built-in embedder's failure mode, so giving two
 * clusters the same name has to actually merge them — otherwise the user
 * fixes the labels and the attendee list still shows the person twice. */
function attendees() {
  const by = new Map();
  for (const l of session.lines) {
    const name = nameFor(l.clusterId);
    const e = by.get(name) || { name, speaking_seconds: 0, clusterId: l.clusterId };
    e.speaking_seconds += l.seconds;
    by.set(name, e);
  }
  return [...by.values()].sort((a, b) => b.speaking_seconds - a.speaking_seconds);
}

/* ───────────────────────────────────────────────────────────── controls ── */

async function start() {
  if (!settings.consentAcknowledged) { openConsent(); return; }
  setStatus('starting…');
  session = newSession();
  session.targetSpeakers = parseInt($('speakerCount').value, 10) || null;
  segIndex = 0;
  unplaced = [];
  embedder = new Embedder();

  if (settings.embedBackend === 'onnx' && settings.embedOnnxUrl) {
    try {
      // The wasm build, and this version: see ORT_GRAPH_OPT in embed.js.
      const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
      const ort = await import(`${ORT}ort.wasm.min.mjs`);
      ort.env.wasm.wasmPaths = ORT;                 // else it looks for .wasm on this origin
      ort.env.logLevel = 'error';                   // EP-assignment warnings are noise in a user's console
      setStatus('loading speaker model…');
      await embedder.useOnnx(settings.embedOnnxUrl, ort);
    } catch (e) {
      warn(`Speaker model not loaded (${e.message.slice(0, 80)}). Run tools/fetch_models.py — using the approximate embedder, which is unreliable past 4 people.`);
    }
  }
  $('embedBadge').textContent = embedder.label;
  $('embedBadge').classList.toggle('warn', embedder.backend !== 'onnx');
  // The thresholds belong to the embedding geometry, not to the design, and
  // the two backends do not share one. Picking SHIPPING for the fallback
  // collapsed a four-person meeting into a single cluster when tested.
  clusterer = new OnlineClusterer(embedder.backend === 'onnx' ? SPEAKER_MODEL : SPECTRAL_FALLBACK);

  asr = makeASR(settings.asrEngine, {
    lang: settings.asrLang,
    model: settings.whisperModel,
    device: settings.whisperDevice,
  });

  const loadWhisper = async () => {
    setStatus(`loading Whisper on ${asr.device} — the first run fetches ${asr.device === 'webgpu' ? '~300' : '~410'} MB, then it is cached`);
    await asr.load(({ loaded, total, file }) => {
      setStatus(`loading ${file?.split('/').pop() ?? 'model'} — ${Math.round(100 * loaded / total)}%`);
    });
  };
  try {
    if (settings.asrEngine === 'whisper') {
      try { await loadWhisper(); }
      catch (e) {
        if (asr.device !== 'webgpu') throw e;
        // No WebGPU here — Firefox, older Safari, some GPUs. wasm is slower
        // (about real time on one thread) but it works everywhere.
        warn(`WebGPU unavailable (${e.message.slice(0, 60)}) — Whisper will run on wasm, which is slower`);
        asr = makeASR('whisper', { model: settings.whisperModel, device: 'wasm' });
        await loadWhisper();
      }
    } else {
      await asr.load();
    }
  } catch (e) { fail(e.message); return; }

  asr.addEventListener('interim', (e) => showInterim(e.detail.text));
  asr.addEventListener('final', (e) => onWebSpeechFinal(e.detail));
  asr.addEventListener('asrerror', (e) => warn(`Recogniser: ${e.detail.message}`));

  capture = new AudioCapture({ gate: settings.vadGate, processing: settings.micProcessing });
  capture.addEventListener('segment', (e) => onSegment(e.detail));
  capture.addEventListener('gatefallback', (e) =>
    warn(`Silero VAD did not load (${e.detail.reason.slice(0, 60)}) — using the energy gate; expect more false triggers.`));
  capture.addEventListener("started", (e) => { $("gateBadge").hidden = false;
    $('gateBadge').textContent = e.detail.gate === 'silero' ? 'Silero VAD' : 'energy gate';
    $('gateBadge').classList.toggle('warn', e.detail.gate !== 'silero');
  });
  capture.addEventListener('level', (e) => meter(e.detail));
  capture.addEventListener('speechstart', () => $('mic').classList.add('live'));
  capture.addEventListener('speechend', () => $('mic').classList.remove('live'));

  try { await capture.start(); } catch (e) {
    const why = e.name === 'NotAllowedError' ? 'permission denied — allow the microphone for this page and press Start again'
              : e.name === 'NotFoundError' ? 'no microphone found'
              : e.name === 'NotReadableError' ? 'the microphone is in use by another app'
              : !window.isSecureContext ? 'the page must be on https or localhost for the microphone to work'
              : e.message;
    fail(`Microphone: ${why}.`); return;
  }
  asr.start();
  setRunning(true);
  setStatus('recording');
}

function togglePause() {
  if (!capture) return;
  if (capture.paused) { capture.resume(); asr.start?.(); setStatus('recording'); $('pause').textContent = 'Pause'; }
  else { capture.pause(); asr.stop?.(); setStatus('paused'); $('pause').textContent = 'Resume'; }
}

async function end() {
  if (!capture) return;
  setStatus('finishing…');
  asr.stop();
  await capture.stop();
  clearTimeout(runTimer);
  await flushRun();
  placeFinals(true);
  clusterer.finish();
  if (session.targetSpeakers) clusterer.reclusterTo(session.targetSpeakers);
  syncLabels();
  session.durationSeconds = capture.t;
  await matchVoiceprints();
  render();
  await store.saveSession(session);
  setRunning(false);
  capture = null;
  const found = new Set(session.lines.map(l => l.clusterId)).size;
  if (session.droppedFinals) warn(`${session.droppedFinals} recognised phrase${session.droppedFinals === 1 ? '' : 's'} had no matching turn and were dropped`);
  if (embedder.backend !== 'onnx' && found > SPECTRAL_RELIABLE_SPEAKERS) {
    warn(`${session.lines.length} lines · ${found} speakers — past ${SPECTRAL_RELIABLE_SPEAKERS} voices the built-in embedder mislabels a lot. Expect to merge and rename, or switch to the ECAPA model.`);
  } else {
    setStatus(`ended · ${session.lines.length} lines · ${found} speakers`);
  }
  $('generate').disabled = false;
}

/* ─────────────────────────────────────────────────────────────── the loop ── */

async function onSegment(seg) {
  const index = segIndex++;
  const emb = await embedder.embed(seg.embedPcm);
  const cid = clusterer.add(index, emb, seg.seconds);

  const line = {
    index, t0: seg.start, seconds: seg.seconds, text: '',
    // A deferred segment gets its nearest centroid as a provisional label
    // rather than a blank; resolveDeferred() settles it at End.
    clusterId: cid ?? clusterer.nearest(emb) ?? 0, provisional: cid === null,
  };
  session.lines.push(line);
  syncLabels();
  store.putClip(`${session.id}:${index}`, toWav(seg.pcm)).catch(() => {});

  if (settings.asrEngine === 'whisper') queueForWhisper(seg, line, cid);
  else { placeFinals(); render(); }
}

/** Re-clustering and deferral rewrite assignments after the fact. The
 *  transcript follows them as they happen, so an early over-split visibly
 *  heals when the recluster pass merges it, instead of only at End. */
function syncLabels() {
  for (const l of session.lines) {
    const cid = clusterer.assignment.get(l.index);
    if (cid !== undefined && cid !== null) { l.clusterId = cid; l.provisional = false; }
  }
}

function queueForWhisper(seg, line, cid) {
  clearTimeout(runTimer);
  const joinable = run && cid !== null && run.clusterId === cid
                && seg.start - run.end <= COALESCE_GAP
                && run.seconds + seg.seconds <= COALESCE_MAX;
  if (!joinable) flushRun();
  if (!run) run = { pcms: [], t0: seg.start, end: seg.end, seconds: 0, clusterId: cid, lines: [] };
  run.pcms.push(seg.pcm);
  run.end = seg.end;
  run.seconds += seg.seconds;
  run.lines.push(line);
  render();
  runTimer = setTimeout(flushRun, COALESCE_GAP * 1000);
}

async function flushRun() {
  if (!run) return;
  const r = run; run = null;
  let text = null;
  try { text = await asr.transcribe(flatten(r.pcms), r.seconds); }
  catch (e) { warn(`Transcription failed: ${e.message}`); }
  // One transcript line for the run. The other segments still exist in the
  // clusterer — they trained the centroid — they just do not get a row.
  const head = r.lines[0];
  head.seconds = r.seconds;
  head.text = text || '';
  const rest = new Set(r.lines.slice(1));
  session.lines = session.lines.filter(l => !rest.has(l) && (l !== head || head.text));
  render();
}

function onWebSpeechFinal({ text, wallStart, wallEnd }) {
  const toCapture = (ms) => (ms - capture.startedAt) / 1000;
  unplaced.push({ text, s: toCapture(wallStart) - WS_START_LAG, e: toCapture(wallEnd) - WS_END_LAG });
  placeFinals();
  showInterim('');
  // If no segment arrives to trigger placement (the gate is still in its
  // hangover, or trimmed the turn away), try again shortly.
  clearTimeout(placeTimer);
  placeTimer = setTimeout(() => placeFinals(), 3500);
}

/** Match each waiting result to the segment it overlaps most. One result per
 *  segment is the norm; a second is appended, and a result nothing overlaps
 *  goes to the nearest segment within 4 s or is counted as dropped. */
function placeFinals(force = false) {
  const now = force || !capture ? Infinity : capture.t;
  const keep = [];
  const append = (l, t) => { l.text = l.text ? `${l.text} ${t}` : t; };
  for (const f of unplaced) {
    let best = null, bestOv = 0;
    for (const l of session.lines) {
      const ov = Math.min(f.e, l.t0 + l.seconds) - Math.max(f.s, l.t0);
      if (ov > bestOv) { bestOv = ov; best = l; }
    }
    if (best && bestOv >= Math.min(0.5, 0.3 * Math.max(0.1, f.e - f.s))) { append(best, f.text); continue; }
    if (now - f.e < 3.0) { keep.push(f); continue; }          // its segment may still be coming
    const mid = (f.s + f.e) / 2;
    let near = null, nd = 4.0;
    for (const l of session.lines) {
      const d = Math.abs(l.t0 + l.seconds / 2 - mid);
      if (d < nd) { nd = d; near = l; }
    }
    if (near) append(near, f.text);
    else session.droppedFinals = (session.droppedFinals || 0) + 1;
  }
  unplaced = keep;
  render();
}

/* ────────────────────────────────────────────────────────────── rendering ── */

function render() {
  const box = $('transcript');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  box.innerHTML = '';
  for (const l of session.lines) {
    // Whisper: a turn that is cut but not yet transcribed shows as a pending
    // row in the speaker's colour, so the page visibly heard them.
    const pending = !l.text && settings.asrEngine === 'whisper' && capture;
    if (!l.text && !pending) continue;
    const row = document.createElement('div');
    row.className = 'line' + (pending ? ' pending' : '');
    const cid = l.clusterId;
    const tag = document.createElement('button');
    tag.className = 'who' + (l.provisional ? ' provisional' : '');
    tag.style.background = colourFor(cid);
    tag.textContent = nameFor(cid);
    tag.title = 'Double-click to rename this speaker everywhere';
    tag.ondblclick = () => rename(cid);
    const t = document.createElement('span');
    t.className = 'at'; t.textContent = hhmm(l.t0);
    t.onclick = () => playClip(l.index);
    const txt = document.createElement('span');
    txt.className = 'text'; txt.textContent = pending ? '…' : l.text;
    row.append(t, tag, txt);
    box.appendChild(row);
  }
  if (atBottom) box.scrollTop = box.scrollHeight;
  renderSpeakers();
}

function renderSpeakers() {
  const box = $('speakers');
  box.innerHTML = '';
  for (const a of attendees()) {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.style.borderColor = colourFor(a.clusterId);
    chip.innerHTML = `<span class="dot" style="background:${colourFor(a.clusterId)}"></span>` +
                     `${a.name} <span class="secs">${hhmm(a.speaking_seconds)}</span>`;
    chip.onclick = () => rename(a.clusterId);
    box.appendChild(chip);
  }
}

function showInterim(text) {
  const el = $('interim');
  el.textContent = text;
  el.classList.toggle('on', !!text);
}

/** The user knows how many people are in the room; the thresholds only guess. */
function applySpeakerCount() {
  const n = parseInt($('speakerCount').value, 10) || 0;
  if (session) session.targetSpeakers = n || null;
  if (!clusterer || !session) return;
  if (!n) { setStatus('speaker count: automatic'); return; }
  const changed = clusterer.reclusterTo(n);
  syncLabels();
  render();
  const sus = clusterer.lastRecluster?.suspicious ?? [];
  if (sus.length) {
    const m = sus[0];
    warn(`told ${n}, but the audio sounds like ${n + sus.length}: merged two voices that spoke for ${Math.round(m.secondsA)} s and ${Math.round(m.secondsB)} s and did not sound alike (similarity ${m.cosine.toFixed(2)}). Try ${n + sus.length}.`);
  } else {
    setStatus(`re-clustered to ${n} speakers — ${changed} line${changed === 1 ? '' : 's'} relabelled`);
  }
  store.saveSession(session).catch(() => {});
}

async function rename(cid) {
  const current = session.names[cid] ?? '';
  const name = prompt(`Who is ${nameFor(cid)}?`, current);
  if (name === null) return;
  const clean = name.trim();
  if (!clean) { delete session.names[cid]; }
  else {
    session.names[cid] = clean;
    // Bank the voiceprint so this person is recognised in the next meeting (§4.3).
    const c = clusterer?.clusters.find(x => x.id === cid);
    if (c) store.putVoiceprint({ name: clean, centroid: Array.from(c.centroid), updated: Date.now() }).catch(() => {});
  }
  render();
  if (session.minutes) renderMinutes(session.minutes);
  store.saveSession(session).catch(() => {});
}

/** Name clusters automatically from people seen in previous meetings (§4.3). */
async function matchVoiceprints() {
  let prints = [];
  try { prints = await store.allVoiceprints(); } catch { return; }
  if (!prints.length) return;
  for (const c of clusterer.clusters) {
    if (session.names[c.id]) continue;
    let best = null, bestSim = -1;
    for (const p of prints) {
      const v = p.centroid;
      if (!v || v.length !== c.centroid.length) continue;
      let s = 0;
      for (let i = 0; i < v.length; i++) s += v[i] * c.centroid[i];
      if (s > bestSim) { bestSim = s; best = p; }
    }
    // Deliberately stricter than the live clustering threshold: carrying a
    // wrong name across meetings is worse than leaving one blank.
    if (best && bestSim >= 0.75) session.names[c.id] = best.name;
  }
}

async function playClip(index) {
  const blob = await store.getClip(`${session.id}:${index}`);
  if (!blob) return;
  const a = new Audio(URL.createObjectURL(blob));
  a.play().catch(() => {});
}

function meter({ level, speech }) {
  $('meterFill').style.width = `${Math.min(100, level * 600)}%`;
  $('meterFill').classList.toggle('speech', speech);
}

/* ──────────────────────────────────────────────────────────────── minutes ── */

async function generate() {
  if (busy) return;
  if (!settings.llmBaseUrl || !settings.llmModel) { openSettings(); warn('Set an API endpoint and model first.'); return; }
  const lines = session.lines.filter(l => l.text).map(l => ({ t0: l.t0, speaker: nameFor(l.clusterId), text: l.text }));
  if (!lines.length) { warn('Nothing was transcribed.'); return; }

  busy = true;
  $('generate').disabled = true;
  const client = new MinutesClient({
    baseUrl: settings.llmBaseUrl, apiKey: settings.llmApiKey,
    model: settings.llmModel, format: settings.llmFormat,
  });
  try {
    const m = await client.generate(lines, {
      date: session.date,
      durationSeconds: Math.round(session.durationSeconds),
      attendees: attendees().map(({ name, speaking_seconds }) => ({ name, speaking_seconds })),
    }, { onStage: (s) => setStatus(`minutes · ${s}`) });
    session.minutes = groundMinutes(m, lines);
    renderMinutes(session.minutes);
    await store.saveSession(session);
    const g = session.minutes._grounding;
    const flagged = g.weak + g.unsourced;
    if (flagged) warn(`minutes ready — ${flagged} of ${g.total} items could not be verified against the transcript, check those first`);
    else setStatus(`minutes ready — all ${g.total} decisions and actions verified against the transcript`);
  } catch (e) {
    fail(`Minutes failed: ${e.message}`);
  } finally { busy = false; $('generate').disabled = false; }
}

function renderMinutes(m) {
  $('minutesPane').hidden = false;
  $('minutesBody').innerHTML = '';
  const md = toMarkdown({ ...m, attendees: attendees().map(({ name, speaking_seconds }) => ({ name, speaking_seconds })) });
  const pre = document.createElement('pre');
  pre.className = 'md';
  pre.textContent = md;
  $('minutesBody').appendChild(pre);
  $('minutesBody').dataset.md = md;
}

function download(name, text, type = 'text/markdown') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
}

/* ─────────────────────────────────────────────────────────────── settings ── */

function openSettings() { $('settings').showModal(); }
function openConsent() { $('consent').showModal(); }

function bindSettings() {
  const map = {
    asrEngine: 'asrEngine', asrLang: 'asrLang', vadGate: 'vadGate', micProcessing: 'micProcessing',
    whisperModel: 'whisperModel', whisperDevice: 'whisperDevice',
    embedBackend: 'embedBackend', embedOnnxUrl: 'embedOnnxUrl',
    llmBaseUrl: 'llmBaseUrl', llmApiKey: 'llmApiKey', llmModel: 'llmModel', llmFormat: 'llmFormat',
  };
  for (const [key, id] of Object.entries(map)) {
    const el = $(id);
    if (!el) continue;
    if (el.type === 'checkbox') el.checked = !!settings[key]; else el.value = settings[key] ?? '';
    el.onchange = () => {
      settings[key] = el.type === 'checkbox' ? el.checked : el.value;
      store.saveSettings(settings);
      reflectEngine();
      reflectEmbedder();
    };
  }
  reflectEngine();

  $('testApi').onclick = async () => {
    const out = $('apiResult');
    out.textContent = 'testing…'; out.className = 'result';
    try {
      const c = new MinutesClient({
        baseUrl: settings.llmBaseUrl, apiKey: settings.llmApiKey,
        model: settings.llmModel, format: settings.llmFormat,
      });
      out.textContent = `OK — replied: ${await c.test()}`;
      out.className = 'result ok';
    } catch (e) {
      out.textContent = `Failed: ${e.message}`;
      out.className = 'result bad';
    }
  };
}

function reflectEngine() {
  const whisper = settings.asrEngine === 'whisper';
  $('whisperRow').hidden = !whisper;
  $('webspeechRow').hidden = whisper;
  $('privacyNote').textContent = whisper
    ? 'Local. Audio never leaves this machine; the model is downloaded once and cached.'
    : 'Not local. Chrome sends microphone audio to Google for recognition.';
  $('privacyNote').className = whisper ? 'note ok' : 'note warn';
  $('onnxRow').hidden = settings.embedBackend !== 'onnx';
}

/* ────────────────────────────────────────────────────────────────── chrome ── */

function setRunning(on) {
  $('start').hidden = on;
  $('pause').hidden = !on;
  $('end').hidden = !on;
  $('mic').classList.toggle('on', on);
  if (!on) $('pause').textContent = 'Pause';
}
const setStatus = (t) => { $('status').textContent = t; $('status').className = 'status'; };
const warn = (t) => { $('status').textContent = t; $('status').className = 'status warn'; };
const fail = (t) => { $('status').textContent = t; $('status').className = 'status bad'; setRunning(false); };

function reflectEmbedder() {
  const onnx = settings.embedBackend === 'onnx' && settings.embedOnnxUrl;
  const name = onnx ? settings.embedOnnxUrl.split('/').pop().replace(/\.onnx$/, '') : '';
  $('embedBadge').textContent = onnx ? `${name} (ONNX)` : 'spectral fallback — approximate';
  $('embedBadge').classList.toggle('warn', !onnx);
}

export function init() {
  bindSettings();
  reflectEmbedder();
  $('start').onclick = start;
  $('pause').onclick = togglePause;
  $('end').onclick = end;
  $('generate').onclick = generate;
  $('openSettings').onclick = openSettings;
  $('speakerCount').onchange = applySpeakerCount;
  $('exportMd').onclick = () => download(`minutes-${session?.date ?? 'draft'}.md`, $('minutesBody').dataset.md || '');
  $('exportTxt').onclick = () => {
    const lines = session.lines.filter(l => l.text).map(l => ({ t0: l.t0, speaker: nameFor(l.clusterId), text: l.text }));
    download(`transcript-${session?.date ?? 'draft'}.txt`, renderTranscript(lines), 'text/plain');
  };
  $('acceptConsent').onclick = () => {
    settings.consentAcknowledged = true;
    store.saveSettings(settings);
    $('consent').close();
    start();
  };

  if (!WebSpeechASR.available && settings.asrEngine === 'webspeech') {
    warn('This browser has no built-in speech recognition — use Chrome, or switch to Whisper in settings.');
  }
  setStatus('ready');
}

document.addEventListener('DOMContentLoaded', init);
