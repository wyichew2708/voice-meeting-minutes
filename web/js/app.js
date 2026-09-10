/* Session orchestration — the gateway's job (§2 ORCH), running in the page.
 *
 * One audio segment produces, in order: an embedding, a cluster assignment, a
 * speaker label, and a transcript line. The two recognisers reach that last
 * step differently and the difference is visible in the code, because it is
 * visible to the user too (see asr.js).
 */
import { AudioCapture, toWav, SAMPLE_RATE } from './audio.js';
import { OnlineClusterer, SHIPPING, SPECTRAL_FALLBACK, SPECTRAL_RELIABLE_SPEAKERS } from './clustering.js';
import { Embedder } from './embed.js';
import { makeASR, WebSpeechASR } from './asr.js';
import { MinutesClient, toMarkdown, renderTranscript } from './minutes.js';
import * as store from './store.js';

const $ = (id) => document.getElementById(id);
const hhmm = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const PALETTE = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899',
                 '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#06b6d4', '#a855f7'];

let settings = store.loadSettings();
let capture = null, asr = null, clusterer = null, embedder = null;
let session = null, segIndex = 0, pendingText = [], busy = false;

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
  segIndex = 0;
  pendingText = [];
  embedder = new Embedder();

  if (settings.embedBackend === 'onnx' && settings.embedOnnxUrl) {
    try {
      const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.mjs');
      await embedder.useOnnx(settings.embedOnnxUrl, ort);
    } catch (e) {
      warn(`Speaker model failed to load (${e.message}). Falling back to the approximate embedder.`);
    }
  }
  $('embedBadge').textContent = embedder.label;
  $('embedBadge').classList.toggle('warn', embedder.backend !== 'onnx');
  // The thresholds belong to the embedding geometry, not to the design, and
  // the two backends do not share one. Picking SHIPPING for the fallback
  // collapsed a four-person meeting into a single cluster when tested.
  clusterer = new OnlineClusterer(embedder.backend === 'onnx' ? SHIPPING : SPECTRAL_FALLBACK);

  asr = makeASR(settings.asrEngine, {
    lang: settings.asrLang,
    model: settings.whisperModel,
    quantized: settings.whisperDtype,
    device: settings.whisperDevice,
  });

  try {
    if (settings.asrEngine === 'whisper') {
      setStatus('loading Whisper — first run downloads the model');
      await asr.load(({ loaded, total, file }) => {
        setStatus(`downloading ${file?.split('/').pop() ?? 'model'} — ${Math.round(100 * loaded / total)}%`);
      });
    } else {
      await asr.load();
    }
  } catch (e) { fail(e.message); return; }

  asr.addEventListener('interim', (e) => showInterim(e.detail.text));
  asr.addEventListener('final', (e) => onWebSpeechFinal(e.detail));
  asr.addEventListener('asrerror', (e) => warn(`Recogniser: ${e.detail.message}`));

  capture = new AudioCapture();
  capture.addEventListener('segment', (e) => onSegment(e.detail));
  capture.addEventListener('level', (e) => meter(e.detail));
  capture.addEventListener('speechstart', () => $('mic').classList.add('live'));
  capture.addEventListener('speechend', () => $('mic').classList.remove('live'));

  try { await capture.start(); } catch (e) {
    fail(`Microphone: ${e.message}. The page must be on https or localhost.`); return;
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
  clusterer.finish();
  // Deferral and re-clustering both rewrite assignments after the fact, so the
  // rendered transcript has to be rebuilt from the final map, not trusted as
  // it was drawn live. This is what makes a late merge fix earlier lines.
  for (const l of session.lines) {
    const cid = clusterer.assignment.get(l.index);
    if (cid !== undefined && cid !== null) l.clusterId = cid;
  }
  session.durationSeconds = capture.t;
  await matchVoiceprints();
  render();
  await store.saveSession(session);
  setRunning(false);
  capture = null;
  const found = new Set(session.lines.map(l => l.clusterId)).size;
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
    index, t0: seg.start, seconds: seg.seconds,
    clusterId: cid ?? -1, text: '', pending: cid === null,
  };
  session.lines.push(line);
  store.putClip(`${session.id}:${index}`, toWav(seg.pcm)).catch(() => {});

  if (settings.asrEngine === 'whisper') {
    render();
    try {
      const text = await asr.transcribe(seg.pcm);
      line.text = text || '';
    } catch (e) { warn(`Transcription failed: ${e.message}`); }
    if (!line.text) session.lines = session.lines.filter(l => l !== line);
    render();
  } else {
    // The browser recogniser is on its own clock. Text that already arrived is
    // claimed here; text that has not yet is claimed when it does.
    const waiting = pendingText.shift();
    if (waiting) { line.text = waiting; }
    render();
  }
}

function onWebSpeechFinal({ text }) {
  const line = [...session.lines].reverse().find(l => !l.text);
  if (line) { line.text = text; render(); }
  else pendingText.push(text);       // audio segment has not been cut yet
  showInterim('');
}

/* ────────────────────────────────────────────────────────────── rendering ── */

function render() {
  const box = $('transcript');
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
  box.innerHTML = '';
  for (const l of session.lines) {
    if (!l.text) continue;
    const row = document.createElement('div');
    row.className = 'line';
    const cid = l.clusterId;
    const tag = document.createElement('button');
    tag.className = 'who';
    tag.style.background = colourFor(cid);
    tag.textContent = nameFor(cid);
    tag.title = 'Double-click to rename this speaker everywhere';
    tag.ondblclick = () => rename(cid);
    const t = document.createElement('span');
    t.className = 'at'; t.textContent = hhmm(l.t0);
    t.onclick = () => playClip(l.index);
    const txt = document.createElement('span');
    txt.className = 'text'; txt.textContent = l.text;
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
    session.minutes = m;
    renderMinutes(m);
    await store.saveSession(session);
    setStatus('minutes ready — check them before circulating');
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
    asrEngine: 'asrEngine', asrLang: 'asrLang', whisperModel: 'whisperModel',
    whisperDtype: 'whisperDtype', whisperDevice: 'whisperDevice',
    embedBackend: 'embedBackend', embedOnnxUrl: 'embedOnnxUrl',
    llmBaseUrl: 'llmBaseUrl', llmApiKey: 'llmApiKey', llmModel: 'llmModel', llmFormat: 'llmFormat',
  };
  for (const [key, id] of Object.entries(map)) {
    const el = $(id);
    if (!el) continue;
    el.value = settings[key] ?? '';
    el.onchange = () => {
      settings[key] = el.value;
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
  $('embedBadge').textContent = onnx ? 'ECAPA-TDNN (ONNX)' : 'spectral fallback — approximate';
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
