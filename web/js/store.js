/* Local persistence. The design's SQLite (api-contract.md) becomes IndexedDB;
 * the shape is deliberately the same so a later port is a rewrite of this file
 * and nothing else.
 *
 * Audio stays as WAV blobs in the same store as the transcript, because §6's
 * click-back from a minute line to the moment in the audio only works if both
 * survive a reload together.
 *
 * Everything is on this machine. Nothing here is uploaded anywhere; the only
 * outbound request the app makes at all is the minutes call to the API the
 * user configured.
 */

const DB = 'voice-minutes';
const VERSION = 1;

function open() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, VERSION);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains('sessions')) {
        db.createObjectStore('sessions', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('clips')) {
        db.createObjectStore('clips', { keyPath: 'key' });
      }
      if (!db.objectStoreNames.contains('voiceprints')) {
        // Cross-meeting identity (§4.3). Centroid + the name a human gave it.
        db.createObjectStore('voiceprints', { keyPath: 'name' });
      }
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

const tx = async (store, mode, fn) => {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(store, mode);
    const out = fn(t.objectStore(store));
    t.oncomplete = () => res(out?.result ?? out);
    t.onerror = () => rej(t.error);
  });
};

export const saveSession = (s) => tx('sessions', 'readwrite', (o) => o.put(s));
export const getSession = (id) => tx('sessions', 'readonly', (o) => o.get(id));
export const allSessions = () => tx('sessions', 'readonly', (o) => o.getAll());
export const deleteSession = async (id) => {
  await tx('sessions', 'readwrite', (o) => o.delete(id));
  const db = await open();
  const t = db.transaction('clips', 'readwrite');
  const store = t.objectStore('clips');
  const req = store.openCursor();
  req.onsuccess = () => {
    const c = req.result;
    if (!c) return;
    if (String(c.key).startsWith(`${id}:`)) c.delete();
    c.continue();
  };
};

export const putClip = (key, blob) => tx('clips', 'readwrite', (o) => o.put({ key, blob }));
export const getClip = async (key) => (await tx('clips', 'readonly', (o) => o.get(key)))?.blob ?? null;

export const putVoiceprint = (v) => tx('voiceprints', 'readwrite', (o) => o.put(v));
export const allVoiceprints = () => tx('voiceprints', 'readonly', (o) => o.getAll());
export const deleteVoiceprint = (name) => tx('voiceprints', 'readwrite', (o) => o.delete(name));

/* ─────────────────────────────────────────────────────────────── settings ── */

const KEY = 'voice-minutes.settings';

export const DEFAULTS = {
  asrEngine: 'webspeech',
  asrLang: 'en-SG',
  whisperModel: 'mjwong/whisper-small-singlish',
  whisperDtype: 'q8',
  whisperDevice: 'webgpu',
  embedBackend: 'spectral',
  embedOnnxUrl: '',
  llmBaseUrl: '',
  llmApiKey: '',
  llmModel: '',
  llmFormat: 'auto',
  consentAcknowledged: false,
};

export function loadSettings() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') }; }
  catch { return { ...DEFAULTS }; }
}

export function saveSettings(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch {}
}
