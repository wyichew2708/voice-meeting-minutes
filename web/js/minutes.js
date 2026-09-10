/* Minutes generation against a user-supplied API.
 *
 * The design points this at the Qwen3.6 already running on the GPU box (§6).
 * There is no GPU box here, so the endpoint is configuration: base URL, key,
 * model. Anything speaking the OpenAI chat-completions shape works unchanged —
 * OpenAI, OpenRouter, Groq, Together, a local vLLM or Ollama — and Anthropic's
 * /v1/messages is supported as a second wire format.
 *
 * The schema, the token budget and both prompts are §6's, kept deliberately
 * intact so a later move onto the real endpoint is a config change.
 *
 * ⚠ §6's warning is the one that matters most and is reproduced in the UI:
 *   hallucinated action items are a bigger risk than transcription errors.
 *   Every decision and action carries `t0` so a reader can click back and
 *   check it, and the output is an editable draft, never a finished record.
 *
 * ⚠ The key is sent to whatever base URL is configured, from the browser. It
 *   lives in localStorage on this machine and is never sent anywhere else, but
 *   a wrong base URL is a leaked key — the settings panel says so too.
 */

export const MAX_TOKENS = 3000;              // §6: 2000 was tight at ten people
export const SINGLE_PASS_TOKEN_LIMIT = 24000; // chunking starts ~2.5 h (§6)

const SYSTEM = `You write meeting minutes from a speaker-attributed transcript.

Rules, without exception:
- Extract ONLY what is present in the transcript. Never infer an action item,
  a decision, an owner or a date that is not actually said.
- If nothing in a category was discussed, return an empty array. An empty
  array is a correct answer; an invented entry is not.
- Every decision and action MUST carry t0: the start time in seconds of the
  transcript line it came from, copied from the [t=...] marker on that line.
- Owners must be one of the named speakers, copied exactly. If the transcript
  says only "someone will chase this", leave owner null.
- Dates only if stated. Resolve relative dates ("next Friday") against the
  meeting date given, and otherwise leave due null.

Reply with a single JSON object and nothing else.`;

const schemaBlock = (meta) => `Meeting date: ${meta.date}. Duration: ${meta.durationSeconds}s.
Attendees, with speaking time: ${meta.attendees.map(a => `${a.name} (${Math.round(a.speaking_seconds)}s)`).join(', ') || 'unnamed speakers only'}.

Return exactly this shape:
{
  "title": "short descriptive title",
  "summary": "3-5 sentences",
  "topics":    [{"heading": "...", "points": ["..."], "t0": 0.0}],
  "decisions": [{"text": "...", "t0": 0.0, "speaker": "..."}],
  "actions":   [{"text": "...", "owner": "... or null", "due": "YYYY-MM-DD or null", "t0": 0.0}],
  "open_questions": ["..."],
  "risks": ["..."]
}`;

const MAP_SYSTEM = `Extract from this passage of a meeting transcript, as JSON:
{"topics":[],"decisions":[],"actions":[],"questions":[],"quotes":[]}
Only what is explicitly present. Keep the t0 markers. Reply with JSON only.`;

/** Rough token estimate. Good enough to choose a strategy, not for billing. */
export const estimateTokens = (s) => Math.ceil(s.length / 3.6);

export class MinutesClient {
  /** @param cfg {{baseUrl, apiKey, model, format?: 'openai'|'anthropic'}} */
  constructor(cfg) { this.cfg = cfg; }

  get format() {
    if (this.cfg.format && this.cfg.format !== 'auto') return this.cfg.format;
    return /anthropic\.com/i.test(this.cfg.baseUrl || '') ? 'anthropic' : 'openai';
  }

  async _chat(system, user, { maxTokens = MAX_TOKENS, signal } = {}) {
    const base = (this.cfg.baseUrl || '').replace(/\/+$/, '');
    if (!base) throw new Error('No API base URL configured.');
    if (this.format === 'anthropic') {
      const r = await fetch(`${base}/v1/messages`, {
        method: 'POST', signal,
        headers: {
          'content-type': 'application/json',
          'x-api-key': this.cfg.apiKey,
          'anthropic-version': '2023-06-01',
          'anthropic-dangerous-direct-browser-access': 'true',
        },
        body: JSON.stringify({
          model: this.cfg.model, max_tokens: maxTokens, system,
          messages: [{ role: 'user', content: user }],
        }),
      });
      if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
      const j = await r.json();
      return (j.content || []).map(c => c.text || '').join('');
    }

    const r = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST', signal,
      headers: {
        'content-type': 'application/json',
        ...(this.cfg.apiKey ? { authorization: `Bearer ${this.cfg.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: this.cfg.model,
        max_tokens: maxTokens,
        temperature: 0.2,
        messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
      }),
    });
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
    const j = await r.json();
    return j.choices?.[0]?.message?.content ?? '';
  }

  /** Send a one-token request so a misconfigured endpoint fails in settings,
   *  not thirty minutes into a meeting. */
  async test() {
    const t = await this._chat('Reply with the word: ok', 'ok', { maxTokens: 16 });
    return t.trim().slice(0, 40) || '(empty reply)';
  }

  async generate(transcript, meta, { signal, onStage } = {}) {
    const flat = renderTranscript(transcript);
    const tokens = estimateTokens(flat);

    if (tokens <= SINGLE_PASS_TOKEN_LIMIT) {
      onStage?.(`single pass · ~${tokens.toLocaleString()} tokens`);
      const raw = await this._chat(SYSTEM, `${schemaBlock(meta)}\n\nTRANSCRIPT\n${flat}`, { signal });
      return finalise(parseJson(raw), meta);
    }

    // Map-reduce (§6). Chunks are ~4000 tokens with 200 of overlap.
    const chunks = chunk(transcript, 4000, 200);
    onStage?.(`map-reduce · ${chunks.length} chunks · ~${tokens.toLocaleString()} tokens`);
    const partials = [];
    for (let i = 0; i < chunks.length; i++) {
      onStage?.(`map ${i + 1}/${chunks.length}`);
      const raw = await this._chat(MAP_SYSTEM, renderTranscript(chunks[i]), { maxTokens: 1500, signal });
      partials.push(parseJson(raw));
    }
    onStage?.('reduce');
    const raw = await this._chat(
      SYSTEM,
      `${schemaBlock(meta)}\n\nThese are extracts from consecutive passages of one meeting. ` +
      `Deduplicate, order them, and write the meeting document. Keep every t0.\n\n` +
      JSON.stringify(partials),
      { signal });
    return finalise(parseJson(raw), meta);
  }
}

/** Lines the model can cite: the t0 marker is what makes click-back work. */
export function renderTranscript(lines) {
  return lines.map(l => `[t=${l.t0.toFixed(1)}] ${l.speaker}: ${l.text}`).join('\n');
}

function chunk(lines, size, overlap) {
  const out = [];
  let cur = [], tok = 0;
  for (const l of lines) {
    const t = estimateTokens(l.text) + 8;
    if (tok + t > size && cur.length) {
      out.push(cur);
      let back = [], b = 0;
      for (let i = cur.length - 1; i >= 0 && b < overlap; i--) {
        back.unshift(cur[i]); b += estimateTokens(cur[i].text) + 8;
      }
      cur = back; tok = b;
    }
    cur.push(l); tok += t;
  }
  if (cur.length) out.push(cur);
  return out;
}

/** Models wrap JSON in prose and fences however they like. Dig it out. */
export function parseJson(raw) {
  const text = String(raw || '').trim();
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : text;
  try { return JSON.parse(body); } catch {}
  const a = body.indexOf('{'), b = body.lastIndexOf('}');
  if (a >= 0 && b > a) {
    try { return JSON.parse(body.slice(a, b + 1)); } catch {}
  }
  throw new Error('The model did not return usable JSON.');
}

function finalise(m, meta) {
  const arr = (x) => (Array.isArray(x) ? x : []);
  return {
    title: m.title || 'Meeting minutes',
    date: meta.date,
    duration_seconds: meta.durationSeconds,
    attendees: meta.attendees,
    summary: m.summary || '',
    topics: arr(m.topics),
    decisions: arr(m.decisions),
    actions: arr(m.actions),
    open_questions: arr(m.open_questions ?? m.questions),
    risks: arr(m.risks),
  };
}

export function toMarkdown(m) {
  const hhmm = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const L = [`# ${m.title}`, '', `**Date** ${m.date} · **Duration** ${hhmm(m.duration_seconds)}`, ''];
  if (m.attendees?.length) {
    L.push('**Attendees** ' + m.attendees.map(a => `${a.name} (${hhmm(a.speaking_seconds)})`).join(', '), '');
  }
  L.push('## Summary', '', m.summary, '');
  if (m.topics?.length) {
    L.push('## Topics', '');
    for (const t of m.topics) {
      L.push(`### ${t.heading}${t.t0 != null ? ` — ${hhmm(t.t0)}` : ''}`, '');
      for (const p of t.points || []) L.push(`- ${p}`);
      L.push('');
    }
  }
  if (m.decisions?.length) {
    L.push('## Decisions', '');
    for (const d of m.decisions) L.push(`- ${d.text}${d.speaker ? ` — *${d.speaker}*` : ''}${d.t0 != null ? ` (${hhmm(d.t0)})` : ''}`);
    L.push('');
  }
  if (m.actions?.length) {
    L.push('## Action items', '', '| Owner | Action | Due | At |', '|---|---|---|---|');
    for (const a of m.actions) L.push(`| ${a.owner || '_unassigned_'} | ${a.text} | ${a.due || '—'} | ${a.t0 != null ? hhmm(a.t0) : '—'} |`);
    L.push('');
  }
  if (m.open_questions?.length) { L.push('## Open questions', ''); for (const q of m.open_questions) L.push(`- ${q}`); L.push(''); }
  if (m.risks?.length) { L.push('## Risks', ''); for (const r of m.risks) L.push(`- ${r}`); L.push(''); }
  L.push('---', '', '*Draft generated from an automatic transcript. Verify before circulating.*');
  return L.join('\n');
}
