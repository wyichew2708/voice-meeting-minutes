/* Does the grounding pass flag what it should and leave the rest alone?
 * node sim/verify_grounding.mjs */
import { groundMinutes, toMarkdown } from '../web/js/minutes.js';

const lines = [
  { t0: 12.5, speaker: 'Jimmy', text: 'The claims backlog is sitting at about three hundred and forty cases this week.' },
  { t0: 61.0, speaker: 'Jimmy', text: "Let's hire two temps for six weeks and review the numbers again at the end of October." },
  { t0: 88.2, speaker: 'Aisha', text: "I'll draft the JD by next Tuesday." },
  { t0: 120.0, speaker: 'Wei', text: 'Someone needs to chase the medical reports, they are blocking the whole queue.' },
  { t0: 300.0, speaker: 'Jimmy', text: 'Right, anything else before we close?' },
];
const m = {
  title: 'T', summary: 'S', topics: [], open_questions: [], risks: [],
  decisions: [
    { text: 'Hire two temps for six weeks', t0: 61.0, speaker: 'Jimmy' },              // verbatim
    { text: 'Bring on two temporary staff, revisit in October', t0: 62.0 },            // paraphrase
    { text: 'Migrate the claims system to the new vendor platform', t0: 63.0 },        // invented, plausible t0
  ],
  actions: [
    { text: 'Draft the JD', owner: 'Aisha', due: '2026-09-15', t0: 88.2 },             // verbatim
    { text: 'Chase medical reports', owner: null, due: null, t0: 120.0 },              // verbatim-ish
    { text: 'Schedule the vendor security review', owner: 'Wei', due: null, t0: 240.0 }, // nothing said within 20 s
    { text: 'Book the offsite venue', owner: 'Jimmy', due: null },                     // no t0 at all
  ],
};
const g = groundMinutes(m, lines);
const expect = { decisions: ['ok', 'ok', 'weak'], actions: ['ok', 'ok', 'no-source', 'no-source'] };
let bad = 0;
for (const k of ['decisions', 'actions']) {
  g[k].forEach((x, i) => {
    const ok = x._grounding === expect[k][i];
    if (!ok) bad++;
    console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${k.slice(0, -1).padEnd(8)} ${x._grounding.padEnd(9)} overlap ${x._overlap == null ? '  — ' : x._overlap.toFixed(2)}  ${JSON.stringify(x.text)}`);
  });
}
const s = g._grounding;
const sumOk = s.total === 7 && s.ok === 4 && s.weak === 1 && s.unsourced === 2;
if (!sumOk) bad++;
console.log(`  ${sumOk ? 'PASS' : 'FAIL'}  summary   ${JSON.stringify(s)}`);
const md = toMarkdown({ ...g, date: '2026-09-12', duration_seconds: 600, attendees: [] });
const mdOk = md.includes('**2 flagged**') === false && md.includes('**3 flagged**') && md.includes('⚠ *unverified — check 01:03*') && (md.match(/no source in transcript/g) || []).length === 2;
if (!mdOk) bad++;
console.log(`  ${mdOk ? 'PASS' : 'FAIL'}  markdown  banner + per-item flags rendered`);
console.log(bad ? `\n${bad} wrong` : '\nall cases pass');
process.exit(bad ? 1 : 0);
