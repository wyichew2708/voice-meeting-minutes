/* Does the Whisper hallucination filter drop the right things — and, more
 * importantly, keep backchannels? node sim/verify_hallucination_filter.mjs */
import { looksHallucinated } from '../web/js/asr.js';

const cases = [
  // [text, clip seconds, should be dropped, why]
  ['Thank you.', 5.0, true,  'filler over a long clip'],
  ['Thank you.', 0.8, false, 'a person actually saying thank you'],
  ['Yes, agreed.', 1.7, false, 'backchannel — must survive'],
  ['Right.', 0.5, false, 'backchannel — must survive'],
  ['Mm-hmm.', 0.6, false, 'backchannel — must survive'],
  ['you', 3.0, true,  'the classic silence hallucination'],
  ['Subtitles by the Amara.org community', 3.0, true, 'subtitle credit'],
  ['Please like and subscribe!', 2.0, true, 'video outro'],
  ['[Music]', 3.0, true, 'bracketed non-speech tag'],
  ['(applause)', 2.0, true, 'bracketed non-speech tag'],
  ['', 2.0, true, 'empty'],
  ['...', 1.0, true, 'punctuation only'],
  ['the the the the the the the the the the', 3.0, true, 'token loop'],
  ['so we said so we said so we said so we said so we said', 4.0, true, 'phrase loop'],
  [Array(40).fill('word').join(' '), 3.0, true, 'impossible word rate'],
  ['I think Aisha should own the medical reports item, she has the context from last quarter.', 5.5, false, 'ordinary turn'],
  ['The claims backlog is sitting at about three hundred and forty cases this week.', 6.3, false, 'ordinary turn'],
  ['Can we come back to that after the break?', 2.9, false, 'ordinary short turn'],
  ['谢谢大家，我们下周再讨论这个问题。', 3.0, false, 'Chinese content, not a credit'],
  ['字幕由 Amara.org 社区提供', 3.0, true, 'Chinese subtitle credit'],
];
let bad = 0;
for (const [text, secs, expect, why] of cases) {
  const got = looksHallucinated(text, secs);
  const ok = got === expect;
  if (!ok) bad++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${expect ? 'drop' : 'keep'}  ${JSON.stringify(text.slice(0, 44))}${text.length > 44 ? '…' : ''}  (${secs}s) — ${why}`);
}
console.log(bad ? `\n${bad} wrong` : '\nall cases pass');
process.exit(bad ? 1 : 0);
