/* Does trailingSpeechEnd find the last word and not the gate's hangover?
 * node sim/verify_trim.mjs */
import { trailingSpeechEnd } from '../web/js/audio.js';
const SR = 16000;
let seed = 7;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
/** Pseudo-speech: noise with a syllabic envelope. */
const speech = (s, amp = 0.3) => Float32Array.from({ length: Math.round(SR * s) }, (_, i) =>
  amp * rnd() * (0.55 + 0.45 * Math.sin(2 * Math.PI * 4 * i / SR)));
const quiet = (s, amp = 0) => Float32Array.from({ length: Math.round(SR * s) }, () => amp * rnd());
const cat = (...xs) => { const o = new Float32Array(xs.reduce((a, x) => a + x.length, 0)); let p = 0; for (const x of xs) { o.set(x, p); p += x.length; } return o; };

const cases = [
  ['3 s speech + 1.3 s digital silence',       cat(speech(3), quiet(1.3)),          3.0],
  ['3 s speech + 1.3 s room hiss (-50 dB)',     cat(speech(3), quiet(1.3, 0.002)),  3.0],
  ['3 s speech + 1.3 s louder hiss (-35 dB)',   cat(speech(3), quiet(1.3, 0.012)),  3.0],
  ['2 s speech, no tail',                       speech(2),                           2.0],
  ['0.6 s backchannel + 0.7 s hangover',        cat(speech(0.6), quiet(0.7)),        0.6],
  ['quiet speaker (-20 dB) + 1 s silence',      cat(speech(3, 0.03), quiet(1.0)),    3.0],
];
let bad = 0;
for (const [name, pcm, expect] of cases) {
  const got = trailingSpeechEnd(pcm) / SR;
  const ok = Math.abs(got - expect) <= 0.06;
  if (!ok) bad++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(44)} expect ${expect.toFixed(2)} s  got ${got.toFixed(2)} s`);
}
const allSilence = trailingSpeechEnd(quiet(2)) / SR;
const ok = allSilence === 0; if (!ok) bad++;
console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${'2 s of nothing'.padEnd(44)} expect 0.00 s  got ${allSilence.toFixed(2)} s`);
console.log(bad ? `\n${bad} wrong` : '\nall cases pass');
process.exit(bad ? 1 : 0);
