/* Is web/js/fbank.js actually Kaldi fbank?
 *
 * Compares it frame-for-frame against torchaudio.compliance.kaldi.fbank on a
 * real utterance. A speaker model fed features that are subtly not Kaldi's
 * still returns numbers, and still clusters synthetic tones — which is why a
 * cosine-only check is not enough and this exists.
 *
 *   python3 tools/make_fbank_ref.py some_16k_mono.wav   # writes the reference
 *   node sim/verify_fbank.mjs [web/models/_test/fbank_ref.json]
 */
import { readFileSync } from 'node:fs';
import { kaldiFbank } from '../web/js/fbank.js';

const ref = JSON.parse(readFileSync(process.argv[2] || 'web/models/_test/fbank_ref.json', 'utf8'));
const pcm = Float32Array.from(ref.wav);
const t0 = performance.now();
const { data, frames } = kaldiFbank(pcm);
const ms = performance.now() - t0;

const T = ref.fbank.length, D = ref.fbank[0].length;
let maxAbs = 0, sumAbs = 0, worst = [0, 0];
for (let f = 0; f < Math.min(T, frames); f++) {
  for (let m = 0; m < D; m++) {
    const d = Math.abs(data[f * D + m] - ref.fbank[f][m]);
    sumAbs += d;
    if (d > maxAbs) { maxAbs = d; worst = [f, m]; }
  }
}
const meanAbs = sumAbs / (Math.min(T, frames) * D);
const framesOk = frames === T;
const valuesOk = maxAbs < 1e-2;
console.log(`frames      js ${frames}  torchaudio ${T}   ${framesOk ? 'PASS' : 'FAIL'}`);
console.log(`max |diff|  ${maxAbs.toExponential(2)} at frame ${worst[0]} bin ${worst[1]}   ${valuesOk ? 'PASS' : 'FAIL'}`);
console.log(`mean |diff| ${meanAbs.toExponential(2)}`);
console.log(`speed       ${(pcm.length / 16000).toFixed(1)} s of audio in ${ms.toFixed(1)} ms`);
console.log(framesOk && valuesOk ? '\nfbank.js matches torchaudio' : '\nMISMATCH — do not trust embeddings from this');
process.exit(framesOk && valuesOk ? 0 : 1);
