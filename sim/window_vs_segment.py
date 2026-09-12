#!/usr/bin/env python3
"""Does the rolling 4 s embedding window help or hurt, on real embeddings?

sim/windowed.py showed, on the synthetic geometry, that embedding on a rolling
window rescues one speaker chopped into 2 s pieces. But a window that ends at
a short reply reaches back into the PREVIOUS speaker's turn. This measures
both on CAM++ over meeting-shaped conversations built from the TTS corpus:
45% chance the same speaker continues, 0.3 s gaps, six minutes, five seeds.

    ./sim/make_tts_corpus.sh && python3 tools/fetch_models.py    # once
    python3 sim/window_vs_segment.py

Result that changed the app (web/js/audio.js): the window loses at every
size, and 61-75% of short cross-speaker replies embed nearer to whoever
spoke before. The embedder now gets the turn alone.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clustering import Config, OnlineClusterer  # noqa: E402
from meeting import Segment, score  # noqa: E402

VOICES = ['Samantha', 'Daniel', 'Karen', 'Moira', 'Tessa', 'Fred',
          'Rishi', 'Tara', 'Aman', 'Kathy', 'Ralph', 'Albert']
CFG = Config(threshold=0.65, min_centroid_seconds=1.5, margin=0.06,
             defer_under_seconds=1.5, recluster_every=100, recluster_threshold=0.80)
SR, WIN, GAP = 16000, 4.0, 0.3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default=str(HERE / 'corpus'))
    ap.add_argument('--model', default=str(HERE.parent / 'web' / 'models' / 'wespeaker_en_voxceleb_CAM++.onnx'))
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    import torch, torchaudio, onnxruntime as ort  # noqa: E401

    def load_wav(p):
        with wave.open(p) as w:
            return torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768)

    def fbank(x):
        return torchaudio.compliance.kaldi.fbank((x * 32768).unsqueeze(0), num_mel_bins=80, frame_length=25,
                                                 frame_shift=10, dither=0.0, energy_floor=0.0, sample_frequency=16000,
                                                 window_type='povey', preemphasis_coefficient=0.97, remove_dc_offset=True,
                                                 use_energy=False, low_freq=20, high_freq=0, snip_edges=True)

    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])

    def embed(x):
        f = fbank(x).numpy()
        f = f - f.mean(0, keepdims=True)
        e = sess.run(['embs'], {'feats': f[None].astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)

    W = {v: [load_wav(p) for p in sorted(glob.glob(f'{a.corpus}/{v}/*.wav'), key=lambda p: int(Path(p).stem))] for v in VOICES}
    if not all(W.values()):
        print(f'no corpus at {a.corpus} — run sim/make_tts_corpus.sh', file=sys.stderr)
        return 1
    E = {v: [embed(x) for x in W[v]] for v in VOICES}
    C = {v: np.mean(E[v], 0) for v in VOICES}

    def conversation(people, seed, seconds=360):
        rng, cur, out, t = random.Random(seed), None, [], 0
        while t < seconds:
            spk = cur if (cur is not None and rng.random() < 0.45) else rng.choice(people)
            u = rng.randrange(len(W[spk]))
            out.append((spk, u)); cur = spk; t += len(W[spk][u]) / SR + GAP
        return out

    def run(people, seed, windowed):
        cl, segs, prev = OnlineClusterer(CFG), [], None
        cross = wrong = 0
        for idx, (spk, u) in enumerate(conversation(people, seed)):
            x = W[spk][u]; secs = len(x) / SR
            if windowed and secs < WIN and prev is not None:
                need = int(SR * WIN) - len(x) - int(SR * GAP)
                tail = prev[1][-need:] if need > 0 else prev[1][:0]
                e = embed(torch.cat([tail, torch.zeros(int(SR * GAP)), x]))
                if prev[0] != spk:
                    cross += 1
                    if float(e @ C[prev[0]]) > float(e @ C[spk]):
                        wrong += 1
            else:
                e = E[spk][u]
            segs.append(Segment(idx, people.index(spk), secs)); cl.add(idx, e, secs); prev = (spk, x)
        cl.finish()
        return score(segs, cl.assignment), cross, wrong

    print(f'rolling {WIN:.0f} s window vs segment-only, CAM++, SPEAKER_MODEL, 6-minute conversations, {a.seeds} seeds')
    print(f"{'people':>6} | {'segment-only':^24} | {'rolling window':^24} | short cross-speaker replies")
    print(f"{'':>6} | {'clusters':>9} {'confusion':>10}   | {'clusters':>9} {'confusion':>10}   | nearer to the PREVIOUS speaker")
    for n in (4, 8, 10):
        people = VOICES[:n]
        seg = [run(people, s, False)[0] for s in range(a.seeds)]
        win = [run(people, s, True) for s in range(a.seeds)]
        cs, fs = np.mean([r['clusters'] for r in seg]), np.mean([r['confusion'] for r in seg])
        cw, fw = np.mean([r[0]['clusters'] for r in win]), np.mean([r[0]['confusion'] for r in win])
        cross, wrong = sum(r[1] for r in win), sum(r[2] for r in win)
        print(f"{n:>6} | {cs:9.1f} {fs * 100:9.1f}%   | {cw:9.1f} {fw * 100:9.1f}%   | {wrong}/{cross}  ({100 * wrong / max(1, cross):.0f}%)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
