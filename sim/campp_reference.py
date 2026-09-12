#!/usr/bin/env python3
"""CAM++ on real speech, through the reference pipeline.

torchaudio's Kaldi fbank is the ground truth web/js/fbank.js is checked
against; sim/clustering.py is the validated clusterer. This answers, before
trusting the browser: does CMN matter, what geometry does CAM++ produce, do the
design's thresholds hold on it, and who gets confused with whom. It also
writes the reference embeddings the in-browser check compares against.

    ./sim/make_tts_corpus.sh                   # once, macOS
    python3 tools/fetch_models.py              # once
    python3 sim/campp_reference.py [--corpus sim/corpus] [--model web/models/…CAM++.onnx]

Needs torch, torchaudio, onnxruntime, numpy. Reads WAVs with the stdlib `wave`
module because torchaudio.load needs an I/O backend that is often missing.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import random
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from clustering import Config, OnlineClusterer  # noqa: E402
from meeting import Segment, score  # noqa: E402

VOICES = ['Samantha', 'Daniel', 'Karen', 'Moira', 'Tessa', 'Fred',
          'Rishi', 'Tara', 'Aman', 'Kathy', 'Ralph', 'Albert']
# sim/test_scale.py SHIPPING — calibrated for ECAPA's synthetic geometry
SHIPPING = Config(threshold=0.60, min_centroid_seconds=1.5, margin=0.06,
                  defer_under_seconds=1.5, recluster_every=100, recluster_threshold=0.72)
# web/js/clustering.js SPEAKER_MODEL — what this script measured for CAM++
SPEAKER_MODEL = Config(threshold=0.65, min_centroid_seconds=1.5, margin=0.06,
                       defer_under_seconds=1.5, recluster_every=100, recluster_threshold=0.80)
# the three files the in-browser check embeds
BROWSER_TEST = [('samantha3.wav', 'Samantha', 3), ('daniel5.wav', 'Daniel', 5), ('rishi7.wav', 'Rishi', 7)]


def load_wav(path):
    import torch
    with wave.open(path) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return torch.from_numpy(x)


def fbank(x):
    import torchaudio
    return torchaudio.compliance.kaldi.fbank(
        (x * 32768).unsqueeze(0), num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000, window_type='povey',
        preemphasis_coefficient=0.97, remove_dc_offset=True, use_energy=False,
        low_freq=20, high_freq=0, snip_edges=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default=str(HERE / 'corpus'))
    ap.add_argument('--model', default=str(ROOT / 'web' / 'models' / 'wespeaker_en_voxceleb_CAM++.onnx'))
    ap.add_argument('--ref-out', default=str(ROOT / 'web' / 'models' / '_test'),
                    help='where campp_ref.json and the three test wavs go (served to the browser check)')
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])

    def embed(feats, cmn):
        f = feats.numpy()
        if cmn:
            f = f - f.mean(0, keepdims=True)
        e = sess.run(['embs'], {'feats': f[None].astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)

    data = {}
    for v in VOICES:
        files = sorted(glob.glob(f'{a.corpus}/{v}/*.wav'), key=lambda p: int(Path(p).stem))
        if not files:
            print(f'no corpus at {a.corpus} — run sim/make_tts_corpus.sh', file=sys.stderr)
            return 1
        data[v] = [(load_wav(p).shape[0] / 16000, fbank(load_wav(p))) for p in files]

    t0 = time.time()
    E = {cmn: {v: [(s, embed(f, cmn)) for s, f in data[v]] for v in VOICES} for cmn in (False, True)}
    n = 2 * sum(len(x) for x in data.values())
    print(f'embedded {n} utterances in {time.time() - t0:.1f}s ({(time.time() - t0) / n * 1000:.0f} ms each, CPU)\n')

    # ---- geometry
    def stats(emb):
        within, between, short = [], [], []
        for i, va in enumerate(VOICES):
            for j, vb in enumerate(VOICES):
                for sa, ea in emb[va]:
                    for sb, eb in emb[vb]:
                        if va == vb and ea is eb:
                            continue
                        c = float(ea @ eb)
                        if va == vb:
                            within.append(c)
                            if sa < 2 or sb < 2:
                                short.append(c)
                        elif i < j:
                            between.append(c)
        q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
        return (np.mean(within), q(within, 5), np.mean(short), np.mean(between), q(between, 95), max(between))

    print('geometry (cosine)')
    for cmn in (False, True):
        w, w05, ws, b, b95, bmax = stats(E[cmn])
        print(f'  CMN={cmn!s:5}  within {w:.3f} (p05 {w05:.3f}, short clips {ws:.3f})'
              f'   between {b:.3f} (p95 {b95:.3f}, max {bmax:.3f})   gap {w - b:.3f}')
    print('  ECAPA ref   within 0.720                              between 0.320                     gap 0.400')

    # ---- clustering, meeting-shaped
    def run(emb, people, seed, cfg):
        rng = random.Random(seed)
        order = [(i, v, u) for _ in range(3) for i, v in enumerate(people) for u in range(len(emb[v]))]
        rng.shuffle(order)
        cl, segs = OnlineClusterer(cfg), []
        for idx, (i, v, u) in enumerate(order):
            secs, e = emb[v][u]
            segs.append(Segment(idx, i, secs))
            cl.add(idx, e, secs)
        cl.finish()
        return score(segs, cl.assignment), segs, cl.assignment

    def report(emb, label, people, cfg):
        rs = [run(emb, people, s, cfg)[0] for s in range(a.seeds)]
        c = np.mean([r['clusters'] for r in rs])
        conf = np.mean([r['confusion'] for r in rs])
        found = np.mean([r['speakers_found'] for r in rs])
        ok = abs(c - len(people)) <= 0.5 and conf < 0.05
        print(f"  {'OK ' if ok else '!! '}{label:<40} {len(people):2d} true -> {c:5.1f} clusters, "
              f"{conf * 100:5.1f}% confusion, {found:.1f} found")

    for name, cfg in (('SHIPPING (0.60 / 0.72), no CMN', SHIPPING), ):
        print(f'\n{name}')
        for k in (2, 4, 8, 10, 12):
            report(E[False], f'{k} people', VOICES[:k], cfg)
    for name, cfg in (('SHIPPING (0.60 / 0.72), CMN', SHIPPING), ('SPEAKER_MODEL (0.65 / 0.80), CMN', SPEAKER_MODEL)):
        print(f'\n{name}')
        for k in (2, 4, 8, 10, 12):
            report(E[True], f'{k} people', VOICES[:k], cfg)
        report(E[True], '10 people, Aman swapped for Kathy', VOICES[:8] + ['Kathy', 'Ralph'], cfg)
        report(E[True], '11 people, all but Aman', [v for v in VOICES if v != 'Aman'], cfg)

    # ---- who merges with whom
    pairs = {}
    for s in range(a.seeds):
        _, segs, asg = run(E[True], VOICES, s, SPEAKER_MODEL)
        by = {}
        for sg in segs:
            by.setdefault(asg.get(sg.index), {}).setdefault(sg.speaker, 0.0)
            by[asg.get(sg.index)][sg.speaker] += sg.seconds
        for d in by.values():
            if len(d) > 1:
                top = sorted(d, key=d.get, reverse=True)[:2]
                k = tuple(sorted(VOICES[t] for t in top))
                pairs[k] = pairs.get(k, 0) + 1
    print(f'\nmerged pairs, SPEAKER_MODEL, 12 people, {a.seeds} seeds:')
    for k, c in sorted(pairs.items(), key=lambda x: -x[1])[:5]:
        print(f'  {c}x  {k[0]} + {k[1]}')
    cent = {v: np.mean([e for _, e in E[True][v]], 0) for v in VOICES}
    cent = {v: c / np.linalg.norm(c) for v, c in cent.items()}
    sims = sorted(((float(cent[x] @ cent[y]), x, y) for x, y in itertools.combinations(VOICES, 2)), reverse=True)
    print('closest voice pairs by centroid cosine:')
    for s_, x, y in sims[:4]:
        print(f'  {s_:.3f}  {x} + {y}')
    print(f'  median pair {np.median([s_ for s_, _, _ in sims]):.3f}')

    # ---- reference for the browser check
    out = Path(a.ref_out)
    out.mkdir(parents=True, exist_ok=True)
    ref = {}
    for name, v, i in BROWSER_TEST:
        ref[name] = E[True][v][i - 1][1].tolist()
        shutil.copy(f'{a.corpus}/{v}/{i}.wav', out / name)
    json.dump(ref, open(out / 'campp_ref.json', 'w'))
    print(f'\nwrote {out}/campp_ref.json and {len(BROWSER_TEST)} test wavs for the in-browser check')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
