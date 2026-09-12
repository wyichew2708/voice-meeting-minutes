#!/usr/bin/env python3
"""Write the torchaudio Kaldi-fbank reference that sim/verify_fbank.mjs checks
web/js/fbank.js against.

    python3 tools/make_fbank_ref.py some_16k_mono.wav [web/models/_test/fbank_ref.json]

Then:  node sim/verify_fbank.mjs [that json]

The parameters here are wespeaker's, and therefore the ones fbank.js
implements. Change one and the JS must change with it — that is what the
check is for. Reads the WAV with the stdlib `wave` module because
torchaudio.load needs an I/O backend that is frequently not installed.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("web/models/_test/fbank_ref.json")

    import numpy as np
    import torch
    import torchaudio

    with wave.open(str(src)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            print(f"need 16 kHz mono 16-bit, got {w.getframerate()} Hz, "
                  f"{w.getnchannels()} ch, {w.getsampwidth() * 8} bit", file=sys.stderr)
            return 1
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    x = torch.from_numpy(x)

    fb = torchaudio.compliance.kaldi.fbank(
        (x * 32768).unsqueeze(0),        # int16 scale: normalize_samples=0 in the model metadata
        num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, energy_floor=0.0, sample_frequency=16000,
        window_type="povey", preemphasis_coefficient=0.97, remove_dc_offset=True,
        use_energy=False, low_freq=20, high_freq=0, snip_edges=True,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"file": str(src), "sr": 16000, "wav": x.tolist(), "fbank": fb.tolist()}, open(out, "w"))
    print(f"{src}: {x.shape[0]} samples -> fbank {list(fb.shape)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
