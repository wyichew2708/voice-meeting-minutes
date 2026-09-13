"""CAM++ speaker embeddings, the reference way: torchaudio's Kaldi fbank
(web/js/fbank.js is verified against it), per-utterance mean subtraction,
onnxruntime. Shared by the back-test and the measurement scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "web" / "models" / "wespeaker_en_voxceleb_CAM++.onnx"


class CamPP:
    def __init__(self, model: str | Path = DEFAULT_MODEL):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def embed(self, pcm: np.ndarray) -> np.ndarray | None:
        """Unit-norm 512-d embedding of float PCM at 16 kHz, or None if < 0.1 s."""
        import torch
        import torchaudio
        x = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))
        if x.shape[0] < 400 + 9 * 160:
            return None
        f = torchaudio.compliance.kaldi.fbank(
            (x * 32768).unsqueeze(0), num_mel_bins=80, frame_length=25, frame_shift=10,
            dither=0.0, energy_floor=0.0, sample_frequency=16000, window_type="povey",
            preemphasis_coefficient=0.97, remove_dc_offset=True, use_energy=False,
            low_freq=20, high_freq=0, snip_edges=True).numpy()
        f = f - f.mean(0, keepdims=True)                 # not optional: docs/html-version.md
        e = self.sess.run(["embs"], {"feats": f[None].astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)


def trailing_speech_end(pcm: np.ndarray, chunk: int = 320, rel: float = 0.06, floor: float = 0.0015) -> int:
    """Index just past the last audible sample — the port of audio.js trailingSpeechEnd."""
    n = len(pcm) // chunk
    if not n:
        return len(pcm)
    lv = np.sqrt((pcm[: n * chunk].reshape(n, chunk) ** 2).mean(1))
    thr = max(floor, rel * float(lv.max()))
    idx = np.nonzero(lv >= thr)[0]
    return int((idx[-1] + 1) * chunk) if len(idx) else 0
