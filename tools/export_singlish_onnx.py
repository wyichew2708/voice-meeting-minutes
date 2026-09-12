#!/usr/bin/env python3
"""Export a Singlish-finetuned Whisper to ONNX for transformers.js — the recipe
that actually worked, not the one that looked right.

None of the Singlish finetunes publish ONNX weights and transformers.js needs
them. This drives the official transformers.js conversion script inside an
isolated venv, with the two pins it took three attempts to find:

  * torch 2.5.1 — newer torch routes torch.onnx.export through the dynamo
    exporter, which writes encoder_model.onnx_data; optimum expects the legacy
    exporter's .onnx.data and crashes with FileNotFoundError after the export.
  * onnxscript — imported by torch.onnx on newer torch even on the legacy path.

    python3 tools/export_singlish_onnx.py                                  # small, q8
    python3 tools/export_singlish_onnx.py --model mjwong/whisper-large-v3-turbo-singlish

Output: web/models/<model id>/ with the configs and tokenizer beside four
graphs — encoder + merged decoder in q4 (~300 MB, what WebGPU loads) and in q8
(~410 MB, what wasm loads) — the layout transformers.js reads as
`local/<model id>`, the value already in Settings. Weights download once
(~1 GB for small, ~3.2 GB for the turbo). fp32 graphs are removed unless
--keep-fp32; q8 on WebGPU is never loaded because it produces garbage.

Why these models — WER on SASRBench-v1, spontaneous Singlish:

    openai/whisper-small                     147.80%   unusable
    mjwong/whisper-small-singlish             18.49%   Apache-2.0, 0.2B
    mjwong/whisper-large-v3-turbo-singlish    13.35%   MIT, 0.8B

sim/singlish_wer_reference.py reproduces 18.5% on a 30-clip sample through
PyTorch, and the self-test page reproduces it through the browser export.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "web" / "models"
TJS_TAG = "3.7.5"                      # the transformers.js the app loads from the CDN
UNUSED = ["decoder_model.onnx", "decoder_with_past_model.onnx",
          "decoder_model_quantized.onnx", "decoder_with_past_model_quantized.onnx",
          "decoder_model_q4.onnx", "decoder_with_past_model_q4.onnx"]
FP32 = ["encoder_model.onnx", "decoder_model_merged.onnx"]


def sh(cmd, cwd=None, quiet=False):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE if quiet else None,
                       stderr=subprocess.STDOUT if quiet else None, text=True)
    if r.returncode != 0:
        if quiet and r.stdout:
            print(r.stdout[-3000:])
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mjwong/whisper-small-singlish")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "voice-meeting-minutes" / "tjs"),
                    help="where the venv and the transformers.js checkout live")
    ap.add_argument("--keep-fp32", action="store_true", help="keep the fp32 graphs next to the q8 ones")
    a = ap.parse_args()

    cache = Path(a.cache)
    repo, venv = cache / "repo", cache / "venv"
    cache.mkdir(parents=True, exist_ok=True)
    py = venv / "bin" / "python"

    print("1/4  transformers.js conversion script")
    if not repo.exists():
        try:
            sh(["git", "clone", "-q", "--depth", "1", "--branch", TJS_TAG,
                "https://github.com/huggingface/transformers.js.git", str(repo)])
        except SystemExit:
            sh(["git", "clone", "-q", "--depth", "1", "https://github.com/huggingface/transformers.js.git", str(repo)])

    print("2/4  isolated venv (your global Python is not touched)")
    if not py.exists():
        sh([sys.executable, "-m", "venv", str(venv)])
    sh([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], quiet=True)
    sh([str(py), "-m", "pip", "install", "-q", "-r", str(repo / "scripts" / "requirements.txt")], quiet=True)
    sh([str(py), "-m", "pip", "install", "-q", "onnxscript", "torch==2.5.1"], quiet=True)

    print(f"3/4  export {a.model} -> {OUT_ROOT / a.model}  (downloads the weights on first run)")
    sh([str(py), "-m", "scripts.convert", "--quantize", "--modes", "q8", "q4",
        "--model_id", a.model, "--task", "automatic-speech-recognition-with-past",
        "--output_parent_dir", str(OUT_ROOT), "--skip_validation"], cwd=repo)

    print("4/4  prune graphs transformers.js does not load")
    onnx = OUT_ROOT / a.model / "onnx"
    for f in UNUSED + ([] if a.keep_fp32 else FP32):
        p = onnx / f
        if p.exists():
            p.unlink()
    for f in sorted(onnx.glob("*.onnx")):
        print(f"     {f.stat().st_size / 1e6:7.1f} MB  {f.relative_to(OUT_ROOT)}")
    need = [onnx / f for f in ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx",
                               "encoder_model_q4.onnx", "decoder_model_merged_q4.onnx")]
    if not all(p.exists() for p in need):
        raise SystemExit("export did not produce the q8 and q4 encoder + merged decoder pairs")
    print(f"\ndone. In the app: Settings -> Speech recognition -> Whisper -> 'local/{a.model}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
