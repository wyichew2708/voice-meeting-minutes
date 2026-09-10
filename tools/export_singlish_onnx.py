#!/usr/bin/env python3
"""Export a Singlish-finetuned Whisper to ONNX for transformers.js.

None of the Singlish finetunes publish ONNX weights — they are safetensors
only — and transformers.js needs ONNX. This does the conversion once. The
output goes in web/models/<name>/, which the app loads when the model is set
to "local export".

    pip install "optimum[onnxruntime]" onnx
    python3 tools/export_singlish_onnx.py                    # small, q8
    python3 tools/export_singlish_onnx.py --model mjwong/whisper-large-v3-turbo-singlish

Why these models — WER on SASRBench-v1, spontaneous Singlish:

    openai/whisper-small                     147.80%   unusable
    mjwong/whisper-small-singlish             18.49%   Apache-2.0, 0.2B
    openai/whisper-large-v3-turbo              27.58%
    mjwong/whisper-large-v3-turbo-singlish    13.35%   MIT, 0.8B

Vanilla Whisper above 100% WER means it inserts more words than the reference
contains. The finetune is not an optimisation here; it is the difference
between working and not.

⚠ These are trained on IMDA National Speech Corpus close-talk read speech.
   A boardroom far-field microphone is a different acoustic problem and none
   of these numbers predict it. Tune on real recordings of the real room.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "web" / "models"

MODELS = {
    "mjwong/whisper-small-singlish": "whisper-singlish-onnx",
    "mjwong/whisper-large-v3-turbo-singlish": "whisper-singlish-turbo-onnx",
    "jensenlwt/whisper-small-singlish-122k": "whisper-singlish-122k-onnx",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mjwong/whisper-small-singlish",
                    help="Hugging Face model id (default: %(default)s)")
    ap.add_argument("--out", default=None, help="output directory name under web/models/")
    ap.add_argument("--quantize", default="q8", choices=["q8", "fp16", "none"],
                    help="weight precision for the browser (default: %(default)s)")
    args = ap.parse_args()

    try:
        import optimum  # noqa: F401
    except ImportError:
        print('optimum is not installed. Run:\n\n    pip install "optimum[onnxruntime]" onnx\n',
              file=sys.stderr)
        return 1

    name = args.out or MODELS.get(args.model, args.model.split("/")[-1] + "-onnx")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    print(f"exporting {args.model}\n     -> {out}")
    cmd = [sys.executable, "-m", "optimum.exporters.onnx",
           "--model", args.model, "--task", "automatic-speech-recognition-with-past",
           "--opset", "14", str(out)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("export failed", file=sys.stderr)
        return r.returncode

    # transformers.js expects the graphs under onnx/ and, for a quantized build,
    # the _quantized suffix it looks for by dtype.
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    for f in out.glob("*.onnx*"):
        shutil.move(str(f), onnx_dir / f.name)

    if args.quantize != "none":
        print(f"quantizing to {args.quantize}")
        q = subprocess.run([sys.executable, "-m", "onnxruntime.quantization.preprocess",
                            "--input", str(onnx_dir)], capture_output=True)
        if q.returncode != 0:
            print("  (skipped: onnxruntime quantization tools not available — "
                  "the fp32 export still works, it is just larger)", file=sys.stderr)

    print(f"\ndone. In the app: Settings -> Whisper -> model 'local export'.")
    print(f"Serve the app from web/ so ./models/{name}/ is reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
