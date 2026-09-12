#!/usr/bin/env python3
"""Download the speaker-embedding model(s) the browser build can use.

sherpa-onnx publishes speaker models as ready-made ONNX — no conversion — but
GitHub release assets send no CORS header, so the page cannot fetch them
itself. This puts them in web/models/, which the static server serves.

    python3 tools/fetch_models.py                 # CAM++ English (default, 29 MB)
    python3 tools/fetch_models.py --all           # every model listed below
    python3 tools/fetch_models.py campplus_zh_en  # by short name

Why CAM++: 29 MB, 512-d output, 80-dim Kaldi fbank in, VoxCeleb-trained, and
the model the app's thresholds were measured against (docs/html-version.md).
The zh+en variant exists for open question §12.3 in the design.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
OUT = Path(__file__).resolve().parent.parent / "web" / "models"

MODELS = {
    # short name: (file name on the release, note)
    "campplus_en": ("wespeaker_en_voxceleb_CAM++.onnx", "English, VoxCeleb, 29 MB — default"),
    "campplus_zh_en": ("3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx", "Chinese + English, 28 MB"),
    "resnet34_en": ("wespeaker_en_voxceleb_resnet34_LM.onnx", "English, 27 MB"),
    "eres2net_en": ("3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx", "English, 27 MB"),
}


def fetch(short: str) -> Path:
    name, note = MODELS[short]
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  have   {name}")
        return dest
    OUT.mkdir(parents=True, exist_ok=True)
    url = BASE + urllib.parse.quote(name)
    print(f"  fetch  {name}  ({note})")

    def hook(blocks, bs, total):
        if total > 0:
            done = min(100, blocks * bs * 100 // total)
            sys.stdout.write(f"\r         {done:3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    sys.stdout.write("\r         done\n")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"one or more of: {', '.join(MODELS)}")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = list(MODELS) if a.all else (a.names or ["campplus_en"])
    bad = [n for n in names if n not in MODELS]
    if bad:
        print(f"unknown: {', '.join(bad)}. Known: {', '.join(MODELS)}", file=sys.stderr)
        return 2
    print(f"-> {OUT}")
    for n in names:
        fetch(n)
    print("\nIn the app: Settings -> Speaker identification -> pick the model.")
    return 0


if __name__ == "__main__":
    import urllib.parse
    raise SystemExit(main())
