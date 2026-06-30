"""Validate spatial extraction on the reference frame.

Runs the extractor on ``data/reference_frame.png``, writes an overlay image, and
prints contrast metrics. This is the quickest visual + numeric check that the
extraction tracks the band.

Reusable across sessions:
    uv run python scripts/validate_extraction.py
    uv run python scripts/validate_extraction.py --frame data/reference_frame.png --out output/overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from wave_from_video.extract import band_contrast, extract_waveform
from wave_from_video.render import render_overlay


def load_gray(path: str) -> np.ndarray:
    gray = imageio.imread(path).astype(np.float64)
    return gray.mean(axis=2) if gray.ndim == 3 else gray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default="data/reference_frame.png")
    parser.add_argument("--out", default="output/overlay.png")
    args = parser.parse_args()

    gray = load_gray(args.frame)
    wf = extract_waveform(gray)
    metrics = band_contrast(gray, wf)
    present = int((~np.isnan(wf.centerline)).sum())

    out = render_overlay(gray, wf, args.out, title="Extracted waveform overlay")
    print(f"frame: {args.frame}  present_columns: {present}")
    print(
        f"under_curve: {metrics['under_curve']:.1f}  background: {metrics['background']:.1f}  "
        f"delta: {metrics['delta']:.1f}"
    )
    print(f"overlay written: {Path(out)}")


if __name__ == "__main__":
    main()
