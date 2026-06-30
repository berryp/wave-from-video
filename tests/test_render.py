"""Tests for overlay rendering and the contrast metric."""

import imageio.v2 as imageio
import numpy as np

from wave_from_video.extract import band_contrast, extract_waveform
from wave_from_video.render import render_overlay

from .synthetic import make_band_frame


def test_render_overlay_writes_png(tmp_path):
    frame, _, _ = make_band_frame()
    wf = extract_waveform(frame)
    out = render_overlay(frame, wf, tmp_path / "overlay.png")
    assert out.exists()
    assert out.stat().st_size > 0
    img = imageio.imread(out)
    assert img.ndim >= 2 and min(img.shape[:2]) > 10


def test_band_contrast_on_reference(reference_frame_path):
    gray = imageio.imread(reference_frame_path).astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    wf = extract_waveform(gray)
    metrics = band_contrast(gray, wf)
    assert metrics["delta"] > 20.0
