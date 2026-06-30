"""Synthetic frame generator with a known band shape, for ground-truth tests."""

from __future__ import annotations

import numpy as np


def make_band_frame(
    *,
    width: int = 640,
    height: int = 360,
    baseline: float = 180.0,
    amplitude: float = 40.0,
    cycles: float = 3.0,
    band_start: int = 120,
    band_end: int = 520,
    band_sigma: float = 8.0,
    background: float = 140.0,
    peak: float = 80.0,
    noise: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, slice]:
    """Render a frame whose band follows a known sine amplitude.

    Returns:
        frame: float64 grayscale ``(H, W)``.
        true_amplitude: the ground-truth signed displacement per column ``(W,)``
            (0 outside the band region).
        band: slice of columns where the band is present.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(width)
    true_amplitude = np.zeros(width)

    band = slice(band_start, band_end)
    span = band_end - band_start
    phase = 2.0 * np.pi * cycles * (x[band] - band_start) / span
    true_amplitude[band] = amplitude * np.sin(phase)

    rows = np.arange(height)[:, None]
    frame = np.full((height, width), background, dtype=np.float64)
    center = baseline - true_amplitude[band]  # centerline = baseline - amplitude
    gauss = peak * np.exp(-((rows - center[None, :]) ** 2) / (2.0 * band_sigma**2))
    frame[:, band] += gauss
    frame += rng.normal(0.0, noise, size=frame.shape)
    return frame, true_amplitude, band
