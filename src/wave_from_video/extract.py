"""Spatial waveform extraction — the core algorithm.

For a single grayscale frame, find the bright reflection band's vertical
position for every column (``centerline``), expressed as a signed displacement
from its resting level (``amplitude``). A per-column ``confidence`` marks where
the band is actually present (it only spans the central part of the width).

The band is blurry, so an intensity-weighted centroid (sub-pixel, noise-robust)
is used rather than an ``argmax``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d


@dataclass(frozen=True)
class Waveform:
    """Per-frame spatial waveform.

    Attributes:
        x: Column indices, shape ``(W,)``.
        centerline: Band row position per column, shape ``(W,)``. NaN where the
            band is absent (confidence below threshold).
        amplitude: Signed displacement ``baseline - centerline`` per column,
            shape ``(W,)``. Zero where the band is absent.
        confidence: Normalised per-column band intensity in ``[0, 1]``.
        baseline: The band's resting row level (weighted median centerline).
        height: Frame height in pixels (for plotting / overlay).
    """

    x: np.ndarray
    centerline: np.ndarray
    amplitude: np.ndarray
    confidence: np.ndarray
    baseline: float
    height: int


def _column_background(gray: np.ndarray, percentile: float) -> np.ndarray:
    """Estimate per-column background brightness (the grey water surface)."""
    return np.percentile(gray, percentile, axis=0)


def extract_waveform(
    gray: np.ndarray,
    *,
    background_percentile: float = 70.0,
    confidence_threshold: float = 0.35,
    frame_smooth_sigma: float = 2.0,
    weight_power: float = 2.0,
    smooth_sigma: float = 2.0,
) -> Waveform:
    """Extract the spatial waveform from one grayscale frame.

    Args:
        gray: 2-D float array ``(H, W)``.
        background_percentile: Per-column percentile used as background; the band
            is what rises above it. Higher values suppress the diffuse water
            texture so the faint band stands out.
        confidence_threshold: Columns whose peak band height (relative to the
            brightest column) is below this are treated as band-absent
            (centerline NaN, amplitude 0).
        frame_smooth_sigma: Gaussian denoising applied to the frame before
            tracing, to suppress the grainy water texture.
        weight_power: Exponent on the above-background brightness used for the
            centroid. ``> 1`` concentrates weight on the bright ridge and
            rejects diffuse background residual.
        smooth_sigma: Gaussian smoothing (in columns) applied to the amplitude.

    Returns:
        A :class:`Waveform`.
    """
    if gray.ndim != 2:
        raise ValueError("expected a 2-D grayscale frame")
    gray = gray.astype(np.float64)
    height, width = gray.shape
    rows = np.arange(height, dtype=np.float64)[:, None]  # (H, 1)

    # Denoise, then subtract per-column background: keep brightness above it.
    smoothed = gaussian_filter(gray, frame_smooth_sigma) if frame_smooth_sigma > 0 else gray
    background = _column_background(smoothed, background_percentile)[None, :]  # (1, W)
    bright = np.clip(smoothed - background, 0.0, None)  # (H, W)

    # Per-column centroid weighted by the band ridge (peak-emphasised).
    weight = bright**weight_power  # (H, W)
    weight_sum = weight.sum(axis=0)  # (W,)
    with np.errstate(invalid="ignore", divide="ignore"):
        centerline = (weight * rows).sum(axis=0) / weight_sum  # (W,)

    # Confidence = normalised per-column peak band height above background.
    peak = bright.max(axis=0)  # (W,)
    confidence = peak / peak.max() if peak.max() > 0 else np.zeros(width)
    present = confidence >= confidence_threshold

    # Baseline = confidence-weighted median of present centerlines.
    if present.any():
        baseline = _weighted_median(centerline[present], confidence[present])
    else:
        baseline = height / 2.0

    amplitude = np.where(present, baseline - centerline, 0.0)
    if smooth_sigma > 0:
        amplitude = gaussian_filter1d(amplitude, smooth_sigma)

    centerline_out = np.where(present, centerline, np.nan)
    return Waveform(
        x=np.arange(width),
        centerline=centerline_out,
        amplitude=amplitude,
        confidence=confidence,
        baseline=float(baseline),
        height=height,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``values`` with non-negative ``weights``."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum = np.cumsum(weights)
    cutoff = cum[-1] / 2.0
    return float(values[np.searchsorted(cum, cutoff)])
