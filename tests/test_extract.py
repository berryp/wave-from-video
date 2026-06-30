"""Tests for spatial waveform extraction."""

import imageio.v2 as imageio
import numpy as np

from wave_from_video.extract import extract_waveform

from .synthetic import make_band_frame


def test_recovers_known_sine_amplitude():
    """Ground truth: recovered amplitude matches the injected sine."""
    frame, true_amp, band = make_band_frame()
    wf = extract_waveform(frame)

    # Compare only where the band is present and confidently detected.
    mask = np.zeros(frame.shape[1], dtype=bool)
    mask[band] = True
    mask &= wf.confidence >= 0.15
    assert mask.sum() > 300  # most of the band region detected

    recovered = wf.amplitude[mask]
    truth = true_amp[mask]

    corr = np.corrcoef(recovered, truth)[0, 1]
    rmse = np.sqrt(np.mean((recovered - truth) ** 2))
    assert corr >= 0.95, f"correlation too low: {corr:.3f}"
    assert rmse <= 4.0, f"rmse too high: {rmse:.2f}px"


def test_baseline_near_truth():
    frame, _, _ = make_band_frame(baseline=180.0)
    wf = extract_waveform(frame)
    assert abs(wf.baseline - 180.0) <= 3.0


def test_confidence_high_in_band_low_outside():
    frame, _, band = make_band_frame()
    wf = extract_waveform(frame)
    in_band = wf.confidence[band].mean()
    outside = np.concatenate([wf.confidence[:band.start], wf.confidence[band.stop:]])
    assert in_band > 0.5
    assert outside.mean() < 0.15


def test_flat_band_has_zero_amplitude():
    """A perfectly horizontal band should yield ~zero amplitude."""
    frame, _, band = make_band_frame(amplitude=0.0)
    wf = extract_waveform(frame)
    assert np.nanmax(np.abs(wf.amplitude[band])) < 2.0


# --- real reference frame ---


def test_reference_frame_contrast(reference_frame_path):
    """On the real frame, brightness under the extracted centerline beats background."""
    gray = imageio.imread(reference_frame_path).astype(np.float64)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    wf = extract_waveform(gray)

    present = ~np.isnan(wf.centerline)
    assert present.sum() > 50  # the band is found across a meaningful span

    cols = np.where(present)[0]
    rows = np.clip(np.round(wf.centerline[cols]).astype(int), 0, gray.shape[0] - 1)
    under_curve = gray[rows, cols].mean()
    background = np.median(gray)
    assert under_curve > background + 20, (
        f"under-curve {under_curve:.1f} not brighter than background {background:.1f}"
    )

    # The band actually wiggles.
    assert np.nanstd(wf.amplitude[present]) > 1.0
