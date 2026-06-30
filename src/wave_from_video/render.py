"""Rendering and output: overlay images, waveform plots, and data files.

The overlay (extracted centerline drawn on a frame) is the primary visual
validation. Plotting uses a non-interactive backend so it runs headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample  # noqa: E402

from .extract import Waveform  # noqa: E402
from .temporal import TemporalResult  # noqa: E402

WAV_SAMPLE_RATE = 44100
WAV_DURATION = 2.0


def render_overlay(
    gray: np.ndarray, waveform: Waveform, out_path: str | Path, *, title: str | None = None
) -> Path:
    """Draw the extracted centerline over the frame and save a PNG.

    The line is coloured by confidence so faint/absent regions are visible.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(gray.shape[1] / 100, gray.shape[0] / 100), dpi=100)
    ax.imshow(gray, cmap="gray", aspect="auto")

    present = ~np.isnan(waveform.centerline)
    x = waveform.x[present]
    y = waveform.centerline[present]
    ax.scatter(x, y, c=waveform.confidence[present], cmap="autumn", s=4, vmin=0, vmax=1)
    ax.axhline(waveform.baseline, color="cyan", lw=0.5, alpha=0.6, ls="--")

    ax.set_xlim(0, gray.shape[1])
    ax.set_ylim(gray.shape[0], 0)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Mean-remove and peak-normalise a 1-D signal to ~[-1, 1]."""
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float64))
    sig = sig - sig.mean()
    peak = np.max(np.abs(sig))
    return (sig / peak * 0.95) if peak > 0 else sig


def signal_to_wav(
    signal: np.ndarray,
    out_path: str | Path,
    *,
    sample_rate: int = WAV_SAMPLE_RATE,
    duration: float = WAV_DURATION,
) -> Path:
    """Write a 1-D signal to a WAV, normalised and resampled to an audible rate.

    The waveform values are not audio to begin with; they are normalised and
    stretched over ``duration`` seconds so the file renders as an inspectable
    audio waveform (matplotlib / ``IPython.display.Audio``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_signal(signal)
    n_samples = int(round(sample_rate * duration))
    if norm.size < 2:
        audio = np.zeros(n_samples, dtype=np.float32)
    else:
        audio = resample(norm, n_samples).astype(np.float32)
    sf.write(out_path, audio, sample_rate)
    return out_path


def save_waveform_npz(
    out_path: str | Path,
    *,
    spatial_x: np.ndarray,
    spatial_amplitude: np.ndarray,
    temporal: TemporalResult,
    sample_rate: int = WAV_SAMPLE_RATE,
    duration: float = WAV_DURATION,
) -> Path:
    """Save all waveform arrays + metadata to a ``.npz`` for notebook rendering."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spatial_x=spatial_x,
        spatial_amplitude=spatial_amplitude,
        # 2-D raw arrays stored as float32 to keep the file small; the 1-D
        # signals that drive plots/audio stay float64.
        centerlines=temporal.centerlines.astype(np.float32),
        amplitudes=temporal.amplitudes.astype(np.float32),
        envelope=temporal.envelope,
        fps=temporal.fps,
        stride=temporal.stride,
        wav_sample_rate=sample_rate,
        wav_duration=duration,
    )
    return out_path


def render_waveform_plot(
    out_path: str | Path,
    *,
    spatial_x: np.ndarray,
    spatial_amplitude: np.ndarray,
    envelope: np.ndarray,
    fps: float,
) -> Path:
    """Plot the representative spatial waveform and the temporal envelope."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

    ax1.plot(spatial_x, spatial_amplitude, color="tab:blue", lw=1)
    ax1.axhline(0, color="grey", lw=0.5)
    ax1.set_title("Spatial waveform (representative frame)")
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("amplitude (px)")

    t = np.arange(envelope.size) / fps if fps else np.arange(envelope.size)
    ax2.plot(t, envelope, color="tab:red", lw=1)
    ax2.set_title("Temporal waveform (oscillation over time)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("envelope (px)")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
