"""Temporal analysis: process every frame into a waveform over time.

Streams the video, extracts each frame's spatial waveform, and assembles:
- ``centerlines`` / ``amplitudes`` — 2-D arrays ``(n_frames, width)``.
- ``envelope`` — one scalar per frame summarising that frame's wiggle, i.e. how
  the reflection oscillates over time (the most audio-like signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .extract import extract_waveform
from .io_video import iter_frames, read_meta


@dataclass(frozen=True)
class TemporalResult:
    """Whole-video waveform data."""

    centerlines: np.ndarray  # (n_frames, width)
    amplitudes: np.ndarray  # (n_frames, width)
    envelope: np.ndarray  # (n_frames,)
    fps: float
    stride: int


def _rms(amp: np.ndarray) -> float:
    finite = amp[np.isfinite(amp)]
    finite = finite[finite != 0.0]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0


def _mean_abs(amp: np.ndarray) -> float:
    finite = amp[np.isfinite(amp)]
    finite = finite[finite != 0.0]
    return float(np.mean(np.abs(finite))) if finite.size else 0.0


def _center_column(amp: np.ndarray) -> float:
    return float(amp[amp.size // 2])


REDUCERS: dict[str, Callable[[np.ndarray], float]] = {
    "rms": _rms,
    "mean": _mean_abs,
    "center": _center_column,
}


def analyze_video(
    path: str | Path,
    *,
    reducer: str = "rms",
    stride: int = 1,
    max_frames: int | None = None,
) -> TemporalResult:
    """Extract per-frame waveforms across the whole video.

    Args:
        path: Video file.
        reducer: How to collapse each frame's amplitude to one envelope sample
            (``rms``, ``mean``, or ``center``).
        stride: Process every ``stride``-th frame.
        max_frames: Optional cap on the number of processed frames.
    """
    if reducer not in REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; choose from {sorted(REDUCERS)}")
    reduce = REDUCERS[reducer]
    meta = read_meta(path)

    centerlines: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    envelope: list[float] = []
    for i, gray in enumerate(iter_frames(path, gray=True, stride=stride)):
        if max_frames is not None and i >= max_frames:
            break
        wf = extract_waveform(gray)
        centerlines.append(wf.centerline)
        amplitudes.append(wf.amplitude)
        envelope.append(reduce(wf.amplitude))

    return TemporalResult(
        centerlines=np.asarray(centerlines),
        amplitudes=np.asarray(amplitudes),
        envelope=np.asarray(envelope),
        fps=meta.fps / stride,
        stride=stride,
    )
