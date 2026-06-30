"""Video ingestion: decode frames from the source ``.MOV``.

Uses imageio's bundled FFMPEG backend (``imageio-ffmpeg``) so no system ffmpeg
is required. Frames are streamed rather than loaded all at once — the source
clip has thousands of frames and only the per-frame centerline is retained
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True)
class VideoMeta:
    """Basic properties of a video file."""

    path: str
    fps: float
    width: int
    height: int
    n_frames: int  # estimated from duration * fps (ffmpeg cannot always count cheaply)
    duration: float


def read_meta(path: str | Path) -> VideoMeta:
    """Read video metadata without decoding every frame."""
    path = str(path)
    reader = imageio.get_reader(path, format="ffmpeg")
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    fps = float(meta["fps"])
    width, height = (int(v) for v in meta["size"])
    duration = float(meta.get("duration", 0.0))
    n_frames = int(round(duration * fps)) if duration else 0
    return VideoMeta(path, fps, width, height, n_frames, duration)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 frame to a float64 grayscale array.

    The footage is effectively monochrome, so a channel mean is sufficient and
    avoids baking in luminosity weights the source does not carry.
    """
    if frame.ndim == 2:
        return frame.astype(np.float64)
    return frame.astype(np.float64).mean(axis=2)


def iter_frames(
    path: str | Path, *, gray: bool = True, stride: int = 1
) -> Iterator[np.ndarray]:
    """Yield frames one at a time.

    Args:
        path: Video file.
        gray: If True, yield float64 grayscale; else raw RGB uint8.
        stride: Yield every ``stride``-th frame (>= 1) to subsample.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        for i, frame in enumerate(reader):
            if i % stride:
                continue
            yield to_gray(frame) if gray else frame
    finally:
        reader.close()


def read_frame(path: str | Path, index: int, *, gray: bool = True) -> np.ndarray:
    """Read a single frame by index."""
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        frame = reader.get_data(index)
    finally:
        reader.close()
    return to_gray(frame) if gray else frame


def content_bbox(frame: np.ndarray, *, threshold: float = 10.0) -> tuple[slice, slice]:
    """Find the non-letterbox content region of a grayscale frame.

    Returns row/column slices that drop near-black borders. For footage with no
    letterboxing (like the source clip) this is the full frame.
    """
    gray = frame if frame.ndim == 2 else to_gray(frame)
    rows = np.where(gray.mean(axis=1) > threshold)[0]
    cols = np.where(gray.mean(axis=0) > threshold)[0]
    if rows.size == 0 or cols.size == 0:
        return slice(0, gray.shape[0]), slice(0, gray.shape[1])
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
