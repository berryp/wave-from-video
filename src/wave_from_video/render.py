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

from .extract import Waveform  # noqa: E402


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
