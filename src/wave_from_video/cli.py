"""Command-line entry point: video -> overlay + waveform data files.

    wave-from-video data/IMG_4767_2.MOV

Runs the full pipeline and writes (into ``--out``, default ``output/``):
- ``overlay.png``    — extracted waveform on the representative frame
- ``waveform.npz``   — all arrays + metadata for notebook rendering
- ``waveform.png``   — spatial + temporal waveform plot
- ``spatial.wav``    — representative spatial trace as audio
- ``temporal.wav``   — temporal envelope as audio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .extract import band_contrast, extract_waveform
from .io_video import read_frame
from .render import (
    render_overlay,
    render_waveform_plot,
    save_waveform_npz,
    signal_to_wav,
)
from .temporal import REDUCERS, analyze_video


def run(
    video: str | Path,
    out_dir: str | Path = "output",
    *,
    reducer: str = "rms",
    stride: int = 1,
    max_frames: int | None = None,
) -> dict[str, Path]:
    """Run the full pipeline and return the paths of all written artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal = analyze_video(video, reducer=reducer, stride=stride, max_frames=max_frames)

    # Representative frame = the one with the strongest wiggle.
    rep_idx = int(np.argmax(temporal.envelope))
    spatial_amplitude = temporal.amplitudes[rep_idx]
    spatial_x = np.arange(spatial_amplitude.size)

    # Overlay drawn on the actual representative frame.
    rep_gray = read_frame(video, rep_idx * stride, gray=True)
    rep_wf = extract_waveform(rep_gray)
    contrast = band_contrast(rep_gray, rep_wf)

    paths = {
        "overlay": render_overlay(
            rep_gray, rep_wf, out_dir / "overlay.png", title="Extracted waveform overlay"
        ),
        "npz": save_waveform_npz(
            out_dir / "waveform.npz",
            spatial_x=spatial_x,
            spatial_amplitude=spatial_amplitude,
            temporal=temporal,
        ),
        "waveform_plot": render_waveform_plot(
            out_dir / "waveform.png",
            spatial_x=spatial_x,
            spatial_amplitude=spatial_amplitude,
            envelope=temporal.envelope,
            fps=temporal.fps,
        ),
        "spatial_wav": signal_to_wav(spatial_amplitude, out_dir / "spatial.wav"),
        "temporal_wav": signal_to_wav(temporal.envelope, out_dir / "temporal.wav"),
    }

    print(f"processed {temporal.envelope.size} frames (stride={stride}, reducer={reducer})")
    print(f"representative frame index: {rep_idx * stride}  band delta: {contrast['delta']:.1f}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract a waveform from a water-reflection video.")
    parser.add_argument("video", help="path to the source video")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--reducer", default="rms", choices=sorted(REDUCERS))
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="cap processed frames")
    args = parser.parse_args(argv)
    run(
        args.video,
        args.out,
        reducer=args.reducer,
        stride=args.stride,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
