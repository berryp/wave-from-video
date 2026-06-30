"""Extract frames from the source video.

Primary use: produce ``data/reference_frame.png`` — a representative frame where
the reflection band is clearly visible — for tests and the overlay validation.
Can also dump a sample of frames for inspection.

Reusable across sessions:
    uv run python scripts/extract_frames.py                 # write reference_frame.png
    uv run python scripts/extract_frames.py --index 1300    # force a specific frame
    uv run python scripts/extract_frames.py --dump-dir /tmp/frames --stride 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from wave_from_video.io_video import iter_frames, read_frame


def band_score(gray: np.ndarray) -> float:
    """Higher when a bright horizontal band stands out from the background.

    The band concentrates brightness into a few rows, so the spread (std) of the
    row-mean profile peaks when the band is well-defined.
    """
    return float(gray.mean(axis=1).std())


def pick_representative(video: str, stride: int) -> int:
    """Return the index (in original frame numbering) of the clearest band frame."""
    best_idx, best_score = 0, -np.inf
    for i, gray in enumerate(iter_frames(video, gray=True, stride=stride)):
        score = band_score(gray)
        if score > best_score:
            best_idx, best_score = i * stride, score
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="data/IMG_4767_2.MOV")
    parser.add_argument("--out", default="data/reference_frame.png")
    parser.add_argument("--index", type=int, default=None, help="force this frame index")
    parser.add_argument("--stride", type=int, default=30, help="sampling stride when searching")
    parser.add_argument("--dump-dir", default=None, help="also dump sampled frames here")
    args = parser.parse_args()

    if args.dump_dir:
        dump = Path(args.dump_dir)
        dump.mkdir(parents=True, exist_ok=True)
        for i, gray in enumerate(iter_frames(args.video, gray=True, stride=args.stride)):
            imageio.imwrite(dump / f"frame_{i * args.stride:05d}.png", gray.astype(np.uint8))
        print(f"dumped frames to {dump}")

    idx = args.index if args.index is not None else pick_representative(args.video, args.stride)
    gray = read_frame(args.video, idx, gray=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, gray.astype(np.uint8))
    print(f"wrote {out} from frame index {idx} (band_score={band_score(gray):.2f})")


if __name__ == "__main__":
    main()
