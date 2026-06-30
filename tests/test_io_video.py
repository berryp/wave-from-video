"""Tests for video ingestion."""

import numpy as np

from wave_from_video.io_video import (
    content_bbox,
    iter_frames,
    read_frame,
    read_meta,
    to_gray,
)


def test_content_bbox_crops_letterbox():
    frame = np.zeros((100, 200), dtype=np.float64)
    frame[20:80, 30:170] = 150.0  # bright content with black borders
    rows, cols = content_bbox(frame)
    assert rows.start == 20 and rows.stop == 80
    assert cols.start == 30 and cols.stop == 170


def test_content_bbox_full_when_no_border():
    frame = np.full((50, 60), 140.0)
    rows, cols = content_bbox(frame)
    assert (rows.start, rows.stop) == (0, 50)
    assert (cols.start, cols.stop) == (0, 60)


def test_to_gray_reduces_channels():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 60
    rgb[..., 2] = 90
    gray = to_gray(rgb)
    assert gray.shape == (4, 5)
    assert np.allclose(gray, 60.0)


# --- integration tests against the real clip ---


def test_read_meta(video_path):
    meta = read_meta(video_path)
    assert meta.fps > 0
    assert meta.width == 640 and meta.height == 360
    assert meta.n_frames > 100


def test_iter_frames_shape_and_dtype(video_path):
    first = next(iter_frames(video_path, gray=True, stride=1))
    assert first.shape == (360, 640)
    assert first.dtype == np.float64
    assert first.max() > first.min()  # not a blank frame


def test_read_frame_index(video_path):
    frame = read_frame(video_path, 30, gray=True)
    assert frame.shape == (360, 640)
    assert frame.max() > 150  # the bright band is present
