"""Shared test fixtures and paths."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = REPO_ROOT / "data" / "IMG_4767_2.MOV"
REFERENCE_FRAME = REPO_ROOT / "data" / "reference_frame.png"


@pytest.fixture(scope="session")
def video_path() -> Path:
    if not VIDEO_PATH.exists():
        pytest.skip(f"source video not present: {VIDEO_PATH}")
    return VIDEO_PATH


@pytest.fixture(scope="session")
def reference_frame_path() -> Path:
    if not REFERENCE_FRAME.exists():
        pytest.skip(f"reference frame not present: {REFERENCE_FRAME}")
    return REFERENCE_FRAME
