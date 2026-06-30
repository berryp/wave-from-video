"""Smoke test: the package imports and exposes its version."""

import wave_from_video


def test_package_imports():
    assert wave_from_video.__version__
