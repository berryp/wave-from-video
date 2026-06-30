"""Tests for temporal analysis and data-file output."""

import numpy as np
import soundfile as sf

from wave_from_video.render import (
    WAV_SAMPLE_RATE,
    normalize_signal,
    render_waveform_plot,
    save_waveform_npz,
    signal_to_wav,
)
from wave_from_video.temporal import REDUCERS, TemporalResult, analyze_video


def test_reducers_basic():
    # rms ignores NaN and zero entries; mean uses absolute values.
    amp = np.array([3.0, -4.0, 0.0, np.nan])
    assert abs(REDUCERS["rms"](amp) - np.sqrt(12.5)) < 1e-9
    assert REDUCERS["mean"](np.array([3.0, -5.0])) == 4.0
    assert REDUCERS["center"](np.array([1.0, 2.0, 9.0, 4.0, 5.0])) == 9.0


def test_normalize_signal_range():
    sig = np.array([1.0, 2.0, 3.0, 4.0])
    norm = normalize_signal(sig)
    assert abs(norm.mean()) < 1e-9
    assert np.max(np.abs(norm)) <= 0.95 + 1e-9


def test_signal_to_wav_roundtrip(tmp_path):
    sig = np.sin(np.linspace(0, 8 * np.pi, 500))
    out = signal_to_wav(sig, tmp_path / "s.wav", duration=1.0)
    audio, sr = sf.read(out)
    assert sr == WAV_SAMPLE_RATE
    assert audio.shape[0] == WAV_SAMPLE_RATE  # 1.0s
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


def test_save_npz_keys(tmp_path):
    tr = TemporalResult(
        centerlines=np.zeros((4, 10)),
        amplitudes=np.ones((4, 10)),
        envelope=np.array([1.0, 2.0, 3.0, 4.0]),
        fps=30.0,
        stride=1,
    )
    out = save_waveform_npz(
        tmp_path / "w.npz",
        spatial_x=np.arange(10),
        spatial_amplitude=np.ones(10),
        temporal=tr,
    )
    data = np.load(out)
    for key in ("spatial_x", "spatial_amplitude", "centerlines", "amplitudes",
                "envelope", "fps", "wav_sample_rate", "wav_duration"):
        assert key in data
    assert data["amplitudes"].shape == (4, 10)


def test_render_waveform_plot(tmp_path):
    out = render_waveform_plot(
        tmp_path / "wave.png",
        spatial_x=np.arange(50),
        spatial_amplitude=np.sin(np.linspace(0, 6, 50)),
        envelope=np.cos(np.linspace(0, 6, 40)),
        fps=30.0,
    )
    assert out.exists() and out.stat().st_size > 0


# --- integration: a few real frames ---


def test_analyze_video_small(video_path):
    tr = analyze_video(video_path, reducer="rms", stride=1, max_frames=8)
    assert tr.centerlines.shape[0] == 8
    assert tr.amplitudes.shape == tr.centerlines.shape
    assert tr.envelope.shape == (8,)
    assert np.all(np.isfinite(tr.envelope))
    assert tr.fps > 0
