"""End-to-end CLI test on a few real frames."""

import numpy as np

from wave_from_video.cli import run


def test_run_pipeline_writes_all_outputs(video_path, tmp_path):
    paths = run(video_path, tmp_path, reducer="rms", stride=1, max_frames=12)
    expected = {"overlay", "npz", "waveform_plot", "spatial_wav", "temporal_wav"}
    assert set(paths) == expected
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    data = np.load(paths["npz"])
    assert data["amplitudes"].shape[0] == 12
    assert data["envelope"].shape == (12,)
