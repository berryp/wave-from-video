"""Validate the rendering notebook executes end-to-end.

Skips if the committed data files are absent (so it never silently triggers the
multi-minute full-video run in a bare checkout)."""

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "waveform.ipynb"
NPZ = ROOT / "output" / "waveform.npz"


def test_notebook_executes():
    if not NPZ.exists():
        pytest.skip("output/waveform.npz not present; run the pipeline first")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=120, resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    client.execute()  # raises on any cell error
