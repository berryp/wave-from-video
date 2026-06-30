# wave-from-video

Turn a video of a water reflection into a waveform.

The source footage shows light reflected off moving water as a bright band that
wiggles across a textured background — it already looks like an oscilloscope
trace. This project extracts that band from every frame and produces:

- a **spatial** waveform (the band's vertical displacement across each frame), and
- a **temporal** waveform (how the reflection oscillates over time across frames).

The primary validations are an **overlay image** (the extracted waveform drawn on
top of a reference frame) and **waveform data files** (`.npz` for matplotlib and
`.wav` for audio) that render the usual way in a Jupyter notebook.

Built with [mise](https://mise.jdx.dev/) + [uv](https://docs.astral.sh/uv/).

## Quickstart

```bash
bash scripts/setup.sh                 # install Python 3.14 + deps (uv)
uv run wave-from-video data/IMG_4767_2.MOV
```

This writes to `output/`:

| File | Description |
|------|-------------|
| `overlay.png` | Extracted waveform drawn on the representative frame (visual validation) |
| `waveform.npz` | All arrays + metadata (`spatial_*`, `centerlines`, `amplitudes`, `envelope`, `fps`) |
| `waveform.png` | Spatial + temporal waveform plot |
| `spatial.wav`, `temporal.wav` | Signals as audio (normalised, resampled to 44.1 kHz) |

Then open `notebooks/waveform.ipynb` to render the plots and audio the usual way.

## How it works

1. **Ingest** — decode frames via `imageio-ffmpeg` (no system ffmpeg needed).
2. **Extract** — per frame, trace the bright band's centerline with a
   peak-emphasised intensity-weighted centroid → amplitude vs `x` (spatial).
3. **Temporal** — reduce each frame to one envelope sample → oscillation over time.
4. **Render** — overlay, `.npz`/`.wav` data files, and a notebook.

## Development

```bash
uv run pytest -q        # tests (synthetic ground-truth + real-frame checks)
uv run ruff check       # lint
```

See `scripts/README.md` for the helper scripts, `spec.md` for the full design,
and `todo.md` for progress.
