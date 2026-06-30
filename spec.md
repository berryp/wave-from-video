# Specification: Water-Reflection Waveform Extractor

## 1. Problem & Goal

A handheld video (`IMG_4767_2.MOV`, ~30 MB) captures light reflected off moving water.
The reflection appears as a bright, blurry band that wiggles horizontally across a textured
grey background — visually identical to an oscilloscope / audio waveform. The goal is a small
Python project that extracts that band from the video frames and turns it into waveform data,
with strong, self-checking validations.

### Primary deliverables (the definition of done)
1. **Overlay image** — the extracted waveform drawn on top of a reference frame, proving the
   extraction tracks the visible bright band (`output/overlay.png`).
2. **Waveform data files** rendered "the typical way in Jupyter":
   - `output/waveform.npz` — arrays for a matplotlib line plot.
   - `output/spatial.wav`, `output/temporal.wav` — playable audio via `IPython.display.Audio`.

## 2. Confirmed Requirements (from interview)

| Decision | Choice |
|----------|--------|
| Signal axis | **Both** spatial (amplitude vs x within a frame) and temporal (oscillation over time across frames) |
| Output formats | **`.npz`** (matplotlib) **and `.wav`** (IPython audio) |
| Branch | Commit/push directly to **`main`** |
| Dev env | **mise + uv** (Python 3.14, ruff) per `mise.toml` |
| Reference | The attached frame, saved on disk as `data/reference_frame.png` |

## 3. Definitions

- **Band**: the bright reflection region in a frame.
- **Centerline** `y(x)`: the vertical pixel position of the band for each column `x`.
- **Baseline**: the weighted-median centerline (the band's resting level).
- **Amplitude** `a(x) = baseline - y(x)`: signed displacement; positive = band sits higher.
- **Confidence** `w(x)`: per-column total band intensity; low where the band is faint/absent
  (the band only spans the central part of the frame width).
- **Temporal envelope** `e(t)`: one scalar per frame summarising that frame's wiggle.

## 4. Approach

### 4.1 Video ingestion (`io_video.py`)
- Decode frames with **`imageio` + `imageio-ffmpeg`** (bundled ffmpeg; no system ffmpeg needed).
- Convert to grayscale float.
- Detect the **content ROI**, cropping black letterbox bars (columns/rows that are near-zero
  across the whole clip).
- Provide a frame iterator and a `read_all()` helper, plus metadata (fps, size, count).

### 4.2 Spatial extraction (`extract.py`) — core algorithm
For one grayscale frame within the ROI:
1. **Background estimate & subtraction**: subtract a per-column background (e.g. column median
   or a low percentile) so only the bright band remains; clip negatives to 0.
2. **Per-column centroid**: for each column `x`, `y(x) = Σ(row·I) / Σ(I)` over the rows
   (intensity-weighted centroid). Centroid (not argmax) gives sub-pixel, noise-robust results
   for the blurry band.
3. **Confidence** `w(x) = Σ I` per column; normalise. Mask columns below a confidence threshold.
4. **Baseline** = weighted median of `y(x)`; **amplitude** `a(x) = baseline - y(x)`.
5. **Smoothing**: light Savitzky–Golay / Gaussian on `a(x)`.

Returns a small dataclass: `centerline`, `amplitude`, `confidence`, `baseline`, `x` coords.

### 4.3 Temporal signal (`temporal.py`)
- Run 4.2 over **all frames** → `centerlines` 2D array `[n_frames, width]` and matching
  `amplitudes`.
- **Temporal envelope** `e(t)`: default reducer = **RMS of the masked amplitude** per frame
  (configurable: center-column value, or mean). Sampled at native fps.

### 4.4 Rendering & data files (`render.py`)
- **Overlay**: draw `centerline` (or amplitude band) over the reference image → `overlay.png`.
- **Matplotlib waveform**: plot a representative spatial amplitude and the temporal envelope →
  `waveform.png`.
- **`.npz`**: save `x`, `amplitude` (representative frame), `centerlines`, `amplitudes`,
  `temporal_envelope`, `fps`.
- **`.wav`**: map signals to audio. Both the representative spatial trace and the temporal
  envelope are mean-removed, peak-normalised to [-1, 1], and **resampled to 44.1 kHz** over a
  short fixed duration so they are inspectable as audio waveforms. The mapping (sample rate,
  duration) is recorded in the `.npz` metadata and printed by the CLI.

### 4.5 CLI (`cli.py`)
`wave-from-video <video> [--out output/] [--reducer rms|center|mean]` runs the whole pipeline:
ingest → extract all frames → write overlay + `waveform.npz` + `*.wav` + `waveform.png`.

### 4.6 Jupyter notebook (`notebooks/waveform.ipynb`)
Loads `waveform.npz` and the `.wav` files; shows the matplotlib waveform plot and
`IPython.display.Audio` players. Must execute end-to-end headlessly.

## 5. Validation Strategy

Every epic has an automated check. The strongest gate is a **synthetic ground-truth test**.

- **Synthetic frame** (`tests/`): generate a frame with a band following a *known* sine
  `a*(x)`; assert the recovered amplitude matches within tolerance (correlation ≥ 0.95 and
  RMSE small). This proves the extractor is correct independent of the real footage.
- **Reference frame**: assert mean brightness *under* the extracted centerline ≫ background
  (contrast ratio above threshold) and amplitude variance is non-trivial (it actually wiggles).
- **Ingestion**: frame count > 0; expected shape/dtype; ROI crop removes letterbox.
- **Data files**: array shapes/finiteness; `.wav` round-trips (re-read equals written within
  quantisation); `.npz` keys present.
- **Overlay**: file exists and is non-empty; contrast metric passes.
- **Notebook**: `jupyter nbconvert --execute` exits 0 with outputs.
- **End-to-end**: CLI on the real `.MOV` produces all expected files, non-empty.

## 6. Project Layout

```
src/wave_from_video/{__init__,io_video,extract,temporal,render,cli}.py
tests/                # pytest
scripts/              # reusable setup/extraction/validation/pipeline scripts (+ README)
data/                 # reference_frame.png, source video
output/               # generated artifacts (gitignored; a small sample may be committed)
notebooks/waveform.ipynb
spec.md  todo.md  README.md  CLAUDE.md  mise.toml  pyproject.toml
```

## 7. Tooling & Dependencies

- Runtime: `numpy`, `scipy`, `matplotlib`, `pillow`, `imageio`, `imageio-ffmpeg`, `soundfile`.
- Dev: `pytest`, `ruff`, `jupyter`/`nbconvert`/`nbclient`.
- Managed by `uv`; Python pinned to 3.14 via `mise.toml` + `pyproject.toml`. The project must
  also run via plain `uv` (no mise) so fresh/CI sessions work.

## 8. Non-Goals

- Real-time processing, GUI, or web app.
- Audio fidelity / musicality of the `.wav` (it is a data-inspection artifact, not a track).
- Handling videos with materially different framing than the provided clip (single horizontal
  band on a plain background). Generalisation is out of scope unless requested.

## 9. Open Defaults (chosen; adjustable)

- Temporal reducer default = RMS of masked amplitude.
- `.wav` resample target = 44.1 kHz, fixed short duration (documented in metadata).
- Whole-video processing may subsample frames for speed; if so, the stride is logged and stored.
