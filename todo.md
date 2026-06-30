# TODO — Water-Reflection Waveform Extractor

Status legend: `[ ]` pending · `[~]` in progress · `[x]` done

Each epic ends with: **tests + validation pass → commit → push to `main`**.

---

## Epic 0 — Bootstrap & write-access check
- [x] 0.1 Update `README.md` with summary → commit → **push** (verify write access)
- [x] 0.2 Add `spec.md` + `todo.md` → commit → push
- **Validation:** push to `origin/main` succeeds; files on remote. ✅

## Epic 1 — Dev env & scaffolding
- [x] 1.1 Fill `pyproject.toml` (runtime + dev deps, package config, CLI entry point)
- [x] 1.2 Create `src/wave_from_video/` package skeleton + smoke test
- [x] 1.3 `scripts/setup.sh` (bootstrap mise/uv + `uv sync`); `data/`, `output/`, `.gitignore`
- [x] 1.4 Save `data/reference_frame.png` (from the video) — done in Epic 2 (needs the reader)
- **Validation:** `uv sync`; `uv run python -c "import wave_from_video"`; `uv run pytest -q`
  collects; `uv run ruff check` clean. ✅

## Epic 2 — Video ingestion
- [x] 2.1 `io_video.py`: frame iterator, metadata (fps/size/count), grayscale, ROI/letterbox crop
- [x] 2.2 `scripts/extract_frames.py`: dump/sample frames, produce `reference_frame.png`
- **Validation:** test — frame count > 0, expected shape/dtype; ROI crop removes letterbox. ✅
- **Note:** real frames are clean 640×360 (no letterbox); the attached image's black
  bars/chevron were the phone's screen UI. `reference_frame.png` is real frame #30.

## Epic 3 — Spatial waveform extraction (core)
- [ ] 3.1 `extract.py`: bg-subtract + per-column centroid → centerline/amplitude/confidence
- [ ] 3.2 Synthetic sine-band fixture + ground-truth test (corr ≥ 0.95, low RMSE)
- [ ] 3.3 Reference-frame contrast test (brightness under curve ≫ background; non-trivial variance)
- **Validation:** both tests green.

## Epic 4 — Overlay validation output
- [ ] 4.1 `render.py` overlay: centerline drawn on reference image → `output/overlay.png`
- [ ] 4.2 `scripts/validate_extraction.py`: run extraction + emit overlay + contrast metrics
- **Validation:** overlay file exists, non-empty; contrast metric passes; artifact saved.

## Epic 5 — Temporal signal + data files
- [ ] 5.1 `temporal.py`: all-frames centerlines (2D) + temporal envelope (1D)
- [ ] 5.2 `render.py` writers: `waveform.npz`, `spatial.wav`, `temporal.wav`, `waveform.png`
- **Validation:** array shapes/finite; `.wav` round-trips; `.npz` keys present.

## Epic 6 — Jupyter rendering notebook
- [ ] 6.1 `notebooks/waveform.ipynb`: load npz + wav, matplotlib plot + `IPython.display.Audio`
- **Validation:** `jupyter nbconvert --execute` exits 0 with outputs.

## Epic 7 — CLI end-to-end
- [ ] 7.1 `cli.py` + `wave-from-video` entry point; `scripts/run_pipeline.sh`
- **Validation:** run on real `.MOV` → overlay + npz + wav all exist, non-empty.

## Epic 8 — Docs & session continuity
- [ ] 8.1 README usage section
- [ ] 8.2 `scripts/README.md` (what each script does / when to reuse)
- [ ] 8.3 Pointer in `CLAUDE.md` so new sessions discover `scripts/`, `spec.md`, `todo.md`
- [ ] 8.4 Keep `spec.md` / `todo.md` in sync with any direction changes
- **Validation:** docs present and accurate; new-session discoverability confirmed.
