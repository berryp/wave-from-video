# scripts/

Reusable helper scripts for data extraction, validation, and running the
pipeline. Prefer reusing these over re-creating ad-hoc commands in new sessions.

| Script | What it does | When to use |
|--------|--------------|-------------|
| `setup.sh` | Bootstrap mise/uv, install Python 3.14, `uv sync`. | First-time setup or a fresh checkout/session. |
| `extract_frames.py` | Pick the clearest band frame and write `data/reference_frame.png`; optionally dump sampled frames. | Regenerate the reference frame, or inspect frames. |
| `validate_extraction.py` | Run extraction on the reference frame, write `output/overlay.png`, print contrast metrics. | Quick visual + numeric check that extraction tracks the band. |
| `run_pipeline.sh` | Full pipeline on the source video → all outputs in `output/`. | Regenerate the deliverables (overlay, npz, wavs, plot). |

## Examples

```bash
bash scripts/setup.sh                      # set up the environment
uv run python scripts/extract_frames.py    # data/reference_frame.png
uv run python scripts/validate_extraction.py
bash scripts/run_pipeline.sh               # full run on data/IMG_4767_2.MOV
```

All Python scripts take `--help`.
