#!/usr/bin/env bash
# Run the full waveform pipeline on the source video and write all outputs.
#
# Usage: bash scripts/run_pipeline.sh [VIDEO] [OUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="${1:-data/IMG_4767_2.MOV}"
OUT="${2:-output}"

uv run wave-from-video "$VIDEO" --out "$OUT"
echo "outputs in: $OUT"
