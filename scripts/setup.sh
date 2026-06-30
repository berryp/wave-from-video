#!/usr/bin/env bash
# Bootstrap the dev environment for wave-from-video.
#
# Works with or without mise. uv is the source of truth for the Python
# environment; mise (if installed) just pins tool versions via mise.toml.
#
# Usage: bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v mise >/dev/null 2>&1; then
  echo "==> mise found; installing pinned tools"
  mise install
else
  echo "==> mise not found; continuing with uv only (see mise.toml for pins)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> ensuring Python 3.14 is available"
uv python install 3.14

echo "==> syncing dependencies"
uv sync

echo "==> done. Run tests with: uv run pytest -q"
