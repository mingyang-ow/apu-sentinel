#!/usr/bin/env bash
# One command to reconstruct the ephemeral Colab environment: clone the
# repo, install uv, uv sync with the CUDA torch extra, ready to run.
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL to this repo's git remote}"
REPO_DIR="${REPO_DIR:-apu-sentinel}"

if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python pin 3.11
uv sync --extra cuda

echo "Ready. Run: uv run python scripts/train.py --config colab"
