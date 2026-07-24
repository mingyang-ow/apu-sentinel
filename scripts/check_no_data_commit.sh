#!/usr/bin/env bash
# Pre-commit guard (Tier 1): refuse to commit anything under data/, except
# the .gitkeep placeholders that keep the empty dirs in git.
set -euo pipefail

staged_data_files=$(git diff --cached --name-only | grep -E '^data/' | grep -vE '\.gitkeep$' || true)

if [ -n "$staged_data_files" ]; then
  echo "ERROR: refusing to commit files under data/:"
  echo "$staged_data_files"
  exit 1
fi

exit 0
