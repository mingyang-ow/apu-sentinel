#!/usr/bin/env bash
# Blocking leakage-guard hook (Tier 2 of the testing strategy in CLAUDE.md).
#
# Fires after any Edit/Write/MultiEdit whose target file is under
# src/apu_sentinel/data/. Runs the two leakage-guard tests and BLOCKS
# (exit 2) if either fails, per CLAUDE.md hard rules 1 and 2:
#   1. time-based split only, no shuffle
#   2. scalers fit on the training window only
#
# Exit code 2 is Claude Code's "blocking" signal: stderr is surfaced back to
# the model, which must resolve the failure before continuing.
set -euo pipefail

INPUT_JSON=$(cat)
FILE_PATH=$(printf '%s' "$INPUT_JSON" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
")

case "$FILE_PATH" in
  */src/apu_sentinel/data/*) ;;
  *) exit 0 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if ! uv run pytest tests/test_split_no_leakage.py tests/test_scaler_train_only.py -q; then
  echo "BLOCKING: leakage-guard tests failed after edit to $FILE_PATH -- resolve before continuing (see CLAUDE.md rules 1-2)." >&2
  exit 2
fi

exit 0
