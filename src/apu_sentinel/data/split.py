"""Time-based train/val/test split. CROWN-JEWEL FILE.

Hard rule (CLAUDE.md #1): split by time only. No training sample's
timestamp may exceed the train/val boundary, and no shuffle/random split is
permitted anywhere in this project. This is enforced by
tests/test_split_no_leakage.py, which runs as a blocking Claude Code hook on
every edit to src/apu_sentinel/data/ (see .claude/hooks/check_leakage.sh)
and again in the full pytest suite.

Stub: split logic implemented in a later pass. The `strategy` parameter
exists only to make the "time-based only" contract explicit and rejectable
at the signature level -- any non-"time" strategy must raise, never merely
warn.
"""

from __future__ import annotations

import pandas as pd


def split_by_time(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    strategy: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) using time-based boundaries only.

    Returns dataframes such that
    max(train.timestamp) < min(val.timestamp) < min(test.timestamp).

    Raises:
        ValueError: if strategy is anything other than "time" (e.g. any
            shuffle/random split request).
    """
    if strategy != "time":
        raise ValueError(
            f"split_by_time only supports strategy='time', got {strategy!r} -- "
            "shuffle/random splits are not permitted (CLAUDE.md rule 1)."
        )
    raise NotImplementedError
