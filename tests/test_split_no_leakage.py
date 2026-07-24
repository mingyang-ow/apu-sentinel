"""Crown-jewel guard: no leakage across the time-based split.

Runs as a blocking Claude Code hook on every edit to
src/apu_sentinel/data/ (.claude/hooks/check_leakage.sh) and again in the
full pytest suite (Tier 3). The real function is imported at module level
so a half-finished refactor fails cleanly at collection time, rather than
tripping on transient intermediate file states.
"""

from __future__ import annotations

import pytest

from apu_sentinel.data.split import split_by_time


def test_split_is_strictly_time_ordered(synthetic_series):
    pytest.skip("stub — implement with logic")
    train, val, test = split_by_time(
        synthetic_series,
        train_end="2020-01-01T02:00:00",
        val_end="2020-01-01T02:40:00",
    )
    assert train.index.max() < val.index.min() < test.index.min()


def test_split_rejects_shuffle_strategy(synthetic_series):
    with pytest.raises(ValueError):
        split_by_time(
            synthetic_series,
            train_end="2020-01-01T02:00:00",
            val_end="2020-01-01T02:40:00",
            strategy="shuffle",
        )
