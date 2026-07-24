"""Crown-jewel guard: scalers are fit on the TRAINING window ONLY.

Runs as a blocking Claude Code hook on every edit to
src/apu_sentinel/data/ (.claude/hooks/check_leakage.sh) and again in the
full pytest suite (Tier 3). The real functions are imported at module level
so a half-finished refactor fails cleanly at collection time.
"""

from __future__ import annotations

import pytest

from apu_sentinel.data.split import split_by_time
from apu_sentinel.data.windows import fit_scaler


def test_scaler_matches_train_window_not_full_series(synthetic_series):
    pytest.skip("stub — implement with logic")
    train, _val, _test = split_by_time(
        synthetic_series,
        train_end="2020-01-01T02:00:00",
        val_end="2020-01-01T02:40:00",
    )
    train_scaler = fit_scaler(train)
    full_series_scaler = fit_scaler(synthetic_series)
    assert train_scaler.mean_ == pytest.approx(train.mean().to_numpy())
    assert train_scaler.mean_ != pytest.approx(full_series_scaler.mean_)
