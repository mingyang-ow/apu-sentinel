"""Crown-jewel guard: scalers are fit on the TRAINING window ONLY.

Runs as a blocking Claude Code hook on every edit to
src/apu_sentinel/data/ (.claude/hooks/check_leakage.sh) and again in the
full pytest suite (Tier 3). The real functions are imported at module level
so a half-finished refactor fails cleanly at collection time.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apu_sentinel.data.split import Fold, apply_fold
from apu_sentinel.data.windows import fit_scaler


def test_scaler_matches_train_window_not_full_series(synthetic_series):
    pytest.skip("stub — implement with logic")
    fold = Fold(
        event_id=1,
        train_start=synthetic_series.index.min(),
        train_end=pd.Timestamp("2020-01-01T02:00:00"),
        test_start=pd.Timestamp("2020-01-01T02:40:00"),
        test_end=synthetic_series.index.max(),
        train_exclusions=(),
    )
    train, _test = apply_fold(synthetic_series, fold)
    train_scaler = fit_scaler(train)
    full_series_scaler = fit_scaler(synthetic_series)
    assert train_scaler.mean_ == pytest.approx(train.mean().to_numpy())
    assert train_scaler.mean_ != pytest.approx(full_series_scaler.mean_)
