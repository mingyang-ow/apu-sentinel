"""Crown-jewel guard: scalers are fit on the TRAINING window ONLY.

Runs as a blocking Claude Code hook on every edit to
src/apu_sentinel/data/ (.claude/hooks/check_leakage.sh) and again in the
full pytest suite (Tier 3). The real functions are imported at module level
so a half-finished refactor fails cleanly at collection time.

Uses only small synthetic fixtures (tests/conftest.py) -- never the real
MetroPT-3 dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from apu_sentinel.data import scaling as scaling_module
from apu_sentinel.data.scaling import FoldScaler, fit_scaler, transform
from apu_sentinel.data.split import apply_fold, make_folds


def _expected_robust_stats(values: np.ndarray) -> tuple[float, float]:
    q75, q25 = np.percentile(values, [75, 25])
    return float(np.median(values)), float(q75 - q25)


def test_fitted_stats_match_clean_training_slice_exactly(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)

    for fold in folds:
        train, _test = apply_fold(synthetic_scaling_df, fold)
        scaler = fit_scaler(train, synthetic_scaling_settings)

        expected_center, expected_scale = _expected_robust_stats(train["channel_0"].to_numpy())
        assert scaler.center_["channel_0"] == pytest.approx(expected_center)
        assert scaler.scale_["channel_0"] == pytest.approx(expected_scale)


def test_fitted_stats_differ_from_full_series_stats(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold1 = next(f for f in folds if f.event_id == 1)
    train, _test = apply_fold(synthetic_scaling_df, fold1)

    fold_scaler = fit_scaler(train, synthetic_scaling_settings)
    full_series_scaler = fit_scaler(synthetic_scaling_df, synthetic_scaling_settings)

    assert fold_scaler.center_["channel_0"] != pytest.approx(
        full_series_scaler.center_["channel_0"]
    )


def test_fitted_stats_differ_from_train_plus_test_stats(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold1 = next(f for f in folds if f.event_id == 1)

    train, _test = apply_fold(synthetic_scaling_df, fold1)
    train_only_scaler = fit_scaler(train, synthetic_scaling_settings)

    combined_mask = (synthetic_scaling_df.index >= fold1.train_start) & (
        synthetic_scaling_df.index <= fold1.test_end
    )
    combined = synthetic_scaling_df.loc[combined_mask]
    train_plus_test_scaler = fit_scaler(combined, synthetic_scaling_settings)

    assert train_only_scaler.center_["channel_0"] != pytest.approx(
        train_plus_test_scaler.center_["channel_0"]
    )


def test_exclusions_applied_before_fitting_prevents_contamination(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    """The test that proves contamination was actually prevented: fit on
    the fold's clean (exclusion-removed) slice vs. fit on the SAME time
    span with the exclusion left in. The planted extreme values inside the
    exclusion window must measurably move the statistics.
    """
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold3 = next(f for f in folds if f.event_id == 3)
    assert fold3.train_exclusions, "fixture must exercise a fold with exclusions"

    clean_train, _test = apply_fold(synthetic_scaling_df, fold3)
    clean_scaler = fit_scaler(clean_train, synthetic_scaling_settings)

    dirty_mask = (synthetic_scaling_df.index >= fold3.train_start) & (
        synthetic_scaling_df.index < fold3.train_end
    )
    dirty_train = synthetic_scaling_df.loc[dirty_mask]
    dirty_scaler = fit_scaler(dirty_train, synthetic_scaling_settings)

    assert clean_scaler.center_["channel_0"] != pytest.approx(dirty_scaler.center_["channel_0"])
    assert clean_scaler.scale_["channel_0"] != pytest.approx(dirty_scaler.scale_["channel_0"])


def test_each_fold_gets_a_distinct_scaler(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold_first = min(folds, key=lambda f: f.event_id)
    fold_last = max(folds, key=lambda f: f.event_id)

    train_first, _ = apply_fold(synthetic_scaling_df, fold_first)
    train_last, _ = apply_fold(synthetic_scaling_df, fold_last)
    scaler_first = fit_scaler(train_first, synthetic_scaling_settings)
    scaler_last = fit_scaler(train_last, synthetic_scaling_settings)

    assert scaler_first is not scaler_last
    assert scaler_first.center_ is not scaler_last.center_
    assert scaler_first.center_["channel_0"] != pytest.approx(scaler_last.center_["channel_0"])


def test_transform_never_fits(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    assert not hasattr(scaling_module, "fit_transform")

    unfitted = FoldScaler(
        method="robust", analog_columns=("channel_0",), passthrough_columns=("flag_0",)
    )
    with pytest.raises(ValueError):
        transform(synthetic_scaling_df, unfitted)

    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold = folds[0]
    train, test = apply_fold(synthetic_scaling_df, fold)
    scaler = fit_scaler(train, synthetic_scaling_settings)

    center_before = dict(scaler.center_)
    scale_before = dict(scaler.scale_)
    _ = transform(test, scaler)
    assert scaler.center_ == center_before
    assert scaler.scale_ == scale_before


def test_digital_columns_pass_through_and_unlisted_column_raises(
    synthetic_scaling_settings, synthetic_split_data_bounds, synthetic_scaling_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_scaling_settings, data_start, data_end)
    fold = folds[0]
    train, _test = apply_fold(synthetic_scaling_df, fold)
    scaler = fit_scaler(train, synthetic_scaling_settings)

    transformed = transform(train, scaler)
    assert (transformed["flag_0"].to_numpy() == train["flag_0"].to_numpy()).all()
    assert transformed["flag_0"].dtype == train["flag_0"].dtype

    with_extra_column = train.copy()
    with_extra_column["mystery_col"] = 1.0
    with pytest.raises(ValueError):
        fit_scaler(with_extra_column, synthetic_scaling_settings)
    with pytest.raises(ValueError):
        transform(with_extra_column, scaler)
