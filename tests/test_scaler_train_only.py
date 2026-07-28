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
import pandas as pd
import pytest

from apu_sentinel.data import scaling as scaling_module
from apu_sentinel.data.scaling import (
    FoldScaler,
    fit_regime_scalers,
    fit_scaler,
    transform,
    transform_by_regime,
)
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


def test_constant_channel_gets_epsilon_guarded_scale_and_logs_warning(
    synthetic_scaling_settings, caplog
):
    index = pd.date_range("2020-01-01", periods=10, freq="1h")
    train = pd.DataFrame(
        {"channel_0": [5.0] * len(index), "flag_0": [1.0] * len(index)},
        index=index,
    )

    with caplog.at_level("WARNING"):
        scaler = fit_scaler(train, synthetic_scaling_settings)

    assert scaler.center_["channel_0"] == pytest.approx(5.0)
    assert scaler.scale_["channel_0"] == pytest.approx(1.0)

    transformed = transform(train, scaler)
    assert np.isfinite(transformed["channel_0"].to_numpy()).all()

    assert any(
        "channel_0" in record.message and "computed scale" in record.message
        for record in caplog.records
    )


def _expected_robust(values: np.ndarray) -> tuple[float, float]:
    q75, q25 = np.percentile(values, [75, 25])
    return float(np.median(values)), float(q75 - q25)


def test_regime_scalers_fitted_per_regime_match_and_differ(
    regime_scaling_settings, regime_scaling_train_df_and_regimes
):
    train_df, train_regimes = regime_scaling_train_df_and_regimes
    scalers = fit_regime_scalers(train_df, train_regimes, regime_scaling_settings, fold_id=1)

    assert set(scalers) == {"LOADED", "STOPPED"}
    for regime in ("LOADED", "STOPPED"):
        subset = train_df.loc[(train_regimes == regime).to_numpy()]
        expected_center, expected_scale = _expected_robust(subset["chan_normal"].to_numpy())
        assert scalers[regime].center_["chan_normal"] == pytest.approx(expected_center)
        assert scalers[regime].scale_["chan_normal"] == pytest.approx(expected_scale)

    assert scalers["LOADED"].center_["chan_normal"] != pytest.approx(
        scalers["STOPPED"].center_["chan_normal"]
    )


def test_regime_scalers_still_train_only(
    regime_scaling_settings, regime_scaling_train_df_and_regimes
):
    train_df, train_regimes = regime_scaling_train_df_and_regimes
    train_scalers = fit_regime_scalers(train_df, train_regimes, regime_scaling_settings)

    # An extreme, clearly-distinguishable "test period" appended after
    # train -- must not move the train-only fitted parameters.
    extra_index = pd.date_range(train_df.index[-1] + pd.Timedelta(hours=1), periods=50, freq="1h")
    extra_df = pd.DataFrame(
        {
            "chan_normal": np.full(50, 5000.0),
            "chan_inactive_when_stopped": np.full(50, 20.0),
            "chan_pathological_but_active": np.full(50, 30.0),
            "flag_digital": np.zeros(50),
        },
        index=extra_index,
    )
    extra_regimes = pd.Series(["LOADED"] * 50, index=extra_index, dtype="category")

    combined_df = pd.concat([train_df, extra_df])
    combined_regimes = pd.concat([train_regimes.astype(str), extra_regimes.astype(str)])
    combined_scalers = fit_regime_scalers(combined_df, combined_regimes, regime_scaling_settings)

    assert train_scalers["LOADED"].center_["chan_normal"] != pytest.approx(
        combined_scalers["LOADED"].center_["chan_normal"]
    )


def test_inactive_channel_becomes_constant_zero(
    regime_scaling_settings, regime_scaling_train_df_and_regimes
):
    train_df, train_regimes = regime_scaling_train_df_and_regimes
    scalers = fit_regime_scalers(train_df, train_regimes, regime_scaling_settings)
    transformed = transform_by_regime(train_df, train_regimes, scalers, regime_scaling_settings)

    stopped_mask = (train_regimes == "STOPPED").to_numpy()
    loaded_mask = (train_regimes == "LOADED").to_numpy()

    # Inactive in STOPPED -- constant 0.0, NOT divided by its tiny scale.
    assert (transformed.loc[stopped_mask, "chan_inactive_when_stopped"] == 0.0).all()
    # Active in LOADED -- must NOT be constant zero there.
    assert not (transformed.loc[loaded_mask, "chan_inactive_when_stopped"] == 0.0).all()


def test_transformed_shape_stable_across_regime_mix(
    regime_scaling_settings, regime_scaling_train_df_and_regimes
):
    train_df, train_regimes = regime_scaling_train_df_and_regimes
    scalers = fit_regime_scalers(train_df, train_regimes, regime_scaling_settings)

    loaded_mask = (train_regimes == "LOADED").to_numpy()
    stopped_mask = (train_regimes == "STOPPED").to_numpy()

    transformed_loaded = transform_by_regime(
        train_df.loc[loaded_mask], train_regimes.loc[loaded_mask], scalers, regime_scaling_settings
    )
    transformed_stopped = transform_by_regime(
        train_df.loc[stopped_mask],
        train_regimes.loc[stopped_mask],
        scalers,
        regime_scaling_settings,
    )
    transformed_full = transform_by_regime(
        train_df, train_regimes, scalers, regime_scaling_settings
    )

    assert list(transformed_loaded.columns) == list(transformed_full.columns)
    assert list(transformed_stopped.columns) == list(transformed_full.columns)


def test_min_samples_per_regime_guard_raises_naming_fold_and_regime(regime_scaling_settings):
    index = pd.date_range("2020-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "chan_normal": [1.0] * 5,
            "chan_inactive_when_stopped": [1.0] * 5,
            "chan_pathological_but_active": [1.0] * 5,
            "flag_digital": [0.0] * 5,
        },
        index=index,
    )
    regimes = pd.Series(["STOPPED"] * 5, index=index, dtype="category")  # 5 < min_samples (10)

    with pytest.raises(ValueError) as excinfo:
        fit_regime_scalers(df, regimes, regime_scaling_settings, fold_id=7)

    message = str(excinfo.value)
    assert "fold 7" in message
    assert "STOPPED" in message


def test_amplification_warning_logged_not_substituted(
    regime_scaling_settings, regime_scaling_train_df_and_regimes, caplog
):
    train_df, train_regimes = regime_scaling_train_df_and_regimes

    with caplog.at_level("WARNING"):
        scalers = fit_regime_scalers(train_df, train_regimes, regime_scaling_settings)

    stopped_scale = scalers["STOPPED"].scale_["chan_pathological_but_active"]
    assert stopped_scale != pytest.approx(1.0)  # NOT substituted
    assert stopped_scale < 0.01  # genuinely tiny, as planted

    assert any(
        "chan_pathological_but_active" in r.message
        and "STOPPED" in r.message
        and "amplification" in r.message
        for r in caplog.records
    )
