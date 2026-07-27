"""Tiny synthetic fixtures for smoke/contract tests. NOT real MetroPT-3
data -- notebooks/exploratory and data/ are the only places real data
belongs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.config import (
    EvaluationConfig,
    FailureEvent,
    RegimesConfig,
    ResampleConfig,
    ScalingConfig,
    SplitConfig,
    TrainingExclusionConfig,
    WindowingConfig,
)
from apu_sentinel.data.split import _exclusion_window


@pytest.fixture
def synthetic_series() -> pd.DataFrame:
    """A tiny, deterministic multi-channel time series with one planted
    anomalous region near the end, standing in for MetroPT-3 in shape-only
    tests.
    """
    n = 200
    index = pd.date_range("2020-01-01", periods=n, freq="1min")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {f"channel_{i}": rng.normal(size=n) for i in range(3)},
        index=index,
    )
    df.index.name = "timestamp"
    df.iloc[-10:] += 10.0  # planted anomalous region
    return df


@pytest.fixture
def synthetic_failure_events() -> list[str]:
    """Documented timestamp(s) matching synthetic_series' planted anomaly."""
    return ["2020-01-01T03:10:00"]


def _tiny_raw_frame() -> pd.DataFrame:
    """A handful of rows shaped like raw MetroPT-3: a timestamp column plus
    a couple of numeric sensor columns, already time-ordered.
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6, freq="1min"),
            "TP2": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "Motor_current": [0.04, 0.05, 0.04, 0.06, 0.05, 0.04],
        }
    )


@pytest.fixture
def tiny_raw_csv(tmp_path: Path) -> Path:
    """Path to a tiny synthetic raw CSV fixture, already time-ordered."""
    path = tmp_path / "tiny_raw.csv"
    _tiny_raw_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def out_of_order_raw_csv(tmp_path: Path) -> Path:
    """Same shape as tiny_raw_csv but with two rows swapped out of time order."""
    df = _tiny_raw_frame()
    df.loc[[2, 3]] = df.loc[[3, 2]].to_numpy()
    path = tmp_path / "out_of_order_raw.csv"
    df.to_csv(path, index=False)
    return path


def _tiny_raw_frame_all_channels() -> pd.DataFrame:
    """Shaped like the real MetroPT-3 raw CSV: timestamp + all 7 analog +
    8 digital channels -- used to prove the loader's column set is exactly
    this after dropping the unnamed-index serialisation artifact.
    """
    n = 4
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="1min"),
            "TP2": [1.0, 1.1, 1.2, 1.3],
            "TP3": [9.0, 9.1, 9.2, 9.3],
            "H1": [9.0, 9.0, 9.0, 9.0],
            "DV_pressure": [0.0, 0.0, 0.0, 0.0],
            "Reservoirs": [9.0, 9.0, 9.0, 9.0],
            "Oil_temperature": [53.0, 53.1, 53.2, 53.3],
            "Motor_current": [0.04, 0.04, 0.04, 0.04],
            "COMP": [1.0, 1.0, 1.0, 1.0],
            "DV_eletric": [0.0, 0.0, 0.0, 0.0],
            "Towers": [1.0, 1.0, 1.0, 1.0],
            "MPG": [1.0, 1.0, 1.0, 1.0],
            "LPS": [0.0, 0.0, 0.0, 0.0],
            "Pressure_switch": [1.0, 1.0, 1.0, 1.0],
            "Oil_level": [1.0, 1.0, 1.0, 1.0],
            "Caudal_impulses": [1.0, 1.0, 1.0, 1.0],
        }
    )


@pytest.fixture
def raw_csv_with_unnamed_index_column(tmp_path: Path) -> Path:
    """Raw CSV written WITH its pandas integer index (index=True) -- the
    "Unnamed: 0" artifact produced when a CSV is written with its index and
    re-read without index_col, as apparently happened for the real
    MetroPT-3 raw file.
    """
    path = tmp_path / "raw_with_unnamed_index.csv"
    _tiny_raw_frame_all_channels().to_csv(path, index=True)
    return path


# --- Fixtures for tests/test_split_no_leakage.py -----------------------
#
# Three synthetic events shaped like the real MetroPT-3 four (id 1 has no
# maintenance entry; the gap between event 1 and event 2 is large, the gap
# between event 2 and event 3 is tight -- ~136h -- mirroring the real
# event-2/event-3 proximity constraint documented in data/split.py).


@pytest.fixture
def synthetic_split_events() -> list[FailureEvent]:
    return [
        FailureEvent(
            id=1,
            start="2020-01-02 00:00",
            end="2020-01-02 04:00",
            maintenance=None,
            note="synthetic: no maintenance recorded, like real event 1",
        ),
        FailureEvent(
            id=2,
            start="2020-01-20 00:00",
            end="2020-01-20 06:00",
            maintenance="2020-01-20 12:00",
        ),
        FailureEvent(
            id=3,
            start="2020-01-26 10:00",
            end="2020-01-28 14:00",
            maintenance="2020-01-29 16:00",
        ),
    ]


@pytest.fixture
def synthetic_training_exclusion() -> TrainingExclusionConfig:
    return TrainingExclusionConfig(pre_margin_hours=2, post_settle_hours=6, fallback_post_hours=12)


@pytest.fixture
def synthetic_split_settings(synthetic_split_events, synthetic_training_exclusion):
    """Duck-typed like apu_sentinel.config.Settings: exposes .split and
    .evaluation. window_widths max out at 48h -- safely under the ~136h
    event-2/event-3 gap given the margins above.
    """
    split = SplitConfig(embargo_hours=4, training_exclusion=synthetic_training_exclusion)
    evaluation = EvaluationConfig(
        window_widths=[6, 12, 24, 48],
        failure_events=synthetic_split_events,
    )
    return SimpleNamespace(split=split, evaluation=evaluation)


@pytest.fixture
def synthetic_split_settings_overlapping(synthetic_split_events, synthetic_training_exclusion):
    """Same as synthetic_split_settings but with a deliberately over-wide
    150h window width -- wider than the ~136h event-2/event-3 gap, so
    make_folds() must raise naming that pair.
    """
    split = SplitConfig(embargo_hours=4, training_exclusion=synthetic_training_exclusion)
    evaluation = EvaluationConfig(
        window_widths=[6, 12, 24, 48, 150],
        failure_events=synthetic_split_events,
    )
    return SimpleNamespace(split=split, evaluation=evaluation)


@pytest.fixture
def synthetic_split_settings_test_start_overlap(
    synthetic_split_events, synthetic_training_exclusion
):
    """A width (134h) that passes the plain window-width overlap check
    (label_start for event 3 lands just after event 2's exclusion ends,
    136h max) but violates the TIGHTER test_start check, which also backs
    off by embargo_hours (4h) -- exactly the previously-unguarded gap this
    pass closes. Reproduces it with a small margin rather than the exact
    real-dataset 2h/22h numbers, but the same shape of bug.
    """
    split = SplitConfig(embargo_hours=4, training_exclusion=synthetic_training_exclusion)
    evaluation = EvaluationConfig(
        window_widths=[6, 12, 24, 48, 134],
        failure_events=synthetic_split_events,
    )
    return SimpleNamespace(split=split, evaluation=evaluation)


@pytest.fixture
def synthetic_split_data_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp("2019-12-01 00:00"), pd.Timestamp("2020-02-05 00:00")


@pytest.fixture
def synthetic_split_df(synthetic_split_data_bounds) -> pd.DataFrame:
    """Small hourly synthetic series spanning synthetic_split_events, for
    materialising and checking actual fold slices (not just boundaries).
    """
    data_start, data_end = synthetic_split_data_bounds
    index = pd.date_range(data_start, data_end, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"channel_0": rng.normal(size=len(index))}, index=index)
    df.index.name = "timestamp"
    return df


# --- Fixtures for tests/test_scaler_train_only.py -----------------------


@pytest.fixture
def synthetic_scaling_config() -> ScalingConfig:
    return ScalingConfig(
        method="robust",
        analog_columns=["channel_0"],
        passthrough_columns=["flag_0"],
    )


@pytest.fixture
def synthetic_scaling_settings(synthetic_split_settings, synthetic_scaling_config):
    """Duck-typed like apu_sentinel.config.Settings: exposes .split,
    .evaluation, and .scaling.
    """
    return SimpleNamespace(
        split=synthetic_split_settings.split,
        evaluation=synthetic_split_settings.evaluation,
        scaling=synthetic_scaling_config,
    )


@pytest.fixture
def synthetic_scaling_df(
    synthetic_split_data_bounds, synthetic_split_events, synthetic_training_exclusion
) -> pd.DataFrame:
    """Like synthetic_split_df, but with an extreme spike planted inside
    EVERY event's training-exclusion window, so "fit including the
    exclusion" vs. "fit on the clean, exclusion-removed slice" produce
    measurably different statistics -- proving contamination is actually
    prevented, not just that exclusions are recorded.
    """
    data_start, data_end = synthetic_split_data_bounds
    index = pd.date_range(data_start, data_end, freq="1h")
    rng = np.random.default_rng(0)
    channel_0 = rng.normal(loc=0.0, scale=1.0, size=len(index))
    flag_0 = rng.integers(0, 2, size=len(index)).astype(float)
    df = pd.DataFrame({"channel_0": channel_0, "flag_0": flag_0}, index=index)
    df.index.name = "timestamp"

    for event in synthetic_split_events:
        excl_start, excl_end = _exclusion_window(event, synthetic_training_exclusion)
        mask = (df.index >= excl_start) & (df.index < excl_end)
        df.loc[mask, "channel_0"] = 10_000.0
    return df


# --- Fixtures for tests/test_windows.py ---------------------------------
#
# A regular 1-minute-cadence series with window_duration=10min so
# window_length=10 samples, train_stride=2min (2 samples), score_stride=1min
# (1 sample) -- small round numbers that make expected window counts easy to
# hand-derive.


@pytest.fixture
def synthetic_windowing_config() -> WindowingConfig:
    return WindowingConfig(
        window_duration="10min",
        train_stride="2min",
        score_stride="1min",
        gap_tolerance=0.1,
        gap_threshold="5min",
        resample=ResampleConfig(enabled=False, interval="1min"),
    )


@pytest.fixture
def synthetic_windows_scaling_config() -> ScalingConfig:
    return ScalingConfig(
        method="robust",
        analog_columns=["channel_0", "channel_1"],
        passthrough_columns=["flag_0"],
    )


@pytest.fixture
def synthetic_windows_split_config() -> SplitConfig:
    # embargo_hours=1 (60min) comfortably covers window_duration=10min.
    return SplitConfig(
        embargo_hours=1,
        training_exclusion=TrainingExclusionConfig(
            pre_margin_hours=1, post_settle_hours=1, fallback_post_hours=1
        ),
    )


@pytest.fixture
def synthetic_windows_settings(
    synthetic_windows_split_config, synthetic_windows_scaling_config, synthetic_windowing_config
):
    """Duck-typed like apu_sentinel.config.Settings: exposes .split,
    .scaling, and .windowing.
    """
    return SimpleNamespace(
        split=synthetic_windows_split_config,
        scaling=synthetic_windows_scaling_config,
        windowing=synthetic_windowing_config,
    )


@pytest.fixture
def synthetic_windows_settings_embargo_violation(
    synthetic_windows_split_config, synthetic_windows_scaling_config
):
    """windowing.window_duration (2h) exceeds split.embargo_hours (1h) --
    make_windows must raise, naming both values.
    """
    windowing = WindowingConfig(
        window_duration="2h",
        train_stride="30min",
        score_stride="10min",
        gap_tolerance=0.1,
        gap_threshold="30min",
        resample=ResampleConfig(enabled=False, interval="10min"),
    )
    return SimpleNamespace(
        split=synthetic_windows_split_config,
        scaling=synthetic_windows_scaling_config,
        windowing=windowing,
    )


@pytest.fixture
def synthetic_windows_df() -> pd.DataFrame:
    """200 minutes of regular 1-minute-cadence data, no gaps."""
    index = pd.date_range("2020-01-01 00:00", periods=200, freq="1min")
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "channel_0": rng.normal(size=len(index)),
            "channel_1": rng.normal(size=len(index)),
            "flag_0": rng.integers(0, 2, size=len(index)).astype(float),
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_windows_df_with_gap(synthetic_windows_df) -> pd.DataFrame:
    """Same 200-minute series, but rows for minutes 100-119 (inclusive) are
    removed -- a deliberate 21-minute native gap between the row at minute
    99 and the row at minute 120.
    """
    minute = (synthetic_windows_df.index - synthetic_windows_df.index[0]) // pd.Timedelta(minutes=1)
    keep_mask = ~((minute >= 100) & (minute <= 119))
    return synthetic_windows_df.loc[keep_mask]


@pytest.fixture
def synthetic_windows_exclusion_settings(synthetic_split_settings):
    """Combines the existing split+evaluation fixtures (3 events, embargo
    4h) with scaling/windowing config sized to make a real, multi-sample
    train/test fold from synthetic_windows_exclusion_df meaningful:
    window_duration=2h fits under embargo=4h.
    """
    scaling = ScalingConfig(
        method="robust", analog_columns=["channel_0"], passthrough_columns=["flag_0"]
    )
    windowing = WindowingConfig(
        window_duration="2h",
        train_stride="30min",
        score_stride="10min",
        gap_tolerance=0.1,
        gap_threshold="1h",
        resample=ResampleConfig(enabled=False, interval="10min"),
    )
    return SimpleNamespace(
        split=synthetic_split_settings.split,
        evaluation=synthetic_split_settings.evaluation,
        scaling=scaling,
        windowing=windowing,
    )


@pytest.fixture
def synthetic_windows_exclusion_df(synthetic_split_data_bounds) -> pd.DataFrame:
    """10-minute-cadence series spanning synthetic_split_data_bounds, fine
    enough grained that a fold's multi-hour exclusion regions remove many
    rows -- enough to test that no window bridges across them.
    """
    data_start, data_end = synthetic_split_data_bounds
    index = pd.date_range(data_start, data_end, freq="10min")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "channel_0": rng.normal(size=len(index)),
            "flag_0": rng.integers(0, 2, size=len(index)).astype(float),
        },
        index=index,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def synthetic_resample_settings(synthetic_windows_split_config):
    """resample enabled, 1-minute grid -- for testing maybe_resample()'s
    mean/max aggregation and NaN-for-empty-interval behaviour.
    """
    scaling = ScalingConfig(
        method="robust", analog_columns=["analog_a"], passthrough_columns=["digital_a"]
    )
    windowing = WindowingConfig(
        window_duration="2min",
        train_stride="1min",
        score_stride="1min",
        gap_tolerance=0.1,
        gap_threshold="2min",
        resample=ResampleConfig(enabled=True, interval="1min"),
    )
    return SimpleNamespace(
        split=synthetic_windows_split_config, scaling=scaling, windowing=windowing
    )


@pytest.fixture
def synthetic_resample_raw_df() -> pd.DataFrame:
    """Irregular sub-minute samples: minute bin 0 gets two samples (mean/max
    aggregation is checkable), minute bins 1-4 get NONE (must resample to
    NaN, never forward-filled), minute bin 5 gets one sample.
    """
    timestamps = pd.to_datetime(
        [
            "2020-01-01 00:00:10",
            "2020-01-01 00:00:40",
            "2020-01-01 00:05:00",
        ]
    )
    return pd.DataFrame(
        {"analog_a": [2.0, 4.0, 9.0], "digital_a": [0.0, 1.0, 1.0]},
        index=timestamps,
    )


# --- Fixtures for tests/test_metrics.py and tests/test_eval_contract.py --
#
# One failure event with a short, easy-to-hand-compute settle tail
# (post_settle_hours=2h), used across most episode-categorisation tests.


@pytest.fixture
def metrics_training_exclusion() -> TrainingExclusionConfig:
    return TrainingExclusionConfig(pre_margin_hours=1, post_settle_hours=2, fallback_post_hours=4)


@pytest.fixture
def metrics_event() -> FailureEvent:
    return FailureEvent(
        id=1,
        start="2020-01-10 00:00",
        end="2020-01-10 04:00",
        maintenance="2020-01-10 08:00",
    )


@pytest.fixture
def metrics_settings(metrics_event, metrics_training_exclusion):
    """Duck-typed like apu_sentinel.config.Settings: exposes .split
    (training_exclusion) and .evaluation. window_widths=[6] (hours) --
    pre-failure window is [2020-01-09 18:00, 2020-01-10 00:00). The
    event's masked settle tail is [2020-01-10 00:00, 2020-01-10 10:00]
    (maintenance 08:00 + post_settle_hours 2h).
    """
    evaluation = EvaluationConfig(
        window_widths=[6],
        threshold_quantile=0.995,
        threshold_quantiles=[0.99, 0.995],
        episode_hold_time="10min",
        score_gap_threshold="30min",
        min_episode_duration="0min",
        contribution_aggregation="mean",
        failure_events=[metrics_event],
        additional_masked_regions=[],
    )
    return SimpleNamespace(
        split=SimpleNamespace(training_exclusion=metrics_training_exclusion),
        evaluation=evaluation,
    )


@pytest.fixture
def metrics_channel_names() -> tuple[str, ...]:
    return ("chan_a", "chan_b", "chan_c")


# --- Fixtures for tests/test_regimes.py ---------------------------------
#
# A single control flag COMP, polarity COMP=1 -> OFF / COMP=0 -> ON (mirrors
# the real, empirically-verified MetroPT-3 polarity). states only defines
# LOADED (COMP=0); COMP=1 rows fall through to the OFFLOAD/STOPPED current
# threshold split, matching the real four-state scheme. min_duration=30s and
# transition_settle=60s at a 10s cadence -> 3 and 6-7 samples respectively,
# matching the real dataset's measured sampling interval. offload_current_
# threshold=2.0 matches the real config's empirically-justified default.


@pytest.fixture
def regimes_scaling_config() -> ScalingConfig:
    return ScalingConfig(method="robust", analog_columns=["Motor_current"], passthrough_columns=[])


@pytest.fixture
def regimes_config() -> RegimesConfig:
    return RegimesConfig(
        control_columns=["COMP"],
        polarity={"COMP": {1: "OFF", 0: "ON"}},
        states={"LOADED": {"COMP": 0}},
        offload_split_channel="Motor_current",
        offload_current_threshold=2.0,
        min_duration="30s",
        transition_settle="60s",
    )


@pytest.fixture
def regimes_settings(regimes_scaling_config, regimes_config):
    """Duck-typed like apu_sentinel.config.Settings: exposes .scaling and
    .regimes.
    """
    return SimpleNamespace(scaling=regimes_scaling_config, regimes=regimes_config)
