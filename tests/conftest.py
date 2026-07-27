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

from apu_sentinel.config import EvaluationConfig, FailureEvent, SplitConfig, TrainingExclusionConfig


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
