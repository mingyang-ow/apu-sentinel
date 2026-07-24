"""Tiny synthetic fixtures for smoke/contract tests. NOT real MetroPT-3
data -- notebooks/exploratory and data/ are the only places real data
belongs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


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
