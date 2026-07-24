"""Tiny synthetic fixtures for smoke/contract tests. NOT real MetroPT-3
data -- notebooks/exploratory and data/ are the only places real data
belongs.
"""

from __future__ import annotations

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
