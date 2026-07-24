"""Tests for the minimal raw loader (data-brief.md Build Pass 2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from apu_sentinel.data.load import load_raw


def test_load_raw_parses_sorts_and_shapes(tiny_raw_csv: Path):
    df = load_raw(tiny_raw_csv)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert df.shape == (6, 2)
    assert list(df.columns) == ["TP2", "Motor_current"]


def test_load_raw_surfaces_out_of_order_timestamps(out_of_order_raw_csv: Path, caplog):
    with caplog.at_level("WARNING"):
        df = load_raw(out_of_order_raw_csv)

    assert any("NOT monotonic" in record.message for record in caplog.records)
    # still returns a sorted frame -- the disorder is reported, not hidden
    assert df.index.is_monotonic_increasing
