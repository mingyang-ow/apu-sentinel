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


def test_load_raw_drops_unnamed_index_column_and_logs(
    raw_csv_with_unnamed_index_column: Path, caplog
):
    with caplog.at_level("WARNING"):
        df = load_raw(raw_csv_with_unnamed_index_column)

    assert not any(col.startswith("Unnamed:") for col in df.columns)
    assert any(
        "Unnamed" in record.message and "serialisation artifact" in record.message
        for record in caplog.records
    )

    analog_columns = {
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
    }
    digital_columns = {
        "COMP",
        "DV_eletric",
        "Towers",
        "MPG",
        "LPS",
        "Pressure_switch",
        "Oil_level",
        "Caudal_impulses",
    }
    assert set(df.columns) == analog_columns | digital_columns
    assert df.index.name == "timestamp"
