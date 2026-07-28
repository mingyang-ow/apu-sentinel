"""Tests for causal cycle-timing features (features/cycles.py).

The causality test is THE important one: every feature value at t must be
unchanged when all data after t is deleted. Uses small synthetic
DataFrames -- never the real MetroPT-3 dataset.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.features import cycles as cycles_module
from apu_sentinel.features.cycles import baseline_relative, compute_cycle_features


def _cycle_df(
    labels: list[str], reservoirs: np.ndarray, freq: str = "10s"
) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2020-01-01", periods=len(labels), freq=freq)
    regimes = pd.Series(labels, index=index, dtype="category")
    df = pd.DataFrame({"Reservoirs": reservoirs}, index=index)
    return df, regimes


def _two_cycle_scenario() -> tuple[pd.DataFrame, pd.Series]:
    """LOADED(20)->OFFLOAD(10)->STOPPED(30, slope -0.01/s), repeated with a
    second STOPPED run at slope -0.02/s.
    """
    n1_loaded, n1_offload, n1_stopped = 20, 10, 30
    n2_loaded, n2_offload, n2_stopped = 20, 10, 30
    labels = (
        ["LOADED"] * n1_loaded
        + ["OFFLOAD"] * n1_offload
        + ["STOPPED"] * n1_stopped
        + ["LOADED"] * n2_loaded
        + ["OFFLOAD"] * n2_offload
        + ["STOPPED"] * n2_stopped
    )
    n = len(labels)
    reservoirs = np.zeros(n)
    pos = 0
    reservoirs[pos : pos + n1_loaded] = 9.6
    pos += n1_loaded
    reservoirs[pos : pos + n1_offload] = 9.6
    pos += n1_offload
    t1 = np.arange(n1_stopped) * 10.0
    reservoirs[pos : pos + n1_stopped] = 9.6 - 0.01 * t1
    pos += n1_stopped
    reservoirs[pos : pos + n2_loaded] = reservoirs[pos - 1]
    pos += n2_loaded
    reservoirs[pos : pos + n2_offload] = reservoirs[pos - 1]
    pos += n2_offload
    t2 = np.arange(n2_stopped) * 10.0
    reservoirs[pos : pos + n2_stopped] = reservoirs[pos - 1] - 0.02 * t2

    return _cycle_df(labels, reservoirs)


def test_causality_feature_values_unchanged_when_future_deleted(cycles_settings):
    rng = np.random.default_rng(1)
    labels = (
        ["LOADED"] * 20
        + ["OFFLOAD"] * 10
        + ["STOPPED"] * 30
        + ["LOADED"] * 15
        + ["OFFLOAD"] * 8
        + ["STOPPED"] * 25
    )
    n = len(labels)
    reservoirs = 9.6 - 0.001 * np.arange(n) + rng.normal(scale=0.001, size=n)
    df, regimes = _cycle_df(labels, reservoirs)

    full = compute_cycle_features(df, regimes, cycles_settings)

    for cut in (5, 19, 29, 49, 50, 64, 79, 90, n - 1):
        truncated = compute_cycle_features(
            df.iloc[: cut + 1], regimes.iloc[: cut + 1], cycles_settings
        )
        row_full = full.iloc[cut]
        row_trunc = truncated.iloc[-1]
        for col in full.columns:
            a, b = row_full[col], row_trunc[col]
            same = (pd.isna(a) and pd.isna(b)) or (
                not pd.isna(a) and not pd.isna(b) and np.isclose(a, b)
            )
            assert same, f"cut={cut} col={col}: full={a} truncated={b}"


def test_no_lookahead_constructs_in_source():
    source = inspect.getsource(cycles_module)
    assert "center=True" not in source
    assert "centre=True" not in source
    assert "bfill" not in source
    assert "backfill" not in source
    assert re.search(r"shift\(\s*-\d", source) is None


def test_duration_correctness_for_known_run_lengths(cycles_settings):
    df, regimes = _two_cycle_scenario()
    feats = compute_cycle_features(df, regimes, cycles_settings)

    # cycle1: LOADED 20 samples (190s span), OFFLOAD 10 samples (90s span),
    # STOPPED 30 samples (290s span) -- durations become visible once the
    # NEXT run of that kind completes (forward-filled from there).
    cycle2_start = 20 + 10 + 30
    row_after_cycle1 = feats.iloc[cycle2_start]
    assert row_after_cycle1["loaded_duration_last"] == pytest.approx(190.0)
    assert row_after_cycle1["offload_duration_last"] == pytest.approx(90.0)

    cycle2_stopped_start = cycle2_start + 20 + 10
    row_after_stopped1 = feats.iloc[cycle2_stopped_start]
    assert row_after_stopped1["stopped_duration_last"] == pytest.approx(290.0)

    # cycle2's cycle_period_last: gap between the two STOPPED starts.
    assert feats.iloc[cycle2_stopped_start]["cycle_period_last"] == pytest.approx(600.0)


def test_running_reflects_current_run_while_last_completed_holds_previous(cycles_settings):
    df, regimes = _two_cycle_scenario()
    feats = compute_cycle_features(df, regimes, cycles_settings)

    cycle1_stopped_start = 20 + 10
    mid_row = feats.iloc[cycle1_stopped_start + 5]
    # Mid-way through cycle1's own STOPPED run: no STOPPED run has completed
    # yet, so stopped_duration_last is still NaN, while stopped_elapsed
    # reflects time so far in THIS run (5 samples in = 50s).
    assert pd.isna(mid_row["stopped_duration_last"])
    assert mid_row["stopped_elapsed"] == pytest.approx(50.0)

    cycle2_stopped_start = 20 + 10 + 30 + 20 + 10
    mid_row2 = feats.iloc[cycle2_stopped_start + 5]
    # Now inside cycle2's STOPPED run: elapsed reflects THIS run, while
    # duration_last still reports cycle1's completed STOPPED run (290s),
    # not the (incomplete) current one.
    assert mid_row2["stopped_elapsed"] == pytest.approx(50.0)
    assert mid_row2["stopped_duration_last"] == pytest.approx(290.0)


def test_decay_rate_recovers_known_slope_within_tolerance(cycles_settings):
    df, regimes = _two_cycle_scenario()
    feats = compute_cycle_features(df, regimes, cycles_settings)

    cycle2_start = 20 + 10 + 30 + 20 + 10  # first row of cycle2's STOPPED run
    # decay_rate_running, once enough samples have accumulated within
    # cycle1's STOPPED run, must recover its planted slope (-0.01).
    within_cycle1 = feats.iloc[20 + 10 + 5]["decay_rate_running"]
    assert within_cycle1 == pytest.approx(-0.01, abs=1e-6)

    # decay_rate_last, once cycle1's STOPPED run has completed, reports its
    # slope (-0.01); once cycle2's STOPPED run itself later completes, no
    # further row exists here, so check running mid-way through cycle2 to
    # confirm ITS OWN slope (-0.02) is recovered too.
    within_cycle2 = feats.iloc[cycle2_start + 5]["decay_rate_running"]
    assert within_cycle2 == pytest.approx(-0.02, abs=1e-6)
    assert feats.iloc[cycle2_start]["decay_rate_last"] == pytest.approx(-0.01, abs=1e-6)


def test_gap_truncated_run_yields_nan_duration_but_valid_decay_rate(cycles_settings):
    seg1 = pd.date_range("2020-01-01 00:00:00", periods=10, freq="10s")  # LOADED
    seg2 = pd.date_range(seg1[-1] + pd.Timedelta(seconds=10), periods=15, freq="10s")  # STOPPED
    seg3 = pd.date_range(
        seg2[-1] + pd.Timedelta(minutes=5), periods=10, freq="10s"
    )  # STOPPED (post-gap)
    seg4 = pd.date_range(seg3[-1] + pd.Timedelta(seconds=10), periods=10, freq="10s")  # LOADED

    index = seg1.append(seg2).append(seg3).append(seg4)
    labels = ["LOADED"] * 10 + ["STOPPED"] * 15 + ["STOPPED"] * 10 + ["LOADED"] * 10
    regimes = pd.Series(labels, index=index, dtype="category")

    reservoirs = np.zeros(len(index))
    reservoirs[:10] = 9.6
    t2 = np.arange(15) * 10.0
    reservoirs[10:25] = 9.6 - 0.005 * t2
    t3 = np.arange(10) * 10.0
    reservoirs[25:35] = reservoirs[24] - 0.005 * t3
    reservoirs[35:45] = reservoirs[34]

    df = pd.DataFrame({"Reservoirs": reservoirs}, index=index)
    feats = compute_cycle_features(df, regimes, cycles_settings)

    # First row after the gap-truncated run (seg2) completes.
    row = feats.iloc[25]
    assert pd.isna(row["stopped_duration_last"])  # invalid -- NOT the truncated value
    assert row["decay_rate_last"] == pytest.approx(-0.005, abs=1e-6)  # still valid
    assert row["run_gap_truncated"]

    # seg3 (post-gap) completes normally -- its own duration/flag are clean.
    row_after_seg3 = feats.iloc[35]
    assert row_after_seg3["stopped_duration_last"] == pytest.approx(90.0)
    assert not row_after_seg3["run_gap_truncated"]


def test_baseline_relative_ratio_and_causality():
    index = pd.date_range("2020-01-01", periods=5, freq="1D")
    series = pd.Series([10.0, 10.0, 10.0, 10.0, 100.0], index=index)
    window = pd.Timedelta(days=3)

    result = baseline_relative(series, window)
    # Trailing median at day5 over the past 3 days (day3,4,5=[10,10,100]) = 10.
    assert result.iloc[-1] == pytest.approx(10.0)
    assert result.iloc[0] == pytest.approx(1.0)

    # Appending future data must not change earlier baseline-relative values.
    extended_index = index.append(pd.date_range("2020-01-06", periods=2, freq="1D"))
    extended_series = pd.concat([series, pd.Series([1000.0, 1000.0], index=extended_index[-2:])])
    extended_result = baseline_relative(extended_series, window)
    assert extended_result.iloc[: len(series)].equals(result)


def test_alignment_output_index_matches_input_exactly(cycles_settings):
    df, regimes = _two_cycle_scenario()
    feats = compute_cycle_features(df, regimes, cycles_settings)

    assert len(feats) == len(df)
    assert list(feats.index) == list(df.index)
