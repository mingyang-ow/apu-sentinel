"""Tests for the gap-density / STOPPED-run diagnostic (analysis/__init__.py).

Uses small synthetic data -- never the real MetroPT-3 dataset.
"""

from __future__ import annotations

import pandas as pd
import pytest

from apu_sentinel.analysis import monthly_gap_and_stopped_summary


def _segment(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="10s")


def test_monthly_summary_counts_gaps_and_splits_stopped_runs_on_them():
    # January's last segment ends 2020-01-01 00:01:50 and February's first
    # segment starts 2020-02-01 00:00:00 -- almost a month apart. Rather
    # than let that (real, but incidental to this test) gap contaminate the
    # count, build each month as its own DataFrame and analyse separately.
    jan_seg1 = _segment("2020-01-01 00:00:00", 6)
    jan_seg2 = pd.date_range(jan_seg1[-1] + pd.Timedelta(minutes=5), periods=6, freq="10s")
    feb_seg1 = _segment("2020-02-01 00:00:00", 6)
    feb_seg2 = pd.date_range(feb_seg1[-1] + pd.Timedelta(minutes=10), periods=6, freq="10s")

    jan_index = jan_seg1.append(jan_seg2)
    feb_index = feb_seg1.append(feb_seg2)
    jan_df = pd.DataFrame({"x": range(len(jan_index))}, index=jan_index)
    feb_df = pd.DataFrame({"x": range(len(feb_index))}, index=feb_index)
    jan_regimes = pd.Series(["STOPPED"] * len(jan_index), index=jan_index, dtype="category")
    feb_regimes = pd.Series(["STOPPED"] * len(feb_index), index=feb_index, dtype="category")

    jan_report = monthly_gap_and_stopped_summary(
        jan_df, jan_regimes, gap_threshold=pd.Timedelta(minutes=1)
    )
    feb_report = monthly_gap_and_stopped_summary(
        feb_df, feb_regimes, gap_threshold=pd.Timedelta(minutes=1)
    )
    jan = jan_report.loc[pd.Period("2020-01", freq="M")]
    feb = feb_report.loc[pd.Period("2020-02", freq="M")]

    assert jan["n_gaps"] == 1
    assert jan["total_gap_seconds"] == pytest.approx(300.0)
    assert feb["n_gaps"] == 1
    assert feb["total_gap_seconds"] == pytest.approx(600.0)

    # The 5min/10min gaps exceed gap_threshold=1min, so each month's two
    # segments must be reported as two SEPARATE STOPPED runs, not bridged
    # into one -- each segment spans (6-1)*10s = 50s.
    assert jan["n_stopped_runs"] == 2
    assert jan["median_stopped_seconds"] == pytest.approx(50.0)
    assert feb["n_stopped_runs"] == 2
    assert feb["median_stopped_seconds"] == pytest.approx(50.0)


def test_monthly_summary_does_not_split_on_sub_threshold_gaps():
    # Normal 10s cadence throughout -- no gap exceeds the 1min threshold, so
    # the whole stretch is ONE STOPPED run.
    index = _segment("2020-03-01 00:00:00", 20)
    df = pd.DataFrame({"x": range(len(index))}, index=index)
    regimes = pd.Series(["STOPPED"] * len(index), index=index, dtype="category")

    report = monthly_gap_and_stopped_summary(df, regimes, gap_threshold=pd.Timedelta(minutes=1))
    mar = report.loc[pd.Period("2020-03", freq="M")]

    assert mar["n_gaps"] == 0
    assert mar["n_stopped_runs"] == 1
    assert mar["median_stopped_seconds"] == pytest.approx(190.0)  # (20-1)*10s
