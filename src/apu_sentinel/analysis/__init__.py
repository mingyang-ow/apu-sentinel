"""Standalone diagnostic analyses that don't belong in the core pipeline.

Currently one function: a per-calendar-month gap-density vs. STOPPED-run
report, used to check whether the observed drift in median STOPPED
duration (~1400s in Feb -> ~500s in Aug, docs/FINDINGS.md §8) is physical
or partly an artifact of increasing gap density over the recording period
(a data gap falling inside a STOPPED run truncates it, biasing its
measured duration downward -- see docs/FINDINGS.md §9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GAP_THRESHOLD = pd.Timedelta(minutes=1)


def monthly_gap_and_stopped_summary(
    df: pd.DataFrame,
    regimes: pd.Series,
    gap_threshold: pd.Timedelta | None = None,
    stopped_label: str = "STOPPED",
) -> pd.DataFrame:
    """Per calendar month: number of data gaps > gap_threshold, total
    missing duration, number of STOPPED runs, and median STOPPED run
    duration.

    df's index and regimes' index are both used purely for their
    timestamps (df for gap detection, regimes for STOPPED run detection);
    they need not be the same Series, but must share a comparable
    DatetimeIndex.

    A gap is attributed to the calendar month of its START (the last
    good sample before the gap). A STOPPED run is attributed to the
    calendar month of its START.

    Purely descriptive/offline (like characterise_regimes) -- not part of
    the causal assign_regimes() pipeline.
    """
    if gap_threshold is None:
        gap_threshold = DEFAULT_GAP_THRESHOLD

    timestamps = df.index.to_numpy()
    if len(timestamps) >= 2:
        diffs = np.diff(timestamps)
        gap_mask = diffs > np.timedelta64(gap_threshold)
        gap_starts = pd.DatetimeIndex(timestamps[:-1][gap_mask])
        # Divide by a 1-second timedelta64 (not pd.Timedelta.total_seconds()
        # via .apply()) so this stays float64 even when gap_mask is all
        # False -- .apply() on an empty Series leaves it timedelta64-typed,
        # which then can't be cast to float64 downstream.
        gap_durations = diffs[gap_mask] / np.timedelta64(1, "s")
    else:
        gap_starts = pd.DatetimeIndex([])
        gap_durations = np.array([], dtype=float)

    gaps_by_month = pd.DataFrame(
        {"month": gap_starts.to_period("M"), "duration_seconds": gap_durations}
    )
    gap_summary = gaps_by_month.groupby("month")["duration_seconds"].agg(
        n_gaps="count", total_gap_seconds="sum"
    )
    gap_summary = gap_summary.astype({"n_gaps": "int64", "total_gap_seconds": "float64"})

    values = regimes.to_numpy()
    index = regimes.index
    n = len(values)
    if n == 0:
        stopped_summary = pd.DataFrame(columns=["n_stopped_runs", "median_stopped_seconds"])
    else:
        # A run ends at a label change OR a gap exceeding gap_threshold --
        # a gap-truncated STOPPED run undercounts its duration, the exact
        # mechanism this report checks for.
        value_changed = values[1:] != values[:-1]
        gap_occurred = np.diff(index.to_numpy()) > np.timedelta64(gap_threshold)
        boundary = value_changed | gap_occurred
        change_positions = np.flatnonzero(boundary) + 1
        run_starts = np.concatenate(([0], change_positions))
        run_ends = np.concatenate((change_positions - 1, [n - 1]))
        run_labels = values[run_starts]
        is_stopped = run_labels == stopped_label

        stopped_starts = index[run_starts[is_stopped]]
        stopped_ends = index[run_ends[is_stopped]]
        stopped_durations = np.array(
            [(e - s).total_seconds() for s, e in zip(stopped_starts, stopped_ends, strict=True)]
        )
        stopped_by_month = pd.DataFrame(
            {
                "month": pd.PeriodIndex(stopped_starts, freq="M"),
                "duration_seconds": stopped_durations,
            }
        )
        stopped_summary = stopped_by_month.groupby("month")["duration_seconds"].agg(
            n_stopped_runs="count", median_stopped_seconds="median"
        )
        stopped_summary["n_stopped_runs"] = stopped_summary["n_stopped_runs"].astype("int64")

    report = gap_summary.join(stopped_summary, how="outer").sort_index()
    report["n_gaps"] = report["n_gaps"].fillna(0).infer_objects(copy=False).astype("int64")
    report["total_gap_seconds"] = report["total_gap_seconds"].fillna(0.0).infer_objects(copy=False)
    report["n_stopped_runs"] = (
        report["n_stopped_runs"].fillna(0).infer_objects(copy=False).astype("int64")
    )
    return report
