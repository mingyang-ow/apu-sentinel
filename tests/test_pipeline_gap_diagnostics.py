"""Pass 22 (docs/RESULTS.md, event-4 detection validation): gap-adjacency
helpers used by pipeline._build_windowed_input's
exclude_gap_adjacent_windows diagnostic. Small synthetic indices only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apu_sentinel.pipeline import _gap_boundaries, gap_adjacent_mask


def test_gap_boundaries_finds_only_gaps_above_threshold():
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:10",  # 10s -- not a gap
            "2020-01-01 00:10:10",  # 10min -- a gap
            "2020-01-01 00:10:20",
        ]
    )
    gaps = _gap_boundaries(index, pd.Timedelta("5min"))
    assert gaps == [(pd.Timestamp("2020-01-01 00:00:10"), pd.Timestamp("2020-01-01 00:10:10"))]


def test_gap_boundaries_empty_for_fewer_than_two_rows():
    assert _gap_boundaries(pd.DatetimeIndex(["2020-01-01"]), pd.Timedelta("5min")) == []


def test_gap_adjacent_mask_flags_both_sides_within_window_duration():
    gap_start = pd.Timestamp("2020-01-01 12:00:00")
    gap_end = pd.Timestamp("2020-01-01 13:00:00")
    window_duration = pd.Timedelta("30min")

    end_timestamps = pd.DatetimeIndex(
        [
            gap_start - pd.Timedelta("20min"),  # near start -- flagged
            gap_start - pd.Timedelta("2h"),  # far before -- not flagged
            gap_end + pd.Timedelta("20min"),  # near end -- flagged
            gap_end + pd.Timedelta("2h"),  # far after -- not flagged
        ]
    )
    mask = gap_adjacent_mask(end_timestamps, [(gap_start, gap_end)], window_duration)
    assert mask.tolist() == [True, False, True, False]


def test_gap_adjacent_mask_empty_gaps_flags_nothing():
    end_timestamps = pd.date_range("2020-01-01", periods=5, freq="1min")
    mask = gap_adjacent_mask(end_timestamps, [], pd.Timedelta("30min"))
    assert not mask.any()
    assert mask.shape == (5,)
    assert mask.dtype == np.bool_
