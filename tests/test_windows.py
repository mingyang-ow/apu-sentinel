"""Tests for gap-aware sequence windowing (data/windows.py).

Uses only small synthetic fixtures (tests/conftest.py) -- never the real
MetroPT-3 dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.data.split import apply_fold, make_folds
from apu_sentinel.data.windows import make_windows, maybe_resample


def test_no_window_spans_a_gap(synthetic_windows_settings, synthetic_windows_df_with_gap):
    """200 minutes with rows for minutes 100-119 removed -- a 21-minute
    native gap. window_length=10 samples, train_stride=2 samples,
    gap_tolerance=0.1 -> max_allowed_span=11min.

    Hand-derived expectation: 171 stride-1 windows over the 180 remaining
    rows, 86 of which survive the stride=2 subsampling; exactly 4 of those
    (start positions 92, 94, 96, 98) straddle the gap and must be dropped,
    leaving 82.
    """
    windows, end_timestamps = make_windows(
        synthetic_windows_df_with_gap, synthetic_windows_settings, stride_mode="train"
    )

    assert windows.shape[0] == 82

    window_length = 10
    max_allowed_span = pd.Timedelta(minutes=10) * 1.1
    index = synthetic_windows_df_with_gap.index
    for end_ts in end_timestamps:
        pos = index.get_loc(end_ts)
        start_pos = pos - window_length + 1
        span = index[pos] - index[start_pos]
        assert span <= max_allowed_span


def test_exclusion_holes_are_respected(
    synthetic_windows_exclusion_settings,
    synthetic_split_data_bounds,
    synthetic_windows_exclusion_df,
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_windows_exclusion_settings, data_start, data_end)
    fold3 = next(f for f in folds if f.event_id == 3)
    assert fold3.train_exclusions, "fixture must exercise a fold with exclusions"

    train, _test = apply_fold(synthetic_windows_exclusion_df, fold3)
    windows, end_timestamps = make_windows(
        train, synthetic_windows_exclusion_settings, stride_mode="train"
    )
    assert windows.shape[0] > 0

    # A window that silently bridged a removed exclusion region would have
    # an anomalously large internal gap between two of its constituent
    # samples -- so verify every window's samples are exactly one modal
    # interval (10min, this fixture's cadence) apart, i.e. genuinely
    # contiguous, never having skipped over a hole.
    window_length = windows.shape[1]
    modal_interval = pd.Timedelta(minutes=10)
    for end_ts in end_timestamps:
        end_ts = pd.Timestamp(end_ts)
        pos = train.index.get_loc(end_ts)
        start_pos = pos - window_length + 1
        constituent_timestamps = train.index[start_pos : pos + 1]
        diffs = np.diff(constituent_timestamps)
        assert (diffs == modal_interval).all(), (
            f"window ending {end_ts} is not contiguous -- diffs {diffs} -- "
            "it must have bridged a removed exclusion region"
        )

    # And, independently, no exclusion region's timestamps appear at all in
    # train (apply_fold's own contract, reconfirmed here).
    for excl_start, excl_end in fold3.train_exclusions:
        in_excluded_region = (train.index >= excl_start) & (train.index < excl_end)
        assert not in_excluded_region.any()


def test_shape_and_dtype_and_channel_order(synthetic_windows_settings, synthetic_windows_df):
    windows, end_timestamps = make_windows(
        synthetic_windows_df, synthetic_windows_settings, stride_mode="train"
    )

    assert windows.shape == (96, 10, 3)
    assert windows.dtype == np.float32
    assert end_timestamps.shape == (96,)

    expected_columns = ["channel_0", "channel_1", "flag_0"]
    pos = synthetic_windows_df.index.get_loc(pd.Timestamp(end_timestamps[0]))
    expected_last_row = synthetic_windows_df.iloc[pos][expected_columns].to_numpy(dtype=np.float32)
    assert np.allclose(windows[0, -1, :], expected_last_row)


def test_end_timestamp_mapping(synthetic_windows_settings, synthetic_windows_df):
    windows, end_timestamps = make_windows(
        synthetic_windows_df, synthetic_windows_settings, stride_mode="score"
    )

    columns = ["channel_0", "channel_1", "flag_0"]
    for i in (0, len(end_timestamps) // 2, len(end_timestamps) - 1):
        end_ts = pd.Timestamp(end_timestamps[i])
        pos = synthetic_windows_df.index.get_loc(end_ts)
        assert synthetic_windows_df.index[pos] == end_ts

        window_length = windows.shape[1]
        expected_block = synthetic_windows_df.iloc[pos - window_length + 1 : pos + 1][
            columns
        ].to_numpy(dtype=np.float32)
        assert np.allclose(windows[i], expected_block)


def test_stride_behaviour_produces_expected_window_counts(
    synthetic_windows_settings, synthetic_windows_df
):
    train_windows, _ = make_windows(
        synthetic_windows_df, synthetic_windows_settings, stride_mode="train"
    )
    score_windows, _ = make_windows(
        synthetic_windows_df, synthetic_windows_settings, stride_mode="score"
    )

    # 200 samples, window_length=10 -> 191 stride-1 windows.
    assert train_windows.shape[0] == 96  # stride=2 samples -> ceil(191/2)
    assert score_windows.shape[0] == 191  # stride=1 sample -> every window


def test_order_preserved_end_timestamps_strictly_increasing(
    synthetic_windows_settings, synthetic_windows_df
):
    _windows, end_timestamps = make_windows(
        synthetic_windows_df, synthetic_windows_settings, stride_mode="score"
    )
    assert (np.diff(end_timestamps) > np.timedelta64(0)).all()


def test_embargo_validation_raises_naming_both_values(
    synthetic_windows_settings_embargo_violation, synthetic_windows_df
):
    with pytest.raises(ValueError) as excinfo:
        make_windows(synthetic_windows_df, synthetic_windows_settings_embargo_violation)

    message = str(excinfo.value)
    assert "2:00:00" in message  # window_duration=2h
    assert "1:00:00" in message  # embargo_hours=1


def test_resampling_means_maxes_and_nan_gaps(
    synthetic_resample_settings, synthetic_resample_raw_df
):
    resampled = maybe_resample(synthetic_resample_raw_df, synthetic_resample_settings)

    bin0 = pd.Timestamp("2020-01-01 00:00:00")
    bin5 = pd.Timestamp("2020-01-01 00:05:00")
    assert resampled.loc[bin0, "analog_a"] == pytest.approx((2.0 + 4.0) / 2)
    assert resampled.loc[bin0, "digital_a"] == pytest.approx(max(0.0, 1.0))
    assert resampled.loc[bin5, "analog_a"] == pytest.approx(9.0)
    assert resampled.loc[bin5, "digital_a"] == pytest.approx(1.0)

    empty_bins = [pd.Timestamp(f"2020-01-01 00:0{m}:00") for m in (1, 2, 3, 4)]
    for ts in empty_bins:
        assert resampled.loc[ts].isna().all()
        # never forward-filled from bin0's values
        assert resampled.loc[ts, "analog_a"] != pytest.approx(3.0)

    # end-to-end: windows spanning any NaN bin must be dropped entirely --
    # bin0 and bin5 (the only non-NaN bins) aren't adjacent, so NO 2-sample
    # window survives.
    windows, _end_timestamps = make_windows(
        synthetic_resample_raw_df, synthetic_resample_settings, stride_mode="train"
    )
    assert windows.shape[0] == 0
