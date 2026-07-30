"""Gap-aware sequence windowing.

Pipeline position (fixed): apply_fold -> fit_scaler (train only) ->
transform -> make_windows. Windowing is LAST and operates on already-scaled
data; it does not scale anything itself.

Central requirement: windows must not span gaps. After apply_fold removes
a fold's exclusion regions, the training slice has holes where failure
periods were cut out. A naive sliding window over the remaining rows would
splice data from before a failure onto data from after its repair,
presenting a discontinuity to the model as though it were continuous. Gaps
arise from two sources -- apply_fold's exclusion regions, and any native
recording gaps in the raw data -- and both are handled identically here: a
window of n samples may span no more than
n * expected_interval * (1 + gap_tolerance) of wall-clock time; anything
wider is DROPPED, never repaired or interpolated. The number of dropped
windows (and why) is logged at INFO, per the loader's "surfaced, not
silent" philosophy.

expected_interval is never assumed -- it is measured empirically per call
by characterise_sampling() (the modal inter-sample interval), since the raw
~10s-average sampling could in principle be regular subsampling or sparser
data with large gaps; only measuring it tells you which.

Resampling to a regular grid (maybe_resample) is a separate, OFF-by-default
step: enabling it is a modeling decision the user makes knowingly after
reviewing characterise_sampling()'s output. When enabled, empty intervals
become NaN rows -- never forward-filled -- which make_windows treats as
gaps exactly like a native or exclusion-driven gap: a window containing any
NaN is dropped.

Public API:
- `SamplingCharacteristics` -- modal_interval, interval_counts, gaps (tuple
  of (start, end, duration) exceeding gap_threshold), total_span/samples.
- `characterise_sampling(df, gap_threshold) -> SamplingCharacteristics` --
  raises on an empty df or fewer than 2 rows.
- `maybe_resample(df, settings) -> pd.DataFrame` -- no-op unless
  windowing.resample.enabled; raises if a df column is in neither
  scaling.analog_columns nor scaling.passthrough_columns.
- `make_windows(df, settings, stride_mode="train") -> (windows,
  end_timestamps)` -- windows shape (n_windows, window_length, n_channels)
  float32, channel order = analog_columns + passthrough_columns;
  end_timestamps is each window's LAST timestamp (the score/label
  convention downstream episode grouping depends on). `stride_mode` is
  "train" (coarser, windowing.train_stride) or "score" (finer,
  windowing.score_stride). Raises if window_duration exceeds
  split.embargo_hours, if stride_mode is invalid, if df (post-resample) is
  empty, or if an expected column is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingCharacteristics:
    """Empirical sampling profile of a DataFrame's DatetimeIndex."""

    modal_interval: pd.Timedelta
    interval_counts: pd.Series
    # (gap_start, gap_end, duration) for every consecutive pair whose gap
    # exceeds gap_threshold.
    gaps: tuple[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta], ...]
    total_span: pd.Timedelta
    total_samples: int


def characterise_sampling(df: pd.DataFrame, gap_threshold: pd.Timedelta) -> SamplingCharacteristics:
    """Report df's empirical sampling profile: the distribution of
    inter-sample intervals, the modal interval (used downstream as
    expected_interval), gaps exceeding gap_threshold, and total span vs.
    sample count. Logs the finding at INFO. Never assumes a sampling rate.
    """
    if df.empty:
        raise ValueError("cannot characterise sampling of an empty DataFrame")

    timestamps = df.index.to_numpy()
    if len(timestamps) < 2:
        raise ValueError("cannot characterise sampling with fewer than 2 samples")

    deltas = np.diff(timestamps)
    deltas_td = pd.Series(deltas)
    interval_counts = deltas_td.value_counts()
    modal_interval = pd.Timedelta(interval_counts.index[0])

    gap_mask = deltas > np.timedelta64(gap_threshold)
    gap_positions = np.flatnonzero(gap_mask)
    gaps = tuple(
        (pd.Timestamp(timestamps[i]), pd.Timestamp(timestamps[i + 1]), pd.Timedelta(deltas[i]))
        for i in gap_positions
    )

    total_span = pd.Timedelta(timestamps[-1] - timestamps[0])
    total_samples = len(df)

    largest_gap = max((g[2] for g in gaps), default=pd.Timedelta(0))
    logger.info(
        "characterise_sampling: modal_interval=%s samples=%d span=%s gaps(>%s)=%d largest_gap=%s",
        modal_interval,
        total_samples,
        total_span,
        gap_threshold,
        len(gaps),
        largest_gap,
    )

    return SamplingCharacteristics(
        modal_interval=modal_interval,
        interval_counts=interval_counts,
        gaps=gaps,
        total_span=total_span,
        total_samples=total_samples,
    )


def maybe_resample(df: pd.DataFrame, settings) -> pd.DataFrame:
    """Resample df onto a regular grid if windowing.resample.enabled;
    otherwise return df unchanged.

    Analog columns (settings.scaling.analog_columns) aggregate by mean;
    digital/status columns (settings.scaling.passthrough_columns) aggregate
    by max -- a flag that was ON at any point in the interval is ON; it is
    never meaningful to average a binary. Intervals with no underlying
    samples produce NaN rows -- NEVER forward-filled or otherwise
    fabricated -- which make_windows treats as gaps.

    Raises:
        ValueError: if a df column is in neither scaling.analog_columns nor
            scaling.passthrough_columns.
    """
    if not settings.windowing.resample.enabled:
        return df

    analog_columns = set(settings.scaling.analog_columns)
    passthrough_columns = set(settings.scaling.passthrough_columns)
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in analog_columns:
            agg[col] = "mean"
        elif col in passthrough_columns:
            agg[col] = "max"
        else:
            raise ValueError(
                f"column {col!r} is in neither scaling.analog_columns nor "
                "scaling.passthrough_columns -- cannot decide a resample "
                "aggregation for it."
            )

    interval = pd.Timedelta(settings.windowing.resample.interval)
    resampled = df.resample(interval).agg(agg)
    logger.info(
        "maybe_resample: resampled to interval=%s -> %d rows (%d NaN rows from empty intervals)",
        interval,
        len(resampled),
        int(resampled.isna().any(axis=1).sum()),
    )
    return resampled


def _check_window_fits_embargo(settings) -> pd.Timedelta:
    """Raise if windowing.window_duration exceeds split.embargo_hours -- a
    window that long could straddle the train/test boundary, exactly the
    leakage the embargo exists to prevent. Returns window_duration as a
    Timedelta for reuse by the caller.
    """
    window_duration = pd.Timedelta(settings.windowing.window_duration)
    embargo = pd.Timedelta(hours=settings.split.embargo_hours)
    if window_duration > embargo:
        raise ValueError(
            f"windowing.window_duration ({window_duration}) exceeds "
            f"split.embargo_hours ({embargo}) -- a window this long could "
            "straddle the train/test boundary, exactly the leakage the "
            "embargo exists to prevent. Shorten window_duration or widen "
            "split.embargo_hours."
        )
    return window_duration


def make_windows(
    df: pd.DataFrame,
    settings,
    stride_mode: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a scaled fold slice into fixed-length, gap-aware sequences.

    Returns (windows, end_timestamps):
        windows: shape (n_windows, window_length, n_channels), float32.
            Channels are scaling.analog_columns + scaling.passthrough_columns
            from config, in that order -- a documented, stable column order.
        end_timestamps: shape (n_windows,), datetime64[ns]. CONVENTION: a
            window is history up to "now", so its score/label belongs at
            its LAST timestamp, not its first -- downstream episode
            grouping depends on this.

    Order is preserved: windows are emitted in ascending end_timestamp
    order and are NEVER shuffled here. (CLAUDE.md's no-shuffle rule governs
    train/test SPLITTING, not minibatch shuffling in an already-correctly
    split training loop -- that is not this function's concern and must not
    be added here.)

    Gap-aware: expected_interval is the empirically modal inter-sample
    interval (see characterise_sampling()); a window of window_length
    samples spanning more than
    window_length * expected_interval * (1 + windowing.gap_tolerance) of
    wall-clock time is DROPPED, never repaired. This covers apply_fold's
    exclusion regions and native recording gaps identically. A window
    containing any NaN (e.g. an empty resampled interval, see
    maybe_resample()) is dropped for the same reason. Counts and reasons
    are logged at INFO.

    `stride_mode` selects windowing.train_stride ("train", the default --
    coarser, less redundant, smaller tensors) or windowing.score_stride
    ("score" -- finer, so scores map back to timestamps densely). Both are
    time durations, converted to a sample count using expected_interval.

    Uses numpy.lib.stride_tricks.sliding_window_view for a zero-copy view
    over all stride-1 positions, then materialises only the
    stride-selected, gap-filtered subset -- avoiding the full stride-1
    tensor a naive implementation would build. Logs the resulting shape
    and estimated memory.

    Raises:
        ValueError: if windowing.window_duration exceeds split.embargo_hours
            (see _check_window_fits_embargo), if stride_mode is not "train"
            or "score", if df (after any resampling) is empty, or if a
            scaling.analog_columns/passthrough_columns entry is missing
            from df.
    """
    window_duration = _check_window_fits_embargo(settings)

    if stride_mode not in ("train", "score"):
        raise ValueError(f"stride_mode must be 'train' or 'score', got {stride_mode!r}")

    df = maybe_resample(df, settings)
    if df.empty:
        raise ValueError("cannot window an empty DataFrame")

    gap_threshold = pd.Timedelta(settings.windowing.gap_threshold)
    characteristics = characterise_sampling(df, gap_threshold)
    expected_interval = characteristics.modal_interval
    if expected_interval <= pd.Timedelta(0):
        raise ValueError(
            f"measured expected_interval ({expected_interval}) is non-positive -- "
            "cannot derive a window length from it."
        )

    stride_duration_str = (
        settings.windowing.train_stride
        if stride_mode == "train"
        else settings.windowing.score_stride
    )
    stride_duration = pd.Timedelta(stride_duration_str)

    window_length = max(1, round(window_duration / expected_interval))
    stride = max(1, round(stride_duration / expected_interval))
    logger.info(
        "make_windows: expected_interval=%s -> window_length=%d samples "
        "(window_duration=%s), stride=%d samples (%s_stride=%s)",
        expected_interval,
        window_length,
        window_duration,
        stride,
        stride_mode,
        stride_duration,
    )

    columns = tuple(settings.scaling.analog_columns) + tuple(settings.scaling.passthrough_columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"column(s) {missing} from scaling.analog_columns/passthrough_columns "
            f"not found in df (columns: {list(df.columns)})"
        )

    values = df[list(columns)].to_numpy(dtype=np.float32)
    timestamps = df.index.to_numpy()
    n_samples = values.shape[0]
    n_channels = values.shape[1]

    if n_samples < window_length:
        logger.info(
            "make_windows: only %d samples, fewer than window_length=%d -- no windows produced",
            n_samples,
            window_length,
        )
        return (
            np.empty((0, window_length, n_channels), dtype=np.float32),
            np.empty((0,), dtype=timestamps.dtype),
        )

    all_windows = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(values, window_length, axis=0), -1, 1
    )  # (n_windows_stride1, window_length, n_channels)

    end_positions_stride1 = np.arange(window_length - 1, n_samples)
    start_positions_stride1 = end_positions_stride1 - (window_length - 1)

    selected = np.arange(0, all_windows.shape[0], stride)
    candidate_windows = all_windows[selected]
    candidate_end_pos = end_positions_stride1[selected]
    candidate_start_pos = start_positions_stride1[selected]

    span = timestamps[candidate_end_pos] - timestamps[candidate_start_pos]
    max_allowed_span = expected_interval * (window_length * (1 + settings.windowing.gap_tolerance))
    span_ok = span <= max_allowed_span.to_timedelta64()

    nan_ok = ~np.isnan(candidate_windows).any(axis=(1, 2))
    keep = span_ok & nan_ok

    n_dropped_span = int(np.sum(~span_ok))
    n_dropped_nan_only = int(np.sum(span_ok & ~nan_ok))
    n_dropped_total = int(np.sum(~keep))

    windows_kept = candidate_windows[keep]
    end_timestamps_kept = timestamps[candidate_end_pos[keep]]

    logger.info(
        "make_windows: %d windows kept, %d dropped (%d for exceeding gap-tolerance span, "
        "%d for containing NaN from an empty resampled interval)",
        int(np.sum(keep)),
        n_dropped_total,
        n_dropped_span,
        n_dropped_nan_only,
    )
    logger.info(
        "make_windows: output shape=%s dtype=%s estimated_memory=%.2f MB",
        windows_kept.shape,
        windows_kept.dtype,
        windows_kept.nbytes / 1e6,
    )

    return windows_kept, end_timestamps_kept
