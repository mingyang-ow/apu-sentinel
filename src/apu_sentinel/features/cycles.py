"""Causal cycle-timing features: duration and pressure-decay families.

Per docs/FINDINGS.md §8, median STOPPED duration is the strongest leak
signal found so far (a clear ~10-day precursor before event 2, a sharp dip
at event 4), but it is not fixed to a single obvious form: duration is a
simple, interpretable signal; pressure decay RATE is the physically more
direct alternative (an air leak decays pressure faster, which shows up as
both a shorter STOPPED period AND a steeper slope while it lasts). This
module builds BOTH families so a later baseline pass can choose between
them on evidence, rather than picking one now on intuition.

CRITICAL -- causality. At time t inside a run, the run's eventual total
duration is NOT yet known -- only what has been observed up to and
including t. Every feature therefore comes in one of two forms:

  - "last-completed": the value from the most recently FINISHED run,
    forward-filled until the NEXT run of that kind finishes. Never
    reflects the run containing t itself, even if that run will turn out
    to match -- whether t's own run has finished is exactly the thing t
    cannot know about itself.
  - "running": the value accumulated so far within the CURRENT run (NaN
    when t is not in the relevant state), computed only from samples at or
    before t.

Run-boundary detection (_run_segments) itself IS safe to compute in one
vectorised pass over the full array: a boundary at position i is decided
purely by comparing position i to position i-1 (regimes.assign_regimes'
labels are themselves causal, and a timestamp gap is a purely local
diff) -- never by anything at position > i. What is NOT safe is treating a
run as "complete" before its true end has been observed; _last_completed_*
below handles that by only updating its running "last known" value at the
moment a run's END boundary is crossed, which is exactly when that
completion first becomes knowable.

Only pandas.Series.shift(1) (backward: pulls a PAST value into the current
row) and time-based .rolling() (which pandas implements as backward-looking
only, unlike a centered window) are used. Never a CENTERED rolling window,
a BACKWARD fill, or a positive-lag/future shift.

Gap-truncation -- the discriminator between the two feature families. A
STOPPED run cut short by a data gap (features.gap_threshold) yields an
INVALID duration (the run did not really end; it was truncated -- set to
NaN, never the truncated value) but a STILL VALID decay rate (the slope
over the samples actually observed remains meaningful regardless of
whether more of the run existed unobserved). Both cases are flagged via
`run_gap_truncated` so downstream analysis can see why a duration is NaN
distinctly from "no run has completed yet at all".

Drift normalisation -- the trailing baseline. Per docs/FINDINGS.md §8,
median STOPPED duration drifts ~1400s (Feb) -> ~500s (Aug), a 3x shift
larger than most event-related dips, so absolute thresholds on these
features cannot work. baseline_relative() computes value / trailing_median
(value) over features.baseline_window, itself a rolling (never frozen)
computation -- using test-period history for this baseline is legitimate;
it is causal, exactly what would be available in production, unlike
scaler statistics which must never see test data. The window is a
deliberate tradeoff: a baseline that adapts too fast absorbs gradual
degradation and loses the signal (boiling-frog) -- event 2's ~3x step
change survives a 7-day baseline; slow, gradual drift would not.

Public API:
- `compute_cycle_features(df, regimes, settings) -> pd.DataFrame` -- the
  main entry point. df + regimes must share an index (raises otherwise);
  settings.features.decay_source_channel must be a column of df (raises
  otherwise). Returns a frame aligned exactly to df.index with
  stopped_duration_last, offload_duration_last, loaded_duration_last,
  stopped_elapsed, cycle_period_last, duty_ratio_trailing, decay_rate_last,
  decay_rate_running, run_gap_truncated, and *_rel_baseline variants of the
  three drift-prone columns.
- `baseline_relative(series, window) -> pd.Series` -- value /
  trailing_median(value); reusable by callers (e.g. models/rule_based.py)
  that need the same drift-normalisation on their OWN derived quantities.
- `baseline_relative_lagged(series, window, lag) -> pd.Series` -- same
  idea, but the reference is anchored `lag` back so a degradation lasting
  less than `lag` cannot pull its own baseline down to meet it (pass 16's
  boiling-frog fix, docs/findings/12-event2-error-analysis.md). NaN during
  warm-up (less than `lag + window` of history), never a partial baseline.
- `last_completed_run_peak(df, regimes, settings, target_label, channel)
  -> pd.Series` -- forward-filled peak of `channel` over the most recently
  COMPLETED run of `target_label`; NaN'd (not kept) when that run was
  gap-truncated, same causal discipline as stopped_duration_last.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _RunSegment:
    start_idx: int
    end_idx: int  # inclusive
    label: str
    # This run's END boundary coincided with a data gap (>= gap_threshold)
    # -- its true end/exit condition was not observed, so its duration is
    # invalid (must be NaNed), though its decay rate over observed samples
    # remains valid.
    gap_truncated: bool
    # This run's START boundary was forced by a data gap following a run
    # of the SAME label (not a genuine transition into this label) -- used
    # to avoid treating a gap-split of one physical run as a new cycle.
    start_is_gap_continuation: bool


def _run_segments(regimes: pd.Series, gap_threshold: pd.Timedelta) -> list[_RunSegment]:
    """Split regimes into contiguous runs, breaking at EITHER a label
    change or a timestamp gap >= gap_threshold. Purely local (backward)
    comparisons only -- see module docstring for why this is causal.
    """
    values = regimes.astype(str).to_numpy()
    index = regimes.index
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [_RunSegment(0, 0, values[0], False, False)]

    label_changed = values[1:] != values[:-1]
    gap_occurred = np.diff(index.to_numpy()) > np.timedelta64(gap_threshold)
    boundary = label_changed | gap_occurred
    boundary_positions = np.flatnonzero(boundary) + 1
    starts = np.concatenate(([0], boundary_positions))
    ends = np.concatenate((boundary_positions - 1, [n - 1]))

    segments = []
    for i in range(len(starts)):
        s, e = int(starts[i]), int(ends[i])
        gap_truncated = bool(e < n - 1 and gap_occurred[e])
        start_is_gap_continuation = False
        if i > 0:
            prev_label = values[starts[i - 1]]
            boundary_before = s - 1
            if (
                gap_occurred[boundary_before]
                and not label_changed[boundary_before]
                and prev_label == values[s]
            ):
                start_is_gap_continuation = True
        segments.append(_RunSegment(s, e, values[s], gap_truncated, start_is_gap_continuation))
    return segments


def _last_completed_forward_fill(
    n: int,
    segments: list[_RunSegment],
    target_label: str,
    value_fn,
    null_if_gap_truncated: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """For every row, the value_fn() of the most recently COMPLETED run
    matching target_label, forward-filled from the moment that run
    completed until the next completion -- NEVER the run containing the
    row itself (see module docstring). NaN before any completion.

    null_if_gap_truncated: if True, a completed run's contributed value is
    replaced with NaN when that run was gap_truncated (duration features);
    if False, the computed value is kept regardless (decay-rate features).
    """
    values = np.full(n, np.nan)
    gap_flags = np.zeros(n, dtype=bool)
    last_value = np.nan
    last_gap_truncated = False
    for seg in segments:
        values[seg.start_idx : seg.end_idx + 1] = last_value
        gap_flags[seg.start_idx : seg.end_idx + 1] = last_gap_truncated
        if seg.label == target_label:
            computed = value_fn(seg)
            last_gap_truncated = seg.gap_truncated
            last_value = np.nan if (null_if_gap_truncated and seg.gap_truncated) else computed
    return values, gap_flags


def _running_elapsed(
    n: int, segments: list[_RunSegment], index: pd.DatetimeIndex, target_label: str
) -> np.ndarray:
    """Time (seconds) since the CURRENT run began, for rows whose run
    matches target_label; NaN elsewhere.
    """
    values = np.full(n, np.nan)
    for seg in segments:
        if seg.label != target_label:
            continue
        seg_index = index[seg.start_idx : seg.end_idx + 1]
        elapsed = np.asarray((seg_index - seg_index[0]).total_seconds())
        values[seg.start_idx : seg.end_idx + 1] = elapsed
    return values


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0:
        return float("nan")
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _running_decay_rate(
    n: int,
    segments: list[_RunSegment],
    index: pd.DatetimeIndex,
    channel_values: np.ndarray,
    target_label: str,
    min_samples: int,
) -> np.ndarray:
    """Expanding-window least-squares slope of channel_values vs. elapsed
    time, within the CURRENT run (target_label only); NaN until
    min_samples have accumulated. Vectorised via cumulative sums (an
    O(run_length) expanding regression, not an O(run_length^2) per-row
    refit).
    """
    values = np.full(n, np.nan)
    for seg in segments:
        if seg.label != target_label:
            continue
        seg_index = index[seg.start_idx : seg.end_idx + 1]
        x = np.asarray((seg_index - seg_index[0]).total_seconds())
        y = channel_values[seg.start_idx : seg.end_idx + 1]
        m = len(x)
        cum_n = np.arange(1, m + 1, dtype=float)
        cum_x = np.cumsum(x)
        cum_y = np.cumsum(y)
        cum_xy = np.cumsum(x * y)
        cum_x2 = np.cumsum(x * x)
        denom = cum_n * cum_x2 - cum_x**2
        slope = np.full(m, np.nan)
        valid = (cum_n >= min_samples) & (denom != 0)
        slope[valid] = (cum_n[valid] * cum_xy[valid] - cum_x[valid] * cum_y[valid]) / denom[valid]
        values[seg.start_idx : seg.end_idx + 1] = slope
    return values


def _cycle_period_last(n: int, segments: list[_RunSegment], index: pd.DatetimeIndex) -> np.ndarray:
    """Forward-filled duration of the last COMPLETED full
    STOPPED->OFFLOAD->LOADED->STOPPED cycle: the time between the starts of
    two consecutive GENUINE STOPPED runs (a STOPPED run that only exists
    because a gap split an already-ongoing STOPPED run does not count as a
    new cycle boundary -- see _RunSegment.start_is_gap_continuation).

    Unlike the duration/decay families, this "completes" (becomes
    knowable) at the FIRST sample of the new STOPPED run, not at that run's
    end -- both start timestamps needed for the interval are already in
    hand at that row, using only current-and-past data. So the freshly
    computed period is visible starting from that same run's own rows,
    not deferred to the run after it.
    """
    values = np.full(n, np.nan)
    last_period = np.nan
    last_stopped_start: pd.Timestamp | None = None
    for seg in segments:
        if seg.label == "STOPPED" and not seg.start_is_gap_continuation:
            current_start = index[seg.start_idx]
            if last_stopped_start is not None:
                last_period = (current_start - last_stopped_start).total_seconds()
            last_stopped_start = current_start
        values[seg.start_idx : seg.end_idx + 1] = last_period
    return values


def last_completed_run_peak(
    df: pd.DataFrame,
    regimes: pd.Series,
    settings,
    target_label: str,
    channel: str,
) -> pd.Series:
    """Forward-filled peak (max) of `channel` over the most recently
    COMPLETED run labelled `target_label` -- same last-completed-forward-
    fill causality as stopped_duration_last (module docstring): a run's
    peak only becomes knowable once its END boundary is crossed, never
    from the run containing the current row itself.

    Treated like the duration family (null_if_gap_truncated=True), not the
    decay-rate family: a run cut short by a data gap may not have reached
    its true peak yet, so a gap-truncated run's observed peak is dropped
    (NaN) rather than kept as if it were the genuine peak.

    `settings` must expose `settings.features.gap_threshold` -- the shape
    of apu_sentinel.config.Settings.

    Raises:
        ValueError: if df and regimes are misaligned, or if `channel` is
            not a column of df.
    """
    if len(df) != len(regimes) or not df.index.equals(regimes.index):
        raise ValueError("df and regimes must share the same index")
    if channel not in df.columns:
        raise ValueError(f"channel {channel!r} not found in df (columns: {list(df.columns)})")

    gap_threshold = pd.Timedelta(settings.features.gap_threshold)
    segments = _run_segments(regimes, gap_threshold)
    channel_values = df[channel].to_numpy(dtype=float)

    def peak_fn(seg: _RunSegment) -> float:
        return float(np.max(channel_values[seg.start_idx : seg.end_idx + 1]))

    values, _ = _last_completed_forward_fill(
        len(df), segments, target_label, peak_fn, null_if_gap_truncated=True
    )
    return pd.Series(values, index=df.index)


def _duty_ratio_trailing(regimes: pd.Series, window: pd.Timedelta) -> pd.Series:
    """LOADED fraction of time over a trailing window. pandas time-based
    .rolling() is backward-looking only (never centered), so this is
    causal by construction. Each sample is weighted equally regardless of
    its interval to the next -- a simplification given the dataset's ~10s
    near-regular cadence, not a time-weighted duty ratio.
    """
    is_loaded = (regimes.astype(str) == "LOADED").astype(float)
    return is_loaded.rolling(window, min_periods=1).mean()


def baseline_relative(series: pd.Series, window: pd.Timedelta) -> pd.Series:
    """value / trailing_median(value) over `window` -- see module
    docstring for the drift-normalisation tradeoff this encodes.
    """
    baseline = series.rolling(window, min_periods=1).median()
    return series / baseline


def baseline_relative_lagged(
    series: pd.Series, window: pd.Timedelta, lag: pd.Timedelta
) -> pd.Series:
    """value / trailing_median(value) computed over a window that EXCLUDES
    the recent past: the reference for row t is the median of `series` over
    [t - lag - window, t - lag], never [t - window, t].

    Fixes a structural blind spot in `baseline_relative()` (pass 16,
    docs/findings/12-event2-error-analysis.md): a ratio to a TRAILING median
    can only see the *rate of change* of a signal, never its *level* --
    degradation that persists longer than `window` pulls the reference down
    to meet it, so the ratio returns to ~1.0 while the absolute value stays
    depressed (boiling-frog). Anchoring the reference `lag` back in time
    means a degradation lasting less than `lag` cannot contaminate its own
    baseline, at the cost of the reference itself being `lag` stale.

    Still CAUSAL: the reference for row t only ever reads samples at or
    before t - lag, strictly before t. Implemented as an as-of (backward)
    lookup, not a fixed-offset shift, so it degrades gracefully across the
    dataset's irregular sampling/gaps exactly like `baseline_relative`'s own
    rolling median does.

    WARM-UP: rows less than `lag + window` past the series' own start have
    no valid reference and are set to NaN -- never a partial/shortened
    baseline (see module docstring's gap-truncation discipline: an invalid
    reference must read as "unknown", not be silently substituted with one
    computed from less data than requested). This requires strictly more
    than `min_periods=1`'s "some data" -- the trailing median feeding the
    reference is itself masked to NaN until a full `window` of elapsed
    CALENDAR time (not sample count) has passed since `series.index[0]`.

    Drift note (docs/findings/08-cycle-timing.md): the ~3x seasonal
    STOPPED-duration drift plays out over ~7 months: over any single
    `lag`-sized gap (default 14 days) it amounts to roughly a 7% shift --
    negligible against the ~3.6x signal this fixes (findings/12). A 14-day
    lag does not reintroduce the drift problem `baseline_window` exists to
    solve.
    """
    trailing = series.rolling(window, min_periods=1).median()
    elapsed_since_start = trailing.index - trailing.index[0]
    trailing = trailing.where(elapsed_since_start >= window)

    shifted_index = trailing.index + lag
    lookup = pd.Series(trailing.to_numpy(), index=shifted_index)
    baseline = lookup.reindex(series.index, method="ffill")
    return series / baseline


def compute_cycle_features(df: pd.DataFrame, regimes: pd.Series, settings) -> pd.DataFrame:
    """Both cycle-timing feature families (duration and pressure-decay),
    plus baseline-relative variants of the three features most subject to
    drift (docs/FINDINGS.md §8): stopped_duration_last, cycle_period_last,
    decay_rate_last.

    `settings` must expose `settings.features` (decay_source_channel,
    decay_min_samples, gap_threshold, baseline_window, duty_ratio_window)
    -- the shape of apu_sentinel.config.Settings.

    Returns a DataFrame aligned exactly to df.index (same length, order,
    no rows added/dropped/reordered) with columns: stopped_duration_last,
    offload_duration_last, loaded_duration_last, stopped_elapsed,
    cycle_period_last, duty_ratio_trailing, decay_rate_last,
    decay_rate_running, run_gap_truncated, and the three
    *_rel_baseline columns.

    Raises:
        ValueError: if df and regimes are misaligned, or if
            features.decay_source_channel is not a column of df.
    """
    if len(df) != len(regimes) or not df.index.equals(regimes.index):
        raise ValueError("df and regimes must share the same index")

    decay_channel = settings.features.decay_source_channel
    if decay_channel not in df.columns:
        raise ValueError(
            f"features.decay_source_channel {decay_channel!r} not found in df "
            f"(columns: {list(df.columns)})"
        )

    index = regimes.index
    n = len(regimes)
    gap_threshold = pd.Timedelta(settings.features.gap_threshold)
    segments = _run_segments(regimes, gap_threshold)
    channel_values = df[decay_channel].to_numpy(dtype=float)

    def duration_fn(seg: _RunSegment) -> float:
        return (index[seg.end_idx] - index[seg.start_idx]).total_seconds()

    def decay_fn(seg: _RunSegment) -> float:
        seg_index = index[seg.start_idx : seg.end_idx + 1]
        x = np.asarray((seg_index - seg_index[0]).total_seconds())
        y = channel_values[seg.start_idx : seg.end_idx + 1]
        return _linear_slope(x, y)

    out = pd.DataFrame(index=index)

    stopped_duration_last, run_gap_truncated = _last_completed_forward_fill(
        n, segments, "STOPPED", duration_fn, null_if_gap_truncated=True
    )
    out["stopped_duration_last"] = stopped_duration_last
    out["run_gap_truncated"] = run_gap_truncated

    offload_duration_last, _ = _last_completed_forward_fill(
        n, segments, "OFFLOAD", duration_fn, null_if_gap_truncated=True
    )
    out["offload_duration_last"] = offload_duration_last

    loaded_duration_last, _ = _last_completed_forward_fill(
        n, segments, "LOADED", duration_fn, null_if_gap_truncated=True
    )
    out["loaded_duration_last"] = loaded_duration_last

    out["stopped_elapsed"] = _running_elapsed(n, segments, index, "STOPPED")
    out["cycle_period_last"] = _cycle_period_last(n, segments, index)

    duty_ratio_window = pd.Timedelta(settings.features.duty_ratio_window)
    out["duty_ratio_trailing"] = _duty_ratio_trailing(regimes, duty_ratio_window).to_numpy()

    decay_rate_last, _ = _last_completed_forward_fill(
        n, segments, "STOPPED", decay_fn, null_if_gap_truncated=False
    )
    out["decay_rate_last"] = decay_rate_last
    out["decay_rate_running"] = _running_decay_rate(
        n, segments, index, channel_values, "STOPPED", settings.features.decay_min_samples
    )

    baseline_window = pd.Timedelta(settings.features.baseline_window)
    out["stopped_duration_last_rel_baseline"] = baseline_relative(
        out["stopped_duration_last"], baseline_window
    )
    out["cycle_period_last_rel_baseline"] = baseline_relative(
        out["cycle_period_last"], baseline_window
    )
    out["decay_rate_last_rel_baseline"] = baseline_relative(out["decay_rate_last"], baseline_window)

    return out
