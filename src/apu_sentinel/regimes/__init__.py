"""Operating-state (regime) segmentation.

The compressor cycles LOADED -> OFFLOAD -> STOPPED -> LOADED (a pressure
duty cycle, see docs/FINDINGS.md §7); most raw signal variance is
mode-switching, not anomaly. Models must condition on regime (CLAUDE.md
rule 4) -- this module derives the per-timestamp labels needed to do that.
It does NOT itself scale, filter, or otherwise consume the labels --
regime-conditional scaling is a later pass, decided after seeing this
pass's evidence (characterise_regimes' global-vs-within-regime variance
comparison). Four labels are produced: LOADED, OFFLOAD, STOPPED,
TRANSITION.

States are derived from the compressor's own DIGITAL control signals
(regimes.control_columns) wherever possible, never by clustering the
analog channels: the digital flags are the machine's own control state,
are free, interpretable, and available at inference time, while clustering
the analog signals to infer state is partly circular -- those same signals
are later judged normal-or-abnormal against regime-conditional statistics.

The one deliberate, documented exception: OFFLOAD vs STOPPED (both COMP=1,
"no air intake") is NOT resolvable by any available digital flag (see
docs/FINDINGS.md §7 -- all eight flags are saturated during this period).
The split uses regimes.offload_split_channel (Motor_current) against
regimes.offload_current_threshold, sitting in a near-total (99.98%-clean)
separation between the two modes' current draw. This makes that channel
partially self-referential if it is also scored for anomalies during
OFFLOAD/STOPPED -- see CLAUDE.md rule 7 and
regimes.exclude_motor_current_when_off.

CRITICAL -- causal only. No information may flow backward in time.
assign_regimes() and its helpers use only pandas.Series.shift(1)/.ffill()
(both backward-looking: shift(1) pulls a PAST value into the current row,
ffill propagates the most recent PAST non-null value forward) -- never a
CENTERED rolling window, a BACKWARD fill, or a positive-lag/future shift,
any of which would let a later sample influence an earlier timestamp's
label. This is not currently enforced by the blocking Claude Code hook
(unlike data/split.py and data/scaling.py), so tests/test_regimes.py
enforces it both structurally (source inspection for the literal forbidden
spellings) and behaviourally (labels for timestamp t are proven unchanged
when all data after t is deleted).

Flag polarity is NEVER assumed. MetroPT-3's own documentation describes
COMP as active when there is NO air intake -- i.e. COMP == 1 plausibly
means OFF/offloaded, the opposite of a naive reading. verify_flag_semantics
determines polarity empirically (cross-referenced against Motor_current)
and the derived polarity is recorded in config (regimes.polarity), never
hardcoded here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def verify_flag_semantics(df: pd.DataFrame, settings) -> dict:
    """Empirically determine each control flag's behaviour -- do NOT take
    polarity on trust from documentation or config.

    For every column in regimes.control_columns: group by its raw value and
    report mean/median for every analog channel (scaling.analog_columns),
    logging a concise Motor_current-based finding for each. Also reports
    the pairwise agreement rate between control columns after normalising
    each via its CONFIGURED polarity (regimes.polarity) -- high agreement
    justifies treating them as redundant/cross-confirming; anything else is
    a finding to surface, not paper over.

    Returns a dict: {"per_column_stats": {col: {"value_counts": {...},
    "stats_by_level": {level: {channel: {"mean":..., "median":...}}}}},
    "pairwise_agreement": {"colA_vs_colB": rate, ...}}.
    """
    analog_columns = list(settings.scaling.analog_columns)
    control_columns = list(settings.regimes.control_columns)

    per_column_stats: dict[str, dict] = {}
    for col in control_columns:
        levels = sorted(df[col].dropna().unique())
        stats_by_level = {}
        for level in levels:
            subset = df.loc[df[col] == level, analog_columns]
            stats_by_level[level] = {
                ch: {"mean": float(subset[ch].mean()), "median": float(subset[ch].median())}
                for ch in analog_columns
            }
        per_column_stats[col] = {
            "value_counts": {float(k): int(v) for k, v in df[col].value_counts().items()},
            "stats_by_level": stats_by_level,
        }

        if "Motor_current" in analog_columns and len(levels) == 2:
            lo, hi = levels
            m_lo = stats_by_level[lo]["Motor_current"]["mean"]
            m_hi = stats_by_level[hi]["Motor_current"]["mean"]
            logger.info(
                "verify_flag_semantics: %s=%s -> mean Motor_current %.4f; "
                "%s=%s -> mean Motor_current %.4f",
                col,
                lo,
                m_lo,
                col,
                hi,
                m_hi,
            )

    polarity = settings.regimes.polarity
    normalized = {
        col: df[col].map(lambda v, col=col: polarity[col].get(int(v))) for col in control_columns
    }
    pairwise_agreement: dict[str, float] = {}
    for i, col_a in enumerate(control_columns):
        for col_b in control_columns[i + 1 :]:
            rate = float((normalized[col_a] == normalized[col_b]).mean())
            pairwise_agreement[f"{col_a}_vs_{col_b}"] = rate
            logger.info(
                "verify_flag_semantics: normalized(%s) vs normalized(%s) agreement = %.4f",
                col_a,
                col_b,
                rate,
            )

    return {"per_column_stats": per_column_stats, "pairwise_agreement": pairwise_agreement}


def _raw_state_labels(df: pd.DataFrame, regimes_settings) -> pd.Series:
    """Vectorised, order-sensitive: the FIRST state in regimes.states (in
    config order) whose listed flag conditions ALL match a row wins.

    Rows matching NONE of regimes.states (with the current config, this
    means COMP=1 -- "no air intake") are then split into OFFLOAD/STOPPED by
    comparing regimes.offload_split_channel against
    regimes.offload_current_threshold -- see docs/FINDINGS.md §7 for why
    this is a deliberate, documented exception to "states come from digital
    flags only," not a silent one: no digital flag resolves this split, so
    the alternative is not detecting a real, physically-distinct state at
    all.
    """
    states = regimes_settings.states
    labels = pd.Series(index=df.index, dtype=object)
    unmatched = pd.Series(True, index=df.index)
    for state_name, condition in states.items():
        match = pd.Series(True, index=df.index)
        for flag, required_value in condition.items():
            match &= df[flag] == required_value
        assign_mask = match & unmatched
        labels.loc[assign_mask] = state_name
        unmatched &= ~match

    if unmatched.any():
        channel = regimes_settings.offload_split_channel
        threshold = regimes_settings.offload_current_threshold
        offload_mask = unmatched & (df[channel] >= threshold)
        stopped_mask = unmatched & ~offload_mask
        labels.loc[offload_mask] = "OFFLOAD"
        labels.loc[stopped_mask] = "STOPPED"

    return labels


def _time_since_last_change(labels: pd.Series) -> pd.Series:
    """For each row, wall-clock time since `labels` most recently changed
    value (0 at the first sample of a run). Causal: shift(1) pulls the
    PAST value forward for comparison, and ffill propagates only PAST
    non-null values forward -- never a BACKWARD fill.
    """
    index = labels.index
    changed = labels.ne(labels.shift(1))
    changed.iloc[0] = True
    change_times = pd.Series(index, index=index).where(changed)
    last_change_time = change_times.ffill()
    return pd.Series(index, index=index) - last_change_time


def _debounce_min_duration(raw_state: pd.Series, min_duration: pd.Timedelta) -> pd.Series:
    """Causal minimum-state-duration debounce.

    A newly-changed raw state is committed only once it has persisted
    (checked by looking BACKWARD from "now") for at least min_duration;
    until then, samples keep whatever the PREVIOUSLY COMMITTED state was.

    This means: a run shorter than min_duration is absorbed into the
    preceding committed state and NEVER emitted, regardless of what
    happens after it ends; and a run that eventually persists long enough
    only starts being reported as its own state from the min_duration mark
    onward, not retroactively from its true start. Both properties are
    required for causality -- deciding a run's fate from its full
    (start-to-end) duration would use, for its early samples, information
    from strictly later in that same run.

    The very first raw state has no preceding committed state to fall back
    on, so it is accepted immediately regardless of its own duration.
    """
    persisted_enough = _time_since_last_change(raw_state) >= min_duration
    committed = raw_state.where(persisted_enough).ffill()
    return committed.fillna(raw_state.iloc[0])


def _apply_transition_settle(committed: pd.Series, transition_settle: pd.Timedelta) -> pd.Series:
    """Causal: samples within transition_settle of the most recent
    PRECEDING committed-state change are labelled TRANSITION.
    """
    in_transition = _time_since_last_change(committed) <= transition_settle
    return committed.mask(in_transition, "TRANSITION")


def assign_regimes(df: pd.DataFrame, settings) -> pd.Series:
    """Per-timestamp regime label -- one of LOADED, OFFLOAD, STOPPED,
    TRANSITION. CAUSAL: assignment for timestamp t uses ONLY data at or
    before t (see tests/test_regimes.py's causality test, which asserts
    labels are unchanged when all data after t is deleted).

    Does not modify df, scale anything, or filter rows -- returns labels
    only, as a categorical Series aligned exactly to df.index (same
    length, same order).

    `settings` must expose `settings.regimes` (states, offload_split_channel,
    offload_current_threshold, min_duration, transition_settle) -- the
    shape of apu_sentinel.config.Settings.
    """
    raw = _raw_state_labels(df, settings.regimes)
    min_duration = pd.Timedelta(settings.regimes.min_duration)
    committed = _debounce_min_duration(raw, min_duration)
    transition_settle = pd.Timedelta(settings.regimes.transition_settle)
    final = _apply_transition_settle(committed, transition_settle)
    return final.astype("category")


def _channel_stats(values: np.ndarray) -> dict[str, float]:
    q75, q25 = np.percentile(values, [75, 25])
    return {"center": float(np.median(values)), "iqr": float(q75 - q25)}


def characterise_regimes(df: pd.DataFrame, regimes: pd.Series, settings) -> dict:
    """Occupancy, contiguous-run-length distribution, and per-regime
    analog statistics -- the evidence for whether conditioning on regime
    actually collapses the pathological global spread (CLAUDE.md rule 4).

    This is DESCRIPTIVE, offline analysis of already-produced (causal)
    labels -- unlike assign_regimes, it may freely use a run's full
    start-to-end duration.

    Returns {"occupancy": {state: {"count", "percent"}}, "run_lengths":
    {state: {"n_runs", "median_seconds", "q25_seconds", "q75_seconds"}},
    "global_stats": {channel: {"center", "iqr"}}, "regime_stats":
    {state: {channel: {"center", "iqr"}}}}.
    """
    analog_columns = list(settings.scaling.analog_columns)
    total = len(regimes)
    if total == 0:
        raise ValueError("cannot characterise regimes for an empty Series")

    occupancy = {
        str(state): {"count": int(count), "percent": float(count) / total * 100.0}
        for state, count in regimes.value_counts().items()
    }

    values = regimes.to_numpy()
    index = regimes.index
    n = len(values)
    change_positions = np.flatnonzero(values[1:] != values[:-1]) + 1
    run_starts = np.concatenate(([0], change_positions))
    run_ends = np.concatenate((change_positions - 1, [n - 1]))
    run_labels = values[run_starts]
    run_durations_seconds = np.array(
        [(index[e] - index[s]).total_seconds() for s, e in zip(run_starts, run_ends, strict=True)]
    )

    durations_by_state: dict[str, list[float]] = {}
    for label, duration in zip(run_labels, run_durations_seconds, strict=True):
        durations_by_state.setdefault(str(label), []).append(duration)

    run_lengths = {}
    for state, durations in durations_by_state.items():
        arr = np.array(durations)
        run_lengths[state] = {
            "n_runs": len(arr),
            "median_seconds": float(np.median(arr)),
            "q25_seconds": float(np.percentile(arr, 25)),
            "q75_seconds": float(np.percentile(arr, 75)),
        }

    global_stats = {ch: _channel_stats(df[ch].to_numpy(dtype=float)) for ch in analog_columns}

    regime_stats: dict[str, dict] = {}
    for state in occupancy:
        subset = df.loc[regimes.to_numpy() == state, analog_columns]
        if len(subset) == 0:
            continue
        regime_stats[state] = {
            ch: _channel_stats(subset[ch].to_numpy(dtype=float)) for ch in analog_columns
        }

    logger.info(
        "characterise_regimes: occupancy=%s",
        {state: f"{v['percent']:.2f}%" for state, v in occupancy.items()},
    )

    return {
        "occupancy": occupancy,
        "run_lengths": run_lengths,
        "global_stats": global_stats,
        "regime_stats": regime_stats,
    }
