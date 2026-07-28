"""EPISODE-LEVEL evaluation. THE metric every model is scored on.

Never per-timestamp, never per-parameter (CLAUDE.md rule 5). All of
thresholding, grouping flagged timestamps into episodes via a documented
hold-time/hysteresis + gap-awareness rule, categorising episodes
(early_warning / concurrent / masked / false_alarm), detection + lead
time, false-alarm rate, and attaching each episode's mixed ranked
diagnosis (from explain/) happens here -- models (models/) never touch
this. If the harness needs something a model doesn't provide (see
models/base.py's AnomalyModel contract: scores + contributions),
that is a contract problem to raise, not to work around.

Primary objective is RECALL; false-alarm rate is a monitored secondary
under a ceiling that starts null and is set only after baseline behaviour
is seen. Point accuracy/F1 are never computed as headline outputs here.

Public API:
- `ScoredTestData` -- a model's already-computed (timestamps, scores,
  contributions, contributor_names, expected_interval) for one fold's
  scored period. The harness never scores anything itself; callers
  (pipeline.py) always supply this.
- `Episode` / `FoldEvaluation` -- see docs/ARCHITECTURE.md's Contracts
  section for exactly what each field carries and who produces it.
- `fit_threshold(train_scores, settings) -> float` /
  `fit_threshold_sweep(train_scores, settings) -> {quantile: float}` --
  train-scores-only, never test data (see fit_threshold's own docstring
  for the structural guard).
- `evaluate_fold_at_threshold(fold, event, window_width_hours, threshold,
  test_data, settings) -> FoldEvaluation` -- the core scoring function,
  given an ALREADY-FITTED threshold. `evaluate_fold(...)` fits the
  threshold first via fit_threshold; `evaluate_fold_sweep(...)` does both
  for every evaluation.threshold_quantiles value.
- `categorise_episode(episode_start, episode_end, event, window_width_hours,
  masked) -> str` -- one of CATEGORIES, precedence early_warning >
  concurrent > masked > false_alarm.
- Pass 13 null comparison: `p_chance_poisson(false_alarms_per_day,
  window_width_hours) -> float`; `p_chance_permutation(episodes,
  test_start, test_end, window_width_hours, n_samples) -> float` (raises
  if no candidate placement's window fits the test period); `evaluate_chance
  (result, fold, settings) -> ChanceComparison` combines both plus the
  chance_threshold flag.
- Pass 13 dual false-alarm estimation: `evaluate_pooled_stretches(stretches,
  scored_stretches, threshold, settings) -> PooledEvaluation` -- pools
  false-alarm rate across evaluation/events.py's pooled_normal_stretches;
  raises if scored_stretches doesn't align 1:1 with stretches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apu_sentinel.evaluation.events import (
    NormalStretch,
    masked_regions,
    pre_failure_window,
    window_coverage,
)
from apu_sentinel.explain import rank_channel_contributions

CATEGORIES = ("early_warning", "concurrent", "masked", "false_alarm")


@dataclass(frozen=True)
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp
    peak_score: float
    category: str  # one of CATEGORIES
    matched_event_id: int | None
    channel_ranking: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FoldEvaluation:
    event_id: int
    window_width: pd.Timedelta
    threshold: float
    detected: bool
    lead_time: pd.Timedelta | None
    concurrent_only: bool
    episodes: tuple[Episode, ...]
    false_episode_count: int
    evaluated_days: float
    false_alarms_per_day: float
    window_coverage: float


@dataclass(frozen=True)
class ScoredTestData:
    """A model's already-computed output for one fold's test period, plus
    the context needed to score it -- exactly the models/base.py
    AnomalyModel contract (scores, contributions) plus the make_windows
    end-timestamps and contributor names/expected_interval, and nothing
    more. Bundled to keep evaluate_fold's signature manageable.

    channel_names is populated from the MODEL's own contributor_names
    (models/base.py), never from config -- a rule-based model's
    contributors are rule names, not scaling.analog_columns, so hardcoding
    a config channel list here would be wrong for any non-channel-attributing
    model.
    """

    timestamps: pd.DatetimeIndex
    scores: np.ndarray
    contributions: np.ndarray  # shape (n_timestamps, n_contributors)
    channel_names: tuple[str, ...]
    expected_interval: pd.Timedelta


# --- Step 1: thresholding -- fitted on TRAIN scores only -------------------


def fit_threshold(train_scores: np.ndarray, settings) -> float:
    """Fit the alert threshold as a high quantile of TRAINING scores ONLY.

    This is a leakage vector exactly like data/scaling.py's fit_scaler:
    the signature accepts only train_scores, never test scores or test
    outcomes, so there is no code path by which the test period can
    influence the threshold. Applied UNCHANGED to test scores by the
    caller (evaluate_fold_at_threshold).
    """
    return float(np.quantile(train_scores, settings.evaluation.threshold_quantile))


def fit_threshold_sweep(train_scores: np.ndarray, settings) -> dict[float, float]:
    """fit_threshold() for every quantile in evaluation.threshold_quantiles
    -- still fit on train_scores ONLY, one threshold per quantile.

    Produces a curve of results (see evaluate_fold_sweep). Selecting the
    test-optimal quantile from that curve and reporting it as "the" result
    is FORBIDDEN: quantiles must be chosen from train/validation behaviour
    (e.g. the false-alarm rate they produce on validation folds), never
    from which one happens to score best against these few test failures.
    """
    return {q: float(np.quantile(train_scores, q)) for q in settings.evaluation.threshold_quantiles}


# --- Step 2: episode grouping -- hysteresis AND gap-awareness --------------


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a 1-D boolean array, as inclusive
    (start_idx, end_idx) index pairs.
    """
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _coverage_segments(
    timestamps: np.ndarray, score_gap_threshold: pd.Timedelta
) -> list[tuple[int, int]]:
    """Split timestamps into contiguous (start_idx, end_idx) segments,
    breaking wherever two consecutive timestamps differ by more than
    score_gap_threshold. These breaks are the "no data" gaps (from windows
    dropped for spanning a data gap, see data/windows.py) that must never
    be bridged by episode grouping, independent of episode_hold_time.
    """
    n = len(timestamps)
    if n == 0:
        return []
    diffs = np.diff(timestamps)
    gap_positions = np.flatnonzero(diffs > np.timedelta64(score_gap_threshold))
    boundaries = [0, *(gap_positions + 1).tolist(), n]
    return [(boundaries[i], boundaries[i + 1] - 1) for i in range(len(boundaries) - 1)]


def _group_episode_index_ranges(
    timestamps: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    hold_time: pd.Timedelta,
    score_gap_threshold: pd.Timedelta,
    min_duration: pd.Timedelta,
) -> list[tuple[int, int]]:
    """Group above-threshold timestamps into episodes: (start_idx, end_idx)
    inclusive index ranges into the global timestamps/scores/contributions
    arrays.

    Hysteresis: adjacent above-threshold bursts within the SAME coverage
    segment merge into one episode if the below-threshold gap between them
    is <= hold_time; a below-threshold stretch longer than that ends the
    episode. Gap-awareness: bursts in DIFFERENT coverage segments (i.e.
    separated by a data gap > score_gap_threshold) NEVER merge, regardless
    of hold_time or elapsed time -- absence of evidence is not evidence of
    continuation. min_duration (0 = off) then drops episodes shorter than
    it.
    """
    above = scores >= threshold
    episodes: list[list[int]] = []
    for seg_start, seg_end in _coverage_segments(timestamps, score_gap_threshold):
        seg_mask = above[seg_start : seg_end + 1]
        runs = [(seg_start + a, seg_start + b) for a, b in _find_runs(seg_mask)]
        merged: list[list[int]] = []
        for run_start, run_end in runs:
            if merged and (timestamps[run_start] - timestamps[merged[-1][1]]) <= hold_time:
                merged[-1][1] = run_end
            else:
                merged.append([run_start, run_end])
        episodes.extend(merged)

    if min_duration > pd.Timedelta(0):
        episodes = [(a, b) for a, b in episodes if (timestamps[b] - timestamps[a]) >= min_duration]
    return [(a, b) for a, b in episodes]


# --- Step 3: categorisation -- four categories, not two --------------------


def _overlaps(
    a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp
) -> bool:
    return a_start <= b_end and b_start <= a_end


def categorise_episode(
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    event,
    window_width_hours: float,
    masked: tuple,
) -> str:
    """One of CATEGORIES, checked in this precedence order:

    1. early_warning -- overlaps [event.start - width, event.start). The
       successful-detection case, checked first.
    2. concurrent -- overlaps [event.start, event.end]. Checked BEFORE
       masked because the event's own masked region (events.masked_regions)
       necessarily CONTAINS its concurrent period -- without this
       precedence every concurrent detection would be miscategorised as
       masked.
    3. masked -- falls inside ANY masked region: this event's post-failure
       settling tail (beyond event.end), or a different event's masked
       region entirely.
    4. false_alarm -- everything else.
    """
    pre_start, pre_end = pre_failure_window(event, window_width_hours)
    if _overlaps(episode_start, episode_end, pre_start, pre_end):
        return "early_warning"

    event_start, event_end = pd.Timestamp(event.start), pd.Timestamp(event.end)
    if _overlaps(episode_start, episode_end, event_start, event_end):
        return "concurrent"

    for region in masked:
        if _overlaps(episode_start, episode_end, region.start, region.end):
            return "masked"

    return "false_alarm"


def _build_episodes(
    episode_ranges: list[tuple[int, int]],
    timestamps: np.ndarray,
    scores: np.ndarray,
    contributions: np.ndarray,
    channel_names: tuple[str, ...],
    event,
    window_width_hours: float,
    masked: tuple,
    contribution_aggregation: str,
) -> tuple[Episode, ...]:
    episodes = []
    for start_idx, end_idx in episode_ranges:
        start_ts = pd.Timestamp(timestamps[start_idx])
        end_ts = pd.Timestamp(timestamps[end_idx])
        peak = float(scores[start_idx : end_idx + 1].max())
        ranking = rank_channel_contributions(
            contributions[start_idx : end_idx + 1], channel_names, contribution_aggregation
        )
        category = categorise_episode(start_ts, end_ts, event, window_width_hours, masked)
        matched_id = event.id if category in ("early_warning", "concurrent") else None
        episodes.append(
            Episode(
                start=start_ts,
                end=end_ts,
                peak_score=peak,
                category=category,
                matched_event_id=matched_id,
                channel_ranking=ranking,
            )
        )
    return tuple(episodes)


# --- Step 4: detection and lead time ---------------------------------------


def _detection_and_lead_time(
    episodes: tuple[Episode, ...], event
) -> tuple[bool, pd.Timedelta | None]:
    """detected = any early_warning episode exists. lead_time is measured
    from when the EARLIEST such episode's alert FIRED (its start), not its
    peak or end -- that is the actionable warning time.
    """
    early = [e for e in episodes if e.category == "early_warning"]
    if not early:
        return False, None
    earliest = min(early, key=lambda e: e.start)
    return True, pd.Timestamp(event.start) - earliest.start


# --- Step 5: false-alarm rate -- normalised by COVERED time ----------------


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _evaluated_days(
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    timestamps: np.ndarray,
    masked: tuple,
    score_gap_threshold: pd.Timedelta,
) -> float:
    """Wall-clock duration of [test_start, test_end] MINUS masked regions
    MINUS data gaps (from actual score coverage), in days. NEVER calendar
    days: ~18% of the timeline is missing data and masked regions are
    excluded too, so calendar normalisation understates the true
    false-alarm rate. Masked-region and gap intervals are merged before
    subtracting so an overlap between the two is never double-counted.
    """
    excluded: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for region in masked:
        start = max(region.start, test_start)
        end = min(region.end, test_end)
        if end > start:
            excluded.append((start, end))

    ts_in_period = pd.DatetimeIndex(timestamps)
    ts_in_period = ts_in_period[(ts_in_period >= test_start) & (ts_in_period <= test_end)]
    if len(ts_in_period) >= 2:
        diffs = np.diff(ts_in_period.to_numpy())
        for i in np.flatnonzero(diffs > np.timedelta64(score_gap_threshold)):
            excluded.append((pd.Timestamp(ts_in_period[i]), pd.Timestamp(ts_in_period[i + 1])))

    excluded_seconds = sum(
        (end - start).total_seconds() for start, end in _merge_intervals(excluded)
    )
    total_seconds = (test_end - test_start).total_seconds()
    covered_seconds = max(total_seconds - excluded_seconds, 0.0)
    return covered_seconds / 86400.0


# --- Orchestration ----------------------------------------------------------


def evaluate_fold_at_threshold(
    fold,
    event,
    window_width_hours: float,
    threshold: float,
    test_data: ScoredTestData,
    settings,
) -> FoldEvaluation:
    """Evaluate one fold's ALREADY-SCORED test period at a single,
    pre-fitted threshold. Never fits the threshold itself -- see
    fit_threshold() / fit_threshold_sweep(), which take train scores only.
    """
    timestamps = np.asarray(pd.DatetimeIndex(test_data.timestamps))
    hold_time = pd.Timedelta(settings.evaluation.episode_hold_time)
    score_gap_threshold = pd.Timedelta(settings.evaluation.score_gap_threshold)
    min_duration = pd.Timedelta(settings.evaluation.min_episode_duration)

    episode_ranges = _group_episode_index_ranges(
        timestamps, test_data.scores, threshold, hold_time, score_gap_threshold, min_duration
    )
    masked = masked_regions(settings)
    episodes = _build_episodes(
        episode_ranges,
        timestamps,
        test_data.scores,
        test_data.contributions,
        test_data.channel_names,
        event,
        window_width_hours,
        masked,
        settings.evaluation.contribution_aggregation,
    )

    detected, lead_time = _detection_and_lead_time(episodes, event)
    concurrent_only = (not detected) and any(e.category == "concurrent" for e in episodes)
    false_episode_count = sum(1 for e in episodes if e.category == "false_alarm")
    evaluated_days = _evaluated_days(
        fold.test_start, fold.test_end, timestamps, masked, score_gap_threshold
    )
    false_alarms_per_day = false_episode_count / evaluated_days if evaluated_days > 0 else 0.0

    pre_start, pre_end = pre_failure_window(event, window_width_hours)
    coverage = window_coverage(pre_start, pre_end, timestamps, test_data.expected_interval)

    return FoldEvaluation(
        event_id=event.id,
        window_width=pd.Timedelta(hours=window_width_hours),
        threshold=threshold,
        detected=detected,
        lead_time=lead_time,
        concurrent_only=concurrent_only,
        episodes=episodes,
        false_episode_count=false_episode_count,
        evaluated_days=evaluated_days,
        false_alarms_per_day=false_alarms_per_day,
        window_coverage=coverage,
    )


def evaluate_fold(
    fold,
    event,
    window_width_hours: float,
    train_scores: np.ndarray,
    test_data: ScoredTestData,
    settings,
) -> FoldEvaluation:
    """Fit the threshold on train_scores ONLY (fit_threshold), then
    evaluate the already-scored test period at that single threshold.
    """
    threshold = fit_threshold(train_scores, settings)
    return evaluate_fold_at_threshold(
        fold, event, window_width_hours, threshold, test_data, settings
    )


def evaluate_fold_sweep(
    fold,
    event,
    window_width_hours: float,
    train_scores: np.ndarray,
    test_data: ScoredTestData,
    settings,
) -> dict[float, FoldEvaluation]:
    """evaluate_fold_at_threshold() for every quantile in
    evaluation.threshold_quantiles, each threshold fit on train_scores ONLY
    (fit_threshold_sweep). Returns the full curve keyed by quantile.

    Picking the test-optimal quantile out of this dict and reporting it as
    "the" result is FORBIDDEN -- see fit_threshold_sweep.
    """
    thresholds = fit_threshold_sweep(train_scores, settings)
    return {
        q: evaluate_fold_at_threshold(
            fold, event, window_width_hours, threshold, test_data, settings
        )
        for q, threshold in thresholds.items()
    }


# --- Pass 13, Part A: null (chance) comparison ------------------------------
#
# A detection's false-alarm rate alone cannot show skill: a fold firing
# 0.5-1/day, checked against a short pre-failure window, can "detect" by pure
# chance most of the time (docs/FINDINGS.md §13). Two independent estimates
# of what chance alone would give, reported alongside every detection.


@dataclass(frozen=True)
class ChanceComparison:
    p_chance_poisson: float
    p_chance_permutation: float
    # True only when result.detected AND at least one of the two estimates
    # exceeds evaluation.chance_threshold -- a fold with no detection has
    # nothing to flag, even if its own noise level is high.
    not_distinguishable_from_chance: bool


def p_chance_poisson(false_alarms_per_day: float, window_width_hours: float) -> float:
    """Poisson chance baseline: treating false alarms as a Poisson process
    at the fold's OWN measured rate, the probability of at least one
    episode landing in a window this wide by pure chance.

    expected_by_chance = false_alarms_per_day * window_width_days
    p_chance_poisson    = 1 - exp(-expected_by_chance)

    Assumes episodes arrive independently (no clustering) -- see
    p_chance_permutation for a more robust estimate when that assumption
    doesn't hold (docs/FINDINGS.md records real clustering, e.g. 8
    episodes in 9 days in early March).
    """
    expected_by_chance = false_alarms_per_day * (window_width_hours / 24.0)
    return 1.0 - float(np.exp(-expected_by_chance))


def p_chance_permutation(
    episodes: tuple[Episode, ...],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    window_width_hours: float,
    n_samples: int,
) -> float:
    """Empirical null: holding the model's ACTUAL episodes fixed, place a
    window of this width at n_samples candidate times spread evenly across
    [test_start, test_end], keep only placements whose full window
    [t - width, t) fits inside the test period, and report the fraction of
    those that overlap ANY of the given episodes (any category -- a
    randomly-placed failure would be "caught" by whatever the model raised
    there, independent of how that episode was categorised against the
    REAL documented event).

    This is more robust than p_chance_poisson because it accounts for
    episode CLUSTERING that a Poisson process assumes away: this asks "if
    the failure had occurred at a random time, how often would this model
    have caught it?" directly from the actual episode layout, rather than
    from a rate + independence assumption. Skill exists only when the real
    detection is unlikely under this null. If the two estimates diverge
    materially, that itself indicates clustering -- report both, never
    pick one (docs/FINDINGS.md §13).

    Deliberately a DETERMINISTIC, evenly-spaced grid over the test period,
    not a random sample: "spread across" the period is exactly what a grid
    guarantees, and it makes the result reproducible with no seed to
    manage. n_samples (evaluation.permutation_samples) still names how many
    candidate placements are evaluated.

    Raises:
        ValueError: if n_samples < 2, or if no candidate placement's window
            fits inside [test_start, test_end] at this width (the test
            period is shorter than window_width_hours).
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")

    width = pd.Timedelta(hours=window_width_hours)
    candidates = pd.date_range(test_start, test_end, periods=n_samples)
    valid = [t for t in candidates if (t - width) >= test_start]
    if not valid:
        raise ValueError(
            f"no candidate placement's window (width={width}) fits inside the "
            f"test period [{test_start}, {test_end}] -- widen the test period "
            "or narrow window_width_hours."
        )

    hits = sum(any(_overlaps(t - width, t, ep.start, ep.end) for ep in episodes) for t in valid)
    return hits / len(valid)


def evaluate_chance(result: FoldEvaluation, fold, settings) -> ChanceComparison:
    """Both null estimates for one FoldEvaluation, plus the chance flag.

    Always computed regardless of result.detected -- a fold's own noise
    level is informative even without a detection -- but
    not_distinguishable_from_chance can only be True when result.detected
    is also True (see ChanceComparison).

    `fold` supplies test_start/test_end (not carried by FoldEvaluation
    itself); `settings` must expose evaluation.permutation_samples and
    evaluation.chance_threshold.

    Raises:
        ValueError: from p_chance_permutation if the fold's test period is
            shorter than this result's window_width -- callers must not
            request a chance comparison for a width the test period
            cannot even fit once.
    """
    width_hours = result.window_width / pd.Timedelta(hours=1)
    poisson = p_chance_poisson(result.false_alarms_per_day, width_hours)
    permutation = p_chance_permutation(
        result.episodes,
        fold.test_start,
        fold.test_end,
        width_hours,
        settings.evaluation.permutation_samples,
    )
    threshold = settings.evaluation.chance_threshold
    flagged = bool(result.detected and (poisson > threshold or permutation > threshold))
    return ChanceComparison(
        p_chance_poisson=poisson,
        p_chance_permutation=permutation,
        not_distinguishable_from_chance=flagged,
    )


# --- Pass 13, Part B2: pooled normal-operation false-alarm rate -------------


@dataclass(frozen=True)
class PooledEvaluation:
    false_episode_count: int
    evaluated_days: float
    false_alarms_per_day: float


def evaluate_pooled_stretches(
    stretches: tuple[NormalStretch, ...],
    scored_stretches: list[ScoredTestData],
    threshold: float,
    settings,
) -> PooledEvaluation:
    """False-alarm rate pooled across every given normal-operation stretch,
    at a single pre-fitted threshold.

    Each stretch is grouped into episodes INDEPENDENTLY of every other --
    stretches are, by construction (events.pooled_normal_stretches), already
    mutually separated by excluded/masked time, so grouping never bridges
    across a stretch boundary. Every episode found is counted as a false
    alarm: a normal stretch has no documented event to check early_warning/
    concurrent/masked against by definition.

    `scored_stretches` must align 1:1 with `stretches`, already scored (by
    the model whose false-alarm behaviour is being characterised) by the
    caller -- this harness never scores anything itself.

    CAVEAT (docs/FINDINGS.md §8/§13): pooling mixes operating conditions
    from across the whole series (e.g. February vs. August) -- more
    statistical power, but not directly comparable to the in-fold rate.
    Report both, never merge them into one number.

    Raises:
        ValueError: if len(scored_stretches) != len(stretches).
    """
    if len(scored_stretches) != len(stretches):
        raise ValueError(
            f"scored_stretches ({len(scored_stretches)}) must align 1:1 with "
            f"stretches ({len(stretches)})"
        )

    hold_time = pd.Timedelta(settings.evaluation.episode_hold_time)
    score_gap_threshold = pd.Timedelta(settings.evaluation.score_gap_threshold)
    min_duration = pd.Timedelta(settings.evaluation.min_episode_duration)

    total_false = 0
    total_days = 0.0
    for stretch, test_data in zip(stretches, scored_stretches, strict=True):
        timestamps = np.asarray(pd.DatetimeIndex(test_data.timestamps))
        episode_ranges = _group_episode_index_ranges(
            timestamps, test_data.scores, threshold, hold_time, score_gap_threshold, min_duration
        )
        total_false += len(episode_ranges)
        total_days += _evaluated_days(
            stretch.start, stretch.end, timestamps, (), score_gap_threshold
        )

    rate = total_false / total_days if total_days > 0 else 0.0
    return PooledEvaluation(
        false_episode_count=total_false,
        evaluated_days=total_days,
        false_alarms_per_day=rate,
    )
