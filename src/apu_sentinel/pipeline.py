"""Ties data -> regimes -> features -> model -> evaluation -> explain
together, entirely config-driven.

Wires the rule-based baseline (run_pipeline, settings.model.rule_based) and,
as of pass 21, Isolation Forest (run_pipeline_isolation_forest,
settings.model.isolation_forest) -- the model progression in CLAUDE.md is
sequential (rule-based -> isolation forest -> autoencoder), and later models
add their own branch here as they are implemented, not all at once.

The rule-based model deliberately reads RAW (unscaled) channel values, not
the regime-conditional SCALED tensors data/scaling.py produces: its rules
are stated in physical units (e.g. peak ~9.64 bar, decay rate in bar/s),
and it already conditions on regime intrinsically (each rule is computed
per regime-labelled run, see features/cycles.py), satisfying CLAUDE.md
rule 4 without needing data/scaling.py's per-regime normalisation, which
exists for models that need cross-channel comparability (an autoencoder's
reconstruction error) instead. data/windows.py's windowing is likewise
skipped: the rule-based model scores per raw timestamp, not per sliding
window.

Scoring uses CONTINUOUS history from a fold's train_start through its
test_end -- not just the isolated test slice apply_fold() would produce --
because the rules' trailing-baseline calibration is causal and legitimately
uses test-period history that would be available in production (see
features/cycles.py's module docstring, and docs/FINDINGS.md §8). Only
FITTING (calibrating each rule's training-distribution quantiles) is
restricted to the fold's clean, exclusion-purged training slice. The
resulting scores are then trimmed back down to just the fold's test period
before being handed to the evaluation harness.

`data.subset` (configs/local.yaml's CPU/fast-iteration knob) is
DELIBERATELY not applied here: walk-forward folds need the full Feb-Aug
span to have any lead-in/test data at all for events 2-4, and no
"subset" semantics (time-based head-slice vs. random rows) has been
decided or implemented yet -- random subsampling would violate CLAUDE.md
rule 1 regardless. The rule-based model is cheap enough that running the
full local CSV is still fast; a future pass should either wire a
time-respecting subset or accept that this pipeline always uses the full
series locally.

Pass 13 (docs/FINDINGS.md §13): a detection's false-alarm rate alone cannot
show skill, and the pre-pass-13 harness's false-alarm denominator (a few
days per fold) couldn't support a rate estimate anyway. run_pipeline() now
reports, per fold:
  - the common sweep widths (evaluation.window_widths, shared across every
    fold -- the cross-model-comparable result), each with a null (chance)
    comparison (evaluation/metrics.py evaluate_chance) and false-alarm rate
    measured over an EXTENDED, contemporaneous test period (data/split.py
    extend_test_end_for_false_alarms -- Part B1);
  - a POOLED false-alarm rate measured across normal-operation stretches
    spanning the whole series (evaluation/events.py pooled_normal_stretches
    -- Part B2), reported ALONGSIDE the in-fold rate, never merged into it;
  - a SEPARATE sensitivity result at this event's own maximum feasible
    width (data/split.py event_max_width_hours -- Part C), with its own
    chance comparison.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from apu_sentinel.config import Settings
from apu_sentinel.data.load import load_raw
from apu_sentinel.data.scaling import fit_regime_scalers, transform_by_regime
from apu_sentinel.data.split import (
    Fold,
    apply_fold,
    event_max_width_hours,
    extend_test_end_for_false_alarms,
    make_folds,
)
from apu_sentinel.data.windows import characterise_sampling, make_windows
from apu_sentinel.evaluation.events import pooled_normal_stretches
from apu_sentinel.evaluation.metrics import (
    PooledEvaluation,
    ScoredTestData,
    evaluate_chance,
    evaluate_fold_at_threshold,
    evaluate_fold_sweep,
    evaluate_pooled_stretches,
    fit_threshold,
    fit_threshold_sweep,
)
from apu_sentinel.features.cycles import compute_cycle_features
from apu_sentinel.models.autoencoder import AutoencoderModel
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.isolation_forest import IsolationForestModel, WindowedInput
from apu_sentinel.models.rule_based import RuleBasedModel
from apu_sentinel.regimes import assign_regimes

logger = logging.getLogger(__name__)


def _rule_based_input_columns(settings: Settings) -> list[str]:
    """Raw channel columns the rule-based model needs, given its config.
    low_peak_pressure hardcodes Reservoirs (models/rule_based.py); the
    decay-rate family's channel is configurable (rules.fast_pressure_decay.
    source_channel, falling back to features.decay_source_channel).
    """
    columns = {"Reservoirs"}
    rule_based_cfg = settings.model.rule_based
    decay_channel = None
    if rule_based_cfg is not None:
        decay_cfg = rule_based_cfg.rules.get("fast_pressure_decay")
        if decay_cfg is not None and decay_cfg.source_channel is not None:
            decay_channel = decay_cfg.source_channel
    columns.add(decay_channel or settings.features.decay_source_channel)
    return sorted(columns)


def _model_input(df: pd.DataFrame, regimes: pd.Series, columns: list[str]) -> pd.DataFrame:
    model_df = df[columns].copy()
    model_df["regime"] = regimes.loc[df.index]
    return model_df


def _fit_fold_model(
    df: pd.DataFrame,
    regimes: pd.Series,
    fold: Fold,
    columns: list[str],
    settings: Settings,
) -> tuple[RuleBasedModel, pd.DataFrame, pd.DataFrame]:
    """Fit-on-train-only, then return (model, train_input, fold_input) --
    fold_input spans train_start..fold.test_end (continuous history for
    causal scoring, see module docstring); train_input is the clean,
    exclusion-purged training slice alone.
    """
    train_raw, _ = apply_fold(df, fold)
    fold_full = df.loc[(df.index >= fold.train_start) & (df.index <= fold.test_end)]

    train_input = _model_input(train_raw, regimes, columns)
    fold_input = _model_input(fold_full, regimes, columns)

    model = RuleBasedModel(settings)
    model.fit(train_input)
    return model, train_input, fold_input


def _evaluate_at_widths(
    fold: Fold,
    event,
    widths: list[float],
    model: AnomalyModel,
    train_input,
    fold_input,
    expected_interval: pd.Timedelta,
    settings: Settings,
) -> dict[float, dict[str, object]]:
    """evaluate_fold_at_threshold() + evaluate_chance() for every width in
    `widths`, sharing ONE fit + one scoring pass (fit_threshold and the
    score/contributions arrays don't depend on window_width_hours at all
    -- only post-hoc categorisation does).

    Model-agnostic (pass 21): `train_input`/`fold_input` need only expose
    `.index` (a DatetimeIndex) alongside whatever shape `model.score`/
    `model.contributions` themselves expect -- the rule-based model's plain
    per-timestamp DataFrame and IsolationForestModel's WindowedInput (whose
    `.index` aliases its end_timestamps) both satisfy this.
    """
    train_scores = model.score(train_input)
    full_scores = model.score(fold_input)
    full_contributions = model.contributions(fold_input)

    test_mask = (fold_input.index >= fold.test_start) & (fold_input.index <= fold.test_end)
    test_data = ScoredTestData(
        timestamps=fold_input.index[test_mask],
        scores=full_scores[test_mask],
        contributions=full_contributions[test_mask],
        channel_names=model.contributor_names,
        expected_interval=expected_interval,
    )

    threshold = fit_threshold(train_scores, settings)
    results: dict[float, dict[str, object]] = {}
    for width in widths:
        result = evaluate_fold_at_threshold(fold, event, width, threshold, test_data, settings)
        chance = evaluate_chance(result, fold, settings)
        results[width] = {"result": result, "chance": chance}
    return results


def _evaluate_pooled_for_fold(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: RuleBasedModel,
    columns: list[str],
    threshold: float,
    stretches,
    settings: Settings,
) -> PooledEvaluation:
    """Score every pooled normal stretch with this fold's ALREADY-FITTED
    model and threshold, then pool the false-alarm rate across all of them
    (Part B2). Each stretch is scored independently with its own
    continuous history (same causal-scoring rationale as fold_input).
    """
    scored_stretches = []
    for stretch in stretches:
        stretch_df = df.loc[(df.index >= stretch.start) & (df.index <= stretch.end)]
        if stretch_df.empty:
            scored_stretches.append(
                ScoredTestData(
                    timestamps=pd.DatetimeIndex([]),
                    scores=pd.Series(dtype=float).to_numpy(),
                    contributions=pd.Series(dtype=float).to_numpy().reshape(0, 0),
                    channel_names=model.contributor_names,
                    expected_interval=pd.Timedelta(seconds=1),
                )
            )
            continue
        stretch_input = _model_input(stretch_df, regimes, columns)
        scores = model.score(stretch_input)
        contributions = model.contributions(stretch_input)
        scored_stretches.append(
            ScoredTestData(
                timestamps=stretch_input.index,
                scores=scores,
                contributions=contributions,
                channel_names=model.contributor_names,
                expected_interval=pd.Timedelta(seconds=10),
            )
        )
    return evaluate_pooled_stretches(stretches, scored_stretches, threshold, settings)


def run_pipeline(settings: Settings) -> dict[int, dict[str, object]]:
    """Run the rule-based baseline across every walk-forward fold and
    report, per event (docs/FINDINGS.md §13):

    {event_id: {
        "common_widths": {width_hours: {"result": FoldEvaluation,
                                         "chance": ChanceComparison}},
        "extended_test_end": pd.Timestamp,
        "pooled": PooledEvaluation,
        "per_event_max": {"width_hours": float,
                           "result": FoldEvaluation,
                           "chance": ChanceComparison},
    }}

    "common_widths" uses the SAME shared-max fold boundaries as before
    pass 13 (cross-model-comparable), but with test_end EXTENDED (Part B1)
    so false-alarm counting has adequate normal-operation time; detection/
    lead_time are unaffected (categorise_episode never reads fold.test_end).
    "per_event_max" is a SEPARATE sensitivity fold set built at this
    event's own maximum feasible width (Part C) -- never used for
    "common_widths", and never silently presented as replacing the swept
    common widths.

    Raises:
        NotImplementedError: if settings.model.rule_based is not
            configured -- no other model type is wired yet (CLAUDE.md's
            model progression).
    """
    if settings.model.rule_based is None:
        raise NotImplementedError(
            "run_pipeline currently only wires the rule-based baseline "
            "(settings.model.rule_based) -- isolation forest / autoencoder "
            "are later passes in CLAUDE.md's model progression."
        )

    raw_path = Path(settings.data.raw_dir) / settings.data.raw_filename
    df = load_raw(raw_path)
    data_start, data_end = df.index.min(), df.index.max()

    regimes = assign_regimes(df, settings)
    common_folds = make_folds(settings, data_start, data_end)
    max_widths = event_max_width_hours(settings, data_start)
    max_width_folds = make_folds(settings, data_start, data_end, width_hours_by_event=max_widths)

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    columns = _rule_based_input_columns(settings)
    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    events_by_id = {event.id: event for event in events_sorted}
    training_exclusion = settings.split.training_exclusion

    stretches = pooled_normal_stretches(settings, data_start, data_end)

    max_width_folds_by_event = {fold.event_id: fold for fold in max_width_folds}

    results: dict[int, dict[str, object]] = {}
    for fold in common_folds:
        event = events_by_id[fold.event_id]

        extended_fold = extend_test_end_for_false_alarms(
            fold, event, events_sorted, training_exclusion, data_end
        )
        model, train_input, fold_input = _fit_fold_model(
            df, regimes, extended_fold, columns, settings
        )
        common_widths = _evaluate_at_widths(
            extended_fold,
            event,
            settings.evaluation.window_widths,
            model,
            train_input,
            fold_input,
            expected_interval,
            settings,
        )

        threshold = fit_threshold(model.score(train_input), settings)
        pooled = _evaluate_pooled_for_fold(
            df, regimes, model, columns, threshold, stretches, settings
        )

        max_fold = max_width_folds_by_event[fold.event_id]
        max_model, max_train_input, max_fold_input = _fit_fold_model(
            df, regimes, max_fold, columns, settings
        )
        max_width = max_widths[fold.event_id]
        per_event_max = _evaluate_at_widths(
            max_fold,
            event,
            [max_width],
            max_model,
            max_train_input,
            max_fold_input,
            expected_interval,
            settings,
        )[max_width]
        per_event_max["width_hours"] = max_width

        results[fold.event_id] = {
            "common_widths": common_widths,
            "extended_test_end": extended_fold.test_end,
            "pooled": pooled,
            "per_event_max": per_event_max,
        }

    return results


# --- Pass 21: Isolation Forest (windowed) ------------------------------------
#
# Unlike the rule-based baseline, this model needs the FULL pipeline sequence
# (docs/ARCHITECTURE.md): scale (per-fold, per-regime, train-fitted only) ->
# compute cycle features (causal, on the RAW channel) -> window. The helpers
# below mirror _fit_fold_model/_evaluate_pooled_for_fold's shapes above, but
# build a models.isolation_forest.WindowedInput instead of a rule-based
# per-timestamp DataFrame. _evaluate_at_widths is REUSED UNCHANGED for both
# models -- it only ever calls model.score/contributions and reads
# `fold_input.index`, and WindowedInput.index aliases end_timestamps for
# exactly this reason.


def _windowed_channel_names(settings: Settings) -> tuple[str, ...]:
    """make_windows()'s own documented channel order -- analog_columns then
    passthrough_columns.
    """
    return tuple(settings.scaling.analog_columns) + tuple(settings.scaling.passthrough_columns)


# --- Pass 22: gap-adjacency (docs/RESULTS.md, event-4 validation) ----------
#
# make_windows() already drops a window whose OWN span crosses a gap; it
# does NOT flag a window that sits just before/after one -- exactly where a
# cycle feature can still be NaN (a STOPPED run gap-truncated at its exit,
# module docstring in features/cycles.py) and get imputed to the training
# median (IsolationForestModel._fill_nan). This section measures and,
# opt-in only, removes that vicinity from SCORING (never training).


def _gap_boundaries(
    index: pd.DatetimeIndex, gap_threshold: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(gap_start, gap_end) for every consecutive pair in `index` spaced
    more than gap_threshold apart -- the same definition
    data/windows.py characterise_sampling() uses, recomputed directly over
    whatever slice is at hand (a fold's own train_start..test_end span).
    """
    values = index.to_numpy()
    if len(values) < 2:
        return []
    deltas = np.diff(values)
    gap_positions = np.flatnonzero(deltas > np.timedelta64(gap_threshold))
    return [(pd.Timestamp(values[i]), pd.Timestamp(values[i + 1])) for i in gap_positions]


def gap_adjacent_mask(
    end_timestamps,
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]],
    window_duration: pd.Timedelta,
) -> np.ndarray:
    """True where an end_timestamp falls within one window_duration of
    EITHER boundary of any gap -- "just after a gap ended" or "about to hit
    one", the two ways a window's history can be gap-disrupted without its
    own span crossing the gap (that case make_windows() already drops).
    """
    end_timestamps = pd.DatetimeIndex(end_timestamps)
    mask = np.zeros(len(end_timestamps), dtype=bool)
    for gap_start, gap_end in gaps:
        near_start = (end_timestamps >= gap_start - window_duration) & (
            end_timestamps <= gap_start + window_duration
        )
        near_end = (end_timestamps >= gap_end - window_duration) & (
            end_timestamps <= gap_end + window_duration
        )
        mask |= near_start | near_end
    return mask


def _build_windowed_input(
    raw_df: pd.DataFrame,
    regimes: pd.Series,
    scalers: dict,
    settings: Settings,
    stride_mode: str,
    exclude_gap_adjacent: bool = False,
) -> WindowedInput:
    """Scale raw_df with an ALREADY-FITTED (train-only) scaler set, compute
    cycle features on the RAW channel (causal, same discipline as the
    rule-based model), then window -- the bundle IsolationForestModel reads.
    `raw_df`/`regimes` need not share an index (regimes is sliced onto
    raw_df's here); scalers must come from fit_regime_scalers() on this
    fold's clean training slice.

    `exclude_gap_adjacent` (pass 22 diagnostic, model.isolation_forest.
    exclude_gap_adjacent_windows) drops windows via gap_adjacent_mask()
    after windowing -- callers must only set this for a SCORED input, never
    for training data (see _fit_fold_isolation_forest).
    """
    df_regimes = regimes.loc[raw_df.index]
    scaled = transform_by_regime(raw_df, df_regimes, scalers, settings)
    windows, end_timestamps = make_windows(scaled, settings, stride_mode=stride_mode)

    if exclude_gap_adjacent and len(end_timestamps) > 0:
        gap_threshold = pd.Timedelta(settings.windowing.gap_threshold)
        window_duration = pd.Timedelta(settings.windowing.window_duration)
        gaps = _gap_boundaries(scaled.index, gap_threshold)
        keep = ~gap_adjacent_mask(end_timestamps, gaps, window_duration)
        n_dropped = int((~keep).sum())
        logger.info(
            "_build_windowed_input: exclude_gap_adjacent_windows dropped %d/%d windows",
            n_dropped,
            len(end_timestamps),
        )
        windows, end_timestamps = windows[keep], end_timestamps[keep]

    cycle_features = None
    if settings.model.isolation_forest.include_cycle_features:
        decay_channel = settings.features.decay_source_channel
        cycle_features = compute_cycle_features(raw_df[[decay_channel]], df_regimes, settings)

    return WindowedInput(
        windows=windows,
        end_timestamps=pd.DatetimeIndex(end_timestamps),
        channel_names=_windowed_channel_names(settings),
        cycle_features=cycle_features,
    )


def _fit_fold_isolation_forest(
    df: pd.DataFrame,
    regimes: pd.Series,
    fold: Fold,
    settings: Settings,
) -> tuple[IsolationForestModel, dict, WindowedInput, WindowedInput]:
    """Fit-on-train-only scalers AND model, then return (model, scalers,
    train_input, fold_input) -- fold_input spans train_start..fold.test_end
    (continuous history for causal scoring, same rationale as the
    rule-based model's fold_input); train_input is the clean,
    exclusion-purged training slice alone. `scalers` is returned too since
    _evaluate_pooled_for_fold_windowed needs the SAME fitted scalers to
    score pooled stretches with this fold's model.

    `model.isolation_forest.exclude_gap_adjacent_windows` (pass 22
    diagnostic) applies ONLY to fold_input (what gets SCORED) -- train_input
    always sees every window, gap-adjacent or not, since the question is
    whether gap-adjacent windows are being flagged as anomalies, not
    whether the model should learn from them.
    """
    train_raw, _ = apply_fold(df, fold)
    fold_full = df.loc[(df.index >= fold.train_start) & (df.index <= fold.test_end)]

    train_regimes = regimes.loc[train_raw.index]
    scalers = fit_regime_scalers(train_raw, train_regimes, settings, fold_id=fold.event_id)

    train_input = _build_windowed_input(train_raw, regimes, scalers, settings, stride_mode="train")
    fold_input = _build_windowed_input(
        fold_full,
        regimes,
        scalers,
        settings,
        stride_mode="score",
        exclude_gap_adjacent=settings.model.isolation_forest.exclude_gap_adjacent_windows,
    )

    model = IsolationForestModel(settings)
    model.fit(train_input)
    return model, scalers, train_input, fold_input


def _score_pooled_stretches_windowed(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: IsolationForestModel,
    scalers: dict,
    stretches,
    settings: Settings,
) -> list[ScoredTestData]:
    """Score every pooled normal stretch with this fold's ALREADY-FITTED
    model and scalers -- scoring only, no threshold applied yet, so the
    result can be evaluated at several thresholds (pooled_at_quantiles)
    without re-scoring. A stretch shorter than one window (or empty) yields
    zero windows -- make_windows() already returns empty tensors for that
    case, never raises, EXCEPT for a genuinely empty DataFrame, guarded
    here the same way the rule-based version guards it.

    Contributions are NEVER computed here: evaluate_pooled_stretches() only
    ever reads test_data.scores/timestamps (episode counting, no ranked
    diagnosis is attached to a pooled stretch) -- a zero placeholder of the
    right shape satisfies ScoredTestData without paying ablation's
    O(n_features) re-scoring cost across every pooled stretch, which for
    this model (unlike the rule-based baseline's cheap percentile lookup)
    would dominate runtime for no observable effect on the result.
    """
    scored_stretches = []
    n_contributors = len(model.contributor_names)
    for stretch in stretches:
        stretch_df = df.loc[(df.index >= stretch.start) & (df.index <= stretch.end)]
        if stretch_df.empty:
            scored_stretches.append(
                ScoredTestData(
                    timestamps=pd.DatetimeIndex([]),
                    scores=np.empty(0),
                    contributions=np.empty((0, n_contributors)),
                    channel_names=model.contributor_names,
                    expected_interval=pd.Timedelta(seconds=1),
                )
            )
            continue
        stretch_input = _build_windowed_input(
            stretch_df, regimes, scalers, settings, stride_mode="score"
        )
        scores = model.score(stretch_input) if stretch_input.windows.shape[0] else np.empty(0)
        scored_stretches.append(
            ScoredTestData(
                timestamps=stretch_input.end_timestamps,
                scores=scores,
                contributions=np.zeros((len(scores), n_contributors)),
                channel_names=model.contributor_names,
                expected_interval=pd.Timedelta(seconds=10),
            )
        )
    return scored_stretches


def _evaluate_pooled_for_fold_windowed(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: IsolationForestModel,
    scalers: dict,
    threshold: float,
    stretches,
    settings: Settings,
) -> PooledEvaluation:
    """Windowed equivalent of _evaluate_pooled_for_fold: pools the
    false-alarm rate across every pooled normal stretch (Part B2), at a
    single pre-fitted threshold.
    """
    scored_stretches = _score_pooled_stretches_windowed(
        df, regimes, model, scalers, stretches, settings
    )
    return evaluate_pooled_stretches(stretches, scored_stretches, threshold, settings)


def pooled_at_quantiles(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: IsolationForestModel,
    scalers: dict,
    train_scores: np.ndarray,
    stretches,
    settings: Settings,
) -> dict[float, PooledEvaluation]:
    """Pass 22 (docs/RESULTS.md, event-4 validation Part B): pooled
    false-alarm rate at EVERY quantile in evaluation.threshold_quantiles,
    from ONE scoring pass over the pooled stretches -- needed to pick "the
    tightest quantile whose pooled rate stays under a ceiling" without
    re-scoring per quantile. Thresholds come from
    fit_threshold_sweep(train_scores, settings) -- train-scores-only, same
    discipline as everywhere else.
    """
    scored_stretches = _score_pooled_stretches_windowed(
        df, regimes, model, scalers, stretches, settings
    )
    thresholds = fit_threshold_sweep(train_scores, settings)
    return {
        q: evaluate_pooled_stretches(stretches, scored_stretches, threshold, settings)
        for q, threshold in thresholds.items()
    }


def run_pipeline_isolation_forest(settings: Settings) -> dict[int, dict[str, object]]:
    """Isolation Forest across every walk-forward fold, common widths only
    (no per-event-max sensitivity set -- not needed for this model's own
    evaluation, unlike the rule-based baseline's pass-13 Part C). Mirrors
    run_pipeline()'s per-fold reporting shape: {event_id: {"common_widths":
    ..., "extended_test_end": ..., "pooled": ...}}.

    Raises:
        NotImplementedError: if settings.model.isolation_forest is not
            configured.
    """
    if settings.model.isolation_forest is None:
        raise NotImplementedError(
            "run_pipeline_isolation_forest requires settings.model.isolation_forest "
            "to be configured."
        )

    raw_path = Path(settings.data.raw_dir) / settings.data.raw_filename
    df = load_raw(raw_path)
    data_start, data_end = df.index.min(), df.index.max()

    regimes = assign_regimes(df, settings)
    common_folds = make_folds(settings, data_start, data_end)

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    events_by_id = {event.id: event for event in events_sorted}
    training_exclusion = settings.split.training_exclusion

    stretches = pooled_normal_stretches(settings, data_start, data_end)

    results: dict[int, dict[str, object]] = {}
    for fold in common_folds:
        event = events_by_id[fold.event_id]

        extended_fold = extend_test_end_for_false_alarms(
            fold, event, events_sorted, training_exclusion, data_end
        )
        model, scalers, train_input, fold_input = _fit_fold_isolation_forest(
            df, regimes, extended_fold, settings
        )
        common_widths = _evaluate_at_widths(
            extended_fold,
            event,
            settings.evaluation.window_widths,
            model,
            train_input,
            fold_input,
            expected_interval,
            settings,
        )

        threshold = fit_threshold(model.score(train_input), settings)
        pooled = _evaluate_pooled_for_fold_windowed(
            df, regimes, model, scalers, threshold, stretches, settings
        )

        results[fold.event_id] = {
            "common_widths": common_widths,
            "extended_test_end": extended_fold.test_end,
            "pooled": pooled,
        }

    return results


# --- Pass 21: Isolation Forest quantile sweep, checkpointed per fold -------
#
# scripts/isolation_forest_experiment.py is a thin CLI/checkpoint wrapper
# around evaluate_isolation_forest_fold() below: fit once per fold, then
# sweep evaluation.threshold_quantiles at every evaluation.window_widths
# entry from that SAME score array (fit_threshold_sweep/evaluate_fold_sweep
# already do this cheaply -- no re-fitting per quantile). Flagged
# detections (chance.p_chance_permutation < evaluation.chance_threshold)
# get IsolationForestModel.explain_episode() called immediately, while the
# fitted model and fold_input are still in scope -- the checkpoint written
# by the script holds only the (small) evaluation results, never the model
# or the windows tensor, so per-fold checkpoints stay cheap to persist.


def _evaluate_at_widths_and_quantiles(
    fold: Fold,
    event,
    widths: list[float],
    model: AnomalyModel,
    train_input,
    fold_input,
    expected_interval: pd.Timedelta,
    settings: Settings,
) -> dict[float, dict[float, dict[str, object]]]:
    """Like _evaluate_at_widths, but sweeps evaluation.threshold_quantiles
    (evaluate_fold_sweep) at every width too, from ONE shared fit/score
    pass. Returns {width: {quantile: {"result":, "chance":}}}.
    """
    train_scores = model.score(train_input)
    full_scores = model.score(fold_input)
    full_contributions = model.contributions(fold_input)

    test_mask = (fold_input.index >= fold.test_start) & (fold_input.index <= fold.test_end)
    test_data = ScoredTestData(
        timestamps=fold_input.index[test_mask],
        scores=full_scores[test_mask],
        contributions=full_contributions[test_mask],
        channel_names=model.contributor_names,
        expected_interval=expected_interval,
    )

    results: dict[float, dict[float, dict[str, object]]] = {}
    for width in widths:
        per_quantile = evaluate_fold_sweep(fold, event, width, train_scores, test_data, settings)
        results[width] = {
            q: {"result": result, "chance": evaluate_chance(result, fold, settings)}
            for q, result in per_quantile.items()
        }
    return results


def evaluate_isolation_forest_fold(
    df: pd.DataFrame,
    regimes: pd.Series,
    fold: Fold,
    event,
    events_sorted,
    training_exclusion,
    data_end: pd.Timestamp,
    stretches,
    expected_interval: pd.Timedelta,
    settings: Settings,
) -> dict[str, object]:
    """One fold's full Isolation Forest evaluation: fit once, sweep
    (width, quantile) from that one fit, explain flagged detections
    immediately (while the model is in scope), then the pooled false-alarm
    rate. Never returns the fitted model or fold_input -- the caller
    (scripts/isolation_forest_experiment.py) checkpoints this return value
    directly, and both of those are too large/heavy to persist per fold.

    "Flagged" = detected AND p_chance_permutation < evaluation.chance_threshold
    (the same threshold ChanceComparison.not_distinguishable_from_chance
    itself uses) -- explained per (quantile, episode start, episode end),
    deduped across widths since episode boundaries depend only on the
    quantile's threshold, not on window_width_hours (categorise_episode is
    the only width-dependent step).
    """
    extended_fold = extend_test_end_for_false_alarms(
        fold, event, events_sorted, training_exclusion, data_end
    )
    model, scalers, train_input, fold_input = _fit_fold_isolation_forest(
        df, regimes, extended_fold, settings
    )

    common_widths = _evaluate_at_widths_and_quantiles(
        extended_fold,
        event,
        settings.evaluation.window_widths,
        model,
        train_input,
        fold_input,
        expected_interval,
        settings,
    )

    chance_threshold = settings.evaluation.chance_threshold
    explain_cache: dict[tuple, tuple[tuple[str, float], ...]] = {}
    for width_results in common_widths.values():
        for quantile, entry in width_results.items():
            result, chance = entry["result"], entry["chance"]
            if not (result.detected and chance.p_chance_permutation < chance_threshold):
                continue
            explained: dict[tuple[pd.Timestamp, pd.Timestamp], tuple] = {}
            for ep in result.episodes:
                if ep.category not in ("early_warning", "concurrent"):
                    continue
                cache_key = (quantile, ep.start, ep.end)
                if cache_key not in explain_cache:
                    explain_cache[cache_key] = model.explain_episode(ep, fold_input)
                explained[(ep.start, ep.end)] = explain_cache[cache_key]
            entry["explained"] = explained

    train_scores = model.score(train_input)
    threshold = fit_threshold(train_scores, settings)
    pooled = _evaluate_pooled_for_fold_windowed(
        df, regimes, model, scalers, threshold, stretches, settings
    )
    pooled_by_quantile = pooled_at_quantiles(
        df, regimes, model, scalers, train_scores, stretches, settings
    )

    return {
        "event_id": fold.event_id,
        "common_widths": common_widths,
        "extended_test_end": extended_fold.test_end,
        "pooled": pooled,
        "pooled_by_quantile": pooled_by_quantile,
    }


# --- Pass 23: LSTM Autoencoder (windowed, no cycle features) ---------------
#
# Mirrors _fit_fold_isolation_forest/_build_windowed_input's shape, but never
# computes cycle features (models/autoencoder.py's module docstring: this
# model reads the 15 scaled channels only, sidestepping the NaN-imputation
# path in docs/RESULTS.md §22 Part A1) and never touches the isolation
# forest's own functions -- a parallel section, not a refactor of it.
#
# Evaluation deliberately does NOT sweep (width x quantile) the way
# evaluate_isolation_forest_fold does (docs/RESULTS.md §22's own lesson: a
# 160-cell sweep produces a maximum, not a p-value). Instead, run_pipeline_
# autoencoder selects ONE pre-registered operating point per the pass-22
# Part B rule -- width fixed at 72h, quantile the LOOSEST of
# evaluation.threshold_quantiles whose POOLED false-alarm rate, taken as the
# worst case (max) across every fold, is <= POOLED_FALSE_ALARM_CEILING --
# and evaluates every fold at that single point only.

OPERATING_POINT_WIDTH_HOURS = 72.0
# Reuses docs/RESULTS.md §22 Part B's selection ceiling verbatim -- not
# re-derived here, so this model's operating point is chosen by the same
# rule already applied to Isolation Forest, not a new one picked to flatter
# this model.
POOLED_FALSE_ALARM_CEILING = 0.3


def _build_windowed_input_autoencoder(
    raw_df: pd.DataFrame,
    regimes: pd.Series,
    scalers: dict,
    settings: Settings,
    stride_mode: str,
) -> WindowedInput:
    """Scale + window raw_df with an ALREADY-FITTED (train-only) scaler set
    -- no cycle features, ever (cycle_features=None): unlike
    _build_windowed_input (models/isolation_forest.py's caller), this
    builder has no include_cycle_features branch to take.
    """
    df_regimes = regimes.loc[raw_df.index]
    scaled = transform_by_regime(raw_df, df_regimes, scalers, settings)
    windows, end_timestamps = make_windows(scaled, settings, stride_mode=stride_mode)
    return WindowedInput(
        windows=windows,
        end_timestamps=pd.DatetimeIndex(end_timestamps),
        channel_names=_windowed_channel_names(settings),
        cycle_features=None,
    )


def _fit_fold_autoencoder(
    df: pd.DataFrame,
    regimes: pd.Series,
    fold: Fold,
    settings: Settings,
) -> tuple[AutoencoderModel, dict, WindowedInput, WindowedInput]:
    """Fit-on-train-only scalers AND model, then return (model, scalers,
    train_input, fold_input) -- same shape as _fit_fold_isolation_forest.
    """
    train_raw, _ = apply_fold(df, fold)
    fold_full = df.loc[(df.index >= fold.train_start) & (df.index <= fold.test_end)]

    train_regimes = regimes.loc[train_raw.index]
    scalers = fit_regime_scalers(train_raw, train_regimes, settings, fold_id=fold.event_id)

    train_input = _build_windowed_input_autoencoder(
        train_raw, regimes, scalers, settings, stride_mode="train"
    )
    fold_input = _build_windowed_input_autoencoder(
        fold_full, regimes, scalers, settings, stride_mode="score"
    )

    model = AutoencoderModel(settings)
    model.fit(train_input)
    return model, scalers, train_input, fold_input


def _score_pooled_stretches_autoencoder(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: AutoencoderModel,
    scalers: dict,
    stretches,
    settings: Settings,
) -> list[ScoredTestData]:
    """Windowed equivalent of _score_pooled_stretches_windowed, without
    cycle features. Contributions are never computed here for the same
    reason as the isolation forest version: evaluate_pooled_stretches()
    only ever reads test_data.scores/timestamps.
    """
    scored_stretches = []
    n_contributors = len(model.contributor_names)
    for stretch in stretches:
        stretch_df = df.loc[(df.index >= stretch.start) & (df.index <= stretch.end)]
        if stretch_df.empty:
            scored_stretches.append(
                ScoredTestData(
                    timestamps=pd.DatetimeIndex([]),
                    scores=np.empty(0),
                    contributions=np.empty((0, n_contributors)),
                    channel_names=model.contributor_names,
                    expected_interval=pd.Timedelta(seconds=1),
                )
            )
            continue
        stretch_input = _build_windowed_input_autoencoder(
            stretch_df, regimes, scalers, settings, stride_mode="score"
        )
        scores = model.score(stretch_input) if stretch_input.windows.shape[0] else np.empty(0)
        scored_stretches.append(
            ScoredTestData(
                timestamps=stretch_input.end_timestamps,
                scores=scores,
                contributions=np.zeros((len(scores), n_contributors)),
                channel_names=model.contributor_names,
                expected_interval=pd.Timedelta(seconds=10),
            )
        )
    return scored_stretches


def pooled_at_quantiles_autoencoder(
    df: pd.DataFrame,
    regimes: pd.Series,
    model: AutoencoderModel,
    scalers: dict,
    train_scores: np.ndarray,
    stretches,
    settings: Settings,
) -> dict[float, PooledEvaluation]:
    """pooled_at_quantiles() (models/isolation_forest.py's pipeline
    counterpart), without cycle features -- one scoring pass over the
    pooled stretches, evaluated at every evaluation.threshold_quantiles
    entry, needed by select_operating_quantile() below.
    """
    scored_stretches = _score_pooled_stretches_autoencoder(
        df, regimes, model, scalers, stretches, settings
    )
    thresholds = fit_threshold_sweep(train_scores, settings)
    return {
        q: evaluate_pooled_stretches(stretches, scored_stretches, threshold, settings)
        for q, threshold in thresholds.items()
    }


def select_operating_quantile(
    pooled_by_quantile_per_fold: dict[int, dict[float, PooledEvaluation]],
    ceiling: float,
) -> float:
    """docs/RESULTS.md §22 Part B's selection rule, reused verbatim: the
    LOOSEST swept quantile whose pooled false-alarm rate, taken as the
    WORST CASE (max) across every fold, is <= ceiling -- going tighter than
    necessary sacrifices sensitivity for no operational benefit. Selection
    depends only on false-alarm behaviour, never on detection outcomes.

    Raises:
        ValueError: if no swept quantile satisfies the ceiling in every
            fold.
    """
    quantiles = sorted(next(iter(pooled_by_quantile_per_fold.values())).keys())
    for q in quantiles:
        worst = max(
            pooled_by_quantile_per_fold[fold_id][q].false_alarms_per_day
            for fold_id in pooled_by_quantile_per_fold
        )
        if worst <= ceiling:
            return q
    raise ValueError(
        f"no swept quantile (evaluation.threshold_quantiles={quantiles}) keeps the "
        f"worst-case pooled false-alarm rate across all folds under the ceiling "
        f"({ceiling}/day) -- widen the quantile grid or raise the ceiling deliberately."
    )


def run_pipeline_autoencoder(settings: Settings) -> dict[str, object]:
    """LSTM Autoencoder across every walk-forward fold (arm A only -- March
    included, common folds, no additional_regions), evaluated at ONE
    pre-registered operating point (docs/RESULTS.md §22 Part B's rule,
    reused verbatim -- see select_operating_quantile). Never sweeps
    (width x quantile): docs/RESULTS.md §22's own lesson is that a wide
    sweep reports a maximum, not a p-value.

    Returns {"operating_point": {"width_hours", "quantile"},
             "folds": {event_id: {"result": FoldEvaluation,
                                   "chance": ChanceComparison,
                                   "pooled": PooledEvaluation,
                                   "training_summary": {...}}}}.

    Raises:
        NotImplementedError: if settings.model.autoencoder is not
            configured.
        ValueError: from select_operating_quantile if no swept quantile
            satisfies the pooled false-alarm ceiling in every fold.
    """
    if settings.model.autoencoder is None:
        raise NotImplementedError(
            "run_pipeline_autoencoder requires settings.model.autoencoder to be configured."
        )

    raw_path = Path(settings.data.raw_dir) / settings.data.raw_filename
    df = load_raw(raw_path)
    data_start, data_end = df.index.min(), df.index.max()

    regimes = assign_regimes(df, settings)
    common_folds = make_folds(settings, data_start, data_end)

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    events_by_id = {event.id: event for event in events_sorted}
    training_exclusion = settings.split.training_exclusion

    stretches = pooled_normal_stretches(settings, data_start, data_end)

    per_fold: dict[int, dict[str, object]] = {}
    for fold in common_folds:
        event = events_by_id[fold.event_id]
        extended_fold = extend_test_end_for_false_alarms(
            fold, event, events_sorted, training_exclusion, data_end
        )

        start = time.monotonic()
        model, scalers, train_input, fold_input = _fit_fold_autoencoder(
            df, regimes, extended_fold, settings
        )
        logger.info(
            "run_pipeline_autoencoder: fold %d fit in %.1fs (epochs_run=%d "
            "final_train_loss=%.6f final_val_loss=%s device=%s)",
            fold.event_id,
            time.monotonic() - start,
            model.epochs_run_,
            model.final_train_loss_,
            model.final_val_loss_,
            model.device_,
        )

        train_scores = model.score(train_input)
        pooled_by_quantile = pooled_at_quantiles_autoencoder(
            df, regimes, model, scalers, train_scores, stretches, settings
        )

        per_fold[fold.event_id] = {
            "fold": extended_fold,
            "event": event,
            "model": model,
            "fold_input": fold_input,
            "train_scores": train_scores,
            "pooled_by_quantile": pooled_by_quantile,
        }

    chosen_quantile = select_operating_quantile(
        {fold_id: entry["pooled_by_quantile"] for fold_id, entry in per_fold.items()},
        POOLED_FALSE_ALARM_CEILING,
    )

    results: dict[int, dict[str, object]] = {}
    for fold_id, entry in per_fold.items():
        fold, event, model = entry["fold"], entry["event"], entry["model"]
        fold_input = entry["fold_input"]

        threshold = fit_threshold_sweep(entry["train_scores"], settings)[chosen_quantile]
        full_scores = model.score(fold_input)
        full_contributions = model.contributions(fold_input)

        test_mask = (fold_input.index >= fold.test_start) & (fold_input.index <= fold.test_end)
        test_data = ScoredTestData(
            timestamps=fold_input.index[test_mask],
            scores=full_scores[test_mask],
            contributions=full_contributions[test_mask],
            channel_names=model.contributor_names,
            expected_interval=expected_interval,
        )

        result = evaluate_fold_at_threshold(
            fold, event, OPERATING_POINT_WIDTH_HOURS, threshold, test_data, settings
        )
        chance = evaluate_chance(result, fold, settings)

        results[fold_id] = {
            "result": result,
            "chance": chance,
            "pooled": entry["pooled_by_quantile"][chosen_quantile],
            "training_summary": {
                "epochs_run": model.epochs_run_,
                "final_train_loss": model.final_train_loss_,
                "final_val_loss": model.final_val_loss_,
                "elapsed_seconds": model.elapsed_seconds_,
                "device": model.device_,
            },
        }

    return {
        "operating_point": {
            "width_hours": OPERATING_POINT_WIDTH_HOURS,
            "quantile": chosen_quantile,
        },
        "folds": results,
    }
