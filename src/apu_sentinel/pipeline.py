"""Ties data -> regimes -> features -> model -> evaluation -> explain
together, entirely config-driven.

Currently wires only the rule-based baseline (settings.model.rule_based) --
the model progression in CLAUDE.md is sequential (rule-based -> isolation
forest -> autoencoder), and later models add their own branch here as they
are implemented, not all at once.

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

from pathlib import Path

import pandas as pd

from apu_sentinel.config import Settings
from apu_sentinel.data.load import load_raw
from apu_sentinel.data.split import (
    Fold,
    apply_fold,
    event_max_width_hours,
    extend_test_end_for_false_alarms,
    make_folds,
)
from apu_sentinel.data.windows import characterise_sampling
from apu_sentinel.evaluation.events import pooled_normal_stretches
from apu_sentinel.evaluation.metrics import (
    PooledEvaluation,
    ScoredTestData,
    evaluate_chance,
    evaluate_fold_at_threshold,
    evaluate_pooled_stretches,
    fit_threshold,
)
from apu_sentinel.models.rule_based import RuleBasedModel
from apu_sentinel.regimes import assign_regimes


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
    model: RuleBasedModel,
    train_input: pd.DataFrame,
    fold_input: pd.DataFrame,
    expected_interval: pd.Timedelta,
    settings: Settings,
) -> dict[float, dict[str, object]]:
    """evaluate_fold_at_threshold() + evaluate_chance() for every width in
    `widths`, sharing ONE fit + one scoring pass (fit_threshold and the
    score/contributions arrays don't depend on window_width_hours at all
    -- only post-hoc categorisation does).
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
