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
full local CSV is still fast (~70s end-to-end for all four folds); a
future pass should either wire a time-respecting subset or accept that
this pipeline always uses the full series locally.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from apu_sentinel.config import Settings
from apu_sentinel.data.load import load_raw
from apu_sentinel.data.split import Fold, apply_fold, make_folds
from apu_sentinel.data.windows import characterise_sampling
from apu_sentinel.evaluation.metrics import (
    FoldEvaluation,
    ScoredTestData,
    evaluate_fold_at_threshold,
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


def _evaluate_rule_based_fold(
    df: pd.DataFrame,
    regimes: pd.Series,
    fold: Fold,
    event,
    columns: list[str],
    expected_interval: pd.Timedelta,
    settings: Settings,
) -> dict[float, FoldEvaluation]:
    train_raw, _ = apply_fold(df, fold)
    fold_full = df.loc[(df.index >= fold.train_start) & (df.index <= fold.test_end)]

    train_input = _model_input(train_raw, regimes, columns)
    fold_input = _model_input(fold_full, regimes, columns)

    model = RuleBasedModel(settings)
    model.fit(train_input)

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
    return {
        width: evaluate_fold_at_threshold(fold, event, width, threshold, test_data, settings)
        for width in settings.evaluation.window_widths
    }


def run_pipeline(settings: Settings) -> dict[int, dict[float, FoldEvaluation]]:
    """Run the rule-based baseline across every walk-forward fold and
    every configured pre-failure window width, and return episode-level
    evaluation results keyed by {event_id: {window_width_hours:
    FoldEvaluation}}.

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

    regimes = assign_regimes(df, settings)
    folds = make_folds(settings, df.index.min(), df.index.max())

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    columns = _rule_based_input_columns(settings)
    events_by_id = {event.id: event for event in settings.evaluation.failure_events}

    results: dict[int, dict[float, FoldEvaluation]] = {}
    for fold in folds:
        event = events_by_id[fold.event_id]
        results[fold.event_id] = _evaluate_rule_based_fold(
            df, regimes, fold, event, columns, expected_interval, settings
        )
    return results
