#!/usr/bin/env python3
"""Pass 22 Part A2: gap-adjacency of event 4's detecting episode(s).

Re-fits fold 4 (deterministic given fixed random_state -- reproduces the
same episodes scripts/isolation_forest_experiment.py's checkpoint holds),
then reports, for each flagged detection (p_chance_permutation <
evaluation.chance_threshold):
  - the fraction of ALL scored test-period windows that are gap-adjacent
    (pipeline.gap_adjacent_mask -- within one window_duration of a data
    gap boundary)
  - the fraction of the detecting episode's OWN windows that are
    gap-adjacent

If the episode's fraction is much higher than the fold-wide baseline, the
detection is suspicious of being a gap artifact -- Part A3 (--exclude-
gap-adjacent on scripts/isolation_forest_experiment.py) is the direct test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from apu_sentinel.config import AdditionalExclusionRegion, load_config
from apu_sentinel.data.load import load_raw
from apu_sentinel.data.split import extend_test_end_for_false_alarms, make_folds
from apu_sentinel.data.windows import characterise_sampling
from apu_sentinel.pipeline import (
    _evaluate_at_widths_and_quantiles,
    _fit_fold_isolation_forest,
    _gap_boundaries,
    gap_adjacent_mask,
)
from apu_sentinel.regimes import assign_regimes

# scripts/ has no __init__.py (not a package) -- duplicated, not imported,
# from isolation_forest_experiment.py's build_settings. Keep both in sync
# if the arm/profile definitions change.
THRESHOLD_QUANTILES = [0.995, 0.999, 0.9995, 0.9999]
MARCH_EXCLUSION = AdditionalExclusionRegion(
    start="2020-03-03 00:00",
    end="2020-03-12 00:00",
    reason="pass 21 arm B: early-March cluster sensitivity exclusion",
)


def build_settings(config_name: str, profile: str, arm: str):
    settings = load_config(config_name)
    updates: dict[str, object] = {
        "evaluation": settings.evaluation.model_copy(
            update={"threshold_quantiles": THRESHOLD_QUANTILES}
        )
    }
    if profile == "sweep":
        updates["model"] = settings.model.model_copy(
            update={
                "isolation_forest": settings.model.isolation_forest.model_copy(
                    update={"n_estimators": 100}
                )
            }
        )
        updates["windowing"] = settings.windowing.model_copy(update={"score_stride": "5min"})
    if arm == "b":
        exclusion = settings.split.training_exclusion.model_copy(
            update={"additional_regions": [MARCH_EXCLUSION]}
        )
        updates["split"] = settings.split.model_copy(update={"training_exclusion": exclusion})
    return settings.model_copy(update=updates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    parser.add_argument("--profile", default="sweep", choices=["sweep", "final"])
    parser.add_argument("--arm", required=True, choices=["a", "b"])
    parser.add_argument("--fold", type=int, default=4)
    args = parser.parse_args()

    settings = build_settings(args.config, args.profile, args.arm)

    raw_path = Path(settings.data.raw_dir) / settings.data.raw_filename
    df = load_raw(raw_path)
    data_start, data_end = df.index.min(), df.index.max()

    regimes = assign_regimes(df, settings)
    folds = make_folds(settings, data_start, data_end)
    fold = next(f for f in folds if f.event_id == args.fold)

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    event = next(e for e in events_sorted if e.id == args.fold)
    training_exclusion = settings.split.training_exclusion
    extended_fold = extend_test_end_for_false_alarms(
        fold, event, events_sorted, training_exclusion, data_end
    )

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    model, _scalers, train_input, fold_input = _fit_fold_isolation_forest(
        df, regimes, extended_fold, settings
    )

    fold_full = df.loc[
        (df.index >= extended_fold.train_start) & (df.index <= extended_fold.test_end)
    ]
    gap_threshold = pd.Timedelta(settings.windowing.gap_threshold)
    window_duration = pd.Timedelta(settings.windowing.window_duration)
    gaps = _gap_boundaries(fold_full.index, gap_threshold)
    print(f"fold {args.fold}: {len(gaps)} gaps >= {gap_threshold} in train_start..test_end")

    test_mask = (fold_input.index >= extended_fold.test_start) & (
        fold_input.index <= extended_fold.test_end
    )
    test_end_timestamps = fold_input.end_timestamps[test_mask]
    baseline_mask = gap_adjacent_mask(test_end_timestamps, gaps, window_duration)
    baseline_fraction = baseline_mask.mean() if len(baseline_mask) else float("nan")
    print(
        f"fold {args.fold}: {baseline_mask.sum()}/{len(baseline_mask)} "
        f"({baseline_fraction:.3f}) of ALL scored test-period windows are gap-adjacent"
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
    seen_episodes: set[tuple] = set()
    for width, width_results in common_widths.items():
        for q, entry in width_results.items():
            result, chance = entry["result"], entry["chance"]
            if not (result.detected and chance.p_chance_permutation < chance_threshold):
                continue
            for ep in result.episodes:
                if ep.category not in ("early_warning", "concurrent"):
                    continue
                key = (q, ep.start, ep.end)
                if key in seen_episodes:
                    continue
                seen_episodes.add(key)
                ep_mask = (test_end_timestamps >= ep.start) & (test_end_timestamps <= ep.end)
                ep_gap_mask = gap_adjacent_mask(test_end_timestamps[ep_mask], gaps, window_duration)
                ep_fraction = ep_gap_mask.mean() if len(ep_gap_mask) else float("nan")
                print(
                    f"  flagged width={width}h q={q} p_perm={chance.p_chance_permutation:.3f} "
                    f"episode=[{ep.start} -> {ep.end}] n_windows={len(ep_gap_mask)} "
                    f"gap_adjacent={ep_gap_mask.sum()} ({ep_fraction:.3f}) "
                    f"vs baseline {baseline_fraction:.3f}"
                )

    if not seen_episodes:
        print(
            f"fold {args.fold}: no flagged (p_chance_permutation < {chance_threshold}) detections"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
