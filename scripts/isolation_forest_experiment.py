#!/usr/bin/env python3
"""Pass 21: Isolation Forest arm A / arm B + quantile-sweep experiment
runner (CLAUDE.md brief). A thin CLI/checkpoint wrapper around
apu_sentinel.pipeline.evaluate_isolation_forest_fold -- all evaluation
logic lives there; this script only builds the two settings variants,
loops folds, checkpoints, and logs elapsed time.

Arms (split.training_exclusion.additional_regions, sensitivity-only, never
tied to a documented failure event):
  - arm a: additional_regions: [] (base.yaml default, unchanged)
  - arm b: early-March cluster excluded (docs/findings/12-event2-error-
    analysis.md), a comparison arm -- never a silent replacement of arm a.

Profiles (model fit cost -- NOT the quantile grid, which is always swept):
  - sweep (default): n_estimators=100, windowing.score_stride=5min --
    cheap enough to actually run repeatedly while iterating.
  - final: base.yaml's own settings (n_estimators=200, score_stride=1min)
    -- run ONCE, after the sweep profile's results are already reviewed.

Run arm A to completion, then arm B, as two SEPARATE invocations -- never
pass both --arm values in one process (CLAUDE.md build-pass convention:
fail loud, keep each pass's scope a hard boundary).

Checkpoints one pickle per fold under
data/interim/isolation_forest_runs/{profile}_arm_{arm}{tag}/fold_{event_id}.pkl
(data/interim/ is gitignored) -- a restarted run skips any fold whose
checkpoint already exists, so a killed run only re-does the fold in
progress, not the whole arm.

Pass 22 additions (docs/RESULTS.md, event-4 detection validation):
--widths/--quantiles restrict the swept grid (Part C's "at the Part B
operating point only, not the full sweep" -- re-running the full grid at
full settings defeats the point of picking one operating point first);
--only-folds restricts which folds run (A3's "event 4's fold" only);
--exclude-gap-adjacent turns on the model.isolation_forest.
exclude_gap_adjacent_windows diagnostic (A3); --tag suffixes the
checkpoint directory so a restricted/diagnostic run never collides with
or is silently skipped-as-already-done against a full-sweep run.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import pandas as pd

from apu_sentinel.config import AdditionalExclusionRegion, load_config
from apu_sentinel.data.load import load_raw
from apu_sentinel.data.split import make_folds
from apu_sentinel.data.windows import characterise_sampling
from apu_sentinel.evaluation.events import pooled_normal_stretches
from apu_sentinel.pipeline import evaluate_isolation_forest_fold
from apu_sentinel.regimes import assign_regimes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_ROOT = Path("data/interim/isolation_forest_runs")

# docs/findings/12-event2-error-analysis.md: early-March cluster, not
# anchored to any documented event -- arm B sensitivity comparison only.
MARCH_EXCLUSION = AdditionalExclusionRegion(
    start="2020-03-03 00:00",
    end="2020-03-12 00:00",
    reason="pass 21 arm B: early-March cluster sensitivity exclusion "
    "(findings/12-event2-error-analysis.md), not tied to a documented event",
)

# quantiles asked for explicitly, applied regardless of --profile (this is
# the evaluation grid, not a model-fit-cost knob).
THRESHOLD_QUANTILES = [0.995, 0.999, 0.9995, 0.9999]


def build_settings(
    config_name: str,
    profile: str,
    arm: str,
    widths: list[float] | None = None,
    quantiles: list[float] | None = None,
    exclude_gap_adjacent: bool = False,
):
    settings = load_config(config_name)

    updates: dict[str, object] = {
        "evaluation": settings.evaluation.model_copy(
            update={
                "threshold_quantiles": quantiles if quantiles is not None else THRESHOLD_QUANTILES,
                **({"window_widths": widths} if widths is not None else {}),
            }
        )
    }

    isolation_forest_updates: dict[str, object] = {
        "exclude_gap_adjacent_windows": exclude_gap_adjacent
    }
    if profile == "sweep":
        isolation_forest_updates["n_estimators"] = 100
        updates["windowing"] = settings.windowing.model_copy(update={"score_stride": "5min"})
    elif profile != "final":
        raise ValueError(f"unknown profile {profile!r} -- must be 'sweep' or 'final'")
    updates["model"] = settings.model.model_copy(
        update={
            "isolation_forest": settings.model.isolation_forest.model_copy(
                update=isolation_forest_updates
            )
        }
    )

    if arm == "b":
        exclusion = settings.split.training_exclusion.model_copy(
            update={"additional_regions": [MARCH_EXCLUSION]}
        )
        updates["split"] = settings.split.model_copy(update={"training_exclusion": exclusion})
    elif arm != "a":
        raise ValueError(f"unknown arm {arm!r} -- must be 'a' or 'b'")

    return settings.model_copy(update=updates)


def _parse_float_list(raw: str | None) -> list[float] | None:
    return [float(x) for x in raw.split(",")] if raw is not None else None


def _parse_int_list(raw: str | None) -> list[int] | None:
    return [int(x) for x in raw.split(",")] if raw is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    parser.add_argument("--profile", default="sweep", choices=["sweep", "final"])
    parser.add_argument("--arm", required=True, choices=["a", "b"])
    parser.add_argument("--widths", default=None, help="comma-separated hours, e.g. '72'")
    parser.add_argument("--quantiles", default=None, help="comma-separated, e.g. '0.999'")
    parser.add_argument("--only-folds", default=None, help="comma-separated event ids, e.g. '4'")
    parser.add_argument("--exclude-gap-adjacent", action="store_true")
    parser.add_argument(
        "--tag", default="", help="checkpoint dir suffix for restricted/diagnostic runs"
    )
    args = parser.parse_args()

    widths = _parse_float_list(args.widths)
    quantiles = _parse_float_list(args.quantiles)
    only_folds = _parse_int_list(args.only_folds)

    settings = build_settings(
        args.config,
        args.profile,
        args.arm,
        widths=widths,
        quantiles=quantiles,
        exclude_gap_adjacent=args.exclude_gap_adjacent,
    )

    checkpoint_dir = CHECKPOINT_ROOT / f"{args.profile}_arm_{args.arm}{args.tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "arm=%s profile=%s config=%s checkpoint_dir=%s",
        args.arm,
        args.profile,
        args.config,
        checkpoint_dir,
    )

    raw_path = Path(settings.data.raw_dir) / settings.data.raw_filename
    logger.info("loading raw data from %s", raw_path)
    df = load_raw(raw_path)
    data_start, data_end = df.index.min(), df.index.max()
    logger.info("loaded %d rows, %s -> %s", len(df), data_start, data_end)

    regimes = assign_regimes(df, settings)
    common_folds = make_folds(settings, data_start, data_end)
    if only_folds is not None:
        common_folds = [f for f in common_folds if f.event_id in only_folds]
        logger.info("restricted to folds=%s", only_folds)

    sampling = characterise_sampling(df, pd.Timedelta(settings.windowing.gap_threshold))
    expected_interval = sampling.modal_interval

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    events_by_id = {event.id: event for event in events_sorted}
    training_exclusion = settings.split.training_exclusion

    stretches = pooled_normal_stretches(settings, data_start, data_end)

    for fold in common_folds:
        checkpoint_path = checkpoint_dir / f"fold_{fold.event_id}.pkl"
        if checkpoint_path.exists():
            logger.info(
                "fold %d: checkpoint exists at %s, skipping", fold.event_id, checkpoint_path
            )
            continue

        event = events_by_id[fold.event_id]
        logger.info("fold %d: starting", fold.event_id)
        start = time.monotonic()
        result = evaluate_isolation_forest_fold(
            df,
            regimes,
            fold,
            event,
            events_sorted,
            training_exclusion,
            data_end,
            stretches,
            expected_interval,
            settings,
        )
        elapsed = time.monotonic() - start
        result["elapsed_seconds"] = elapsed
        logger.info("fold %d: done in %.1fs", fold.event_id, elapsed)

        tmp_path = checkpoint_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(result, f)
        tmp_path.rename(
            checkpoint_path
        )  # atomic -- a killed write never leaves a half-written checkpoint

    logger.info(
        "arm %s (%s profile) complete -- %d fold checkpoints in %s",
        args.arm,
        args.profile,
        len(common_folds),
        checkpoint_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
