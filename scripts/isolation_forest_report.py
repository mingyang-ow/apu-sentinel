#!/usr/bin/env python3
"""Pass 21: aggregate the checkpointed Isolation Forest arm A / arm B
sweep results (scripts/isolation_forest_experiment.py) into the
per-(quantile, width) table and the aggregate skill statistic
docs/RESULTS.md's existing sections already use (`expected = Σ
p_chance_permutation`, `observed` = detection count, `p(X>=observed)` the
exact Poisson-binomial survival probability over the 4 folds) -- same
convention as §13/§18/§20, computed here from scratch since no shared
helper for it exists yet in evaluation/metrics.py.

Read-only: loads pickles from data/interim/isolation_forest_runs/, prints
a report. Never writes into docs/ itself -- the numbers are transcribed
into docs/RESULTS.md by hand so prose commentary stays human-authored.
"""

from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

CHECKPOINT_ROOT = Path("data/interim/isolation_forest_runs")
FOLD_IDS = (1, 2, 3, 4)


def load_arm(run_dir: Path) -> dict[int, dict]:
    folds = {}
    for fold_id in FOLD_IDS:
        path = run_dir / f"fold_{fold_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path} -- run is incomplete")
        with open(path, "rb") as f:
            folds[fold_id] = pickle.load(f)
    return folds


def poisson_binomial_survival(p_values: list[float], observed: int) -> float:
    """Exact P(X >= observed) for X = sum of independent Bernoulli(p_i).
    Brute-force over all 2^n subsets -- fine for n=4 folds (16 terms).
    """
    n = len(p_values)
    total = 0.0
    for bits in itertools.product([0, 1], repeat=n):
        if sum(bits) < observed:
            continue
        term = 1.0
        for b, p in zip(bits, p_values, strict=True):
            term *= p if b else (1.0 - p)
        total += term
    return total


def aggregate_table(folds: dict[int, dict], widths: list[float], quantiles: list[float]) -> None:
    for width in widths:
        for q in quantiles:
            detected = 0
            p_perms = []
            for fold_id in FOLD_IDS:
                entry = folds[fold_id]["common_widths"][width][q]
                result, chance = entry["result"], entry["chance"]
                if result.detected:
                    detected += 1
                p_perms.append(chance.p_chance_permutation)
            expected = sum(p_perms)
            p_survival = poisson_binomial_survival(p_perms, detected)
            print(
                f"width={width:>4}h  q={q:<7} observed={detected}  "
                f"expected(Σp_perm)={expected:.3f}  p(X>=observed)={p_survival:.3f}"
            )


def detections_detail(folds: dict[int, dict], widths: list[float], quantiles: list[float]) -> None:
    for fold_id in FOLD_IDS:
        for width in widths:
            for q in quantiles:
                entry = folds[fold_id]["common_widths"][width][q]
                result, chance = entry["result"], entry["chance"]
                if not result.detected:
                    continue
                print(
                    f"fold={fold_id} width={width}h q={q} lead_time={result.lead_time} "
                    f"fa/day={result.false_alarms_per_day:.3f} "
                    f"false_ep={result.false_episode_count} "
                    f"eval_days={result.evaluated_days:.2f} "
                    f"p_poisson={chance.p_chance_poisson:.3f} "
                    f"p_perm={chance.p_chance_permutation:.3f} "
                    f"not_distinguishable={chance.not_distinguishable_from_chance}"
                )
                explained = entry.get("explained") or {}
                for (start, end), ranked in explained.items():
                    top5 = ", ".join(f"{name}={value:.4f}" for name, value in ranked[:5])
                    print(f"    explain [{start} -> {end}]: {top5}")


def pooled_summary(folds: dict[int, dict]) -> None:
    for fold_id in FOLD_IDS:
        pooled = folds[fold_id]["pooled"]
        print(
            f"fold={fold_id} pooled: false_ep={pooled.false_episode_count} "
            f"days={pooled.evaluated_days:.2f} fa/day={pooled.false_alarms_per_day:.4f}"
        )


def elapsed_summary(folds: dict[int, dict]) -> None:
    for fold_id in FOLD_IDS:
        print(f"fold={fold_id} elapsed={folds[fold_id]['elapsed_seconds']:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="sweep", choices=["sweep", "final"])
    args = parser.parse_args()

    arm_a = load_arm(CHECKPOINT_ROOT / f"{args.profile}_arm_a")
    arm_b = load_arm(CHECKPOINT_ROOT / f"{args.profile}_arm_b")

    widths = sorted(arm_a[1]["common_widths"].keys())
    quantiles = sorted(arm_a[1]["common_widths"][widths[0]].keys())

    print(f"widths={widths} quantiles={quantiles}\n")

    print("=== ARM A: aggregate skill statistic ===")
    aggregate_table(arm_a, widths, quantiles)
    print("\n=== ARM A: detections ===")
    detections_detail(arm_a, widths, quantiles)
    print("\n=== ARM A: pooled false-alarm rate ===")
    pooled_summary(arm_a)
    print("\n=== ARM A: elapsed per fold ===")
    elapsed_summary(arm_a)

    print("\n=== ARM B: aggregate skill statistic ===")
    aggregate_table(arm_b, widths, quantiles)
    print("\n=== ARM B: detections ===")
    detections_detail(arm_b, widths, quantiles)
    print("\n=== ARM B: pooled false-alarm rate ===")
    pooled_summary(arm_b)
    print("\n=== ARM B: elapsed per fold ===")
    elapsed_summary(arm_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
