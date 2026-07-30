# ARCHITECTURE.md

How data gets from a raw CSV to an evaluated alert, without opening source.
See `CLAUDE.md` for the rules this architecture exists to enforce, and
`docs/FINDINGS.md` / `docs/RESULTS.md` for what running it has shown so far.

## Pipeline order

This sequence is load-bearing and not obvious from the directory layout —
several steps must happen in exactly this order or a guard test fails.

```
download (checksum-verified)
  → load_raw            drops Unnamed:*, parses timestamps, sorts, surfaces issues
  → assign_regimes      per-timestamp 4-state label (causal), on the FULL raw series
  → make_folds          one Fold per documented event (expanding window + embargo)
  → apply_fold          (train, test); exclusions removed from train only
  → fit_regime_scalers  per (fold, regime), TRAIN ONLY
  → transform_by_regime scaled; regime-inactive channels set to constant 0.0
  → features/cycles     causal cycle-timing features (duration, decay rate, duty ratio)
  → make_windows        (windows, end_timestamps); drops gap-spanning windows
  → model.fit/score/contributions
  → fit_threshold       from TRAIN scores only
  → evaluation          episodes → categories → detection/lead time → rates → null comparison
```

Two branches exist off this spine, both in `pipeline.py`:
- **Windowed models** (autoencoder, isolation forest — not yet implemented)
  follow the full sequence above, through `make_windows`.
- **The rule-based baseline** (`models/rule_based.py`, the only model wired
  today) skips `fit_regime_scalers`/`transform_by_regime`/`make_windows`
  entirely — it reads RAW channel values plus the `regime` label directly,
  because its rules are stated in physical units (bar, seconds) and already
  condition on regime via `features/cycles.py`'s per-run computations. See
  `pipeline.py`'s module docstring for the full rationale.

Evaluation itself (pass 13) additionally branches per fold into: the common
sweep (shared width, `evaluation.window_widths`, cross-model-comparable),
an extended-`test_end` variant for adequate false-alarm denominators, pooled
normal-operation stretches spanning the whole series, and a per-event
maximum-width sensitivity fold — see `docs/RESULTS.md` §13 and
`pipeline.run_pipeline`'s docstring for exactly what each produces.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Typed (pydantic-settings) layered config loader: `base.yaml` → `local`/`colab.yaml` → optional experiment overlay, deep-merged, validated at load time. |
| `data/download.py` | Fetch MetroPT-3, two-mode SHA256 checksum (establish / verify). |
| `data/load.py` | Raw CSV → clean, time-sorted DataFrame. Nothing else — no splitting, scaling, filtering. |
| `data/split.py` | **CROWN-JEWEL.** Walk-forward fold construction (`make_folds`), time-only slicing (`apply_fold`), per-event window caps (`event_max_width_hours`), false-alarm test-period extension (`extend_test_end_for_false_alarms`). |
| `data/scaling.py` | **CROWN-JEWEL.** Per-(fold, regime) robust/standard/minmax scaling, fit on train only, regime-inactive channels zeroed. |
| `data/windows.py` | Gap-aware sliding windows over already-scaled data; measures the real sampling interval rather than assuming one. |
| `regimes/__init__.py` | Causal per-timestamp 4-state operating-regime labels (LOADED/OFFLOAD/STOPPED/TRANSITION), derived from digital control flags plus one documented analog exception. |
| `features/cycles.py` | Causal cycle-timing features: STOPPED-duration and pressure-decay families, trailing-baseline drift normalisation. |
| `models/base.py` | **THE CONTRACT.** `AnomalyModel` protocol every model implements; evaluation never touches a model beyond this interface. |
| `models/rule_based.py` | The rule-based baseline (first, and currently only, implemented model) — four configurable rules, MAX-aggregated, fit-on-train-only percentile calibration. |
| `models/isolation_forest.py`, `models/autoencoder.py` | Stubs for the next two stages of the model progression (CLAUDE.md) — not yet implemented. |
| `evaluation/events.py` | Pre-failure windows, masked regions, coverage reporting, pooled normal-operation stretches. |
| `evaluation/metrics.py` | **THE METRIC.** Threshold fitting, episode grouping, categorisation, detection/lead time, false-alarm rates (in-fold and pooled), null (chance) comparison. Owns everything downstream of a model's score. |
| `explain/__init__.py` | Aggregates per-timestamp contributions into one episode's ranked diagnosis. |
| `analysis/__init__.py` | Standalone diagnostic reports outside the core pipeline (currently: gap-density vs. STOPPED-run analysis). |
| `pipeline.py` | Ties every stage together, config-driven; the only module that calls across `data/`, `regimes/`, `features/`, `models/`, and `evaluation/`. |

## Contracts

### `AnomalyModel` (`models/base.py`)

Every model implements:
- `contributor_names -> tuple[str, ...]` — column names for `contributions()`; channels for a channel-attributing model, rule names for a rule-based one.
- `fit(train_data) -> None`
- `score(data) -> np.ndarray` — per-timestamp anomaly score.
- `contributions(data) -> np.ndarray` — per-timestamp, per-contributor contribution, column order matching `contributor_names`.

`data`'s shape is a per-model choice (untyped in the contract) — see
`models/base.py`'s docstring. Every model must be zero-argument
constructible (`ModelCls()`); a real model's `settings=None` default falls
back to `config.load_config()`.

### `Fold` (`data/split.py`)

`train_start`, `train_end`, `test_start`, `test_end`, `train_exclusions`
(regions purged from training). Produced by `make_folds()` (or
`extend_test_end_for_false_alarms()` for a test_end-extended variant);
consumed by `apply_fold()` and by every `evaluation/` function that needs
the fold's boundaries.

### `Episode` (`evaluation/metrics.py`)

`start`, `end`, `peak_score`, `category` (one of `early_warning` /
`concurrent` / `masked` / `false_alarm`), `matched_event_id`,
`channel_ranking` (the mixed, ranked diagnosis). Produced by
`evaluate_fold_at_threshold()`'s internal episode grouping +
categorisation; consumed by detection/lead-time and false-alarm counting,
and by the null-comparison functions (`p_chance_permutation`,
`evaluate_chance`).

### `FoldEvaluation` (`evaluation/metrics.py`)

One fold, one window width, one threshold: `detected`, `lead_time`,
`episodes`, `false_episode_count`, `evaluated_days`, `false_alarms_per_day`,
`window_coverage`, plus the fields above. Produced by
`evaluate_fold_at_threshold()` / `evaluate_fold()` / `evaluate_fold_sweep()`;
consumed by `evaluate_chance()` (pass 13) and by `pipeline.run_pipeline`'s
reporting.

### Pass 13 additions (`evaluation/metrics.py`, `evaluation/events.py`)

- `ScoredTestData` — a model's already-computed (timestamps, scores,
  contributions, contributor_names, expected_interval) for one scored
  period; the harness never scores anything itself.
- `ChanceComparison` — `p_chance_poisson`, `p_chance_permutation`,
  `not_distinguishable_from_chance`. Produced by `evaluate_chance()` from a
  `FoldEvaluation` + its `Fold`.
- `NormalStretch` / `PooledEvaluation` — normal-operation periods spanning
  the whole series, and the false-alarm rate pooled across them. Produced
  by `evaluation/events.py`'s `pooled_normal_stretches()` and
  `evaluation/metrics.py`'s `evaluate_pooled_stretches()`.

## Invariants and where they are enforced

| Rule | Enforced by |
|---|---|
| Time-only split, no shuffle, no leakage across train/test | `tests/test_split_no_leakage.py` (blocking hook on `data/` edits) |
| Scalers fit on TRAIN ONLY | `tests/test_scaler_train_only.py` (blocking hook on `data/` edits) |
| Regime labels are causal (no future information) | `tests/test_regimes.py` (source inspection + behavioural: unchanged when future data deleted) |
| Cycle-timing features are causal | `tests/test_cycles.py` (same two-pronged check) |
| Rule-based model scoring is causal | `tests/test_rule_based.py` |
| Windows never span a gap (native or exclusion-driven) | `tests/test_windows.py` |
| Every model satisfies `AnomalyModel` | `tests/test_eval_contract.py` |
| Threshold fit on TRAIN scores only | `tests/test_metrics.py` (`fit_threshold`'s structural signature guard) |
| Episode grouping/categorisation, false-alarm denominators, null comparison | `tests/test_metrics.py` |
| Config validates strictly (unknown/wrong-typed keys fail loud) | `tests/test_config.py` |
| No data committed to the repo | pre-commit `no-data-commit` hook |

## Non-obvious design decisions

- Scaling is **per-timestamp before windowing** because LOADED's median run
  (99s) is far shorter than the window (1800s) — there is no single-regime
  window.
- `TRANSITION` is scaled inline and **cannot be excluded**; state changes occur
  every couple of minutes, so gap-treating it would leave no contiguous window.
- Regime-inactive channels are set to constant 0.0, **not scaled**, because
  their within-regime spread is sensor noise around zero and dividing by it
  amplifies noise.
- A window's score attaches to its **END timestamp**.
- The embargo must be **≥ the window duration**, or sliding windows straddle
  the train/test boundary.
- Models produce score + contributions only; **evaluation owns the metric**.
- `assign_regimes` runs on the **full raw series once**, then is sliced per
  fold — not re-derived per fold — because regime labelling only needs
  backward-looking information, unlike scaler statistics (which must never
  see test data at all).
- The rule-based model's severity is a **one-sided ramp**, not a raw
  percentile rank: a value on the "normal or better" side of its training
  distribution scores exactly 0. A symmetric rank would give a merely-typical
  value on the wrong side of the median a misleadingly non-zero score.
- Percentile ranking against the training distribution uses **mean-rank
  (averaged left/right insertion), not interpolation over a fixed quantile
  grid** — several rule quantities are heavily forward-filled/periodic, so
  training data often has a big tied mass; grid interpolation would place a
  tie at its upper edge instead of its true middle.
- The common (shared-width) fold sweep and the per-event-maximum-width fold
  set are **deliberately separate** (`data/split.py`'s
  `width_hours_by_event` parameter) — the former is the cross-model-
  comparable result, the latter a sensitivity analysis; neither silently
  replaces the other.
- A detection's false-alarm rate is only interpretable **next to a null
  (chance) comparison** — a wide enough window makes "detection" nearly
  automatic regardless of model skill (`docs/RESULTS.md` §13).
