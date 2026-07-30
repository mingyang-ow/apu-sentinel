"""Isolation Forest model. Second stage of the model progression in
CLAUDE.md.

Unlike the rule-based baseline (models/rule_based.py), which reads raw
per-timestamp channel values, this model needs FIXED-LENGTH vectors, one per
sliding window (data/windows.py make_windows()) -- Isolation Forest has no
native notion of a sequence. `WindowedInput` bundles a fold's windows +
end timestamps + channel names + (optionally) the per-timestamp cycle
features (features/cycles.py compute_cycle_features()) sampled at each
window's end timestamp, matching the existing score/label convention (a
window's score belongs at its LAST timestamp). `windows` must already be
regime-conditionally SCALED (data/scaling.py) before reaching this model --
same as any other windowed model; scaling is not this module's job.

Feature vector per window: per-channel summary statistics
(model.isolation_forest.window_stats, default mean/std/min/max/slope) over
`windows`, plus cycle features if include_cycle_features is set. Both the
stat list and the cycle-feature toggle are config-driven so the feature set
is a documented, swept choice, never hardcoded.

`contamination` is deliberately never touched here: sklearn's
IsolationForest only uses it to set `predict()`'s own decision offset. This
model calls `score_samples()` directly (negated, see below) and the harness
fits its own threshold from training scores (evaluation/metrics.py
fit_threshold) -- `contamination` has no effect on anything this project
uses, so it is not exposed in config; "tuning" it would be a no-op.

Score direction: sklearn's `score_samples()` is higher = MORE NORMAL (an
average path length convention inherited from the isolation-forest paper).
The AnomalyModel contract requires higher = MORE ANOMALOUS
(models/base.py). `score()` therefore returns `-score_samples(...)` --
getting this backwards silently inverts every downstream result (thresholds,
detections, false alarms all become their own complement) without raising
anywhere, so it is called out here explicitly and guarded by
tests/test_isolation_forest.py's score-direction test.

Contributions: Isolation Forest gives one score, no native per-feature
attribution. Ablation (models.isolation_forest.contributions) substitutes
each feature, one at a time, with its OWN training median, re-scores, and
takes the score DROP (original - ablated) as that feature's contribution --
a feature that was genuinely driving the anomaly becomes less anomalous
once replaced by "typical", producing a positive contribution; an
uninvolved feature's replacement barely moves the score. Cost is
O(n_features) re-scoring calls. `contributions.enabled: false` skips this
entirely (zeros, logged) for speed during sweeps.

Public API:
- `WindowedInput` -- the `data` shape this model's fit/score/contributions
  take: windows, end_timestamps, channel_names, cycle_features (nullable).
  Exposes `.index` (aliasing end_timestamps) so pipeline.py's model-agnostic
  evaluation helpers (built against the rule-based model's plain
  DataFrame-with-.index input) work unchanged for this model too.
- `build_feature_matrix(data, settings) -> (X, feature_names)` -- the
  window-stats + cycle-features vector build; also used directly by
  contributions() for ablation.
- `IsolationForestModel(settings=None)` -- implements AnomalyModel. One
  instance per fold (models/isolation_forest.py has no cross-fold state);
  `fit()` on a fold's clean training windows only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from apu_sentinel.config import load_config

logger = logging.getLogger(__name__)

VALID_WINDOW_STATS = ("mean", "std", "min", "max", "slope")


@dataclass(frozen=True)
class WindowedInput:
    """`data` shape for IsolationForestModel. `windows` (n_windows,
    window_length, n_channels) must already be regime-conditionally scaled;
    `channel_names` gives its channel axis's names, in order (matches
    data/windows.py make_windows()'s analog_columns + passthrough_columns
    order). `cycle_features` is the RAW (unscaled) per-timestamp cycle
    feature frame (features/cycles.py compute_cycle_features()), indexed by
    timestamp -- None when include_cycle_features is off.
    """

    windows: np.ndarray
    end_timestamps: pd.DatetimeIndex
    channel_names: tuple[str, ...]
    cycle_features: pd.DataFrame | None

    @property
    def index(self) -> pd.DatetimeIndex:
        """Duck-types as a DataFrame's `.index` -- lets pipeline.py's
        model-agnostic evaluation helpers (written against the rule-based
        model's plain per-timestamp DataFrame input) work unchanged here.
        """
        return self.end_timestamps


def _window_stat_matrix(
    windows: np.ndarray, channel_names: tuple[str, ...], stats: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Per-channel summary stats over each window's samples, channel-major
    (all stats for one channel, then the next) so contributor_names reads
    as grouped-by-channel, e.g. TP3_mean, TP3_std, ..., Reservoirs_mean, ...
    """
    unknown = [s for s in stats if s not in VALID_WINDOW_STATS]
    if unknown:
        raise ValueError(f"unknown window stat(s) {unknown} -- valid: {VALID_WINDOW_STATS}")

    window_length = windows.shape[1]
    per_stat = {}
    if "mean" in stats:
        per_stat["mean"] = windows.mean(axis=1)
    if "std" in stats:
        per_stat["std"] = windows.std(axis=1)
    if "min" in stats:
        per_stat["min"] = windows.min(axis=1)
    if "max" in stats:
        per_stat["max"] = windows.max(axis=1)
    if "slope" in stats:
        per_stat["slope"] = (windows[:, -1, :] - windows[:, 0, :]) / window_length

    columns = []
    names = []
    for ci, channel in enumerate(channel_names):
        for stat in stats:
            columns.append(per_stat[stat][:, ci])
            names.append(f"{channel}_{stat}")
    matrix = np.column_stack(columns) if columns else np.empty((windows.shape[0], 0))
    return matrix, names


def _cycle_feature_matrix(
    cycle_features: pd.DataFrame, end_timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, list[str]]:
    """Cycle features sampled at each window's end timestamp -- a direct
    lookup, not an as-of/ffill: end_timestamps are themselves raw sample
    timestamps already present in cycle_features' index (make_windows()
    only ever selects real rows), so reindex() is exact, not approximate.
    """
    sampled = cycle_features.reindex(end_timestamps)
    return sampled.to_numpy(dtype=float), list(sampled.columns)


def build_feature_matrix(data: WindowedInput, settings) -> tuple[np.ndarray, tuple[str, ...]]:
    """The fixed-length feature vector per window: window summary stats,
    plus cycle features sampled at the window's end timestamp if
    `model.isolation_forest.include_cycle_features` is set.

    Raises:
        ValueError: if include_cycle_features is True but
            data.cycle_features is None (caller built `data` inconsistently
            with its own config).
    """
    cfg = settings.model.isolation_forest
    stat_matrix, stat_names = _window_stat_matrix(
        data.windows, data.channel_names, cfg.window_stats
    )

    if not cfg.include_cycle_features:
        return stat_matrix, tuple(stat_names)

    if data.cycle_features is None:
        raise ValueError(
            "model.isolation_forest.include_cycle_features is True but data.cycle_features is None"
        )
    cycle_matrix, cycle_names = _cycle_feature_matrix(data.cycle_features, data.end_timestamps)
    matrix = np.column_stack([stat_matrix, cycle_matrix])
    return matrix, tuple(stat_names + cycle_names)


class IsolationForestModel:
    """Isolation Forest AnomalyModel. See module docstring for the input
    shape, score-direction convention, and ablation contributions.

    `settings` defaults to the merged app config (load_config()) so the
    class remains zero-argument constructible (tests/test_eval_contract.py
    parametrizes MODEL_CLASSES this way); pass an explicit duck-typed
    settings object (exposing .model.isolation_forest, the shape of
    apu_sentinel.config.Settings) for tests against synthetic data.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = load_config()
        cfg = settings.model.isolation_forest
        if cfg is None:
            raise ValueError(
                "IsolationForestModel requires settings.model.isolation_forest to be configured"
            )
        self._settings = settings
        self._cfg = cfg
        self._forest: IsolationForest | None = None
        self._feature_names: tuple[str, ...] = ()
        self._medians: np.ndarray | None = None

    @property
    def contributor_names(self) -> tuple[str, ...]:
        return self._feature_names

    def _fill_nan(self, matrix: np.ndarray) -> np.ndarray:
        """NaN (e.g. a cycle feature with no completed run yet) filled with
        its OWN training median -- the same value ablation substitutes, so
        "no signal yet" and "ablated to typical" read identically to the
        forest, and no separate imputation policy needs inventing.
        """
        nan_mask = np.isnan(matrix)
        if not nan_mask.any():
            return matrix
        return np.where(nan_mask, self._medians, matrix)

    def fit(self, train_data: WindowedInput) -> None:
        """Fit-on-train-only: `train_data` must already be restricted to a
        fold's clean training windows (caller's job, same contract as
        data/scaling.py fit_scaler) -- this function fits whatever it is
        given.
        """
        matrix, names = build_feature_matrix(train_data, self._settings)
        self._feature_names = names
        self._medians = np.nanmedian(matrix, axis=0)
        matrix = self._fill_nan(matrix)

        self._forest = IsolationForest(
            n_estimators=self._cfg.n_estimators,
            max_samples=self._cfg.max_samples,
            random_state=self._cfg.random_state,
        )
        self._forest.fit(matrix)
        logger.info(
            "IsolationForestModel.fit: %d features (%s), %d training windows",
            len(names),
            ", ".join(names[:5]) + ("..." if len(names) > 5 else ""),
            matrix.shape[0],
        )

    def score(self, data: WindowedInput) -> np.ndarray:
        """Per-window anomaly score: -score_samples() (see module
        docstring's score-direction convention).
        """
        matrix, _ = build_feature_matrix(data, self._settings)
        matrix = self._fill_nan(matrix)
        return -self._forest.score_samples(matrix)

    def contributions(self, data: WindowedInput) -> np.ndarray:
        """Ablation: for each feature, replace its column with the TRAINING
        median for every row, re-score, and take (original - ablated) as
        that feature's contribution. `contributions.enabled: false` returns
        zeros of the correct shape instead, logged so it's never mistaken
        for "no feature mattered".
        """
        matrix, _ = build_feature_matrix(data, self._settings)
        n_rows = matrix.shape[0]
        n_features = len(self._feature_names)

        if not self._cfg.contributions.enabled:
            logger.info(
                "IsolationForestModel.contributions: attribution disabled "
                "(model.isolation_forest.contributions.enabled=false) -- "
                "returning zeros for %d rows x %d features",
                n_rows,
                n_features,
            )
            return np.zeros((n_rows, n_features))

        matrix = self._fill_nan(matrix)
        full_scores = -self._forest.score_samples(matrix)

        contributions = np.zeros((n_rows, n_features))
        for j in range(n_features):
            ablated = matrix.copy()
            ablated[:, j] = self._medians[j]
            ablated_scores = -self._forest.score_samples(ablated)
            contributions[:, j] = full_scores - ablated_scores
        return contributions
