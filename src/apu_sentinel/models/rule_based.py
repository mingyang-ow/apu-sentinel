"""Rule-based baseline. First stage of the model progression in CLAUDE.md.

Each rule computes a quantity from the causal cycle-timing features
(features/cycles.py) or raw channels, expresses it as its position in the
rule's OWN training-data distribution (fit-on-train-only, see fit()'s
docstring), and the model's overall score is the MAX severity across
enabled rules -- the most-violated rule drives the alert, keeping
attribution interpretable (contributor_names / contributions is "which
rule fired", not a channel).

The model emits a continuous score, not a binary flag -- thresholding is
the evaluation harness's job (evaluation/metrics.py fit_threshold). No
alert threshold lives here.

CRITICAL -- causal only. Every rule's raw quantity comes from
features/cycles.py functions (already proven causal by
tests/test_cycles.py) or from this module's own last_completed_run_peak
call, itself a thin wrapper around the same causal run-segment machinery.
Percentile-rank scoring against an already-fitted calibration curve is a
pointwise transform of a causal series and introduces no lookahead of its
own. Never a CENTERED rolling window, a BACKWARD fill, or a
positive-lag/future shift anywhere in this module.

`data` (the argument to fit/score/contributions) is a per-timestamp
DataFrame with a "regime" column (assign_regimes' output: LOADED/OFFLOAD/
STOPPED/TRANSITION) plus whichever raw analog channel columns the enabled
rules need (default: Reservoirs) -- NOT the windowed tensors make_windows
produces, which are shaped for a model like an autoencoder instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from apu_sentinel.config import load_config
from apu_sentinel.features.cycles import (
    baseline_relative,
    compute_cycle_features,
    last_completed_run_peak,
)

# Canonical order: contributor_names / contributions columns follow this
# order, filtered to whichever rules are enabled.
RULE_ORDER = (
    "short_stopped_duration",
    "fast_pressure_decay",
    "low_peak_pressure",
    "high_duty_ratio",
)

# Direction each rule's raw quantity is abnormal in -- "low" means severity
# rises as the value falls below its training distribution; "high" means
# severity rises as the value rises above it.
_DIRECTION = {
    "short_stopped_duration": "low",
    "fast_pressure_decay": "high",
    "low_peak_pressure": "low",
    "high_duty_ratio": "high",
}

# Peak-pressure rule reads this channel; not config-exposed since (unlike
# fast_pressure_decay) no alternative channel is called for in the brief --
# Reservoirs is the nominal ~9.64 bar OFFLOAD-plateau channel (FINDINGS §7).
_LOW_PEAK_PRESSURE_CHANNEL = "Reservoirs"


def _percentile_rank(sorted_train: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean-rank empirical-CDF percentile of each value against the sorted
    training array, in [0, 1]. Deliberately NOT interpolation against a
    small fixed grid of quantile levels: several of these rules'
    quantities are heavily forward-filled / periodic (e.g. duty_ratio_
    trailing revisits the same value every cycle), so training data is
    often a big tied mass at one point. Interpolating against a coarse
    quantile grid assigns a tie to whichever grid level happens to land on
    it -- easily its UPPER edge -- which can misread "sitting on the mode"
    as "90th percentile". Averaging the left- and right-insertion ranks
    (searchsorted) instead places a tie at the middle of its own occupied
    rank, which is what "percentile position in the training distribution"
    is actually supposed to mean.
    """
    n = len(sorted_train)
    if n == 0:
        return np.full(len(values), 0.5)
    left = np.searchsorted(sorted_train, values, side="left")
    right = np.searchsorted(sorted_train, values, side="right")
    return (left + right) / (2.0 * n)


def _rule_config(rules_cfg: dict, name: str):
    return rules_cfg.get(name)


def _is_enabled(rules_cfg: dict, name: str) -> bool:
    cfg = _rule_config(rules_cfg, name)
    return cfg is not None and cfg.enabled


class RuleBasedModel:
    """Rule-based AnomalyModel. See module docstring for the input shape
    and scoring scheme.

    `settings` defaults to the merged app config (load_config()) so the
    class remains zero-argument constructible (tests/test_eval_contract.py
    parametrizes MODEL_CLASSES this way); pass an explicit duck-typed
    settings object (exposing .features and .model.rule_based, the shape
    of apu_sentinel.config.Settings) for tests against synthetic data.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = load_config()
        self._settings = settings

        rule_based_cfg = settings.model.rule_based
        if rule_based_cfg is None:
            rules_cfg: dict = {}
            baseline_window = pd.Timedelta("7D")
        else:
            rules_cfg = rule_based_cfg.rules
            baseline_window = pd.Timedelta(rule_based_cfg.baseline_window)
        self._rules_cfg = rules_cfg
        self._baseline_window = baseline_window

        self._enabled_rules = tuple(name for name in RULE_ORDER if _is_enabled(rules_cfg, name))
        self._calibration: dict[str, np.ndarray] = {}

    @property
    def contributor_names(self) -> tuple[str, ...]:
        return self._enabled_rules

    # --- raw quantities -----------------------------------------------

    def _decay_channel(self) -> str:
        cfg = _rule_config(self._rules_cfg, "fast_pressure_decay")
        if cfg is not None and cfg.source_channel is not None:
            return cfg.source_channel
        return self._settings.features.decay_source_channel

    def _cycle_settings(self, decay_source_channel: str) -> SimpleNamespace:
        base_features = self._settings.features
        features_ns = SimpleNamespace(
            decay_source_channel=decay_source_channel,
            decay_min_samples=base_features.decay_min_samples,
            gap_threshold=base_features.gap_threshold,
            baseline_window=base_features.baseline_window,
            duty_ratio_window=base_features.duty_ratio_window,
        )
        return SimpleNamespace(features=features_ns)

    def _raw_quantities(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Every enabled rule's raw quantity (pre-calibration), aligned to
        df.index. Computed causally throughout -- see module docstring.
        """
        regimes = df["regime"]
        raw: dict[str, pd.Series] = {}

        needs_cycle_features = any(
            r in self._enabled_rules
            for r in ("short_stopped_duration", "fast_pressure_decay", "high_duty_ratio")
        )
        if needs_cycle_features:
            decay_channel = self._decay_channel()
            cycle_settings = self._cycle_settings(decay_channel)
            cycle_features = compute_cycle_features(df[[decay_channel]], regimes, cycle_settings)

            if "short_stopped_duration" in self._enabled_rules:
                raw["short_stopped_duration"] = baseline_relative(
                    cycle_features["stopped_duration_last"], self._baseline_window
                )
            if "fast_pressure_decay" in self._enabled_rules:
                abs_decay = cycle_features["decay_rate_last"].abs()
                raw["fast_pressure_decay"] = baseline_relative(abs_decay, self._baseline_window)
            if "high_duty_ratio" in self._enabled_rules:
                raw["high_duty_ratio"] = cycle_features["duty_ratio_trailing"]

        if "low_peak_pressure" in self._enabled_rules:
            peak_settings = self._cycle_settings(_LOW_PEAK_PRESSURE_CHANNEL)
            peak = last_completed_run_peak(
                df[[_LOW_PEAK_PRESSURE_CHANNEL]],
                regimes,
                peak_settings,
                "OFFLOAD",
                _LOW_PEAK_PRESSURE_CHANNEL,
            )
            raw["low_peak_pressure"] = baseline_relative(peak, self._baseline_window)

        return raw

    # --- fit-on-train-only calibration ----------------------------------

    def fit(self, train_data: pd.DataFrame) -> None:
        """Record each enabled rule's TRAINING distribution (sorted raw
        values) -- calibrating comparability across rules (so max() is
        meaningful), not learning parameters. Fit-on-train-only, same
        discipline as data/scaling.py's fit_scaler: this function only ever
        sees whatever DataFrame it is called with, so calling it with
        train+test data is a caller error this function cannot detect, same
        contract as fit_scaler.
        """
        raw = self._raw_quantities(train_data)
        self._calibration = {}
        for rule, series in raw.items():
            values = series.to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            self._calibration[rule] = np.sort(values)

    def _severities(self, data: pd.DataFrame) -> pd.DataFrame:
        """Each rule's raw quantity -> a severity in [0, 1] via its
        position in the FITTED training distribution (_percentile_rank
        against the calibration array from fit()).

        Severity is a ONE-SIDED ramp, not the raw percentile rank: a value
        on the "normal or better" half of the training distribution scores
        exactly 0 (the rule is not firing), and severity only rises as the
        value moves further into ITS OWN abnormal direction, reaching 1 at
        the most extreme percentile seen in training. This is what makes
        "zero when not firing" (module/contract docstring) meaningful --
        the symmetric percentile rank alone would give a merely-typical
        value on the wrong side of the median a misleadingly non-zero
        score.
        """
        raw = self._raw_quantities(data)
        severities = pd.DataFrame(index=data.index)
        for rule in self._enabled_rules:
            series = raw[rule]
            sorted_train = self._calibration[rule]
            values = series.to_numpy(dtype=float)
            valid = ~np.isnan(values)

            percentile_rank = np.zeros(len(values))
            percentile_rank[valid] = _percentile_rank(sorted_train, values[valid])

            if _DIRECTION[rule] == "low":
                ramp = 1.0 - 2.0 * percentile_rank
            else:
                ramp = 2.0 * percentile_rank - 1.0
            severity = np.where(valid, np.clip(ramp, 0.0, 1.0), 0.0)

            severities[rule] = severity
        return severities

    # --- scoring ----------------------------------------------------------

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """Per-timestamp anomaly score: the MAX severity across enabled
        rules -- the most-violated rule drives the alert.
        """
        severities = self._severities(data)
        return severities.to_numpy().max(axis=1)

    def contributions(self, data: pd.DataFrame) -> np.ndarray:
        """Per-timestamp, per-rule severity. Column order matches
        contributor_names. Zero where a rule has no signal yet (e.g. no
        run of the relevant kind has completed) rather than NaN.
        """
        return self._severities(data).to_numpy()
