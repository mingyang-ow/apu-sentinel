"""Tests for the rule-based baseline model (models/rule_based.py).

Uses small synthetic cycle data (LOADED->OFFLOAD->STOPPED, shaped like the
real duty cycle in docs/FINDINGS.md §7) -- never the real MetroPT-3
dataset. Each planted-pattern test is constructed to change ONLY the one
quantity its target rule reads, holding every other rule's inputs fixed,
so "fires on its own pattern, near-zero on the others" is a real isolation
check rather than a coincidence of correlated quantities.
"""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.config import FeaturesConfig, RuleBasedModelConfig, RuleConfig
from apu_sentinel.models import rule_based as rule_based_module
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.rule_based import RULE_ORDER, RuleBasedModel

# One baseline cycle: LOADED(10) -> OFFLOAD(60) -> STOPPED(120, decaying
# 9.6 -> ~8.17 bar). loaded_n + offload_n = 70 is held constant across the
# "high_duty_ratio" isolation test (only the LOADED/OFFLOAD split moves);
# stopped_n=120 is held constant across every test except the duration
# one -- these invariants are what make each planted pattern isolated.
BASE_CYCLE = {"loaded_n": 10, "offload_n": 60, "stopped_n": 120, "peak": 9.6, "decay_rate": 0.0012}
N_BASELINE_CYCLES = 20
N_INJECTED_CYCLES = 5


def _repeat(cycle: dict, n: int) -> list[dict]:
    return [dict(cycle) for _ in range(n)]


def _jittered_baseline(n: int, seed: int = 0) -> list[dict]:
    """n baseline cycles with small, deterministic per-cycle jitter around
    BASE_CYCLE's stopped_n/decay_rate/peak -- gives each rule's training
    distribution real spread to calibrate against, rather than a
    degenerate single-point distribution (every cycle bit-identical) that
    would make percentile-rank scoring ill-defined for a barely-different
    test value.
    """
    rng = np.random.default_rng(seed)
    cycles = []
    for _ in range(n):
        cycles.append(
            {
                "loaded_n": BASE_CYCLE["loaded_n"],
                "offload_n": BASE_CYCLE["offload_n"],
                "stopped_n": BASE_CYCLE["stopped_n"] + int(rng.integers(-10, 11)),
                "peak": BASE_CYCLE["peak"] + rng.uniform(-0.05, 0.05),
                "decay_rate": BASE_CYCLE["decay_rate"] * (1 + rng.uniform(-0.05, 0.05)),
            }
        )
    return cycles


def _make_cycle_df(cycles: list[dict], freq_seconds: int = 10) -> pd.DataFrame:
    """LOADED->OFFLOAD->STOPPED synthetic series. Reservoirs holds at
    `peak` through LOADED+OFFLOAD (irrelevant to every rule during LOADED;
    the OFFLOAD plateau IS what low_peak_pressure reads), then decays
    linearly at `decay_rate` per second through STOPPED.
    """
    labels: list[str] = []
    reservoirs: list[float] = []
    for c in cycles:
        loaded_n, offload_n, stopped_n = c["loaded_n"], c["offload_n"], c["stopped_n"]
        peak, decay_rate = c["peak"], c["decay_rate"]

        labels += ["LOADED"] * loaded_n
        reservoirs += [peak] * loaded_n
        labels += ["OFFLOAD"] * offload_n
        reservoirs += [peak] * offload_n

        t = np.arange(stopped_n) * freq_seconds
        reservoirs += list(peak - decay_rate * t)
        labels += ["STOPPED"] * stopped_n

    index = pd.date_range(
        "2020-01-01", periods=len(labels), freq=pd.Timedelta(seconds=freq_seconds)
    )
    df = pd.DataFrame({"Reservoirs": np.asarray(reservoirs, dtype=float)}, index=index)
    df["regime"] = pd.Categorical(labels)
    return df


def _settings(
    enabled: dict[str, bool],
    baseline_window: str = "1D",
    fast_pressure_decay_channel: str | None = None,
    duty_ratio_window: str = "1h",
    baseline_mode: str = "trailing",
    baseline_lag: str = "14D",
):
    rules = {}
    for name in RULE_ORDER:
        kwargs = {"enabled": enabled.get(name, False)}
        if name == "fast_pressure_decay" and fast_pressure_decay_channel:
            kwargs["source_channel"] = fast_pressure_decay_channel
        rules[name] = RuleConfig(**kwargs)

    rule_based = RuleBasedModelConfig(
        baseline_window=baseline_window,
        baseline_mode=baseline_mode,
        baseline_lag=baseline_lag,
        rules=rules,
    )
    features = FeaturesConfig(
        decay_source_channel="Reservoirs",
        decay_min_samples=3,
        gap_threshold="1min",
        baseline_window="7D",
        duty_ratio_window=duty_ratio_window,
    )
    return SimpleNamespace(features=features, model=SimpleNamespace(rule_based=rule_based))


def _all_rules_settings(**kwargs) -> SimpleNamespace:
    return _settings({name: True for name in RULE_ORDER}, **kwargs)


# --- 1. Protocol conformance -------------------------------------------


def test_protocol_conformance():
    model = RuleBasedModel(_all_rules_settings())
    assert isinstance(model, AnomalyModel)

    df = _make_cycle_df(_jittered_baseline(N_BASELINE_CYCLES))
    model.fit(df)
    scores = model.score(df)
    contributions = model.contributions(df)

    assert scores.shape == (len(df),)
    assert contributions.shape == (len(df), len(model.contributor_names))
    assert len(model.contributor_names) == len(RULE_ORDER)


# --- 2. Causality --------------------------------------------------------


def test_causality_score_unchanged_when_future_deleted():
    model = RuleBasedModel(_all_rules_settings())
    train_cycles = _jittered_baseline(N_BASELINE_CYCLES)
    train_df = _make_cycle_df(train_cycles)
    model.fit(train_df)

    injected = {**BASE_CYCLE, "stopped_n": 20, "offload_n": BASE_CYCLE["offload_n"] + 100}
    full_df = _make_cycle_df(train_cycles + _repeat(injected, N_INJECTED_CYCLES))
    full_scores = model.score(full_df)

    for cut in (len(train_df) - 1, len(train_df) + 10, len(train_df) + 50, len(full_df) - 1):
        truncated_scores = model.score(full_df.iloc[: cut + 1])
        assert np.isclose(full_scores[cut], truncated_scores[-1]), f"cut={cut}"


# --- 3. No lookahead constructs in source --------------------------------


def test_no_lookahead_constructs_in_source():
    source = inspect.getsource(rule_based_module)
    assert "center=True" not in source
    assert "centre=True" not in source
    assert "bfill" not in source
    assert "backfill" not in source
    assert re.search(r"shift\(\s*-\d", source) is None


# --- 4. Each rule fires on its own planted pattern, others stay quiet ---


@pytest.mark.parametrize(
    "rule_name,injected_overrides",
    [
        (
            "short_stopped_duration",
            {"stopped_n": 20, "offload_n": BASE_CYCLE["offload_n"] + 100},
        ),
        ("fast_pressure_decay", {"decay_rate": 0.006}),
        ("low_peak_pressure", {"peak": 8.0}),
        (
            "high_duty_ratio",
            {"loaded_n": 60, "offload_n": BASE_CYCLE["offload_n"] - 50},
        ),
    ],
)
def test_each_rule_fires_on_its_planted_pattern(rule_name, injected_overrides):
    settings = _all_rules_settings()
    model = RuleBasedModel(settings)

    baseline_cycles = _jittered_baseline(N_BASELINE_CYCLES)
    train_df = _make_cycle_df(baseline_cycles)
    model.fit(train_df)

    injected_cycle = {**BASE_CYCLE, **injected_overrides}
    full_df = _make_cycle_df(baseline_cycles + _repeat(injected_cycle, N_INJECTED_CYCLES))

    contributions = model.contributions(full_df)
    names = model.contributor_names
    col = {name: i for i, name in enumerate(names)}

    tail = contributions[len(train_df) :]

    assert tail[:, col[rule_name]].mean() > 0.7, f"{rule_name} did not fire on its own pattern"
    for other in RULE_ORDER:
        if other == rule_name:
            continue
        assert (
            tail[:, col[other]].mean() < 0.3
        ), f"{other} fired spuriously on {rule_name}'s pattern"


# --- 5. Fit is train-only -------------------------------------------------


def test_fit_is_train_only():
    settings = _settings({"low_peak_pressure": True})
    baseline_cycles = _jittered_baseline(N_BASELINE_CYCLES)
    train_df = _make_cycle_df(baseline_cycles)

    anomalous_cycle = {**BASE_CYCLE, "peak": 8.0}
    full_df = _make_cycle_df(baseline_cycles + _repeat(anomalous_cycle, N_INJECTED_CYCLES))

    clean_model = RuleBasedModel(settings)
    clean_model.fit(train_df)

    contaminated_model = RuleBasedModel(settings)
    contaminated_model.fit(full_df)  # fit sees the "test" anomaly -- contamination

    clean_severity = clean_model.contributions(full_df)[len(train_df) :, 0]
    contaminated_severity = contaminated_model.contributions(full_df)[len(train_df) :, 0]

    assert clean_severity.mean() > contaminated_severity.mean()
    assert clean_severity.mean() != pytest.approx(contaminated_severity.mean())


# --- 6. Continuous score ---------------------------------------------------


def test_continuous_score_not_binary():
    model = RuleBasedModel(_all_rules_settings())
    df = _make_cycle_df(_jittered_baseline(N_BASELINE_CYCLES))
    model.fit(df)
    scores = model.score(df)
    distinct = np.unique(np.round(scores, 6))
    assert len(distinct) > 5
    assert not set(np.round(scores, 6)) <= {0.0, 1.0}


# --- 7. Disabled rules are absent ------------------------------------------


def test_disabled_rules_absent_from_contributions_and_names():
    settings = _settings({"short_stopped_duration": True, "high_duty_ratio": True})
    model = RuleBasedModel(settings)

    assert model.contributor_names == ("short_stopped_duration", "high_duty_ratio")
    assert "fast_pressure_decay" not in model.contributor_names
    assert "low_peak_pressure" not in model.contributor_names

    df = _make_cycle_df(_jittered_baseline(N_BASELINE_CYCLES))
    model.fit(df)
    contributions = model.contributions(df)
    assert contributions.shape[1] == 2


# --- 8. Drift robustness ----------------------------------------------------


def test_drift_robustness_vs_step_change():
    """FINDINGS.md §8: a trailing baseline absorbs GRADUAL drift but a
    STEP change still stands out for roughly a baseline_window's worth of
    time after it occurs (until the trailing median itself catches up --
    which it eventually does, for either scenario; that's what "trailing"
    means, not "permanent memory"). So the fair comparison is the response
    in the window immediately after each change begins, not the whole
    continuation.
    """
    baseline_window = pd.Timedelta("3h")
    settings = _settings({"short_stopped_duration": True}, baseline_window="3h")

    baseline_cycles = _jittered_baseline(N_BASELINE_CYCLES)
    train_df = _make_cycle_df(baseline_cycles)

    n_continuation = 150
    # Slow 3x downward drift: stopped_n falls linearly 120 -> 40 across the
    # continuation (no injected fault) -- the trailing baseline keeps pace
    # with it throughout, so severity stays low even right after it starts.
    drift_cycles = [
        {**BASE_CYCLE, "stopped_n": round(120 - (i / (n_continuation - 1)) * 80)}
        for i in range(n_continuation)
    ]
    drift_df = _make_cycle_df(baseline_cycles + drift_cycles)

    # Step change: stopped_n drops abruptly from 120 to 40 and STAYS there
    # -- the trailing baseline (still reflecting the old, higher level)
    # lags for about one baseline_window, so severity spikes right away.
    step_cycles = _repeat({**BASE_CYCLE, "stopped_n": 40}, n_continuation)
    step_df = _make_cycle_df(baseline_cycles + step_cycles)

    model = RuleBasedModel(settings)
    model.fit(train_df)

    n_train = len(train_df)
    drift_severity = model.contributions(drift_df)[n_train:, 0]
    step_severity = model.contributions(step_df)[n_train:, 0]

    window_samples = int(baseline_window / pd.Timedelta(seconds=10))
    drift_early = drift_severity[:window_samples].mean()
    step_early = step_severity[:window_samples].mean()

    assert drift_early < 0.3
    assert step_early > 0.4
    assert step_early > drift_early


# --- 9-11. Lagged baseline mechanism (pass 17) -------------------------
#
# These three test the underlying feature function (features/cycles.py
# baseline_relative / baseline_relative_lagged) directly against a plain
# daily pd.Series -- the mechanism pass 16 found broken and pass 17 fixes
# is about the BASELINE ITSELF, independent of the cycle/regime scaffold
# the rest of this file uses. Test 12 below exercises the fix through the
# full RuleBasedModel, where it matters (severity).


def test_lagged_reference_excludes_recent_past():
    """The boiling-frog failure made testable, and its fix: a signal that
    steps down and STAYS down. Trailing mode's baseline catches up (ratio
    back near 1.0) once `window` has elapsed since the step; lagged mode's
    reference is still anchored before the step as long as t - lag hasn't
    reached it yet.
    """
    from apu_sentinel.features.cycles import baseline_relative, baseline_relative_lagged

    index = pd.date_range("2020-01-01", periods=60, freq="1D")
    values = pd.Series([1200.0] * 20 + [300.0] * 40, index=index)  # step down at day 20, stays down

    window = pd.Timedelta("7D")
    lag = pd.Timedelta("14D")

    trailing_ratio = baseline_relative(values, window)
    lagged_ratio = baseline_relative_lagged(values, window, lag)

    # Day 40+ -- 20 days after the step, well past the 7-day window -- the
    # trailing baseline has fully caught up: boiling-frog, ratio near 1.0.
    late_trailing = trailing_ratio.iloc[40:]
    assert late_trailing.mean() == pytest.approx(1.0, abs=0.05)

    # Day 33: t - lag = day 19, still inside the pre-step 1200 regime, so
    # lagged mode's reference is untouched by the step -- strongly abnormal.
    check_at = index[33]
    assert lagged_ratio.loc[check_at] < 0.3


def test_causality_under_lag():
    """Severity (here: the lagged ratio itself, upstream of severity) at
    time t is unchanged when all data after t is deleted -- across several
    cut points, in lagged mode.
    """
    from apu_sentinel.features.cycles import baseline_relative_lagged

    index = pd.date_range("2020-01-01", periods=80, freq="1D")
    rng = np.random.default_rng(0)
    values = pd.Series(1000 + rng.normal(0, 50, size=80), index=index)

    window = pd.Timedelta("7D")
    lag = pd.Timedelta("14D")
    full_ratio = baseline_relative_lagged(values, window, lag)

    for cut in (30, 50, 65, 79):
        truncated_ratio = baseline_relative_lagged(values.iloc[: cut + 1], window, lag)
        assert np.isclose(
            full_ratio.iloc[cut], truncated_ratio.iloc[-1], equal_nan=True
        ), f"cut={cut}"


def test_warm_up_yields_nan_not_partial_baseline():
    """Fewer than lag + window of history -- NaN, never a partial/
    silently-shortened baseline."""
    from apu_sentinel.features.cycles import baseline_relative_lagged

    index = pd.date_range("2020-01-01", periods=40, freq="1D")
    values = pd.Series(1000.0, index=index)

    window = pd.Timedelta("7D")
    lag = pd.Timedelta("14D")
    ratio = baseline_relative_lagged(values, window, lag)

    assert ratio.iloc[:21].isna().all()
    assert ratio.iloc[21:].notna().all()


# --- 12. Mode switch is behavioural (pass 17) -------------------------------


def test_mode_switch_is_behavioural():
    """Same input, two modes: different severities where a sustained shift
    exists (the fix actually changes something), comparable severities for
    a transient spike (neither mode's own window is long enough for a
    single brief blip to move its reference, so they shouldn't diverge
    there). window=4h/lag=8h are chosen so a single ~0.53h injected cycle
    (the spike) is short relative to `window`, while the sustained shift
    (80 cycles, ~42h) comfortably outlasts window+lag=12h.
    """
    baseline_cycles = _jittered_baseline(60)  # long enough to calibrate past warm-up
    train_df = _make_cycle_df(baseline_cycles)

    trailing_settings = _settings(
        {"short_stopped_duration": True}, baseline_window="4h", baseline_mode="trailing"
    )
    lagged_settings = _settings(
        {"short_stopped_duration": True},
        baseline_window="4h",
        baseline_mode="lagged",
        baseline_lag="8h",
    )
    trailing_model = RuleBasedModel(trailing_settings)
    trailing_model.fit(train_df)
    lagged_model = RuleBasedModel(lagged_settings)
    lagged_model.fit(train_df)

    # Sustained shift: stopped_n drops and stays down far longer than window + lag.
    sustained_cycles = _repeat({**BASE_CYCLE, "stopped_n": 40}, 80)
    sustained_df = _make_cycle_df(baseline_cycles + sustained_cycles)

    onset = train_df.index[-1] + pd.Timedelta(seconds=10)
    # > window (4h, trailing has caught up) but < lag (8h, lagged reference
    # is still anchored before onset).
    check_time = onset + pd.Timedelta("6h")
    check_idx = sustained_df.index.get_indexer([check_time], method="nearest")[0]

    trailing_sustained = trailing_model.contributions(sustained_df)[check_idx, 0]
    lagged_sustained = lagged_model.contributions(sustained_df)[check_idx, 0]

    assert (
        lagged_sustained - trailing_sustained > 0.3
    ), f"lagged={lagged_sustained}, trailing={trailing_sustained}"

    # Transient spike: a SINGLE injected cycle (~0.53h, short relative to the
    # 4h window), immediately followed by a return to baseline.
    spike_cycle = {**BASE_CYCLE, "stopped_n": 20, "offload_n": BASE_CYCLE["offload_n"] + 100}
    spike_df = _make_cycle_df(baseline_cycles + [spike_cycle] + baseline_cycles)
    n_train = len(train_df)
    spike_end = n_train + len(_make_cycle_df([spike_cycle]))

    trailing_spike = trailing_model.contributions(spike_df)[n_train:spike_end, 0].mean()
    lagged_spike = lagged_model.contributions(spike_df)[n_train:spike_end, 0].mean()

    assert (
        abs(trailing_spike - lagged_spike) < 0.2
    ), f"trailing={trailing_spike}, lagged={lagged_spike}"
