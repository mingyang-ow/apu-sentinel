"""Tests for the Isolation Forest model (models/isolation_forest.py).

Small synthetic window tensors only -- never the real MetroPT-3 dataset.
Each test isolates one property of the AnomalyModel contract or the
ablation-attribution mechanism, following tests/test_rule_based.py's
existing pattern for this project.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from apu_sentinel.config import IsolationForestContributionsConfig, IsolationForestModelConfig
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.isolation_forest import IsolationForestModel, WindowedInput

N_CHANNELS = 3
WINDOW_LENGTH = 10
CHANNEL_NAMES = ("TP2", "TP3", "Reservoirs")


def _settings(include_cycle_features: bool = False, contributions_enabled: bool = True, **kwargs):
    cfg = IsolationForestModelConfig(
        include_cycle_features=include_cycle_features,
        contributions=IsolationForestContributionsConfig(enabled=contributions_enabled),
        **kwargs,
    )
    return SimpleNamespace(model=SimpleNamespace(isolation_forest=cfg))


def _windows(n_windows: int, seed: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=(n_windows, WINDOW_LENGTH, N_CHANNELS))


def _end_timestamps(n_windows: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_windows, freq="1min")


def _data(windows: np.ndarray, cycle_features: pd.DataFrame | None = None) -> WindowedInput:
    return WindowedInput(
        windows=windows,
        end_timestamps=_end_timestamps(windows.shape[0]),
        channel_names=CHANNEL_NAMES,
        cycle_features=cycle_features,
    )


def _cycle_features_for(end_timestamps: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"stopped_duration_last": rng.normal(size=len(end_timestamps))}, index=end_timestamps
    )


# --- 1. Protocol conformance -------------------------------------------


def test_protocol_conformance():
    model = IsolationForestModel(_settings(include_cycle_features=True))
    windows = _windows(200, seed=0)
    data = _data(windows, cycle_features=_cycle_features_for(_end_timestamps(200)))

    model.fit(data)
    assert isinstance(model, AnomalyModel)

    scores = model.score(data)
    contributions = model.contributions(data)

    assert scores.shape == (200,)
    assert contributions.shape == (200, len(model.contributor_names))
    assert (
        len(model.contributor_names) == N_CHANNELS * 5 + 1
    )  # 5 default window stats + 1 cycle feature


# --- 2. Score direction --------------------------------------------------


def test_score_direction_planted_outlier_scores_higher():
    model = IsolationForestModel(_settings())
    train_windows = _windows(400, seed=1)
    model.fit(_data(train_windows))

    normal_windows = _windows(50, seed=2)
    outlier_windows = normal_windows.copy()
    outlier_windows[0] += 50.0  # a single, sharply displaced window

    scores = model.score(_data(outlier_windows))
    assert scores[0] > np.median(scores[1:])


# --- 3. Fit is train-only -------------------------------------------------


def test_fit_is_train_only():
    settings = _settings()
    train_windows = _windows(300, seed=3)

    contaminating_anomaly = _windows(20, seed=3, loc=100.0, scale=1.0)
    train_plus_test_windows = np.concatenate([train_windows, contaminating_anomaly], axis=0)

    clean_model = IsolationForestModel(settings)
    clean_model.fit(_data(train_windows))

    contaminated_model = IsolationForestModel(settings)
    contaminated_model.fit(
        _data(train_plus_test_windows)
    )  # fit sees the "test" anomaly -- contamination

    probe = _windows(50, seed=4)
    clean_scores = clean_model.score(_data(probe))
    contaminated_scores = contaminated_model.score(_data(probe))

    assert not np.allclose(clean_scores, contaminated_scores)


# --- 4. Per-fold models are distinct ---------------------------------------


def test_per_fold_models_are_distinct():
    """Each fold gets its OWN IsolationForestModel instance, fit on that
    fold's own training data -- no state shared across instances. Two
    models fit on genuinely different training distributions must score
    the same held-out data differently.
    """
    settings = _settings()
    fold_a_windows = _windows(300, seed=10, loc=0.0, scale=1.0)
    fold_b_windows = _windows(300, seed=20, loc=5.0, scale=2.0)

    model_a = IsolationForestModel(settings)
    model_a.fit(_data(fold_a_windows))
    model_b = IsolationForestModel(settings)
    model_b.fit(_data(fold_b_windows))

    probe = _windows(50, seed=30)
    scores_a = model_a.score(_data(probe))
    scores_b = model_b.score(_data(probe))

    assert not np.allclose(scores_a, scores_b)


# --- 5. Determinism ----------------------------------------------------


def test_determinism_same_seed_same_scores_different_seed_different():
    train_windows = _windows(300, seed=5)
    probe = _windows(50, seed=6)

    model_1 = IsolationForestModel(_settings(random_state=42))
    model_1.fit(_data(train_windows))
    model_2 = IsolationForestModel(_settings(random_state=42))
    model_2.fit(_data(train_windows))

    scores_1 = model_1.score(_data(probe))
    scores_2 = model_2.score(_data(probe))
    assert np.array_equal(scores_1, scores_2)

    model_3 = IsolationForestModel(_settings(random_state=7))
    model_3.fit(_data(train_windows))
    scores_3 = model_3.score(_data(probe))
    assert not np.array_equal(scores_1, scores_3)


# --- 6. Ablation attribution ------------------------------------------


def test_ablation_attribution_ranks_planted_feature_top():
    settings = _settings()
    train_windows = _windows(400, seed=8)
    model = IsolationForestModel(settings)
    model.fit(_data(train_windows))

    probe = _windows(30, seed=9)
    probe[0, :, 1] += 40.0  # channel index 1 ("TP3") only, row 0 only

    contributions = model.contributions(_data(probe))
    top_feature = model.contributor_names[np.argmax(contributions[0])]
    assert top_feature.startswith("TP3_")


# --- 7. Contributions disabled ---------------------------------------------


def test_contributions_disabled_returns_zeros(caplog):
    settings = _settings(contributions_enabled=False)
    model = IsolationForestModel(settings)
    train_windows = _windows(200, seed=11)
    model.fit(_data(train_windows))

    probe = _windows(20, seed=12)
    with caplog.at_level("INFO"):
        contributions = model.contributions(_data(probe))

    assert contributions.shape == (20, len(model.contributor_names))
    assert np.all(contributions == 0.0)
    assert any("disabled" in message for message in caplog.messages)


# --- 8. Additional regions excluded, union-merged with event regions ------


def test_additional_regions_excluded_and_union_merged():
    from apu_sentinel.config import (
        EvaluationConfig,
        FailureEvent,
        SplitConfig,
        TrainingExclusionConfig,
    )
    from apu_sentinel.data.split import apply_fold, make_folds

    events = [
        FailureEvent(
            id=1, start="2020-01-05 00:00", end="2020-01-05 04:00", maintenance="2020-01-05 08:00"
        ),
        FailureEvent(
            id=2, start="2020-02-01 00:00", end="2020-02-01 04:00", maintenance="2020-02-01 08:00"
        ),
    ]
    # additional_region overlaps event 1's own exclusion window (pre_margin=6h
    # -> excl starts 2020-01-04 18:00; region runs 2020-01-04 12:00 -> 2020-01-04 20:00).
    training_exclusion = TrainingExclusionConfig(
        pre_margin_hours=6,
        post_settle_hours=6,
        fallback_post_hours=6,
        additional_regions=[
            {"start": "2020-01-04 12:00", "end": "2020-01-04 20:00", "reason": "test region"}
        ],
    )
    settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=2, training_exclusion=training_exclusion),
        evaluation=EvaluationConfig(window_widths=[4], failure_events=events),
    )

    data_start = pd.Timestamp("2020-01-01 00:00")
    data_end = pd.Timestamp("2020-03-01 00:00")
    folds = make_folds(settings, data_start, data_end)
    fold2 = next(f for f in folds if f.event_id == 2)

    # event 1's own exclusion: [2020-01-04 18:00, 2020-01-05 14:00)
    # additional region:        [2020-01-04 12:00, 2020-01-04 20:00)
    # union-merged (overlapping): [2020-01-04 12:00, 2020-01-05 14:00)
    assert len(fold2.train_exclusions) == 1
    merged_start, merged_end = fold2.train_exclusions[0]
    assert merged_start == pd.Timestamp("2020-01-04 12:00")
    assert merged_end == pd.Timestamp("2020-01-05 14:00")

    index = pd.date_range(data_start, data_end, freq="1h")
    df = pd.DataFrame({"channel_0": np.zeros(len(index))}, index=index)
    train, _test = apply_fold(df, fold2)
    in_region = (train.index >= pd.Timestamp("2020-01-04 12:00")) & (
        train.index < pd.Timestamp("2020-01-04 20:00")
    )
    assert not in_region.any()
