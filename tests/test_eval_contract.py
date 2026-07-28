"""Every model implements the AnomalyModel contract (models/base.py), and
the evaluation harness (evaluation/metrics.py) evaluates a model conforming
to that contract end-to-end -- proving the contract is actually usable
before any real model is written.

Uses a mock model with planted values -- never the real dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.data.split import Fold
from apu_sentinel.evaluation.metrics import FoldEvaluation, ScoredTestData, evaluate_fold
from apu_sentinel.models.autoencoder import AutoencoderModel
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.isolation_forest import IsolationForestModel
from apu_sentinel.models.rule_based import RuleBasedModel

MODEL_CLASSES = [RuleBasedModel, IsolationForestModel, AutoencoderModel]


class MockAnomalyModel:
    """Implements AnomalyModel with planted, deterministic outputs --
    stands in for a real model so the harness can be exercised end-to-end
    before one exists.
    """

    def __init__(
        self,
        planted_scores: np.ndarray,
        planted_contributions: np.ndarray,
        contributor_names: tuple[str, ...] = (),
    ):
        self._planted_scores = planted_scores
        self._planted_contributions = planted_contributions
        self._contributor_names = contributor_names
        self.fitted_on = None

    @property
    def contributor_names(self) -> tuple[str, ...]:
        return self._contributor_names

    def fit(self, train_data) -> None:
        self.fitted_on = train_data

    def score(self, data) -> np.ndarray:
        assert len(data) == len(self._planted_scores)
        return self._planted_scores

    def contributions(self, data) -> np.ndarray:
        assert len(data) == len(self._planted_contributions)
        return self._planted_contributions


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_model_implements_contract(model_cls):
    assert isinstance(model_cls(), AnomalyModel)


def test_mock_model_implements_contract():
    assert isinstance(MockAnomalyModel(np.array([0.0]), np.zeros((1, 1))), AnomalyModel)


def test_evaluator_detects_planted_failure(metrics_settings, metrics_channel_names):
    event = metrics_settings.evaluation.failure_events[0]  # start = 2020-01-10 00:00

    timestamps = pd.date_range("2020-01-09 19:00", "2020-01-09 21:00", freq="10min")
    n_channels = len(metrics_channel_names)
    n = len(timestamps)

    scores = np.full(n, 0.1)
    episode_mask = (timestamps >= pd.Timestamp("2020-01-09 20:00")) & (
        timestamps <= pd.Timestamp("2020-01-09 20:10")
    )
    scores[episode_mask] = 0.9

    contributions = np.zeros((n, n_channels))
    contributions[episode_mask, 0] = 5.0  # chan_a dominates the planted episode

    model = MockAnomalyModel(scores, contributions, contributor_names=metrics_channel_names)

    train_data = np.zeros((100, n_channels))
    model.fit(train_data)
    assert model.fitted_on is train_data

    test_data_tensor = np.zeros((n, n_channels))
    model_scores = model.score(test_data_tensor)
    model_contributions = model.contributions(test_data_tensor)

    fold = Fold(
        event_id=event.id,
        train_start=timestamps[0] - pd.Timedelta(days=1),
        train_end=timestamps[0] - pd.Timedelta(hours=1),
        test_start=timestamps[0],
        test_end=timestamps[-1],
        train_exclusions=(),
    )
    test_data = ScoredTestData(
        timestamps=timestamps,
        scores=model_scores,
        contributions=model_contributions,
        channel_names=model.contributor_names,
        expected_interval=pd.Timedelta(minutes=10),
    )
    # A threshold-fitting train score distribution that separates the 0.1
    # baseline from the 0.9 planted spike (99.5th percentile of mostly-0.1
    # with one 0.9 outlier lands strictly between them).
    train_scores = np.array([0.1] * 99 + [0.9])

    result = evaluate_fold(fold, event, 6, train_scores, test_data, metrics_settings)

    assert isinstance(result, FoldEvaluation)
    assert 0.1 < result.threshold < 0.9
    assert result.detected is True
    assert result.lead_time == pd.Timedelta(hours=4)
    assert len(result.episodes) == 1
    assert result.episodes[0].category == "early_warning"
    assert result.episodes[0].matched_event_id == event.id
    assert result.episodes[0].channel_ranking[0][0] == "chan_a"
    assert result.false_episode_count == 0
