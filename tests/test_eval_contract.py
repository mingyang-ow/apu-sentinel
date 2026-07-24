"""Every model implements the AnomalyModel contract (models/base.py), and
the evaluator produces correct episode-level detection on a planted
synthetic failure.
"""

from __future__ import annotations

import pytest

from apu_sentinel.models.autoencoder import AutoencoderModel
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.isolation_forest import IsolationForestModel
from apu_sentinel.models.rule_based import RuleBasedModel

MODEL_CLASSES = [RuleBasedModel, IsolationForestModel, AutoencoderModel]


@pytest.mark.parametrize("model_cls", MODEL_CLASSES)
def test_model_implements_contract(model_cls):
    assert isinstance(model_cls(), AnomalyModel)


def test_evaluator_detects_planted_failure(synthetic_series, synthetic_failure_events):
    pytest.skip("stub — implement with logic")
