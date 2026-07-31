"""Pass 24 (docs/RESULTS.md §23/§24): select_operating_quantile must report
"no swept quantile meets the ceiling" as a result, not a raised ValueError
that discards a completed training run -- this is exactly what happened on
the real Colab autoencoder run. Small synthetic inputs only.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import apu_sentinel.pipeline as pipeline_module
from apu_sentinel.config import FailureEvent
from apu_sentinel.data.split import Fold
from apu_sentinel.evaluation.metrics import PooledEvaluation
from apu_sentinel.models.isolation_forest import WindowedInput
from apu_sentinel.pipeline import OperatingPointSelection, select_operating_quantile


def _pooled(fa_per_day: float) -> PooledEvaluation:
    return PooledEvaluation(
        false_episode_count=1, evaluated_days=1.0, false_alarms_per_day=fa_per_day
    )


# --- select_operating_quantile -----------------------------------------


def test_select_operating_quantile_returns_not_found_when_no_quantile_meets_ceiling():
    pooled_by_quantile_per_fold = {
        1: {0.99: _pooled(5.0), 0.999: _pooled(2.0)},
        2: {0.99: _pooled(4.0), 0.999: _pooled(1.5)},
    }
    selection = select_operating_quantile(pooled_by_quantile_per_fold, ceiling=0.3)

    assert selection == OperatingPointSelection(
        found=False, quantile=None, pooled_fa_per_day_by_quantile={0.99: 5.0, 0.999: 2.0}
    )


def test_select_operating_quantile_returns_found_when_a_quantile_meets_ceiling():
    pooled_by_quantile_per_fold = {
        1: {0.99: _pooled(5.0), 0.999: _pooled(0.1)},
        2: {0.99: _pooled(4.0), 0.999: _pooled(0.2)},
    }
    selection = select_operating_quantile(pooled_by_quantile_per_fold, ceiling=0.3)

    assert selection == OperatingPointSelection(
        found=True, quantile=0.999, pooled_fa_per_day_by_quantile={0.99: 5.0, 0.999: 0.2}
    )


# --- _score_distribution_summary ----------------------------------------


def test_score_distribution_summary_computes_train_and_test_stats():
    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # p99 dominated by the outlier
    test_scores = np.array([1.0, 2.0, 200.0])

    summary = pipeline_module._score_distribution_summary(train_scores, test_scores)

    assert summary["train_median"] == 3.0
    assert summary["train_max"] == 100.0
    assert summary["test_median"] == 2.0
    assert summary["test_max"] == 200.0
    # only the 200.0 test score exceeds train's p99.
    assert summary["frac_test_above_train_p99"] == pytest.approx(1 / 3)


# --- run_pipeline_autoencoder: no-operating-point branch -----------------


def _fake_run_pipeline_autoencoder_dependencies(monkeypatch, folds, fold_inputs):
    """Stub every helper run_pipeline_autoencoder calls except the
    function's own control flow (the fold loop, per-fold bookkeeping,
    branching on OperatingPointSelection.found) -- this test is about that
    orchestration, not about regime assignment, scaling, or windowing.
    """

    class _FakeModel:
        epochs_run_ = 2
        final_train_loss_ = 0.1
        final_val_loss_ = 0.2
        elapsed_seconds_ = 0.01
        device_ = "cpu"
        contributor_names = ("chan_a",)

        def score(self, data):
            return np.linspace(0.1, 1.0, len(data.end_timestamps))

        def contributions(self, data):
            return np.zeros((len(data.end_timestamps), 1))

    monkeypatch.setattr(pipeline_module, "load_raw", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        pipeline_module, "assign_regimes", lambda df, settings: pd.Series(dtype=object)
    )
    monkeypatch.setattr(pipeline_module, "make_folds", lambda settings, start, end: folds)
    monkeypatch.setattr(
        pipeline_module,
        "characterise_sampling",
        lambda df, threshold: SimpleNamespace(modal_interval=pd.Timedelta("10s")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "extend_test_end_for_false_alarms",
        lambda fold, event, events_sorted, training_exclusion, data_end: fold,
    )
    monkeypatch.setattr(pipeline_module, "pooled_normal_stretches", lambda settings, start, end: ())
    monkeypatch.setattr(
        pipeline_module,
        "_fit_fold_autoencoder",
        lambda df, regimes, fold, settings: (
            _FakeModel(),
            {},
            fold_inputs[fold.event_id][0],
            fold_inputs[fold.event_id][1],
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "pooled_at_quantiles_autoencoder",
        lambda *a, **k: {0.99: _pooled(5.0)},
    )


def _fold_and_inputs(
    event_id: int, day_offset: int
) -> tuple[Fold, tuple[WindowedInput, WindowedInput]]:
    base = pd.Timestamp("2020-01-01") + pd.Timedelta(days=day_offset)
    end_timestamps = pd.date_range(base, periods=5, freq="1D")
    fold = Fold(
        event_id=event_id,
        train_start=end_timestamps[0],
        train_end=end_timestamps[2],
        test_start=end_timestamps[3],
        test_end=end_timestamps[4],
        train_exclusions=(),
    )
    train_input = WindowedInput(
        windows=np.zeros((3, 1, 1), dtype=np.float32),
        end_timestamps=end_timestamps[:3],
        channel_names=("chan_a",),
        cycle_features=None,
    )
    fold_input = WindowedInput(
        windows=np.zeros((5, 1, 1), dtype=np.float32),
        end_timestamps=end_timestamps,
        channel_names=("chan_a",),
        cycle_features=None,
    )
    return fold, (train_input, fold_input)


def test_run_pipeline_autoencoder_completes_without_operating_point(monkeypatch, caplog):
    """No swept quantile meeting the ceiling must produce a result -- not
    a raised ValueError -- carrying the per-quantile rates and, per fold,
    the training summary and score-distribution summary. Detection is
    never evaluated in this branch (no chosen threshold to evaluate it at).
    """
    fold1, inputs1 = _fold_and_inputs(1, day_offset=0)
    fold2, inputs2 = _fold_and_inputs(2, day_offset=30)
    _fake_run_pipeline_autoencoder_dependencies(
        monkeypatch, [fold1, fold2], {1: inputs1, 2: inputs2}
    )
    monkeypatch.setattr(
        pipeline_module,
        "select_operating_quantile",
        lambda pooled_by_quantile_per_fold, ceiling: OperatingPointSelection(
            found=False, quantile=None, pooled_fa_per_day_by_quantile={0.99: 5.0}
        ),
    )

    events = [
        FailureEvent(id=1, start="2020-01-04", end="2020-01-04 04:00"),
        FailureEvent(id=2, start="2020-01-31", end="2020-01-31 04:00"),
    ]
    settings = SimpleNamespace(
        model=SimpleNamespace(autoencoder=object()),
        data=SimpleNamespace(raw_dir=".", raw_filename="unused.csv"),
        windowing=SimpleNamespace(gap_threshold="5min"),
        evaluation=SimpleNamespace(failure_events=events),
        split=SimpleNamespace(training_exclusion=object()),
    )

    with caplog.at_level(logging.INFO, logger=pipeline_module.logger.name):
        result = pipeline_module.run_pipeline_autoencoder(settings)

    assert result["operating_point"]["found"] is False
    assert result["operating_point"]["pooled_fa_per_day_by_quantile"] == {0.99: 5.0}
    assert set(result["folds"].keys()) == {1, 2}
    for fold_id in (1, 2):
        fold_result = result["folds"][fold_id]
        assert set(fold_result.keys()) == {"training_summary", "score_distribution"}
        assert fold_result["training_summary"]["epochs_run"] == 2
        assert not np.isnan(fold_result["score_distribution"]["test_median"])

    dist_messages = [r.message for r in caplog.records if "score distribution" in r.message]
    assert len(dist_messages) == 2  # emitted for every fold, not only on failure
