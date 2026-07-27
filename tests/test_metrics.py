"""Tests for the episode-level evaluation harness (evaluation/metrics.py).

Synthetic scores with planted patterns -- never the real MetroPT-3 dataset.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.data.split import Fold
from apu_sentinel.evaluation.events import window_coverage
from apu_sentinel.evaluation.metrics import (
    ScoredTestData,
    categorise_episode,
    evaluate_fold_at_threshold,
    fit_threshold,
)
from apu_sentinel.explain import rank_channel_contributions

THRESHOLD = 0.5


def _fold(test_start: pd.Timestamp, test_end: pd.Timestamp) -> Fold:
    return Fold(
        event_id=1,
        train_start=test_start - pd.Timedelta(days=1),
        train_end=test_start - pd.Timedelta(hours=1),
        test_start=test_start,
        test_end=test_end,
        train_exclusions=(),
    )


def _scored(
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    channel_names: tuple[str, ...],
    expected_interval: pd.Timedelta | None = None,
    contributions: np.ndarray | None = None,
) -> ScoredTestData:
    if expected_interval is None:
        expected_interval = pd.Timedelta(minutes=10)
    if contributions is None:
        contributions = np.zeros((len(timestamps), len(channel_names)))
    return ScoredTestData(
        timestamps=timestamps,
        scores=scores,
        contributions=contributions,
        channel_names=channel_names,
        expected_interval=expected_interval,
    )


def test_contiguous_above_threshold_run_is_one_episode(metrics_settings, metrics_channel_names):
    timestamps = pd.date_range("2020-01-01 00:00", periods=20, freq="1min")
    scores = np.full(20, 0.1)
    scores[5:10] = 0.9  # one contiguous above-threshold run

    fold = _fold(timestamps[0], timestamps[-1])
    test_data = _scored(timestamps, scores, metrics_channel_names, pd.Timedelta(minutes=1))
    result = evaluate_fold_at_threshold(
        fold,
        metrics_settings.evaluation.failure_events[0],
        6,
        THRESHOLD,
        test_data,
        metrics_settings,
    )

    assert len(result.episodes) == 1
    assert result.episodes[0].start == timestamps[5]
    assert result.episodes[0].end == timestamps[9]


def test_hold_time_hysteresis_merges_short_dip_but_not_long_dip(
    metrics_settings, metrics_channel_names
):
    # hold_time=10min. Burst A: minutes 0-4. Burst B starts after a dip.
    def build(dip_minutes: int) -> pd.DatetimeIndex:
        return pd.date_range("2020-01-01 00:00", periods=10 + dip_minutes, freq="1min")

    fold_and_data = []
    for dip_minutes in (5, 15):
        timestamps = build(dip_minutes)
        scores = np.full(len(timestamps), 0.1)
        scores[0:5] = 0.9
        scores[5 + dip_minutes : 10 + dip_minutes] = 0.9
        fold = _fold(timestamps[0], timestamps[-1])
        test_data = _scored(timestamps, scores, metrics_channel_names, pd.Timedelta(minutes=1))
        fold_and_data.append((fold, test_data))

    short_dip_fold, short_dip_data = fold_and_data[0]
    long_dip_fold, long_dip_data = fold_and_data[1]

    short_result = evaluate_fold_at_threshold(
        short_dip_fold,
        metrics_settings.evaluation.failure_events[0],
        6,
        THRESHOLD,
        short_dip_data,
        metrics_settings,
    )
    long_result = evaluate_fold_at_threshold(
        long_dip_fold,
        metrics_settings.evaluation.failure_events[0],
        6,
        THRESHOLD,
        long_dip_data,
        metrics_settings,
    )

    assert len(short_result.episodes) == 1  # 5min dip < 10min hold_time
    assert len(long_result.episodes) == 2  # 15min dip > 10min hold_time


def test_score_gap_ends_episode_without_bridging(metrics_settings, metrics_channel_names):
    # score_gap_threshold=30min. Burst A at minutes 0-4, then a 40-minute
    # DATA gap (no rows at all), then burst B resumes.
    seg1 = pd.date_range("2020-01-01 00:00", periods=5, freq="1min")
    seg2 = pd.date_range(seg1[-1] + pd.Timedelta(minutes=40), periods=5, freq="1min")
    timestamps = seg1.append(seg2)
    scores = np.full(len(timestamps), 0.9)  # above threshold throughout both bursts

    fold = _fold(timestamps[0], timestamps[-1])
    test_data = _scored(timestamps, scores, metrics_channel_names, pd.Timedelta(minutes=1))
    result = evaluate_fold_at_threshold(
        fold,
        metrics_settings.evaluation.failure_events[0],
        6,
        THRESHOLD,
        test_data,
        metrics_settings,
    )

    # 41min gap > 30min score_gap_threshold (and > 10min hold_time too, but
    # the point of this test is the gap rule, not hold_time) -- must NOT
    # bridge into one episode.
    assert len(result.episodes) == 2
    assert result.episodes[0].end == seg1[-1]
    assert result.episodes[1].start == seg2[0]


def test_episode_in_pre_failure_window_is_detected_with_exact_lead_time(
    metrics_settings, metrics_channel_names
):
    event = metrics_settings.evaluation.failure_events[0]  # start = 2020-01-10 00:00
    timestamps = pd.date_range("2020-01-09 19:00", "2020-01-09 21:00", freq="10min")
    scores = np.full(len(timestamps), 0.1)
    episode_start = pd.Timestamp("2020-01-09 20:00")
    episode_end = pd.Timestamp("2020-01-09 20:10")
    scores[(timestamps >= episode_start) & (timestamps <= episode_end)] = 0.9

    fold = _fold(timestamps[0], timestamps[-1])
    test_data = _scored(timestamps, scores, metrics_channel_names)
    result = evaluate_fold_at_threshold(fold, event, 6, THRESHOLD, test_data, metrics_settings)

    assert result.detected is True
    assert result.lead_time == pd.Timedelta(hours=4)
    assert result.episodes[0].category == "early_warning"
    assert result.episodes[0].matched_event_id == event.id


def test_episode_outside_all_windows_is_false_alarm_not_detected(
    metrics_settings, metrics_channel_names
):
    event = metrics_settings.evaluation.failure_events[0]
    # Five days before the event -- well outside its 6h pre-failure window,
    # concurrent period, and masked settle tail.
    timestamps = pd.date_range("2020-01-05 00:00", periods=10, freq="10min")
    scores = np.full(len(timestamps), 0.9)

    fold = _fold(timestamps[0], timestamps[-1])
    test_data = _scored(timestamps, scores, metrics_channel_names)
    result = evaluate_fold_at_threshold(fold, event, 6, THRESHOLD, test_data, metrics_settings)

    assert result.detected is False
    assert len(result.episodes) == 1
    assert result.episodes[0].category == "false_alarm"
    assert result.episodes[0].matched_event_id is None
    assert result.false_episode_count == 1


def test_episode_during_failure_period_is_concurrent_not_false_alarm(metrics_settings):
    event = metrics_settings.evaluation.failure_events[0]  # [00:00, 04:00)
    masked = ()  # categorise_episode's masked check is only reached if concurrent doesn't match
    category = categorise_episode(
        pd.Timestamp("2020-01-10 02:00"), pd.Timestamp("2020-01-10 02:10"), event, 6, masked
    )
    assert category == "concurrent"


def test_episode_in_masked_settle_tail_is_masked(metrics_settings):
    from apu_sentinel.evaluation.events import masked_regions

    event = metrics_settings.evaluation.failure_events[0]
    masked = masked_regions(metrics_settings)
    # settle tail: (event.end=04:00, maintenance 08:00 + post_settle_hours 2h = 10:00]
    category = categorise_episode(
        pd.Timestamp("2020-01-10 06:00"), pd.Timestamp("2020-01-10 06:10"), event, 6, masked
    )
    assert category == "masked"


def test_false_alarms_per_day_uses_covered_time_not_calendar_days(
    metrics_settings, metrics_channel_names
):
    event = metrics_settings.evaluation.failure_events[0]  # far in the future, irrelevant here
    seg1 = pd.date_range("2020-01-01 00:00", "2020-01-01 12:00", freq="10min")
    seg2 = pd.date_range("2020-01-02 00:00", "2020-01-03 00:00", freq="10min")
    timestamps = seg1.append(seg2)
    scores = np.full(len(timestamps), 0.1)
    scores[5:8] = 0.9  # one false-alarm burst inside seg1

    fold = _fold(pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-03 00:00"))
    test_data = _scored(timestamps, scores, metrics_channel_names)
    result = evaluate_fold_at_threshold(fold, event, 6, THRESHOLD, test_data, metrics_settings)

    assert result.false_episode_count == 1
    # calendar span = 2 days; the 12h gap between seg1 and seg2 is excluded
    # -> covered = 1.5 days.
    assert result.evaluated_days == pytest.approx(1.5)
    naive_calendar_rate = 1 / 2.0
    assert result.false_alarms_per_day == pytest.approx(1 / 1.5)
    assert result.false_alarms_per_day > naive_calendar_rate


def test_channel_ranking_reflects_aggregation_method():
    contributions = np.array([[9.0, 5.0, 1.0], [0.0, 5.0, 1.0]])
    channel_names = ("chan_a", "chan_b", "chan_c")

    mean_ranking = rank_channel_contributions(contributions, channel_names, method="mean")
    assert mean_ranking == (("chan_b", 5.0), ("chan_a", 4.5), ("chan_c", 1.0))

    max_ranking = rank_channel_contributions(contributions, channel_names, method="max")
    assert max_ranking == (("chan_a", 9.0), ("chan_b", 5.0), ("chan_c", 1.0))


def test_fit_threshold_uses_train_scores_only(metrics_settings):
    rng = np.random.default_rng(0)
    train_scores = rng.normal(loc=0.0, scale=1.0, size=1000)
    test_scores = rng.normal(loc=5.0, scale=1.0, size=1000)

    train_threshold = fit_threshold(train_scores, metrics_settings)
    test_threshold = fit_threshold(test_scores, metrics_settings)
    assert train_threshold != pytest.approx(test_threshold)

    # Structural guard: fit_threshold has no way to see test data at all.
    params = list(inspect.signature(fit_threshold).parameters)
    assert params == ["train_scores", "settings"]


def test_coverage_reports_expected_fraction_with_planted_gap():
    window_start = pd.Timestamp("2020-01-01 00:00")
    window_end = pd.Timestamp("2020-01-01 06:00")
    expected_interval = pd.Timedelta(minutes=10)
    # Only the first half of the window has data (18 of the expected 36
    # samples); the second half (03:00-06:00) is entirely missing.
    timestamps = pd.date_range("2020-01-01 00:00", "2020-01-01 02:50", freq="10min")

    coverage = window_coverage(window_start, window_end, timestamps, expected_interval)
    assert coverage == pytest.approx(0.5)
