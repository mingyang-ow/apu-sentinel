"""Tests for the episode-level evaluation harness (evaluation/metrics.py).

Synthetic scores with planted patterns -- never the real MetroPT-3 dataset.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from apu_sentinel.config import EvaluationConfig, FailureEvent, TrainingExclusionConfig
from apu_sentinel.data.split import (
    Fold,
    event_max_width_hours,
    extend_test_end_for_false_alarms,
    make_folds,
)
from apu_sentinel.evaluation.events import NormalStretch, pooled_normal_stretches, window_coverage
from apu_sentinel.evaluation.metrics import (
    Episode,
    FoldEvaluation,
    ScoredTestData,
    categorise_episode,
    evaluate_chance,
    evaluate_fold_at_threshold,
    evaluate_pooled_stretches,
    fit_threshold,
    p_chance_permutation,
    p_chance_poisson,
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


# --- Pass 13, Part A: null (chance) comparison ------------------------------


def test_p_chance_poisson_known_rate_and_width():
    rate_per_day = 0.5
    width_hours = 48.0  # 2 days
    expected = 1.0 - np.exp(-(0.5 * 2.0))
    assert p_chance_poisson(rate_per_day, width_hours) == pytest.approx(expected)


def _episode(start: pd.Timestamp, minutes: int = 10) -> Episode:
    return Episode(
        start=start,
        end=start + pd.Timedelta(minutes=minutes),
        peak_score=0.9,
        category="false_alarm",
        matched_event_id=None,
        channel_ranking=(),
    )


def test_p_chance_permutation_noisy_model_near_one():
    test_start = pd.Timestamp("2020-01-01")
    test_end = pd.Timestamp("2020-01-10")
    # An episode every 2h -- shorter than the 6h window under test, so
    # ANY 6h window placed anywhere spans at least one of them.
    episodes = tuple(_episode(s) for s in pd.date_range(test_start, test_end, freq="2h"))

    p = p_chance_permutation(episodes, test_start, test_end, window_width_hours=6, n_samples=300)
    assert p > 0.95


def test_p_chance_permutation_skilful_model_low_p():
    test_start = pd.Timestamp("2020-01-01")
    test_end = pd.Timestamp("2020-01-31")  # 30 days
    # A single narrow episode right at the very end -- nowhere else.
    episode_start = test_end - pd.Timedelta(hours=3)
    episodes = (_episode(episode_start),)

    p = p_chance_permutation(episodes, test_start, test_end, window_width_hours=6, n_samples=500)
    assert p < 0.05


def test_p_chance_permutation_excludes_placements_whose_window_does_not_fit():
    # 24h test period, 20h window -- only candidates at/after 20:00 have a
    # window that fits inside [test_start, test_end] at all. An episode at
    # 19:30 falls inside EVERY one of those valid windows (their start
    # ranges 00:00-04:00, always before 19:30; their end IS the candidate,
    # ranging 20:00-24:00, always after it) -- so the correct answer is
    # p=1.0 using only the valid subset. If invalid placements were wrongly
    # kept in the denominator (as non-detections), the answer would instead
    # be diluted to (valid count / n_samples) < 1.0.
    test_start = pd.Timestamp("2020-01-01 00:00")
    test_end = pd.Timestamp("2020-01-02 00:00")
    episodes = (_episode(pd.Timestamp("2020-01-01 19:30"), minutes=1),)

    p = p_chance_permutation(episodes, test_start, test_end, window_width_hours=20, n_samples=25)
    assert p == pytest.approx(1.0)


def test_p_chance_permutation_raises_when_no_placement_fits():
    test_start = pd.Timestamp("2020-01-01 00:00")
    test_end = pd.Timestamp("2020-01-01 02:00")  # 2h test period
    with pytest.raises(ValueError, match="no candidate placement"):
        p_chance_permutation((), test_start, test_end, window_width_hours=6, n_samples=10)


def test_chance_flag_set_when_p_exceeds_threshold(metrics_settings):
    fold = SimpleNamespace(
        test_start=pd.Timestamp("2020-01-01"), test_end=pd.Timestamp("2020-01-05")
    )
    episode = _episode(pd.Timestamp("2020-01-03 12:00"))
    result = FoldEvaluation(
        event_id=1,
        window_width=pd.Timedelta(hours=72),
        threshold=0.9,
        detected=True,
        lead_time=pd.Timedelta(hours=10),
        concurrent_only=False,
        episodes=(episode,),
        false_episode_count=2,
        evaluated_days=3.0,
        false_alarms_per_day=2.0 / 3.0,  # -> expected_by_chance = 2 over 72h
        window_coverage=1.0,
    )

    comparison = evaluate_chance(result, fold, metrics_settings)

    assert comparison.p_chance_poisson > metrics_settings.evaluation.chance_threshold
    assert comparison.not_distinguishable_from_chance is True


def test_chance_flag_not_set_when_not_detected(metrics_settings):
    fold = SimpleNamespace(
        test_start=pd.Timestamp("2020-01-01"), test_end=pd.Timestamp("2020-01-05")
    )
    result = FoldEvaluation(
        event_id=1,
        window_width=pd.Timedelta(hours=72),
        threshold=0.9,
        detected=False,
        lead_time=None,
        concurrent_only=False,
        episodes=(),
        false_episode_count=2,
        evaluated_days=3.0,
        false_alarms_per_day=2.0 / 3.0,
        window_coverage=1.0,
    )

    comparison = evaluate_chance(result, fold, metrics_settings)
    assert comparison.not_distinguishable_from_chance is False


# --- Pass 13, Part B: dual false-alarm estimation ---------------------------


def test_extended_test_period_changes_denominators_not_detection(
    metrics_settings, metrics_channel_names
):
    event = metrics_settings.evaluation.failure_events[0]  # start = 2020-01-10 00:00
    orig_end = pd.Timestamp("2020-01-10 04:00")  # matches metrics_event.end
    extended_end = pd.Timestamp("2020-01-20 00:00")  # 10 extra "normal" days

    all_timestamps = pd.date_range("2020-01-09 19:00", extended_end, freq="10min")
    all_scores = np.full(len(all_timestamps), 0.1)

    # planted early_warning burst inside the 6h pre-failure window
    ew_mask = (all_timestamps >= pd.Timestamp("2020-01-09 20:00")) & (
        all_timestamps <= pd.Timestamp("2020-01-09 20:10")
    )
    all_scores[ew_mask] = 0.9

    # planted false alarm ONLY in the extension region, well after orig_end
    fa_mask = (all_timestamps >= pd.Timestamp("2020-01-15 00:00")) & (
        all_timestamps <= pd.Timestamp("2020-01-15 00:10")
    )
    all_scores[fa_mask] = 0.9

    fold_short = Fold(
        event_id=event.id,
        train_start=all_timestamps[0] - pd.Timedelta(days=1),
        train_end=all_timestamps[0] - pd.Timedelta(hours=1),
        test_start=all_timestamps[0],
        test_end=orig_end,
        train_exclusions=(),
    )
    fold_extended = Fold(
        event_id=event.id,
        train_start=fold_short.train_start,
        train_end=fold_short.train_end,
        test_start=fold_short.test_start,
        test_end=extended_end,
        train_exclusions=(),
    )

    short_mask = all_timestamps <= orig_end
    extended_mask = all_timestamps <= extended_end

    def _test_data(mask):
        n = int(mask.sum())
        return ScoredTestData(
            timestamps=all_timestamps[mask],
            scores=all_scores[mask],
            contributions=np.zeros((n, len(metrics_channel_names))),
            channel_names=metrics_channel_names,
            expected_interval=pd.Timedelta(minutes=10),
        )

    result_short = evaluate_fold_at_threshold(
        fold_short, event, 6, THRESHOLD, _test_data(short_mask), metrics_settings
    )
    result_extended = evaluate_fold_at_threshold(
        fold_extended, event, 6, THRESHOLD, _test_data(extended_mask), metrics_settings
    )

    assert result_short.detected is True
    assert result_extended.detected is True
    assert result_short.lead_time == result_extended.lead_time

    assert result_extended.false_episode_count > result_short.false_episode_count
    assert result_extended.evaluated_days > result_short.evaluated_days


def test_extend_test_end_never_reaches_into_next_exclusion(
    synthetic_split_events, synthetic_training_exclusion, synthetic_split_data_bounds
):
    _data_start, data_end = synthetic_split_data_bounds
    events_sorted = sorted(synthetic_split_events, key=lambda e: pd.Timestamp(e.start))
    event1, event2, _event3 = events_sorted

    fold1 = Fold(
        event_id=event1.id,
        train_start=pd.Timestamp("2019-12-01"),
        train_end=pd.Timestamp("2020-01-01"),
        test_start=pd.Timestamp("2020-01-01 12:00"),
        test_end=pd.Timestamp(event1.end),
        train_exclusions=(),
    )

    extended = extend_test_end_for_false_alarms(
        fold1, event1, events_sorted, synthetic_training_exclusion, data_end
    )

    # event2 has no maintenance override in this fixture's pre_margin --
    # its exclusion START is exactly what the extension must stop at.
    next_excl_start = pd.Timestamp(event2.start) - pd.Timedelta(
        hours=synthetic_training_exclusion.pre_margin_hours
    )
    assert extended.test_end == next_excl_start
    assert extended.test_end <= next_excl_start

    # every other field is untouched
    assert extended.event_id == fold1.event_id
    assert extended.train_start == fold1.train_start
    assert extended.train_end == fold1.train_end
    assert extended.test_start == fold1.test_start


def test_extend_test_end_uses_data_end_for_final_event(
    synthetic_split_events, synthetic_training_exclusion, synthetic_split_data_bounds
):
    _data_start, data_end = synthetic_split_data_bounds
    events_sorted = sorted(synthetic_split_events, key=lambda e: pd.Timestamp(e.start))
    last_event = events_sorted[-1]

    fold_last = Fold(
        event_id=last_event.id,
        train_start=pd.Timestamp("2019-12-01"),
        train_end=pd.Timestamp("2020-01-20"),
        test_start=pd.Timestamp("2020-01-25"),
        test_end=pd.Timestamp(last_event.end),
        train_exclusions=(),
    )

    extended = extend_test_end_for_false_alarms(
        fold_last, last_event, events_sorted, synthetic_training_exclusion, data_end
    )
    assert extended.test_end == pd.Timestamp(data_end)


def test_pooled_normal_stretches_exclude_events_failures_exclusions_and_buffer():
    events = [
        FailureEvent(
            id=1, start="2020-01-10 00:00", end="2020-01-10 04:00", maintenance="2020-01-10 08:00"
        )
    ]
    training_exclusion = TrainingExclusionConfig(
        pre_margin_hours=1, post_settle_hours=2, fallback_post_hours=4
    )
    evaluation = EvaluationConfig(window_widths=[6], failure_events=events, pooled_buffer_hours=1)
    settings = SimpleNamespace(
        evaluation=evaluation, split=SimpleNamespace(training_exclusion=training_exclusion)
    )

    stretches = pooled_normal_stretches(
        settings, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-20")
    )

    # pre-failure window (widest=6h) starts 2020-01-09 18:00, minus 1h buffer.
    assert stretches[0] == NormalStretch(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-09 17:00")
    )
    # masked region ends at maintenance(08:00) + post_settle(2h) = 10:00, plus 1h buffer.
    assert stretches[1] == NormalStretch(
        pd.Timestamp("2020-01-10 11:00"), pd.Timestamp("2020-01-20")
    )


def test_evaluate_pooled_stretches_denominator_sums_stretches_minus_gaps(metrics_settings):
    stretch_a = NormalStretch(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))  # 2 days
    stretch_b = NormalStretch(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-02"))  # 1 day

    # 10min cadence: finer than this fixture's 30min score_gap_threshold,
    # so normal sampling itself is never mistaken for a gap.
    ts_a = pd.date_range(stretch_a.start, stretch_a.end, freq="10min")
    scores_a = np.full(len(ts_a), 0.1)

    # stretch_b has a native 6h gap in the middle -- must be subtracted
    # from its own evaluated_days (score_gap_threshold=30min in this fixture).
    ts_b = pd.date_range(
        stretch_b.start, stretch_b.start + pd.Timedelta(hours=6), freq="10min"
    ).append(pd.date_range(stretch_b.start + pd.Timedelta(hours=12), stretch_b.end, freq="10min"))
    scores_b = np.full(len(ts_b), 0.1)
    scores_b[0] = 0.99  # one planted false-alarm episode in stretch_b

    def _data(ts, scores):
        return ScoredTestData(
            timestamps=ts,
            scores=scores,
            contributions=np.zeros((len(ts), 1)),
            channel_names=("chan_a",),
            expected_interval=pd.Timedelta(minutes=10),
        )

    result = evaluate_pooled_stretches(
        (stretch_a, stretch_b),
        [_data(ts_a, scores_a), _data(ts_b, scores_b)],
        threshold=0.5,
        settings=metrics_settings,
    )

    expected_days = 2.0 + (1.0 - 6.0 / 24.0)
    assert result.evaluated_days == pytest.approx(expected_days)
    assert result.false_episode_count == 1
    assert result.false_alarms_per_day == pytest.approx(1 / expected_days)


def test_evaluate_pooled_stretches_rejects_misaligned_scored_stretches(metrics_settings):
    stretch = NormalStretch(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    with pytest.raises(ValueError):
        evaluate_pooled_stretches((stretch,), [], threshold=0.5, settings=metrics_settings)


# --- Pass 13, Part C: per-event window width caps ---------------------------


def test_event_max_width_hours_tightest_for_event3_and_feedable_to_make_folds(
    synthetic_split_settings, synthetic_split_data_bounds
):
    data_start, data_end = synthetic_split_data_bounds
    widths = event_max_width_hours(synthetic_split_settings, data_start)

    events_by_id = {e.id: e for e in synthetic_split_settings.evaluation.failure_events}
    assert set(widths) == set(events_by_id)

    # event 3 sits right after event 2's exclusion region -- materially
    # tighter than events 1 and 2, which have much more room.
    assert widths[3] < widths[1]
    assert widths[3] < widths[2]

    # Each event's own derived maximum, fed straight back into
    # make_folds(), must be exactly feasible -- not raise.
    folds = make_folds(synthetic_split_settings, data_start, data_end, width_hours_by_event=widths)
    assert len(folds) == 3


def test_width_exceeding_event_max_raises(synthetic_split_settings, synthetic_split_data_bounds):
    data_start, data_end = synthetic_split_data_bounds
    widths = event_max_width_hours(synthetic_split_settings, data_start)

    over_budget = dict(widths)
    over_budget[3] = widths[3] + 1.0

    with pytest.raises(ValueError):
        make_folds(synthetic_split_settings, data_start, data_end, width_hours_by_event=over_budget)
