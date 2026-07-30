"""Crown-jewel guard: no leakage across the walk-forward split.

Runs as a blocking Claude Code hook on every edit to
src/apu_sentinel/data/ (.claude/hooks/check_leakage.sh) and again in the
full pytest suite (Tier 3). The real functions are imported at module level
so a half-finished refactor fails cleanly at collection time, rather than
tripping on transient intermediate file states.

Uses only small synthetic fixtures (tests/conftest.py) -- never the real
MetroPT-3 dataset.
"""

from __future__ import annotations

import inspect
from itertools import pairwise
from types import SimpleNamespace

import pandas as pd
import pytest

from apu_sentinel.config import EvaluationConfig, SplitConfig
from apu_sentinel.data import split as split_module
from apu_sentinel.data.split import (
    Fold,
    apply_fold,
    event_max_width_hours,
    extend_test_end_for_false_alarms,
    make_folds,
    training_days_remaining,
)


def test_causality_train_end_before_test_start_and_no_train_row_in_test_period(
    synthetic_split_settings, synthetic_split_data_bounds, synthetic_split_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_split_settings, data_start, data_end)

    assert len(folds) == 3
    for fold in folds:
        assert fold.train_end < fold.test_start

        train, _test = apply_fold(synthetic_split_df, fold)
        assert train.index.max() < fold.test_start
        assert (train.index < fold.test_start).all()


def test_embargo_respected(synthetic_split_settings, synthetic_split_data_bounds):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_split_settings, data_start, data_end)

    embargo = pd.Timedelta(hours=synthetic_split_settings.split.embargo_hours)
    for fold in folds:
        assert fold.test_start - fold.train_end >= embargo


def test_expanding_window(synthetic_split_settings, synthetic_split_data_bounds):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_split_settings, data_start, data_end)

    for fold in folds:
        assert fold.train_start == data_start

    for earlier, later in pairwise(folds):
        assert later.train_end > earlier.train_end


def test_exclusions_removed_from_train(
    synthetic_split_settings, synthetic_split_data_bounds, synthetic_split_df
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_split_settings, data_start, data_end)

    # Fold for event 3 is the expanding-window tail -- both event 1's and
    # event 2's exclusion regions should have been purged from its train set.
    fold3 = next(f for f in folds if f.event_id == 3)
    assert len(fold3.train_exclusions) == 2

    train, _test = apply_fold(synthetic_split_df, fold3)
    for excl_start, excl_end in fold3.train_exclusions:
        in_excluded_region = (train.index >= excl_start) & (train.index < excl_end)
        assert not in_excluded_region.any()


def test_test_period_contains_its_event_and_pre_failure_window_fits(
    synthetic_split_settings, synthetic_split_data_bounds
):
    data_start, data_end = synthetic_split_data_bounds
    folds = make_folds(synthetic_split_settings, data_start, data_end)
    events_by_id = {e.id: e for e in synthetic_split_settings.evaluation.failure_events}

    for fold in folds:
        event = events_by_id[fold.event_id]
        event_start = pd.Timestamp(event.start)
        event_end = pd.Timestamp(event.end)
        assert fold.test_start <= event_start
        assert event_end <= fold.test_end

        for width in synthetic_split_settings.evaluation.window_widths:
            label_start = event_start - pd.Timedelta(hours=width)
            assert label_start >= fold.test_start


def test_no_shuffle_parameter_and_rejects_non_time_strategy(synthetic_split_df):
    make_folds_params = inspect.signature(make_folds).parameters
    apply_fold_params = inspect.signature(apply_fold).parameters
    assert "shuffle" not in make_folds_params
    assert "shuffle" not in apply_fold_params
    assert "random_state" not in apply_fold_params

    fold = Fold(
        event_id=1,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-02"),
        test_start=pd.Timestamp("2020-01-03"),
        test_end=pd.Timestamp("2020-01-04"),
        train_exclusions=(),
    )
    with pytest.raises(ValueError):
        apply_fold(synthetic_split_df, fold, strategy="shuffle")


def test_train_test_split_not_used_in_module():
    source = inspect.getsource(split_module)
    assert "train_test_split" not in source
    assert "sklearn" not in source


def test_overlap_detection_raises_naming_event_2_and_3(
    synthetic_split_settings_overlapping, synthetic_split_data_bounds
):
    data_start, data_end = synthetic_split_data_bounds
    with pytest.raises(ValueError, match=r"event 3.*event 2|event 2.*event 3"):
        make_folds(synthetic_split_settings_overlapping, data_start, data_end)


def test_test_start_overlap_detected_even_when_window_width_check_passes(
    synthetic_split_settings_test_start_overlap, synthetic_split_data_bounds
):
    """The gap found in the split pass: a width can pass the plain
    window-width overlap check (label_start lands just after the earlier
    event's exclusion ends) while the fold's actual test_start -- which
    backs off by an EXTRA embargo_hours on top of that width -- still opens
    before the earlier event's exclusion region has ended. This must be
    caught too, not just the label-window case.
    """
    data_start, data_end = synthetic_split_data_bounds
    with pytest.raises(ValueError, match="test_start"):
        make_folds(synthetic_split_settings_test_start_overlap, data_start, data_end)


# --- Pass 18: training-exclusion margin sweep -----------------------------


def test_overlapping_exclusions_merge_into_one_region(
    synthetic_split_settings_overlapping_exclusions,
    synthetic_split_data_bounds_wide,
    synthetic_split_df_wide,
):
    """Event 1's and event 2's own exclusion windows overlap each other
    (see the fixture's docstring) -- event 3's fold must see them merged
    into a SINGLE disjoint region, not two overlapping tuples, and no
    training timestamp may fall inside it. Pass 20: exclusions are
    selected by overlap with the training span, not by event identity, so
    fold 3 ALSO excludes its own event's precursor (clipped to train_end)
    as a second, separate region -- two disjoint exclusions in total, not
    one.
    """
    data_start, data_end = synthetic_split_data_bounds_wide
    folds = make_folds(synthetic_split_settings_overlapping_exclusions, data_start, data_end)
    fold3 = next(f for f in folds if f.event_id == 3)

    assert len(fold3.train_exclusions) == 2
    merged_start, merged_end = fold3.train_exclusions[0]
    assert merged_start == pd.Timestamp("2020-01-03 18:00")
    assert merged_end == pd.Timestamp("2020-01-14 10:00")

    own_precursor_start, own_precursor_end = fold3.train_exclusions[1]
    assert own_precursor_start == pd.Timestamp("2020-02-28 18:00")
    assert own_precursor_end == fold3.train_end

    train, _test = apply_fold(synthetic_split_df_wide, fold3)
    for excl_start, excl_end in fold3.train_exclusions:
        in_excluded_region = (train.index >= excl_start) & (train.index < excl_end)
        assert not in_excluded_region.any()


def test_wider_margin_excludes_strictly_more_and_specific_timestamps(
    synthetic_split_events,
    synthetic_training_exclusion,
    synthetic_split_data_bounds,
    synthetic_split_df,
):
    """A larger pre_margin_hours must exclude strictly more training data
    than a smaller one, and specifically must exclude timestamps the
    smaller margin retained -- not just shift the SAME excluded span.
    """
    data_start, data_end = synthetic_split_data_bounds

    def _settings(pre_margin_hours: float) -> SimpleNamespace:
        training_exclusion = synthetic_training_exclusion.model_copy(
            update={"pre_margin_hours": pre_margin_hours}
        )
        split = SplitConfig(embargo_hours=4, training_exclusion=training_exclusion)
        evaluation = EvaluationConfig(
            window_widths=[6, 12, 24, 48],
            failure_events=synthetic_split_events,
        )
        return SimpleNamespace(split=split, evaluation=evaluation)

    narrow_settings = _settings(pre_margin_hours=2)
    wide_settings = _settings(pre_margin_hours=360)  # 15 days

    narrow_fold3 = next(
        f for f in make_folds(narrow_settings, data_start, data_end) if f.event_id == 3
    )
    wide_fold3 = next(f for f in make_folds(wide_settings, data_start, data_end) if f.event_id == 3)

    assert training_days_remaining(wide_fold3) < training_days_remaining(narrow_fold3)

    # 50h before event 1's start: inside the 360h (15d) margin's exclusion,
    # outside the 2h margin's.
    probe = pd.Timestamp(synthetic_split_events[0].start) - pd.Timedelta(hours=50)

    narrow_train, _ = apply_fold(synthetic_split_df, narrow_fold3)
    wide_train, _ = apply_fold(synthetic_split_df, wide_fold3)
    assert probe in narrow_train.index
    assert probe not in wide_train.index


def test_min_training_days_guard_raises_naming_fold_and_remaining_days(
    synthetic_split_settings, synthetic_split_data_bounds
):
    data_start, data_end = synthetic_split_data_bounds
    permissive_folds = make_folds(synthetic_split_settings, data_start, data_end)
    fold1 = next(f for f in permissive_folds if f.event_id == 1)
    remaining = training_days_remaining(fold1)

    tight_split = synthetic_split_settings.split.model_copy(
        update={"min_training_days": remaining + 1}
    )
    tight_settings = SimpleNamespace(
        split=tight_split, evaluation=synthetic_split_settings.evaluation
    )

    with pytest.raises(ValueError, match=r"event 1.*remaining training days"):
        make_folds(tight_settings, data_start, data_end)


def test_per_event_caps_invariant_to_pre_margin_and_infeasible_width_still_raises(
    synthetic_split_events,
    synthetic_training_exclusion,
    synthetic_split_settings_overlapping,
    synthetic_split_data_bounds,
):
    """`event_max_width_hours()`'s per-event maximum feasible LABEL width is
    governed by each earlier event's exclusion window END (anchored on
    maintenance/post_settle_hours or fallback_post_hours) and by
    data_start/embargo -- never by pre_margin_hours, which only moves an
    exclusion window's START further into the past. Verified directly:
    widening pre_margin_hours must NOT change the derived caps. Separately,
    the existing infeasible-width guard (a configured width overlapping an
    earlier event's exclusion) must still raise once a widened
    pre_margin_hours is layered on top of it -- i.e. the union/merge
    changes in this pass must not have silently defeated it.
    """
    data_start, data_end = synthetic_split_data_bounds

    def _settings(pre_margin_hours: float) -> SimpleNamespace:
        training_exclusion = synthetic_training_exclusion.model_copy(
            update={"pre_margin_hours": pre_margin_hours}
        )
        split = SplitConfig(embargo_hours=4, training_exclusion=training_exclusion)
        evaluation = EvaluationConfig(
            window_widths=[6, 12, 24, 48],
            failure_events=synthetic_split_events,
        )
        return SimpleNamespace(split=split, evaluation=evaluation)

    caps_narrow = event_max_width_hours(_settings(pre_margin_hours=2), data_start)
    caps_wide = event_max_width_hours(_settings(pre_margin_hours=300), data_start)
    assert caps_narrow == caps_wide

    widened_overlapping_settings = SimpleNamespace(
        split=SplitConfig(
            embargo_hours=4,
            training_exclusion=synthetic_training_exclusion.model_copy(
                update={"pre_margin_hours": 300}
            ),
        ),
        evaluation=synthetic_split_settings_overlapping.evaluation,
    )
    with pytest.raises(ValueError, match=r"event 3.*event 2|event 2.*event 3"):
        make_folds(widened_overlapping_settings, data_start, data_end)


def test_extend_test_end_never_moves_before_folds_own_test_end(
    synthetic_split_events, synthetic_training_exclusion, synthetic_split_data_bounds
):
    """extend_test_end_for_false_alarms() (pass 13, Part B1) originally
    assumed the NEXT event's exclusion window always starts safely after
    the CURRENT fold's own test period -- true at the original 24h margin,
    but not once pre_margin_hours widens enough for close event pairs
    (events 2/3 here are ~6 days apart): the next event's exclusion can
    then start BEFORE this fold's own test_start, which would otherwise
    silently produce test_end < test_start. The extension must never move
    test_end earlier than the fold's own (un-extended) test_end.
    """
    data_start, data_end = synthetic_split_data_bounds
    training_exclusion = synthetic_training_exclusion.model_copy(update={"pre_margin_hours": 210})
    split = SplitConfig(embargo_hours=4, training_exclusion=training_exclusion)
    evaluation = EvaluationConfig(
        window_widths=[6, 12, 24, 48], failure_events=synthetic_split_events
    )
    settings = SimpleNamespace(split=split, evaluation=evaluation)

    folds = make_folds(settings, data_start, data_end)
    fold2 = next(f for f in folds if f.event_id == 2)
    event2 = next(e for e in synthetic_split_events if e.id == 2)

    extended = extend_test_end_for_false_alarms(
        fold2, event2, synthetic_split_events, training_exclusion, data_end
    )
    assert extended.test_end >= fold2.test_end
    assert extended.test_end >= extended.test_start


# --- Pass 20: exclusion selection by overlap, not event position -----------


def test_own_event_precursor_excluded_from_own_training(
    synthetic_split_events,
    synthetic_training_exclusion,
    synthetic_split_data_bounds,
    synthetic_split_df,
):
    """The bug pass 20 fixes: a fold's own target event starts AFTER its
    own train_end, so under the old event-identity selection its own
    precursor was never excluded, at any pre_margin_hours. A wide enough
    margin must now reach into the fold's own training span and remove it.
    """
    data_start, data_end = synthetic_split_data_bounds
    training_exclusion = synthetic_training_exclusion.model_copy(update={"pre_margin_hours": 100})
    settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=4, training_exclusion=training_exclusion),
        evaluation=EvaluationConfig(
            window_widths=[6, 12, 24, 48], failure_events=synthetic_split_events
        ),
    )

    folds = make_folds(settings, data_start, data_end)
    fold2 = next(f for f in folds if f.event_id == 2)
    event2 = next(e for e in synthetic_split_events if e.id == 2)

    own_precursor_start = pd.Timestamp(event2.start) - pd.Timedelta(hours=100)
    assert own_precursor_start < fold2.train_end  # the region genuinely reaches into training

    train, _test = apply_fold(synthetic_split_df, fold2)
    in_own_precursor = (train.index >= own_precursor_start) & (train.index < fold2.train_end)
    assert not in_own_precursor.any()


def test_overlap_selection_clips_region_straddling_train_end(
    synthetic_split_events, synthetic_training_exclusion, synthetic_split_data_bounds
):
    """An exclusion region that straddles train_end (starts inside the
    training span, ends well after it -- here, inside event 2's own test
    period) is applied only to the portion inside the training span, not
    the region's full un-clipped extent.
    """
    data_start, data_end = synthetic_split_data_bounds
    training_exclusion = synthetic_training_exclusion.model_copy(update={"pre_margin_hours": 100})
    settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=4, training_exclusion=training_exclusion),
        evaluation=EvaluationConfig(
            window_widths=[6, 12, 24, 48], failure_events=synthetic_split_events
        ),
    )

    folds = make_folds(settings, data_start, data_end)
    fold2 = next(f for f in folds if f.event_id == 2)
    event2 = next(e for e in synthetic_split_events if e.id == 2)

    own_precursor_start = pd.Timestamp(event2.start) - pd.Timedelta(hours=100)
    unclipped_end = pd.Timestamp(event2.maintenance) + pd.Timedelta(
        hours=training_exclusion.post_settle_hours
    )
    assert unclipped_end > fold2.train_end  # confirms the region genuinely straddles train_end

    matching = [(s, e) for s, e in fold2.train_exclusions if s == own_precursor_start]
    assert len(matching) == 1
    _, clipped_end = matching[0]
    assert clipped_end == fold2.train_end
    assert clipped_end < unclipped_end


def test_fold1_gains_own_exclusion_at_wide_enough_margin(
    synthetic_split_events, synthetic_training_exclusion, synthetic_split_data_bounds
):
    """Fold 1 (earliest event, no earlier event to exclude) had ZERO
    exclusions at every margin under the old event-identity selection
    (pass 18's finding). A margin wide enough to reach back before its own
    train_end must now give it its own event's exclusion.
    """
    data_start, data_end = synthetic_split_data_bounds
    evaluation = EvaluationConfig(
        window_widths=[6, 12, 24, 48], failure_events=synthetic_split_events
    )
    event1 = next(e for e in synthetic_split_events if e.id == 1)

    narrow_settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=4, training_exclusion=synthetic_training_exclusion),
        evaluation=evaluation,
    )
    fold1_narrow = next(
        f for f in make_folds(narrow_settings, data_start, data_end) if f.event_id == 1
    )
    assert fold1_narrow.train_exclusions == ()

    wide_training_exclusion = synthetic_training_exclusion.model_copy(
        update={"pre_margin_hours": 60}
    )
    wide_settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=4, training_exclusion=wide_training_exclusion),
        evaluation=evaluation,
    )
    fold1_wide = next(f for f in make_folds(wide_settings, data_start, data_end) if f.event_id == 1)

    assert len(fold1_wide.train_exclusions) == 1
    excl_start, excl_end = fold1_wide.train_exclusions[0]
    assert excl_start == pd.Timestamp(event1.start) - pd.Timedelta(hours=60)
    assert excl_end == fold1_wide.train_end


def test_causality_and_embargo_hold_with_own_event_exclusions(
    synthetic_split_events,
    synthetic_training_exclusion,
    synthetic_split_data_bounds,
    synthetic_split_df,
):
    """Same causality/embargo invariants as the existing checks above, now
    exercised with own-event self-exclusion active (a wide margin) -- the
    overlap-based selection must not weaken either guarantee.
    """
    data_start, data_end = synthetic_split_data_bounds
    training_exclusion = synthetic_training_exclusion.model_copy(update={"pre_margin_hours": 100})
    settings = SimpleNamespace(
        split=SplitConfig(embargo_hours=4, training_exclusion=training_exclusion),
        evaluation=EvaluationConfig(
            window_widths=[6, 12, 24, 48], failure_events=synthetic_split_events
        ),
    )

    embargo = pd.Timedelta(hours=4)
    folds = make_folds(settings, data_start, data_end)
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert fold.test_start - fold.train_end >= embargo

        train, _test = apply_fold(synthetic_split_df, fold)
        assert (train.index < fold.test_start).all()
        for excl_start, excl_end in fold.train_exclusions:
            assert excl_start < excl_end
            assert excl_end <= fold.train_end
