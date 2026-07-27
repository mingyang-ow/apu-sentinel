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

import pandas as pd
import pytest

from apu_sentinel.data import split as split_module
from apu_sentinel.data.split import Fold, apply_fold, make_folds


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
