"""Walk-forward time-based split. CROWN-JEWEL FILE.

Hard rule (CLAUDE.md #1): split by time only. No training sample's timestamp
may exceed its fold's train boundary, and no shuffle/random split is
permitted anywhere in this project. This is enforced by
tests/test_split_no_leakage.py, which runs as a blocking Claude Code hook on
every edit to src/apu_sentinel/data/ (see .claude/hooks/check_leakage.sh)
and again in the full pytest suite.

Scheme: one walk-forward (rolling-origin, expanding-window) fold per
documented failure event (see configs/base.yaml evaluation.failure_events).
For fold k, targeting event k:

- test period = [event.start - max(window_widths) - embargo, test_end],
  where test_end extends to event.maintenance if that is later than
  event.end (and recorded).
- train period = [data_start, test_start - embargo), with all earlier
  events' training-exclusion regions removed.

Embargo rationale: once sliding windows are built (a later pass), a window
starting just before a boundary would span it, letting a training sample
see into the test period (or a test-period window reach back into train).
The embargo gap must be at least as long as the longest sequence window
used downstream; it is enforced here even though windowing does not exist
yet.

Training-exclusion vs. window_widths: these are two DELIBERATELY separate
concepts. training_exclusion.* (config) is a fixed, generous margin that
protects training purity by removing ramp-up-to-failure and post-repair
data from every earlier event. evaluation.window_widths is the SWEPT
pre-failure label width used at evaluation/labelling time. Neither is
derived from the other.

Event-2/event-3 proximity constraint: event 2's repair (2020-05-30 12:00)
and event 3's onset (2020-06-05 10:00) are under six days apart. A
pre-failure window wide enough to reach past event 2's training-exclusion
region would mislabel event 2's post-repair recovery as "pre-failure event
3". make_folds() therefore checks every configured window width against
every earlier event's exclusion region and RAISES, naming the offending
event pair and width, rather than silently truncating or overlapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    event_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_exclusions: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]


def _exclusion_window(event, training_exclusion) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The [start, end) region to purge from training for a given event."""
    start = pd.Timestamp(event.start) - pd.Timedelta(hours=training_exclusion.pre_margin_hours)
    if event.maintenance is not None:
        end = pd.Timestamp(event.maintenance) + pd.Timedelta(
            hours=training_exclusion.post_settle_hours
        )
    else:
        end = pd.Timestamp(event.end) + pd.Timedelta(hours=training_exclusion.fallback_post_hours)
    return start, end


def _check_no_window_overlap(events_sorted, window_widths, training_exclusion) -> None:
    """Raise if any configured window width, applied to any event, would
    reach back far enough to overlap an earlier event's exclusion region.
    """
    for i, earlier in enumerate(events_sorted):
        _, excl_end = _exclusion_window(earlier, training_exclusion)
        for later in events_sorted[i + 1 :]:
            later_start = pd.Timestamp(later.start)
            for width in window_widths:
                label_start = later_start - pd.Timedelta(hours=width)
                if label_start < excl_end:
                    raise ValueError(
                        f"window width {width}h applied to event {later.id} "
                        f"(starts {later_start}) reaches back to {label_start}, "
                        f"which overlaps event {earlier.id}'s training-exclusion "
                        f"region (ends {excl_end}). Reduce evaluation.window_widths "
                        "or the split.training_exclusion margins."
                    )


def make_folds(settings, data_start: pd.Timestamp, data_end: pd.Timestamp) -> list[Fold]:
    """Build one walk-forward fold per documented failure event.

    `settings` must expose `settings.split` (embargo_hours,
    training_exclusion) and `settings.evaluation` (window_widths,
    failure_events) -- the shape of apu_sentinel.config.Settings. Returns
    fold time boundaries only, not materialised DataFrames -- see
    apply_fold() for slicing an actual df.

    Raises:
        ValueError: if evaluation.window_widths is empty, if a fold's
            computed train_end would leave no lead-in data before
            data_start, or if any configured window width would overlap an
            earlier event's training-exclusion region (see
            _check_no_window_overlap).
    """
    window_widths = settings.evaluation.window_widths
    if not window_widths:
        raise ValueError("evaluation.window_widths must be non-empty to build folds")

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    training_exclusion = settings.split.training_exclusion
    _check_no_window_overlap(events_sorted, window_widths, training_exclusion)

    max_width = max(window_widths)
    embargo = pd.Timedelta(hours=settings.split.embargo_hours)
    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)

    folds = []
    for event in events_sorted:
        event_start = pd.Timestamp(event.start)
        test_end = pd.Timestamp(event.end)
        if event.maintenance is not None:
            maintenance = pd.Timestamp(event.maintenance)
            if maintenance > test_end:
                test_end = maintenance
        if test_end > data_end:
            raise ValueError(
                f"event {event.id}: test_end {test_end} is after data_end {data_end} "
                "-- not enough data to cover this fold's test period"
            )

        test_start = event_start - pd.Timedelta(hours=max_width) - embargo
        train_end = test_start - embargo
        if train_end <= data_start:
            raise ValueError(
                f"event {event.id}: computed train_end {train_end} is at/before "
                f"data_start {data_start} -- not enough lead-in data for this fold"
            )

        exclusions = []
        for other in events_sorted:
            if other.id == event.id:
                continue
            excl_start, excl_end = _exclusion_window(other, training_exclusion)
            if excl_end <= data_start or excl_start >= train_end:
                continue
            exclusions.append((max(excl_start, data_start), min(excl_end, train_end)))

        folds.append(
            Fold(
                event_id=event.id,
                train_start=data_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_exclusions=tuple(sorted(exclusions)),
            )
        )
    return folds


def apply_fold(
    df: pd.DataFrame,
    fold: Fold,
    strategy: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice df into (train, test) for one fold, by time only.

    Never shuffles, samples randomly, or reorders. `strategy` exists only
    to make the "time-based only" contract explicit and rejectable at the
    call level -- any non-"time" value (e.g. a shuffle/random request)
    raises rather than merely warning.

    Raises:
        ValueError: if strategy is anything other than "time".
    """
    if strategy != "time":
        raise ValueError(
            f"apply_fold only supports strategy='time', got {strategy!r} -- "
            "shuffle/random splits are not permitted (CLAUDE.md rule 1)."
        )

    index = df.index
    train_mask = (index >= fold.train_start) & (index < fold.train_end)
    for excl_start, excl_end in fold.train_exclusions:
        train_mask &= ~((index >= excl_start) & (index < excl_end))
    test_mask = (index >= fold.test_start) & (index <= fold.test_end)
    return df.loc[train_mask], df.loc[test_mask]
