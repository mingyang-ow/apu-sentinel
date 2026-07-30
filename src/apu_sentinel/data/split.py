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
- train period = [data_start, test_start - embargo), with every documented
  event's training-exclusion region that OVERLAPS this span removed --
  selected by overlap, not by whether the event is chronologically earlier
  or later than the fold's own target event (pass 20, see below).

Embargo rationale: once sliding windows are built (a later pass), a window
starting just before a boundary would span it, letting a training sample
see into the test period (or a test-period window reach back into train).
The embargo gap must be at least as long as the longest sequence window
used downstream; it is enforced here even though windowing does not exist
yet.

Training-exclusion vs. window_widths: these are two DELIBERATELY separate
concepts. training_exclusion.* (config) is a fixed, generous margin that
protects training purity by removing ramp-up-to-failure and post-repair
data from every event whose exclusion region overlaps the training span --
including the fold's OWN target event (pass 20, see below); "earlier" vs.
"later" is not the criterion. evaluation.window_widths is the SWEPT
pre-failure label width used at evaluation/labelling time. Neither is
derived from the other.

Event-2/event-3 proximity constraint: event 2's repair (2020-05-30 12:00)
and event 3's onset (2020-06-05 10:00) are under six days apart. A
pre-failure window wide enough to reach past event 2's training-exclusion
region would mislabel event 2's post-repair recovery as "pre-failure event
3". make_folds() therefore checks every configured window width against
every earlier event's exclusion region and RAISES, naming the offending
event pair and width, rather than silently truncating or overlapping.

A second, TIGHTER check covers the fold's actual test_start (which backs
off by max(window_widths) AND embargo_hours, not just the window width
alone) against every earlier event's exclusion region -- a fold's test
period must not open while an earlier event is still "settling", or that
settling period would be evaluated as if it were normal background for the
current fold's event. This was previously unguarded: fold 3's test period
could open before event 2's training-exclusion region ended even though
the plain window-width check passed, since it doesn't account for the
extra embargo subtracted when deriving test_start. make_folds() raises on
this too, naming the offending event pair and the overlap duration.

Pass 13, Part C: the constraint above binds event 3 alone (its prior
exclusion ends only ~94h before its onset), but before this pass it was
applied GLOBALLY -- every event capped at whatever the tightest event
tolerates, via a single shared max(window_widths). make_folds() now takes
an optional per-event width_hours_by_event mapping (see
event_max_width_hours()) so each event can be evaluated at its OWN maximum
feasible width instead, as a SEPARATE sensitivity fold set -- the common,
shared-max sweep (the default, unchanged) remains the cross-model-
comparable result.

Pass 18 (docs/findings/12-event2-error-analysis.md, docs/RESULTS.md §18):
widening `training_exclusion.pre_margin_hours` to cover a precursor's full
run-up (rather than the original fixed 24h) means TWO earlier events'
exclusion windows can now overlap each other (not just approach a later
event's test_start, which _check_no_window_overlap/_check_no_test_start_
overlap already guarded). `make_folds()` now takes the UNION of overlapping
exclusion regions before storing them on the Fold, so `train_exclusions`
is always a set of disjoint, non-overlapping intervals -- both so
`apply_fold()`'s masking never double-applies the same region (harmless
there; boolean OR is idempotent) and so anything that SUMS exclusion
lengths (`training_days_remaining()`) never double-counts an overlap.
`split.min_training_days` guards against the failure mode pass 13
documented: a fold whose training slice is squeezed too thin fits a
threshold of ~0.0 and turns its entire test period into one continuous
episode. `make_folds()` raises, naming the offending fold and its actual
remaining days, rather than letting that recur silently.

Pass 20 (docs/RESULTS.md §20, docs/findings/10-process-lessons.md): the
exclusion loop previously skipped `other.id == event.id` -- i.e. selected
by event IDENTITY, excluding every OTHER documented event's region but
never the fold's own target event's. A fold's own event starts AFTER its
own train_end, so its precursor was never excluded from its own training,
at any `pre_margin_hours` value -- the exact bug pass 18's flat margin
sweep traced back to. Fixed by selecting by OVERLAP with the training span
instead: every event's exclusion window is built and kept if it
intersects [data_start, train_end), regardless of identity or
chronological order. Safe by construction (removing training data cannot
introduce leakage); the pass-18 union-merge above is unchanged and still
applies to whatever set of regions this selection produces.

Pass 21 (docs/RESULTS.md §21): `training_exclusion.additional_regions`
adds sensitivity-only exclusion regions not tied to any documented event
(e.g. the un-anchored early-March cluster pass 20 identified) -- selected
and clipped by the same overlap-with-training-span logic as event regions
above, then union-merged alongside them. Training only, exactly like every
other exclusion here; never affects test periods.

Public API:
- `Fold` -- one fold's boundaries: train_start/train_end, test_start/
  test_end, train_exclusions (tuple of DISJOINT (start, end) regions to
  drop from training -- selected by overlap with the training span, pass
  20, including the fold's own event; overlapping source regions merged,
  pass 18). Produced by make_folds(); consumed by apply_fold() and every
  evaluation/ function that needs test_start/test_end.
- `training_days_remaining(fold) -> float` -- wall-clock days of training
  time left after `fold.train_exclusions` are removed from
  [train_start, train_end); computed from timestamps alone (no DataFrame
  needed) since train_exclusions are already disjoint.
- `make_folds(settings, data_start, data_end, width_hours_by_event=None)
  -> list[Fold]` -- one Fold per documented failure event. Raises
  ValueError on empty window_widths, insufficient lead-in data, any width
  (shared or per-event) that would overlap an earlier event's exclusion
  region, or a fold whose remaining training days fall below
  `split.min_training_days`.
- `apply_fold(df, fold, strategy="time") -> (train_df, test_df)` -- slices
  df by time only; `strategy` must be "time" (a rejectable guard, not a
  real option). Gotcha: does not scale, window, or otherwise transform --
  callers still need fit_regime_scalers/transform_by_regime and/or
  make_windows downstream.
- `event_max_width_hours(settings, data_start) -> {event_id: hours}` --
  each event's own maximum feasible pre-failure width; feed straight back
  into make_folds()'s width_hours_by_event for the Part C sensitivity fold
  set.
- `extend_test_end_for_false_alarms(fold, event, events_sorted,
  training_exclusion, data_end) -> Fold` -- a copy of `fold` with test_end
  pushed out for false-alarm counting only; does not affect detection
  (categorise_episode never reads test_end).
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


def _merge_exclusion_windows(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    """Union of possibly-overlapping [start, end) regions into disjoint,
    sorted intervals. Same merge idea as evaluation/events.py's own
    _merge_intervals -- duplicated locally rather than imported, since
    data/ sits BELOW evaluation/ in the module layering (CLAUDE.md) and
    must not depend on it.
    """
    if not windows:
        return ()
    ordered = sorted(windows)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def training_days_remaining(fold: Fold) -> float:
    """Wall-clock days of TRAINING time left in `fold` after
    `fold.train_exclusions` are removed from [train_start, train_end).

    Computed from timestamps alone (no DataFrame/actual row count needed):
    exclusions are already disjoint (see make_folds' union step), so
    summing their lengths and subtracting once from the fold's full span is
    exact -- never double-counting an overlap between two source events'
    exclusion windows.
    """
    total_seconds = (fold.train_end - fold.train_start).total_seconds()
    excluded_seconds = sum((end - start).total_seconds() for start, end in fold.train_exclusions)
    return max(total_seconds - excluded_seconds, 0.0) / 86400.0


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


def _check_no_window_overlap(events_sorted, width_hours_by_event, training_exclusion) -> None:
    """Raise if any event's OWN configured width (width_hours_by_event[event.id])
    would reach back far enough to overlap an earlier event's exclusion
    region. Checking each event against only the width actually used to
    derive ITS OWN test_start is equivalent to checking every smaller
    configured width too (a smaller width's label_start is always closer
    to later_start, i.e. less far back), so this is not a weaker check
    than iterating the whole sweep -- just parametrised per event instead
    of assuming every event shares the same global width (pass 13, Part C).
    """
    for i, earlier in enumerate(events_sorted):
        _, excl_end = _exclusion_window(earlier, training_exclusion)
        for later in events_sorted[i + 1 :]:
            later_start = pd.Timestamp(later.start)
            width = width_hours_by_event[later.id]
            label_start = later_start - pd.Timedelta(hours=width)
            if label_start < excl_end:
                raise ValueError(
                    f"window width {width}h applied to event {later.id} "
                    f"(starts {later_start}) reaches back to {label_start}, "
                    f"which overlaps event {earlier.id}'s training-exclusion "
                    f"region (ends {excl_end}). Reduce evaluation.window_widths "
                    "or the split.training_exclusion margins."
                )


def _compute_test_start(
    event_start: pd.Timestamp, width_hours: float, embargo: pd.Timedelta
) -> pd.Timestamp:
    return event_start - pd.Timedelta(hours=width_hours) - embargo


def _check_no_test_start_overlap(
    events_sorted, width_hours_by_event, embargo, training_exclusion
) -> None:
    """Raise if any fold's actual test_start -- which backs off by BOTH
    that event's OWN width (width_hours_by_event[event.id]) and
    embargo_hours -- would open before an earlier event's training-exclusion
    region has ended.

    Tighter than _check_no_window_overlap: that check only validates the
    label window itself, not the extra embargo subtracted when deriving
    test_start, so it can pass while this one would still catch a fold
    whose test period opens mid-settle for a previous event.
    """
    for i, earlier in enumerate(events_sorted):
        _, excl_end = _exclusion_window(earlier, training_exclusion)
        for later in events_sorted[i + 1 :]:
            later_start = pd.Timestamp(later.start)
            width = width_hours_by_event[later.id]
            test_start = _compute_test_start(later_start, width, embargo)
            if test_start < excl_end:
                overlap = excl_end - test_start
                raise ValueError(
                    f"fold for event {later.id}: test_start {test_start} (event "
                    f"start {later_start} minus width {width}h "
                    f"and embargo {embargo}) begins {overlap} before event "
                    f"{earlier.id}'s training-exclusion region ends ({excl_end}). "
                    "Reduce evaluation.window_widths, split.embargo_hours, or the "
                    "split.training_exclusion margins."
                )


# A width computed to land test_start EXACTLY on the make_folds()
# lead-in-data boundary would make train_end land EXACTLY on data_start --
# and make_folds() requires train_end strictly AFTER data_start (an
# unrelated check, about having any training data at all), not merely
# at-or-after. event_max_width_hours() shaves this sliver off so its
# output is always directly, strictly feasible when fed back in.
_LEAD_IN_EPSILON = pd.Timedelta(seconds=1)


def event_max_width_hours(settings, data_start: pd.Timestamp) -> dict[int, float]:
    """Each documented event's OWN maximum feasible pre-failure window
    width (hours), replacing the single GLOBAL cap CLAUDE.md's "pre-failure
    window width" section documents as binding on event 3's tight ~94h
    geometry alone but applied, before pass 13, to every event uniformly.

    Two independent constraints bound each event's width, and the smaller
    one wins:

    1. test_start = event.start - width - embargo must not precede any
       EARLIER event's own training-exclusion region end (the same check
       _check_no_test_start_overlap enforces) -- one candidate per earlier
       event.
    2. train_end (= test_start - embargo) must fall strictly after
       data_start regardless of whether an earlier event exists at all
       (make_folds()'s own, unrelated lead-in-data check) -- always a
       candidate, and the ONLY one for the earliest event.

    Passing this mapping to make_folds() (as width_hours_by_event) builds a
    fold set where each event uses its own maximum instead of a shared one
    -- a SEPARATE, sensitivity-only fold set (CLAUDE.md's "pre-failure
    window width": window_widths is a swept, reported result, never
    silently replaced by a single "best" number).
    """
    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    training_exclusion = settings.split.training_exclusion
    embargo = pd.Timedelta(hours=settings.split.embargo_hours)
    data_start = pd.Timestamp(data_start)

    result = {}
    for event in events_sorted:
        event_start = pd.Timestamp(event.start)
        candidates_hours = []

        for other in events_sorted:
            if pd.Timestamp(other.start) < event_start:
                _, excl_end = _exclusion_window(other, training_exclusion)
                candidates_hours.append((event_start - embargo - excl_end).total_seconds() / 3600.0)

        lead_in_limit = event_start - 2 * embargo - data_start - _LEAD_IN_EPSILON
        candidates_hours.append(lead_in_limit.total_seconds() / 3600.0)

        result[event.id] = max(0.0, min(candidates_hours))
    return result


def make_folds(
    settings,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
    width_hours_by_event: dict[int, float] | None = None,
) -> list[Fold]:
    """Build one walk-forward fold per documented failure event.

    `settings` must expose `settings.split` (embargo_hours,
    training_exclusion) and `settings.evaluation` (window_widths,
    failure_events) -- the shape of apu_sentinel.config.Settings. Returns
    fold time boundaries only, not materialised DataFrames -- see
    apply_fold() for slicing an actual df.

    width_hours_by_event: optional {event_id: width_hours} override of the
    width used to derive THAT event's own test_start. Defaults to None,
    meaning every event uses max(evaluation.window_widths) -- the ORIGINAL
    global-cap behaviour (unchanged), needed for the common, cross-model-
    comparable sweep. Pass event_max_width_hours(settings, data_start)'s
    output to build a SEPARATE fold set at each event's own maximum feasible
    width instead (pass 13, Part C's per-event caps) -- a sensitivity
    result, never a silent replacement of the common sweep.

    Raises:
        ValueError: if evaluation.window_widths is empty, if a fold's
            computed train_end would leave no lead-in data before
            data_start, or if any event's own width -- or the fold's
            actual test_start -- would overlap an earlier event's
            training-exclusion region (see _check_no_window_overlap and
            _check_no_test_start_overlap).
    """
    window_widths = settings.evaluation.window_widths
    if not window_widths:
        raise ValueError("evaluation.window_widths must be non-empty to build folds")

    events_sorted = sorted(settings.evaluation.failure_events, key=lambda e: pd.Timestamp(e.start))
    training_exclusion = settings.split.training_exclusion
    embargo = pd.Timedelta(hours=settings.split.embargo_hours)

    if width_hours_by_event is None:
        shared_max = max(window_widths)
        width_hours_by_event = {event.id: shared_max for event in events_sorted}

    _check_no_window_overlap(events_sorted, width_hours_by_event, training_exclusion)
    _check_no_test_start_overlap(events_sorted, width_hours_by_event, embargo, training_exclusion)

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

        width = width_hours_by_event[event.id]
        test_start = _compute_test_start(event_start, width, embargo)
        train_end = test_start - embargo
        if train_end <= data_start:
            raise ValueError(
                f"event {event.id}: computed train_end {train_end} is at/before "
                f"data_start {data_start} -- not enough lead-in data for this fold"
            )

        exclusions = []
        for other in events_sorted:
            excl_start, excl_end = _exclusion_window(other, training_exclusion)
            if excl_end <= data_start or excl_start >= train_end:
                continue
            exclusions.append((max(excl_start, data_start), min(excl_end, train_end)))

        for region in training_exclusion.additional_regions:
            region_start, region_end = pd.Timestamp(region.start), pd.Timestamp(region.end)
            if region_end <= data_start or region_start >= train_end:
                continue
            exclusions.append((max(region_start, data_start), min(region_end, train_end)))

        fold = Fold(
            event_id=event.id,
            train_start=data_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_exclusions=_merge_exclusion_windows(exclusions),
        )

        min_training_days = settings.split.min_training_days
        remaining_days = training_days_remaining(fold)
        if remaining_days < min_training_days:
            raise ValueError(
                f"event {event.id}: fold's remaining training days "
                f"({remaining_days:.2f}) fall below split.min_training_days "
                f"({min_training_days}) after exclusions -- widen the data span, "
                "narrow split.training_exclusion margins, or lower "
                "split.min_training_days if this is expected."
            )

        folds.append(fold)
    return folds


def extend_test_end_for_false_alarms(
    fold: Fold,
    event,
    events_sorted,
    training_exclusion,
    data_end: pd.Timestamp,
) -> Fold:
    """Returns a NEW Fold (fold.test_end pushed forward) giving false-alarm
    counting far more normal-operation time than the tight test period used
    for detection (pass 13, Part B1) -- CLAUDE.md rule 5's evaluated_days
    denominator cannot support a rate estimate from 1-2 episodes over a
    ~3-day window.

    test_end extends to just BEFORE the next chronologically later event's
    training-exclusion region begins (never INTO it -- that period is that
    other event's own ramp-up/settle time, not this fold's normal
    background), or to data_end for the final fold.

    Detection and lead time are UNCHANGED by this: categorise_episode and
    pre_failure_window are keyed off event.start and window_width_hours
    alone, never fold.test_end. Only evaluated_days/false_alarms_per_day
    change, because more test-period time (and any episodes within it) is
    now visible to the harness.

    The extended period legitimately OVERLAPS later folds' own TRAINING
    periods -- normal in walk-forward (folds overlap by construction, each
    fold remains internally causal on its own) and not leakage; it is not,
    however, allowed to overlap another event's own EXCLUSION region (see
    above), which is the one thing that would make settle-time look like
    normal background.

    Pass 18 (docs/RESULTS.md §18): a widened `training_exclusion.
    pre_margin_hours` moves the next event's exclusion window's START
    (not just its end) further into the past -- for close event pairs
    (event 2/3, ~6 days apart) it can move earlier than THIS fold's own
    test_start once the margin is wide enough, which would otherwise
    silently produce test_end < test_start (invalid) and, propagated
    further, an empty-window p_chance_permutation crash. Never allowed to
    move test_end BEFORE the fold's own (un-extended) test_end -- this
    function only ever extends forward; if the next event's exclusion has
    already swallowed the entire would-be extension, no extra
    false-alarm-counting time is available and test_end is left as-is.
    """
    later_events = [e for e in events_sorted if pd.Timestamp(e.start) > pd.Timestamp(event.start)]
    if later_events:
        next_event = min(later_events, key=lambda e: pd.Timestamp(e.start))
        candidate_test_end, _ = _exclusion_window(next_event, training_exclusion)
        new_test_end = max(candidate_test_end, fold.test_end)
    else:
        new_test_end = pd.Timestamp(data_end)

    return Fold(
        event_id=fold.event_id,
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=new_test_end,
        train_exclusions=fold.train_exclusions,
    )


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
