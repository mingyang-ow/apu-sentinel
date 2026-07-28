"""Pre-failure windows, masked regions, and coverage reporting.

Feeds evaluation/metrics.py's episode categorisation (early_warning /
concurrent / masked / false_alarm) and its false-alarm-rate normalisation.
Pre-failure window width is a SWEPT hyperparameter (CLAUDE.md) -- every
function here accepts a width, never hardcodes one.

Public API:
- `pre_failure_window(event, width_hours) -> (start, end)` -- the SWEPT
  pre-failure label window: [event.start - width_hours, event.start).
- `masked_regions(settings) -> tuple[MaskedRegion, ...]` -- every event's
  own [start, settle-end] region (NOT starting at pre_margin -- the
  pre-failure ramp must stay creditable as early_warning) plus
  evaluation.additional_masked_regions.
- `window_coverage(window_start, window_end, timestamps, expected_interval)
  -> float` -- observed/expected sample fraction in [0, 1].
- `coverage_report(settings, timestamps, expected_interval) -> {event_id:
  {width_hours: coverage}}` -- window_coverage() for every event x every
  configured evaluation.window_widths candidate.
- Pass 13, Part B2: `NormalStretch` -- one [start, end] period outside
  every event's pre-failure window/failure/settle period and any
  additional_masked_regions, each padded by evaluation.pooled_buffer_hours.
  `pooled_normal_stretches(settings, data_start, data_end) ->
  tuple[NormalStretch, ...]` builds them; consumed by evaluation/metrics.py
  evaluate_pooled_stretches().
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MaskedRegion:
    start: pd.Timestamp
    end: pd.Timestamp
    # None for an explicitly configured extra region (not tied to a
    # documented failure event).
    event_id: int | None


def pre_failure_window(event, width_hours: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The SWEPT pre-failure label window for one candidate width:
    [event.start - width_hours, event.start).
    """
    event_start = pd.Timestamp(event.start)
    return event_start - pd.Timedelta(hours=width_hours), event_start


def _event_mask_end(event, training_exclusion) -> pd.Timestamp:
    """End of the region masked for this event: maintenance +
    post_settle_hours, or event.end + fallback_post_hours when maintenance
    is unrecorded (event 1). Anchored on whichever of event.end/maintenance
    is later, matching data/split.py's test_end convention.
    """
    event_end = pd.Timestamp(event.end)
    if event.maintenance is not None:
        maintenance = pd.Timestamp(event.maintenance)
        anchor = max(event_end, maintenance)
        return anchor + pd.Timedelta(hours=training_exclusion.post_settle_hours)
    return event_end + pd.Timedelta(hours=training_exclusion.fallback_post_hours)


def masked_regions(settings) -> tuple[MaskedRegion, ...]:
    """Regions excluded from false-alarm counting: every documented
    failure event's own period through its repair/settle end
    (event.start -> maintenance + post_settle_hours, or event.end +
    fallback_post_hours when maintenance is unrecorded), plus any
    explicitly configured extra region (evaluation.additional_masked_regions).

    Rationale: during a failure and its repair the machine genuinely IS
    abnormal, so an alert there is not a false positive. It is categorised
    `concurrent` if it falls in the event's OWN [start, end] --
    metrics.categorise_episode checks concurrent before masked precisely
    because this masked region necessarily contains that same span -- or
    `masked` for the settling tail beyond `end`, or for a DIFFERENT event's
    masked region entirely.

    Starts at event.start, NOT event.start - pre_margin_hours (unlike
    data/split.py's training-exclusion window): the pre-failure ramp
    before event.start is exactly what an early_warning detection should
    be credited for, so it must never be masked away.

    Forward-looking note: this is also the intended mechanism for letting
    the evaluation.window_widths sweep exceed data/split.py's current 72h
    event-2/event-3 proximity cap in a later pass -- mask the resulting
    overlap between event 2's settling region and event 3's pre-failure
    window here, rather than shrinking the sweep. Not yet exercised, since
    the cap still holds at the configured widths.
    """
    regions = [
        MaskedRegion(
            start=pd.Timestamp(event.start),
            end=_event_mask_end(event, settings.split.training_exclusion),
            event_id=event.id,
        )
        for event in settings.evaluation.failure_events
    ]
    regions.extend(
        MaskedRegion(start=pd.Timestamp(extra.start), end=pd.Timestamp(extra.end), event_id=None)
        for extra in settings.evaluation.additional_masked_regions
    )
    return tuple(sorted(regions, key=lambda r: r.start))


def window_coverage(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    timestamps,
    expected_interval: pd.Timedelta,
) -> float:
    """Fraction of [window_start, window_end) that actually has data:
    observed_samples / expected_samples_at_nominal_interval, clipped to
    [0, 1]. This belongs in results, not discovered mid-analysis -- e.g.
    event 4's pre-failure window is expected around 0.8 given its ~14h
    internal data gap.
    """
    duration = window_end - window_start
    expected_samples = duration / expected_interval
    if expected_samples <= 0:
        return 0.0

    timestamps = pd.DatetimeIndex(timestamps)
    observed_samples = int(((timestamps >= window_start) & (timestamps < window_end)).sum())
    return min(1.0, observed_samples / expected_samples)


def coverage_report(
    settings,
    timestamps,
    expected_interval: pd.Timedelta,
) -> dict[int, dict[float, float]]:
    """window_coverage() for every documented failure event x every
    configured evaluation.window_widths candidate: {event_id: {width_hours:
    coverage}}.
    """
    report: dict[int, dict[float, float]] = {}
    for event in settings.evaluation.failure_events:
        per_width = {}
        for width in settings.evaluation.window_widths:
            start, end = pre_failure_window(event, width)
            per_width[width] = window_coverage(start, end, timestamps, expected_interval)
        report[event.id] = per_width
    return report


# --- Pooled normal-operation stretches (pass 13, Part B2) -------------------


@dataclass(frozen=True)
class NormalStretch:
    """One contiguous [start, end] period outside every event's
    pre-failure window, failure/settle period, and any additional masked
    region -- material for pooled_normal_stretches()'s false-alarm-rate
    estimate.
    """

    start: pd.Timestamp
    end: pd.Timestamp


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Same merge as evaluation/metrics.py's -- duplicated locally (not
    imported) to avoid a circular import, since metrics.py imports FROM
    this module.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def pooled_normal_stretches(
    settings,
    data_start: pd.Timestamp,
    data_end: pd.Timestamp,
) -> tuple[NormalStretch, ...]:
    """Periods across the WHOLE series that fall outside every event's
    pre-failure window, failure-to-settled period, and any
    additional_masked_regions -- each padded by
    evaluation.pooled_buffer_hours on both sides before taking the
    complement over [data_start, data_end].

    The pre-failure window uses max(evaluation.window_widths) -- the
    common SWEEP's widest value, deliberately NOT the much larger
    per-event maxima from data/split.py's event_max_width_hours() (pass 13
    Part C): using those instead would exclude most of the series (event
    1's own maximum is ~76 days) and leave little left to pool.

    CAVEAT (docs/FINDINGS.md §8, pass 13 Part B2): pooling mixes February
    and August operating conditions, which genuinely differ (median
    STOPPED duration drifts ~3x across that span) -- this is reported
    ALONGSIDE the in-fold rate for more statistical power, never as a
    replacement for it.
    """
    widest = max(settings.evaluation.window_widths)
    buffer = pd.Timedelta(hours=settings.evaluation.pooled_buffer_hours)
    events_by_id = {event.id: event for event in settings.evaluation.failure_events}

    excluded: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for region in masked_regions(settings):
        if region.event_id is not None:
            pre_start, _ = pre_failure_window(events_by_id[region.event_id], widest)
            start = pre_start
        else:
            start = region.start
        excluded.append((start - buffer, region.end + buffer))

    data_start = pd.Timestamp(data_start)
    data_end = pd.Timestamp(data_end)

    stretches = []
    cursor = data_start
    for start, end in _merge_intervals(excluded):
        start = max(start, data_start)
        end = min(end, data_end)
        if start > cursor:
            stretches.append(NormalStretch(cursor, start))
        cursor = max(cursor, end)
    if cursor < data_end:
        stretches.append(NormalStretch(cursor, data_end))
    return tuple(stretches)
