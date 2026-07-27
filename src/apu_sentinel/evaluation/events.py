"""Pre-failure windows, masked regions, and coverage reporting.

Feeds evaluation/metrics.py's episode categorisation (early_warning /
concurrent / masked / false_alarm) and its false-alarm-rate normalisation.
Pre-failure window width is a SWEPT hyperparameter (CLAUDE.md) -- every
function here accepts a width, never hardcodes one.
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
