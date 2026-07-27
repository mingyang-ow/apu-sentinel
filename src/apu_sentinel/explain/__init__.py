"""Per-channel contribution attribution -- CORE, not an extension.

Feeds the mixed, ranked per-episode diagnosis in evaluation/metrics.py --
one alert's detail, never separate parallel alarms per channel. EVERY
episode gets a ranking, including false_alarm episodes: those are the
material for the error analysis CLAUDE.md calls for.
"""

from __future__ import annotations

import numpy as np

VALID_AGGREGATIONS = ("mean", "max")


def rank_channel_contributions(
    contributions: np.ndarray,
    channel_names: tuple[str, ...],
    method: str = "mean",
) -> tuple[tuple[str, float], ...]:
    """Aggregate one episode's per-timestamp, per-channel contributions
    (shape (n_timestamps, n_channels)) into a single ranked
    (channel_name, contribution) tuple, descending by contribution.

    method: "mean" (default) or "max" -- driven by config
    (evaluation.contribution_aggregation), never hardcoded by a caller.

    Raises:
        ValueError: if method is not "mean"/"max", if contributions'
            channel dimension doesn't match len(channel_names), or if
            contributions has zero timestamps.
    """
    if method not in VALID_AGGREGATIONS:
        raise ValueError(f"method must be one of {VALID_AGGREGATIONS}, got {method!r}")

    contributions = np.asarray(contributions)
    if contributions.ndim != 2 or contributions.shape[1] != len(channel_names):
        raise ValueError(
            f"contributions shape {contributions.shape} does not match "
            f"{len(channel_names)} channel_names"
        )
    if contributions.shape[0] == 0:
        raise ValueError("cannot rank contributions for an empty episode (0 timestamps)")

    aggregated = contributions.mean(axis=0) if method == "mean" else contributions.max(axis=0)
    order = np.argsort(aggregated)[::-1]
    return tuple((channel_names[i], float(aggregated[i])) for i in order)
