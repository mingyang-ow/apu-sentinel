"""Per-channel reconstruction-error attribution. CORE, not an extension.

Feeds the mixed, ranked per-episode diagnosis in evaluation/ -- one alert's
detail, never separate parallel alarms per channel.

Stub: attribution logic implemented in a later pass.
"""

from __future__ import annotations

import numpy as np


def rank_channel_contributions(
    channel_contributions: np.ndarray,
    channel_names: list[str],
) -> list[tuple[str, float]]:
    """Aggregate per-timestamp, per-channel contributions over an episode
    into a single ranked (channel_name, contribution) list.
    """
    raise NotImplementedError
