"""EPISODE-LEVEL evaluation. THE metric every model is scored on.

Never per-timestamp, never per-parameter (CLAUDE.md rule 5). All of
thresholding, grouping flagged timestamps into episodes via a documented
hold-time/hysteresis rule, checking episodes against pre-failure windows
(detection + lead time), counting false episodes, and attaching each
episode's mixed ranked diagnosis (from explain/) happens here -- models
(models/) never touch this.

Primary objective is RECALL; false-alarm rate is a monitored secondary under
a ceiling that starts null and is set only after baseline behaviour is seen.

Stub: metric logic implemented in a later pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Episode:
    start: pd.Timestamp
    end: pd.Timestamp
    is_detection: bool
    lead_time_minutes: float | None
    ranked_diagnosis: list[tuple[str, float]]


def scores_to_episodes(
    timestamps: pd.DatetimeIndex,
    scores: np.ndarray,
    threshold: float,
    hold_time_seconds: float,
) -> list[Episode]:
    """Group thresholded, flagged timestamps into episodes using a
    hold-time/hysteresis rule.
    """
    raise NotImplementedError


def evaluate_episodes(
    episodes: list[Episode],
    failure_events: list[str],
    window_width_minutes: int,
) -> dict:
    """Compute episode-level detection, lead time, and false-alarm counts."""
    raise NotImplementedError
