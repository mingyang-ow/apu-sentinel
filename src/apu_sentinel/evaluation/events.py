"""Failure-window labels, derived from documented failure dates.

Pre-failure window width is a SWEPT hyperparameter (CLAUDE.md), never a
fixed magic number -- this module must accept a width, not hardcode one.

Stub: label-derivation logic implemented in a later pass.
"""

from __future__ import annotations

import pandas as pd

# Documented MetroPT-3 failure event timestamps. Filled in and versioned
# once the dataset's known failure dates are confirmed against source docs.
FAILURE_EVENTS: list[str] = []


def label_pre_failure_windows(
    index: pd.DatetimeIndex,
    failure_events: list[str],
    window_width_minutes: int,
) -> pd.Series:
    """Return a boolean Series over index: True inside a pre-failure window
    of width window_width_minutes before each event in failure_events.
    """
    raise NotImplementedError
