"""Sliding-window slicing.

Scaling now lives in data/scaling.py (fit-on-train-only, per fold) -- this
module only slices already-scaled data into overlapping windows.

Stub: windowing logic implemented in a later pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    scaler: Any,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Slice df into overlapping windows, scaled with a pre-fit scaler."""
    raise NotImplementedError
