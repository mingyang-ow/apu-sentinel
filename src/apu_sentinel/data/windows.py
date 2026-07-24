"""Windowing and scaling. CROWN-JEWEL FILE.

Hard rule (CLAUDE.md #2): scalers are fit on the TRAINING window ONLY, never
on val/test, never on full-series stats. Enforced by
tests/test_scaler_train_only.py, which runs as a blocking Claude Code hook on
every edit to src/apu_sentinel/data/ (see .claude/hooks/check_leakage.sh)
and again in the full pytest suite.

Stub: windowing/scaling logic implemented in a later pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_scaler(train_df: pd.DataFrame) -> Any:
    """Fit a scaler using ONLY train_df's statistics."""
    raise NotImplementedError


def make_windows(
    df: pd.DataFrame,
    scaler: Any,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Slice df into overlapping windows, scaled with a pre-fit scaler."""
    raise NotImplementedError
