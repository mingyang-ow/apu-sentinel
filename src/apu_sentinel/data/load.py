"""Load MetroPT-3 raw/interim data into a single time-indexed dataframe.

Stub: loading logic implemented in a later pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Load the raw MetroPT-3 series, indexed by timestamp."""
    raise NotImplementedError


def load_interim(interim_dir: Path) -> pd.DataFrame:
    """Load the cleaned / regime-tagged series, indexed by timestamp."""
    raise NotImplementedError
