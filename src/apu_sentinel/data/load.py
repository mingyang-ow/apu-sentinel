"""Minimal MetroPT-3 loader: raw CSV -> clean, time-sorted DataFrame.

Loads, parses the timestamp column, sorts ascending, and runs light
structural sanity checks (shape, time span, all-null columns). If the raw
timestamps are not already monotonic non-decreasing, that is REPORTED (not
silently fixed) before the frame is sorted -- disorder in the source data is
a data-quality signal, not something to hide.

Nothing beyond loading: no splitting, scaling, resampling, windowing,
row-dropping, or feature engineering. Those are later, deliberate passes
(see CLAUDE.md).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Load the raw MetroPT-3 CSV at `path` into a time-sorted DataFrame
    indexed by `timestamp_column`. Returns the DataFrame as-is otherwise --
    no transformation, no derived columns, no value-based filtering.
    """
    path = Path(path)
    df = pd.read_csv(path)

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{timestamp_column}' not found in {path} "
            f"(columns: {list(df.columns)})"
        )

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    if not df[timestamp_column].is_monotonic_increasing:
        n_out_of_order = int((df[timestamp_column].diff() < pd.Timedelta(0)).sum())
        logger.warning(
            "%s: timestamps are NOT monotonic non-decreasing (%d out-of-order rows) "
            "in the raw file -- surfacing this rather than silently reordering and "
            "hiding it. Sorting for the returned frame regardless.",
            path,
            n_out_of_order,
        )

    df = df.sort_values(timestamp_column).set_index(timestamp_column)

    null_columns = df.columns[df.isna().all()].tolist()
    if null_columns:
        logger.warning("%s: all-null columns: %s", path, null_columns)

    logger.info(
        "%s: loaded shape=%s time_span=[%s, %s]",
        path,
        df.shape,
        df.index.min(),
        df.index.max(),
    )

    return df


def load_interim(interim_dir: Path) -> pd.DataFrame:
    """Load the cleaned / regime-tagged series, indexed by timestamp.

    Stub: interim data doesn't exist until the cleaning/regime-tagging pass.
    """
    raise NotImplementedError
