"""Per-fold scaling. CROWN-JEWEL FILE (second guard, alongside split.py).

Hard rule (CLAUDE.md #2): scalers are fit on the TRAINING window ONLY, never
on val/test, never on full-series stats. Enforced by
tests/test_scaler_train_only.py, which runs as a blocking Claude Code hook on
every edit to src/apu_sentinel/data/ (see .claude/hooks/check_leakage.sh)
and again in the full pytest suite.

One scaler per fold: each walk-forward fold (data/split.py) has a different
training slice, so each fold fits its own FoldScaler. There is no code path
that fits one scaler and reuses it across folds -- that would leak a later
fold's statistics into an earlier one.

Required order of operations (non-negotiable):
    train, test = apply_fold(df, fold)   # exclusions already removed
    scaler = fit_scaler(train, settings)  # fit on that CLEAN slice only
    train_scaled = transform(train, scaler)
    test_scaled = transform(test, scaler)  # same fitted parameters

Fitting before exclusions are removed lets the documented failure periods
skew location/spread. Fitting on anything that includes test data is
outright leakage. transform() NEVER fits -- there is no fit_transform in
this module, and transform() raises on an unfitted scaler rather than
lazily fitting.

Column selection is explicit and config-driven (scaling.analog_columns /
scaling.passthrough_columns). MetroPT-3's 7 analog channels are scaled;
its 8 digital/status channels are meaningless to scale (binary 0/1 flags)
and are passed through bit-identical. A DataFrame column present in
neither list raises -- this catches schema drift loudly instead of
guessing.

Default method is `robust` (median / IQR): training data still contains
UNREPORTED anomalies -- only the four documented events can be excluded via
data/split.py's training_exclusion -- so location/spread estimates should
be resistant to the outliers that remain. `standard` and `minmax` are
supported alternatives, selected via config, never hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VALID_METHODS = ("robust", "standard", "minmax")


@dataclass
class FoldScaler:
    """Fitted on ONE fold's clean training slice. Never refits.

    center_/scale_ are None until fit_scaler() populates them; transform()
    raises if either is still None.
    """

    method: str
    analog_columns: tuple[str, ...]
    passthrough_columns: tuple[str, ...]
    center_: dict[str, float] | None = None
    scale_: dict[str, float] | None = None

    @property
    def is_fitted(self) -> bool:
        return self.center_ is not None and self.scale_ is not None


def _check_columns(
    columns,
    analog_columns: tuple[str, ...],
    passthrough_columns: tuple[str, ...],
) -> None:
    known = set(analog_columns) | set(passthrough_columns)
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise ValueError(
            f"column(s) {unknown} are in neither scaling.analog_columns nor "
            "scaling.passthrough_columns -- list every column explicitly "
            "rather than guessing (schema drift guard)."
        )


def fit_scaler(train_df: pd.DataFrame, settings) -> FoldScaler:
    """Fit a FoldScaler on train_df's analog columns, using ONLY train_df's
    own statistics.

    train_df must already be a fold's CLEAN training slice -- i.e. the
    output of apply_fold(df, fold), with exclusions removed. This function
    does not know about folds or exclusions; it fits whatever slice it is
    given, so calling it on an unclean or test-containing slice is a
    caller error, not something this function can detect.

    `settings` must expose `settings.scaling` (method, analog_columns,
    passthrough_columns) -- the shape of apu_sentinel.config.Settings.
    """
    method = settings.scaling.method
    if method not in VALID_METHODS:
        raise ValueError(f"scaling.method must be one of {VALID_METHODS}, got {method!r}")

    analog_columns = tuple(settings.scaling.analog_columns)
    passthrough_columns = tuple(settings.scaling.passthrough_columns)
    _check_columns(train_df.columns, analog_columns, passthrough_columns)

    center: dict[str, float] = {}
    scale: dict[str, float] = {}
    for col in analog_columns:
        values = train_df[col].to_numpy(dtype=float)
        if method == "robust":
            c = float(np.median(values))
            q75, q25 = np.percentile(values, [75, 25])
            s = float(q75 - q25)
        elif method == "standard":
            c = float(values.mean())
            s = float(values.std())
        else:  # minmax
            c = float(values.min())
            s = float(values.max() - values.min())
        center[col] = c
        scale[col] = s if s != 0 else 1.0  # constant column -- center only, no divide-by-zero

    return FoldScaler(
        method=method,
        analog_columns=analog_columns,
        passthrough_columns=passthrough_columns,
        center_=center,
        scale_=scale,
    )


def transform(df: pd.DataFrame, scaler: FoldScaler) -> pd.DataFrame:
    """Scale analog columns with scaler's already-fitted parameters; pass
    digital/status columns through bit-identical. NEVER fits.

    Never reorders rows, never shuffles, never drops rows -- same index and
    columns as df.

    Raises:
        ValueError: if scaler is unfitted (center_/scale_ still None), or
            if df has a column outside scaler's known analog/passthrough
            columns.
    """
    if not scaler.is_fitted:
        raise ValueError(
            "transform() called with an unfitted FoldScaler -- call "
            "fit_scaler() on the fold's training slice first."
        )

    _check_columns(df.columns, scaler.analog_columns, scaler.passthrough_columns)

    out = df.copy()
    for col in scaler.analog_columns:
        out[col] = (df[col] - scaler.center_[col]) / scaler.scale_[col]
    return out
