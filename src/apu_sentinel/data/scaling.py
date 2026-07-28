"""Per-fold, per-regime scaling. CROWN-JEWEL FILE (second guard, alongside
split.py).

Hard rule (CLAUDE.md #2): scalers are fit on the TRAINING window ONLY, never
on val/test, never on full-series stats. Enforced by
tests/test_scaler_train_only.py, which runs as a blocking Claude Code hook on
every edit to src/apu_sentinel/data/ (see .claude/hooks/check_leakage.sh)
and again in the full pytest suite.

Pipeline position (fixed): apply_fold -> assign_regimes -> fit_regime_scalers
(train only) -> transform_by_regime -> make_windows. Regime-conditional
scaling is applied PER-TIMESTAMP, BEFORE windowing -- per docs/FINDINGS.md
§7, a typical LOADED run (median 99s) is far shorter than a window (1800s),
so there is no such thing as a single-regime window; regime handling cannot
happen at window level.

One scaler per (fold, regime): each walk-forward fold (data/split.py) has a
different training slice, and within it, each operating regime
(regimes.assign_regimes: LOADED/OFFLOAD/STOPPED/TRANSITION) has different
statistics -- global per-fold stats are dominated by mode-switching, not
fault behaviour (docs/FINDINGS.md §7). There is no code path that fits one
scaler and reuses it across folds OR across regimes within a fold -- either
would leak statistics from data the fitting call shouldn't see.

TRANSITION gets its own scaler like any other regime. At ~6.25% occupancy
with a state change every couple of minutes (docs/FINDINGS.md §7a), it
cannot be excluded or treated as a gap -- doing so would leave no
contiguous window anywhere. It is a genuine MIXTURE state (samples from
more than one underlying regime, still settling); its statistics describe
that mixture, not a single clean state, and downstream consumers should
treat it as such.

Required order of operations (non-negotiable), per fold:
    train, test = apply_fold(df, fold)         # exclusions already removed
    regimes = assign_regimes(full_df, settings)  # once, on the full raw
                                                  # series -- see
                                                  # fit_regime_scalers'
                                                  # docstring for why this
                                                  # is still causally sound
    train_regimes = regimes.loc[train.index]
    test_regimes = regimes.loc[test.index]
    scalers = fit_regime_scalers(train, train_regimes, settings)  # train only
    train_scaled = transform_by_regime(train, train_regimes, scalers, settings)
    test_scaled = transform_by_regime(test, test_regimes, scalers, settings)

Fitting before exclusions are removed lets the documented failure periods
skew location/spread. Fitting on anything that includes test data is
outright leakage. transform()/transform_by_regime() NEVER fit -- there is
no fit_transform in this module, and both raise on an unfitted scaler
rather than lazily fitting.

Column selection is explicit and config-driven (scaling.analog_columns /
scaling.passthrough_columns). MetroPT-3's 7 analog channels are scaled;
its 8 digital/status channels are meaningless to scale (binary 0/1 flags)
and are passed through bit-identical. A DataFrame column present in
neither list raises -- this catches schema drift loudly instead of
guessing.

Active channels per regime (scaling.active_channels): a channel NOT active
in a regime is set to a constant 0.0 after transform, never scaled --
TP2/DV_pressure vent to ~0 when the compressor stops, so their spread
during STOPPED/OFFLOAD is sensor noise, not signal (docs/FINDINGS.md §7).
They keep their column position (an autoencoder reconstructs a constant
trivially, so reconstruction error is ~0 there and they cannot drive false
alerts) -- the tensor shape made by make_windows must stay stable
regardless of the regime mix in a window.

amplification_warn_factor is WARN-ONLY: it logs, at fit time, any ACTIVE
channel whose within-regime scale implies amplification above the
configured factor. It never substitutes or clamps -- zero_scale_epsilon
remains the ONLY substitution, and only for true division-by-zero.

Default method is `robust` (median / IQR): training data still contains
UNREPORTED anomalies -- only the four documented events can be excluded via
data/split.py's training_exclusion -- so location/spread estimates should
be resistant to the outliers that remain. `standard` and `minmax` are
supported alternatives, selected via config, never hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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
        if abs(s) < settings.scaling.zero_scale_epsilon:
            logger.warning(
                "channel %r: computed scale %.3g is below zero_scale_epsilon "
                "%.3g (constant/near-constant in this fold's training slice) "
                "-- substituting scale=1.0 to avoid dividing by ~0.",
                col,
                s,
                settings.scaling.zero_scale_epsilon,
            )
            s = 1.0
        scale[col] = s

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


def _warn_on_amplification(scaler: FoldScaler, settings, regime: str, fold_id) -> None:
    """Log (never substitute) if an active channel's within-regime scale
    implies amplification above scaling.amplification_warn_factor.
    """
    active = settings.scaling.active_channels.get(regime)
    if active is None:
        raise ValueError(
            f"regime {regime!r} is missing from scaling.active_channels -- "
            "list every regime explicitly rather than guessing which "
            "channels carry information in it."
        )

    warn_factor = settings.scaling.amplification_warn_factor
    fold_label = f"fold {fold_id}" if fold_id is not None else "this fold"
    for col in scaler.analog_columns:
        if col not in active:
            continue
        scale = scaler.scale_[col]
        if scale <= 0:
            continue
        amplification = 1.0 / scale
        if amplification > warn_factor:
            logger.warning(
                "%s, regime %r: active channel %r has scale %.4g -> "
                "amplification %.1fx exceeds scaling.amplification_warn_factor "
                "(%.1f). NOT substituted -- this is a real narrow spread, "
                "surfaced for review, not a numerical bug.",
                fold_label,
                regime,
                col,
                scale,
                amplification,
                warn_factor,
            )


def fit_regime_scalers(
    train_df: pd.DataFrame,
    train_regimes: pd.Series,
    settings,
    fold_id: int | str | None = None,
) -> dict[str, FoldScaler]:
    """Fit one FoldScaler per operating regime present in train_df, each
    restricted to that regime's own rows within this fold's CLEAN training
    slice (train_df, train_regimes -- i.e. the output of apply_fold(df,
    fold) plus regimes sliced onto that same index; see module docstring
    for the full pipeline order).

    fold_id is used only to make error/warning messages identify which
    fold is at fault; pass e.g. a Fold's event_id.

    Raises:
        ValueError: if train_df and train_regimes are misaligned, if a
            (fold, regime) pair has fewer training samples than
            scaling.min_samples_per_regime (a scaler fit on this few
            samples is worse than none), or if a present regime is missing
            from scaling.active_channels.
    """
    if len(train_df) != len(train_regimes) or not train_df.index.equals(train_regimes.index):
        raise ValueError("train_df and train_regimes must share the same index")

    regime_values = train_regimes.astype(str)
    scalers: dict[str, FoldScaler] = {}
    for regime in sorted(regime_values.unique()):
        mask = (regime_values == regime).to_numpy()
        subset = train_df.loc[mask]
        if len(subset) < settings.scaling.min_samples_per_regime:
            fold_label = f"fold {fold_id}" if fold_id is not None else "this fold"
            raise ValueError(
                f"{fold_label}, regime {regime!r}: only {len(subset)} training "
                "sample(s), below scaling.min_samples_per_regime "
                f"({settings.scaling.min_samples_per_regime}) -- a scaler fit on "
                "this few samples is worse than none. Widen the training "
                "window, merge regimes, or lower min_samples_per_regime "
                "deliberately."
            )
        scaler = fit_scaler(subset, settings)
        _warn_on_amplification(scaler, settings, regime, fold_id)
        scalers[regime] = scaler
    return scalers


def transform_by_regime(
    df: pd.DataFrame,
    regimes: pd.Series,
    scalers: dict[str, FoldScaler],
    settings,
) -> pd.DataFrame:
    """Apply each row's OWN regime's fitted FoldScaler (from
    fit_regime_scalers), then zero out any analog channel NOT active in
    that row's regime (scaling.active_channels) -- a constant 0.0, not
    scaled, keeping its column position so the tensor shape make_windows
    produces stays stable across regimes.

    Per-timestamp, BEFORE windowing (see module docstring) -- a typical
    LOADED run (median 99s) is far shorter than a window (1800s), so there
    is no such thing as a single-regime window.

    Never reorders rows, never shuffles, never drops rows -- same index
    and columns as df.

    Raises:
        ValueError: if df and regimes are misaligned, if a regime present
            in `regimes` has no fitted scaler in `scalers` (fit_regime_
            scalers was not run on a training slice that saw this
            regime), or if a present regime is missing from
            scaling.active_channels.
    """
    if len(df) != len(regimes) or not df.index.equals(regimes.index):
        raise ValueError("df and regimes must share the same index")

    active_channels = settings.scaling.active_channels
    regime_values = regimes.astype(str).to_numpy()
    out = df.copy()
    for regime in sorted(set(regime_values)):
        if regime not in scalers:
            raise ValueError(
                f"no fitted scaler for regime {regime!r} -- fit_regime_scalers "
                "must be run on this fold's training slice first, and that "
                "training slice must have contained this regime."
            )
        if regime not in active_channels:
            raise ValueError(
                f"regime {regime!r} is missing from scaling.active_channels -- "
                "list every regime explicitly rather than guessing which "
                "channels carry information in it."
            )

        mask = regime_values == regime
        scaler = scalers[regime]
        transformed = transform(df.loc[mask], scaler)
        inactive = [c for c in scaler.analog_columns if c not in active_channels[regime]]
        for col in inactive:
            transformed[col] = 0.0
        out.loc[mask, transformed.columns] = transformed
    return out
