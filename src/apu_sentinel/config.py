"""Typed config schema (pydantic-settings) for apu_sentinel.

Loading model: configs/base.yaml merged with one of configs/local.yaml or
configs/colab.yaml (selected by CONFIG=), optionally overlaid with a
configs/experiment/*.yaml file. Validation is strict -- unknown keys or
wrong types must fail loud at load time, never silently coerce.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/apu_sentinel/config.py -> src/apu_sentinel -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

VALID_CONFIG_NAMES = ("local", "colab")

# data.* fields holding filesystem paths -- resolved to absolute (against
# REPO_ROOT, not CWD) at load time so callers work regardless of where the
# process was launched from (notebooks, Colab, scripts run from subdirs).
_DATA_PATH_FIELDS = ("raw_dir", "interim_dir", "processed_dir")

_STRICT = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    model_config = _STRICT

    url: str
    raw_dir: Path
    raw_filename: str
    interim_dir: Path
    processed_dir: Path
    # null = establish mode, string = verify mode -- see data/download.py.
    checksum: str | None = None
    subset: float = 1.0


class TrainingExclusionConfig(BaseModel):
    """Fixed, generous margins purging training data around each failure.

    Deliberately kept separate from evaluation.window_widths (the SWEPT
    pre-failure label width) -- one must never be derived from the other.
    All values in hours.
    """

    model_config = _STRICT

    pre_margin_hours: float
    post_settle_hours: float
    # Used instead of post_settle_hours when an event's maintenance
    # timestamp is unrecorded (null) -- see evaluation.failure_events.
    fallback_post_hours: float


class SplitConfig(BaseModel):
    model_config = _STRICT

    # Gap enforced between a fold's train_end and test_start. Must be >=
    # the longest sequence window any later windowing pass will use.
    embargo_hours: float
    training_exclusion: TrainingExclusionConfig


class FailureEvent(BaseModel):
    """One documented MetroPT-3 failure event. Timestamps are kept as
    strings here (parsed to pd.Timestamp downstream in data/split.py),
    matching the existing "timestamps as strings in config" convention.
    """

    model_config = _STRICT

    id: int
    start: str
    end: str
    # null when the source table has no maintenance/repair entry for this
    # event (see event 1's note) -- callers must fall back to
    # training_exclusion.fallback_post_hours in that case.
    maintenance: str | None = None
    note: str | None = None


class MaskedRegionConfig(BaseModel):
    """An explicitly configured region excluded from false-alarm counting,
    in addition to the automatic per-event masked regions -- see
    evaluation/events.py masked_regions().
    """

    model_config = _STRICT

    start: str
    end: str
    note: str | None = None


class EvaluationConfig(BaseModel):
    model_config = _STRICT

    # SWEPT candidate pre-failure window widths (hours) -- a reported
    # result, not a single magic number. The widest value must respect the
    # event-2/event-3 proximity cap enforced by data/split.py.
    window_widths: list[float] = Field(default_factory=list)

    # Threshold is fit on TRAINING scores ONLY (evaluation/metrics.py
    # fit_threshold) -- a high quantile, never a hardcoded score value.
    threshold_quantile: float = 0.995
    # Optional sweep: fit_threshold_sweep() fits ALL of these on train
    # scores and returns the whole curve -- picking the test-optimal one
    # and reporting it as "the" result is forbidden (see fit_threshold_sweep
    # docstring).
    threshold_quantiles: list[float] = Field(default_factory=list)

    # Hysteresis / hold-time: a below-threshold stretch shorter than this
    # does not end an episode. Pandas duration string (e.g. "10min").
    episode_hold_time: str = "10min"
    # A gap in the score timeline (from dropped windows) longer than this
    # ENDS an episode outright, regardless of episode_hold_time -- absence
    # of evidence is not evidence of continuation. Pandas duration string.
    score_gap_threshold: str = "30min"
    # Episodes shorter than this are dropped. "0min" (default) = OFF --
    # enabling this is a knowing choice, not an automatic default.
    min_episode_duration: str = "0min"
    # How per-timestamp channel contributions are aggregated across an
    # episode's timestamps into its ranked diagnosis (explain/).
    contribution_aggregation: Literal["mean", "max"] = "mean"

    # Deferred until baseline behaviour has been observed.
    false_alarm_ceiling: float | None = None
    failure_events: list[FailureEvent] = Field(default_factory=list)
    # Extra masked regions beyond the automatic per-event ones (e.g. a
    # known sensor outage unrelated to a documented failure).
    additional_masked_regions: list[MaskedRegionConfig] = Field(default_factory=list)


class TrainConfig(BaseModel):
    model_config = _STRICT

    epochs: int
    max_minutes: int


class ScalingConfig(BaseModel):
    """Column selection is explicit and config-driven -- see
    data/scaling.py. A DataFrame column absent from BOTH analog_columns and
    passthrough_columns must raise, never be silently guessed at.
    """

    model_config = _STRICT

    method: Literal["robust", "standard", "minmax"] = "robust"
    analog_columns: list[str]
    passthrough_columns: list[str]
    # A computed scale below this is treated as zero (constant/near-constant
    # channel) -- substituted with 1.0 rather than dividing by ~0. See
    # data/scaling.py fit_scaler().
    zero_scale_epsilon: float = 1e-8
    # Regime-conditional scaling (data/scaling.py fit_regime_scalers /
    # transform_by_regime): a (fold, regime) pair with fewer training
    # samples than this raises rather than fitting a near-meaningless
    # scaler. Empty dict default keeps this field opt-in for callers that
    # don't do regime-conditional scaling at all.
    min_samples_per_regime: int = 100
    # regime name -> analog channels that carry information in that state.
    # Every regime assign_regimes() can produce (LOADED/OFFLOAD/STOPPED/
    # TRANSITION) must have an explicit entry when regime-conditional
    # scaling is used -- a channel NOT listed for a regime is set to a
    # constant 0.0 after transform (docs/FINDINGS.md §7: TP2/DV_pressure
    # vent to ~0 when the compressor stops; their OFF-state spread is
    # sensor noise, not signal -- scaling it would raise ~250x
    # amplification to ~500x for no benefit). Empty by default -- opt-in.
    active_channels: dict[str, list[str]] = Field(default_factory=dict)
    # Warn-only (never substitutes/clamps): after fitting a regime scaler,
    # log a WARNING naming any ACTIVE channel whose within-regime scale
    # implies amplification (1/scale) above this factor -- surfaces future
    # instances of the TP2-style pathology instead of hiding them.
    amplification_warn_factor: float = 100.0


class ResampleConfig(BaseModel):
    """Resampling to a regular grid -- OFF by default. Enabling it is a
    modeling decision the user should make knowingly after reviewing
    data/windows.py characterise_sampling()'s output, not an automatic
    default.
    """

    model_config = _STRICT

    enabled: bool = False
    # Pandas duration string (e.g. "1min"), parsed downstream in
    # data/windows.py -- never hardcoded there.
    interval: str = "1min"


class WindowingConfig(BaseModel):
    """Durations are pandas duration strings (e.g. "30min"), parsed to
    pd.Timedelta in data/windows.py and converted to a sample count using
    the empirically measured expected_interval -- never a hardcoded sample
    count here.
    """

    model_config = _STRICT

    window_duration: str
    train_stride: str
    score_stride: str
    # Fractional slack on a window's expected wall-clock span before it is
    # dropped for spanning a gap.
    gap_tolerance: float = 0.1
    # What counts as a "gap" when characterise_sampling() reports on the
    # raw sampling (a separate concern from gap_tolerance, which governs
    # per-window drop decisions).
    gap_threshold: str
    resample: ResampleConfig = Field(default_factory=ResampleConfig)


class RegimesConfig(BaseModel):
    """Operating-regime segmentation, derived from the compressor's own
    digital control signals -- NOT by clustering the analog channels (see
    regimes/__init__.py). No hardcoded flag names, polarities, or durations
    in code; all of it lives here.
    """

    model_config = _STRICT

    # Cross-referenced against each other in regimes.verify_flag_semantics
    # (their pairwise agreement rate, after normalising via `polarity`, is
    # logged as evidence for which one to trust as the deciding flag).
    control_columns: list[str]
    # Empirically-derived (regimes.verify_flag_semantics): raw flag value ->
    # normalised OFF/ON reading. Filled from evidence against Motor_current,
    # NEVER assumed -- a naive reading can be exactly backwards (see the
    # pass-9 write-up on COMP).
    polarity: dict[str, dict[int, str]]
    # Explicit condition -> state name: {state_name: {flag_name: required
    # raw value}}. Checked in this dict's order; the first state whose
    # listed flag(s) all match a row wins. Rows matching none of these
    # (i.e. "no air intake") are further split into OFFLOAD/STOPPED below --
    # see regimes/__init__.py.
    states: dict[str, dict[str, int]]
    # Channel + threshold used to split whatever doesn't match `states`
    # into OFFLOAD (>= threshold) vs STOPPED (< threshold). 2.0A is the
    # empirically-justified default -- see docs/FINDINGS.md §7: only
    # 0.015% of OFF samples fall in the 1-3A valley between the two modes,
    # so 2.0A sits in genuine emptiness, not an arbitrary split point.
    offload_split_channel: str = "Motor_current"
    offload_current_threshold: float = 2.0
    # A newly-changed raw state is committed only once it has persisted (a
    # backward-looking check) for at least this long -- shorter blips are
    # absorbed into the preceding committed state, never emitted as their
    # own regime. Pandas duration string.
    min_duration: str
    # Samples within this long of a committed state change are labelled
    # TRANSITION rather than the new state, since analog channels are
    # still settling. Pandas duration string.
    transition_settle: str
    # Records the INTENT to optionally drop offload_split_channel
    # (Motor_current) from scored channels while in an OFFLOAD/STOPPED
    # state, to remove the circularity of using it both to DEFINE those
    # states and to SCORE them for anomalies (docs/FINDINGS.md §9). Default
    # false so enabling it is a knowing choice. NOT wired into scoring --
    # scoring does not exist yet.
    exclude_motor_current_when_off: bool = False


class FeaturesConfig(BaseModel):
    """Causal cycle-timing features (features/cycles.py) -- duration and
    pressure-decay families, both built so a later baseline pass can
    choose between them on evidence (docs/FINDINGS.md §8).
    """

    model_config = _STRICT

    # Pressure channel the decay-rate family reads. Reservoirs is the
    # default; TP3 is a near-duplicate (docs/FINDINGS.md §5) and is an
    # equally valid alternative, never hardcoded.
    decay_source_channel: str = "Reservoirs"
    # decay_rate_running is NaN until this many samples have accumulated
    # within the current run -- a slope from 1-2 points is noise.
    decay_min_samples: int = 3
    # What counts as a data gap for run-boundary purposes here -- matches
    # the convention established in analysis.monthly_gap_and_stopped_summary
    # (docs/FINDINGS.md §9): a run split by a gap this long or longer gets
    # an invalid (NaN) duration but keeps a valid decay rate over its
    # observed samples. Pandas duration string.
    gap_threshold: str = "1min"
    # Trailing window for baseline-relative (value / trailing_median)
    # variants. Pandas duration string. 7 days is a deliberate choice, not
    # a default to take for granted -- see features/cycles.py docstring:
    # short enough that event 2's ~3x step change survives it, long enough
    # that gradual seasonal drift (docs/FINDINGS.md §8) isn't itself
    # absorbed into "normal".
    baseline_window: str = "7D"
    # Trailing window for duty_ratio_trailing (LOADED fraction of time) --
    # deliberately much shorter than baseline_window: this is a
    # current-state indicator, not a drift baseline.
    duty_ratio_window: str = "1h"


class Settings(BaseSettings):
    """Top-level, merged config for a single run."""

    model_config = SettingsConfigDict(extra="forbid")

    device: str
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    scaling: ScalingConfig
    windowing: WindowingConfig
    regimes: RegimesConfig
    features: FeaturesConfig
    train: TrainConfig
    model: dict = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_data_paths(data: dict[str, Any]) -> None:
    """Resolve data.* path fields to absolute, in place, against REPO_ROOT.

    Relative paths in configs/*.yaml (e.g. "data/raw") are meant relative to
    the repo root, not whatever directory the process happens to be running
    from. An already-absolute value (e.g. an experiment override) is left
    untouched.
    """
    for field in _DATA_PATH_FIELDS:
        value = data.get(field)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            data[field] = str((REPO_ROOT / path).resolve())


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto base; overlay wins on conflicts, but
    keys base has that overlay doesn't are kept (never a shallow replace of
    nested dicts).
    """
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_name: str | None = None,
    experiment: str | None = None,
    config_dir: Path | None = None,
) -> Settings:
    """Load configs/base.yaml, merge configs/{config_name}.yaml on top, then
    optionally overlay configs/experiment/{experiment}.yaml. Later layers
    override earlier ones; nested dicts are merged key-by-key (deep merge),
    not replaced wholesale.

    config_name resolution: explicit argument > CONFIG env var > "local"
    (the safe CPU/subset default). Raises ValueError for an unrecognized
    name.

    data.* path fields (raw_dir, interim_dir, processed_dir) are resolved to
    absolute paths against the repo root before validation, so callers work
    regardless of the process's current working directory.

    The merged dict is validated through the Settings schema -- a wrong
    type or unknown key fails here, loudly, rather than later in a training
    loop.

    config_dir defaults to the repo's configs/ directory (resolved from
    this file's location, not the working directory); overridable for tests.
    """
    if config_name is None:
        config_name = os.environ.get("CONFIG", "local")
    if config_name not in VALID_CONFIG_NAMES:
        raise ValueError(
            f"Unknown CONFIG={config_name!r} -- valid options: {', '.join(VALID_CONFIG_NAMES)}"
        )

    if config_dir is None:
        config_dir = CONFIG_DIR

    merged = _load_yaml(config_dir / "base.yaml")
    merged = _deep_merge(merged, _load_yaml(config_dir / f"{config_name}.yaml"))

    if experiment is not None:
        merged = _deep_merge(merged, _load_yaml(config_dir / "experiment" / f"{experiment}.yaml"))

    if "data" in merged:
        _resolve_data_paths(merged["data"])

    return Settings(**merged)
