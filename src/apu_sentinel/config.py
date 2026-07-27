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


class EvaluationConfig(BaseModel):
    model_config = _STRICT

    # SWEPT candidate pre-failure window widths (hours) -- a reported
    # result, not a single magic number. The widest value must respect the
    # event-2/event-3 proximity cap enforced by data/split.py.
    window_widths: list[float] = Field(default_factory=list)
    episode_hold_time: int | None = None
    # Deferred until baseline behaviour has been observed.
    false_alarm_ceiling: float | None = None
    failure_events: list[FailureEvent] = Field(default_factory=list)


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


class Settings(BaseSettings):
    """Top-level, merged config for a single run."""

    model_config = SettingsConfigDict(extra="forbid")

    device: str
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    scaling: ScalingConfig
    windowing: WindowingConfig
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
