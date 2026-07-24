"""Typed config schema (pydantic-settings) for apu_sentinel.

Loading model: configs/base.yaml merged with one of configs/local.yaml or
configs/colab.yaml (selected by CONFIG=), optionally overlaid with a
configs/experiment/*.yaml file. Validation is strict -- unknown keys or
wrong types must fail loud at load time, never silently coerce.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/apu_sentinel/config.py -> src/apu_sentinel -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

VALID_CONFIG_NAMES = ("local", "colab")

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


class SplitConfig(BaseModel):
    model_config = _STRICT

    # Time-based boundaries only -- see CLAUDE.md rule 1. Never a ratio.
    train_end: str
    val_end: str


class EvaluationConfig(BaseModel):
    model_config = _STRICT

    # SWEPT candidate pre-failure window widths (minutes) -- a reported
    # result, not a single magic number.
    window_widths: list[int] = Field(default_factory=list)
    episode_hold_time: int | None = None
    # Deferred until baseline behaviour has been observed.
    false_alarm_ceiling: float | None = None


class TrainConfig(BaseModel):
    model_config = _STRICT

    epochs: int
    max_minutes: int


class Settings(BaseSettings):
    """Top-level, merged config for a single run."""

    model_config = SettingsConfigDict(extra="forbid")

    device: str
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    train: TrainConfig
    model: dict = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


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

    return Settings(**merged)
