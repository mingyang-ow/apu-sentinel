"""Typed config schema (pydantic-settings) for apu_sentinel.

Loading model: configs/base.yaml merged with one of configs/local.yaml or
configs/colab.yaml (selected by CONFIG=), optionally overlaid with a
configs/experiment/*.yaml file. Validation is strict -- unknown keys or
wrong types must fail loud at load time, never silently coerce.

This module currently defines the schema only; merge/load logic is
implemented in a later pass.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    checksum: str
    subset: float = 1.0


class SplitConfig(BaseModel):
    # Time-based boundaries only -- see CLAUDE.md rule 1. Never a ratio.
    train_end: str
    val_end: str


class EvaluationConfig(BaseModel):
    # SWEPT candidate pre-failure window widths (minutes) -- a reported
    # result, not a single magic number.
    window_widths: list[int] = Field(default_factory=list)
    episode_hold_time: int | None = None
    # Deferred until baseline behaviour has been observed.
    false_alarm_ceiling: float | None = None


class TrainConfig(BaseModel):
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


def load_config(config_name: str = "local", experiment: str | None = None) -> Settings:
    """Load configs/base.yaml, merge configs/{config_name}.yaml on top, then
    optionally overlay configs/experiment/{experiment}.yaml.

    Stub: merge/load logic not yet implemented.
    """
    raise NotImplementedError
