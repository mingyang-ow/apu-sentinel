"""Tests for the layered config loader (config-pass-brief.md Build Pass 3).

Uses small synthetic YAML fixtures in a temp config_dir rather than the
real configs/, so these stay stable if real config values change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apu_sentinel.config import REPO_ROOT, Settings, load_config

BASE_YAML = {
    "device": "cpu",
    "data": {
        "url": "http://example.com/data.zip",
        "raw_dir": "data/raw",
        "raw_filename": "data.csv",
        "interim_dir": "data/interim",
        "processed_dir": "data/processed",
        "checksum": None,
    },
    "split": {
        "embargo_hours": 24,
        "training_exclusion": {
            "pre_margin_hours": 24,
            "post_settle_hours": 24,
            "fallback_post_hours": 48,
        },
    },
    "evaluation": {
        "window_widths": [],
        "episode_hold_time": None,
        "false_alarm_ceiling": None,
        "failure_events": [],
    },
    "train": {"epochs": 1, "max_minutes": 1},
    "model": {},
}

LOCAL_YAML = {"data": {"subset": 0.05}}

COLAB_YAML = {
    "device": "cuda",
    "data": {"subset": 1.0},
    "train": {"epochs": 50, "max_minutes": 120},
}

EXPERIMENT_YAML = {"train": {"epochs": 999}}


def _write_yaml(path: Path, content: dict) -> None:
    path.write_text(yaml.safe_dump(content))


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    _write_yaml(d / "base.yaml", BASE_YAML)
    _write_yaml(d / "local.yaml", LOCAL_YAML)
    _write_yaml(d / "colab.yaml", COLAB_YAML)
    (d / "experiment").mkdir()
    _write_yaml(d / "experiment" / "exp1.yaml", EXPERIMENT_YAML)
    return d


def test_deep_merge_retains_base_keys_env_overrides_one(config_dir: Path):
    """The important one: local.yaml only sets data.subset -- the rest of
    the base data block (url, raw_dir, raw_filename, checksum, ...) must
    survive. A shallow merge would drop them.
    """
    settings = load_config("local", config_dir=config_dir)

    assert settings.data.url == BASE_YAML["data"]["url"]
    assert settings.data.raw_dir == REPO_ROOT / BASE_YAML["data"]["raw_dir"]
    assert settings.data.raw_filename == BASE_YAML["data"]["raw_filename"]
    assert settings.data.checksum is None
    assert settings.data.subset == 0.05


def test_precedence_base_env_experiment(config_dir: Path):
    settings = load_config("colab", experiment="exp1", config_dir=config_dir)

    # base + colab + experiment all set train.epochs -- experiment wins
    assert settings.train.epochs == 999
    # base + colab set train.max_minutes, no experiment override -- colab wins
    assert settings.train.max_minutes == 120
    # base-only key, untouched by colab or experiment
    assert settings.split.embargo_hours == BASE_YAML["split"]["embargo_hours"]


def test_config_selection_local_vs_colab(config_dir: Path):
    local_settings = load_config("local", config_dir=config_dir)
    colab_settings = load_config("colab", config_dir=config_dir)

    assert local_settings.device == "cpu"
    assert colab_settings.device == "cuda"


def test_config_selection_defaults_to_local(config_dir: Path, monkeypatch):
    monkeypatch.delenv("CONFIG", raising=False)
    settings = load_config(config_dir=config_dir)
    assert settings.device == "cpu"


def test_config_selection_reads_config_env_var(config_dir: Path, monkeypatch):
    monkeypatch.setenv("CONFIG", "colab")
    settings = load_config(config_dir=config_dir)
    assert settings.device == "cuda"


def test_config_selection_rejects_unknown_name(config_dir: Path):
    with pytest.raises(ValueError, match="Unknown CONFIG"):
        load_config("bogus", config_dir=config_dir)


def test_validation_failure_on_wrong_type(config_dir: Path):
    bad_base = {**BASE_YAML, "train": {"epochs": "not-a-number", "max_minutes": 1}}
    _write_yaml(config_dir / "base.yaml", bad_base)

    with pytest.raises(Exception, match="epochs"):
        load_config("local", config_dir=config_dir)


def test_load_config_returns_typed_settings(config_dir: Path):
    settings = load_config("local", config_dir=config_dir)
    assert isinstance(settings, Settings)


def test_data_paths_are_absolute_regardless_of_cwd(config_dir: Path, monkeypatch, tmp_path: Path):
    """raw_dir/interim_dir/processed_dir must resolve against the repo root,
    not the process's CWD -- notebooks and scripts run from anywhere.
    """
    other_cwd = tmp_path / "somewhere" / "else"
    other_cwd.mkdir(parents=True)
    monkeypatch.chdir(other_cwd)

    settings = load_config("local", config_dir=config_dir)

    for path in (settings.data.raw_dir, settings.data.interim_dir, settings.data.processed_dir):
        assert path.is_absolute()

    assert settings.data.raw_dir == REPO_ROOT / BASE_YAML["data"]["raw_dir"]
    assert settings.data.interim_dir == REPO_ROOT / BASE_YAML["data"]["interim_dir"]
    assert settings.data.processed_dir == REPO_ROOT / BASE_YAML["data"]["processed_dir"]
