"""Tests for operating-regime segmentation (regimes/__init__.py).

Four states: LOADED (COMP=0), OFFLOAD/STOPPED (COMP=1, split by
Motor_current against offload_current_threshold), TRANSITION (near any
change). The causality test is THE important one here: regimes/ is not
currently watched by the blocking Claude Code hook (unlike data/split.py
and data/scaling.py), so this suite is the only thing enforcing it.

Uses small synthetic DataFrames -- never the real MetroPT-3 dataset.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from apu_sentinel import regimes as regimes_module
from apu_sentinel.regimes import assign_regimes, characterise_regimes


def _make_df(comp: list[int], motor_current: list[float] | None = None) -> pd.DataFrame:
    """Default Motor_current: ~5.6A (LOADED-like) when COMP=0, ~0.0A
    (STOPPED-level, below the 2.0A offload_current_threshold) when COMP=1 --
    override explicitly to exercise OFFLOAD.
    """
    index = pd.date_range("2020-01-01", periods=len(comp), freq="10s")
    if motor_current is None:
        motor_current = [5.6 if c == 0 else 0.0 for c in comp]
    return pd.DataFrame({"COMP": comp, "Motor_current": motor_current}, index=index)


def test_flags_produce_expected_state_labels(regimes_settings):
    # 50 samples COMP=1 (STOPPED-level current), then 50 samples COMP=0
    # (LOADED). Committed change (after min_duration=30s/3 samples) lands at
    # index 53, TRANSITION through index 59 (transition_settle=60s), LOADED
    # from index 60 -- see steady-state tails only, well clear of both
    # debounce windows.
    df = _make_df([1] * 50 + [0] * 50)
    labels = assign_regimes(df, regimes_settings).astype(str)

    assert (labels.iloc[15:50] == "STOPPED").all()
    assert (labels.iloc[70:100] == "LOADED").all()


def test_four_state_assignment_from_flags_and_currents(regimes_settings):
    # Three long, steady blocks: LOADED (COMP=0), STOPPED (COMP=1, 0.5A),
    # OFFLOAD (COMP=1, 3.8A) -- checked well past both debounce windows.
    comp = [0] * 60 + [1] * 60 + [1] * 60
    motor_current = [5.6] * 60 + [0.5] * 60 + [3.8] * 60
    df = _make_df(comp, motor_current)
    labels = assign_regimes(df, regimes_settings).astype(str)

    assert (labels.iloc[20:60] == "LOADED").all()
    assert (labels.iloc[90:120] == "STOPPED").all()
    assert (labels.iloc[150:180] == "OFFLOAD").all()


def test_threshold_boundary_offload_vs_stopped(regimes_settings):
    # COMP=1 throughout; current just below (1.9A) then just above (2.1A)
    # the 2.0A offload_current_threshold.
    df = _make_df([1] * 120, [1.9] * 60 + [2.1] * 60)
    labels = assign_regimes(df, regimes_settings).astype(str)

    assert (labels.iloc[10:60] == "STOPPED").all()
    assert (labels.iloc[80:120] == "OFFLOAD").all()


def test_causality_labels_unchanged_when_future_deleted(regimes_settings):
    n = 200
    comp = np.ones(n, dtype=int)
    comp[50:53] = 0  # a 30s blip, right at the min_duration boundary
    comp[100:150] = 0  # a long, legitimate LOADED stretch
    comp[150:152] = 1  # a short 20s blip back to STOPPED/OFFLOAD
    comp[152:200] = 0

    motor_current = [5.6 if c == 0 else 0.0 for c in comp]
    # Inject an OFFLOAD sub-segment inside the first COMP=1 stretch (indices
    # 0-49) so the four-state case -- not just LOADED/STOPPED -- is
    # exercised by the causality check.
    for i in range(10, 40):
        motor_current[i] = 3.8
    df = _make_df(list(comp), motor_current)

    full_labels = assign_regimes(df, regimes_settings)

    for cut in (10, 15, 39, 52, 75, 105, 151, 180, 199):
        truncated = df.iloc[: cut + 1]
        truncated_labels = assign_regimes(truncated, regimes_settings)
        assert truncated_labels.iloc[-1] == full_labels.iloc[cut], (
            f"label at cut={cut} changed when future data was deleted -- "
            "assign_regimes is using lookahead"
        )


def test_no_centered_or_backward_filling_operations_in_source():
    source = inspect.getsource(regimes_module)
    assert "center=True" not in source
    assert "centre=True" not in source
    assert "bfill" not in source
    assert "backfill" not in source
    # No positive-lag shift used backwards (a future-referencing shift would
    # be shift(-n) for n > 0).
    assert re.search(r"shift\(\s*-\d", source) is None


def test_state_shorter_than_min_duration_is_absorbed(regimes_settings):
    # min_duration=30s=3 samples. A 2-sample (20s) blip must never surface.
    df = _make_df([1] * 10 + [0, 0] + [1] * 8)
    labels = assign_regimes(df, regimes_settings).astype(str)
    assert "LOADED" not in set(labels)


def test_transition_settle_labels_samples_near_a_change(regimes_settings):
    df = _make_df([1] * 50 + [0] * 50)
    labels = assign_regimes(df, regimes_settings).astype(str)

    # Committed change lands at index 53 (50 + min_duration's 3 samples);
    # TRANSITION for transition_settle=60s (indices 53-59); LOADED from 60.
    assert labels.iloc[52] == "STOPPED"
    assert labels.iloc[53] == "TRANSITION"
    assert labels.iloc[59] == "TRANSITION"
    assert labels.iloc[60] == "LOADED"


def test_alignment_index_matches_input_exactly(regimes_settings):
    df = _make_df([1] * 5 + [0] * 5 + [1] * 5)
    labels = assign_regimes(df, regimes_settings)

    assert len(labels) == len(df)
    assert list(labels.index) == list(df.index)


def test_characterisation_occupancy_sums_to_total(regimes_settings):
    rng = np.random.default_rng(0)
    comp = [1] * 40 + [0] * 40 + [1] * 20
    df = _make_df(comp)
    df["Motor_current"] = rng.normal(size=len(df))

    labels = assign_regimes(df, regimes_settings)
    result = characterise_regimes(df, labels, regimes_settings)

    total_count = sum(v["count"] for v in result["occupancy"].values())
    total_percent = sum(v["percent"] for v in result["occupancy"].values())
    assert total_count == len(df)
    assert total_percent == pytest.approx(100.0)


def test_four_state_occupancy_sums_to_100(regimes_settings):
    # A scenario exercising all four labels (LOADED, STOPPED, OFFLOAD,
    # TRANSITION) at once.
    comp = [0] * 60 + [1] * 60 + [1] * 60
    motor_current = [5.6] * 60 + [0.5] * 60 + [3.8] * 60
    df = _make_df(comp, motor_current)

    labels = assign_regimes(df, regimes_settings)
    result = characterise_regimes(df, labels, regimes_settings)

    assert set(result["occupancy"]) >= {"LOADED", "STOPPED", "OFFLOAD", "TRANSITION"}
    total_count = sum(v["count"] for v in result["occupancy"].values())
    total_percent = sum(v["percent"] for v in result["occupancy"].values())
    assert total_count == len(df)
    assert total_percent == pytest.approx(100.0)
