"""Ties data -> regimes -> features -> model -> evaluation -> explain
together, entirely config-driven.

Stub: orchestration logic implemented in a later pass.
"""

from __future__ import annotations

from apu_sentinel.config import Settings


def run_pipeline(settings: Settings) -> dict:
    """Run the full pipeline for one config and return episode-level
    evaluation results.
    """
    raise NotImplementedError
