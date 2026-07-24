"""End-to-end smoke test: run the whole pipeline on the tiny synthetic
fixture and assert output shapes. Shape-only -- no correctness claims.
"""

from __future__ import annotations

import pytest


def test_pipeline_runs_end_to_end_on_fixture(synthetic_series):
    pytest.skip("stub — implement with logic")
