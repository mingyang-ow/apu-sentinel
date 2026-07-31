#!/usr/bin/env python3
"""Thin training entry-point that wraps apu_sentinel.pipeline in three
layers of Colab-credit protection (see CLAUDE.md / scaffolding-brief §8):

1. Self-terminating run: training runs inside try/finally -- on completion
   OR exception, the result is logged, MLflow is flushed, and the process
   exits cleanly. No dangling live computation after the job.
2. Wall-clock budget guard: aborts the run if it exceeds
   config.train.max_minutes (protects against hangs / non-converging loops
   quietly eating GPU hours).
3. Best-effort runtime disconnect: on Colab, attempts runtime.unassign(),
   guarded in try/except. This is NOT a guarantee -- code cannot force
   Google to reclaim the runtime or stop billing, only stop doing compute
   here and ask the user to verify in the Colab UI.
"""

from __future__ import annotations

import argparse
import signal
import sys

from apu_sentinel.config import load_config
from apu_sentinel.pipeline import (
    run_pipeline,
    run_pipeline_autoencoder,
    run_pipeline_isolation_forest,
)

PIPELINES = {
    "rule_based": run_pipeline,
    "isolation_forest": run_pipeline_isolation_forest,
    "autoencoder": run_pipeline_autoencoder,
}


class TrainingBudgetExceeded(Exception):
    """Raised when a run exceeds its configured train.max_minutes budget."""


def _install_wall_clock_guard(max_minutes: float) -> None:
    def _on_alarm(signum, frame):
        raise TrainingBudgetExceeded(f"training exceeded max_minutes={max_minutes}")

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(max_minutes * 60))


def _best_effort_colab_disconnect() -> None:
    try:
        from google.colab import runtime  # type: ignore

        runtime.unassign()
    except Exception:
        pass
    finally:
        print(
            "NOTE: attempted best-effort Colab runtime disconnect. "
            "This is NOT guaranteed -- verify in the Colab UI that the "
            "session actually ended."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local", choices=["local", "colab"])
    parser.add_argument("--model", default="rule_based", choices=list(PIPELINES))
    args = parser.parse_args()

    settings = load_config(args.config)
    _install_wall_clock_guard(settings.train.max_minutes)

    result = None
    try:
        result = PIPELINES[args.model](settings)
    finally:
        signal.alarm(0)
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass
        print(f"train.py finished. result={result!r}")
        if args.config == "colab":
            _best_effort_colab_disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
