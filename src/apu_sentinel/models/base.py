"""THE MODEL CONTRACT.

Every model implements this interface; the evaluation layer does everything
else, so all models are scored identically. Models produce ONLY a
per-timestamp anomaly score and per-channel contributions. Thresholding,
grouping flagged timestamps into episodes (via hold-time), checking episodes
against pre-failure windows (detection + lead time), counting false
episodes, and attaching the ranked diagnosis are ALL done centrally in
evaluation/. Models never touch the metric.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyModel(Protocol):
    def fit(self, train_data) -> None: ...

    def score(self, data) -> np.ndarray:
        """Per-timestamp anomaly score."""
        ...

    def channel_contributions(self, data) -> np.ndarray:
        """Per-timestamp, per-channel contribution to the score.

        Feeds explain/ and the mixed, ranked per-episode diagnosis.
        """
        ...
