"""Rule-based baseline. First stage of the model progression in CLAUDE.md.

Stub: declares the AnomalyModel interface; logic implemented in a later pass.
"""

from __future__ import annotations

import numpy as np


class RuleBasedModel:
    def fit(self, train_data) -> None:
        raise NotImplementedError

    def score(self, data) -> np.ndarray:
        raise NotImplementedError

    def channel_contributions(self, data) -> np.ndarray:
        raise NotImplementedError
