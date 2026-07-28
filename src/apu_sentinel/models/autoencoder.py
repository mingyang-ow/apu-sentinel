"""Autoencoder / temporal model. Third stage of the model progression in
CLAUDE.md.

Stub: declares the AnomalyModel interface; logic implemented in a later pass.
"""

from __future__ import annotations

import numpy as np


class AutoencoderModel:
    @property
    def contributor_names(self) -> tuple[str, ...]:
        return ()

    def fit(self, train_data) -> None:
        raise NotImplementedError

    def score(self, data) -> np.ndarray:
        raise NotImplementedError

    def contributions(self, data) -> np.ndarray:
        raise NotImplementedError
