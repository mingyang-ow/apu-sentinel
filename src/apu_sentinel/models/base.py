"""THE MODEL CONTRACT.

Every model implements this interface; the evaluation layer does everything
else, so all models are scored identically. Models produce ONLY a
per-timestamp anomaly score and per-timestamp, per-contributor contributions
(named by contributor_names -- channels for a model that attributes to
channels, rules for a rule-based model). Thresholding, grouping flagged
timestamps into episodes (via hold-time), checking episodes against
pre-failure windows (detection + lead time), counting false episodes, and
attaching the ranked diagnosis are ALL done centrally in evaluation/. Models
never touch the metric.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyModel(Protocol):
    @property
    def contributor_names(self) -> tuple[str, ...]:
        """Names for the columns of contributions().

        Channel names for models that attribute to channels (e.g. an
        autoencoder's per-channel reconstruction error); rule names for
        rule-based models, whose interpretable attribution is which rule
        fired, not which channel.
        """
        ...

    def fit(self, train_data) -> None: ...

    def score(self, data) -> np.ndarray:
        """Per-timestamp anomaly score."""
        ...

    def contributions(self, data) -> np.ndarray:
        """Per-timestamp contribution per contributor.

        Column order matches contributor_names. Feeds explain/ and the
        mixed, ranked per-episode diagnosis.
        """
        ...
