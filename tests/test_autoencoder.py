"""Tests for the Convolutional Autoencoder model (models/autoencoder.py).

Small synthetic window tensors only -- never the real MetroPT-3 dataset.
Each test isolates one property of the AnomalyModel contract or the
training discipline, following tests/test_isolation_forest.py's existing
pattern for this project. CPU-only, few epochs, small tensors -- must run
fast.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import apu_sentinel.models.autoencoder as autoencoder_module
from apu_sentinel.config import AutoencoderModelConfig
from apu_sentinel.models.autoencoder import AutoencoderModel
from apu_sentinel.models.base import AnomalyModel
from apu_sentinel.models.isolation_forest import WindowedInput

N_CHANNELS = 3
WINDOW_LENGTH = 10
CHANNEL_NAMES = ("TP2", "TP3", "Reservoirs")


def _settings(
    channels: list[int] | None = None,
    kernel_size: int = 3,
    bottleneck_dim: int = 4,
    dropout: float = 0.0,
    activation: str = "relu",
    learning_rate: float = 0.02,
    batch_size: int = 32,
    random_state: int = 42,
    val_fraction: float = 0.15,
    patience: int = 5,
    deterministic: bool = False,
    epochs: int = 5,
    max_minutes: float = 5,
    device: str = "cpu",
):
    cfg = AutoencoderModelConfig(
        channels=channels if channels is not None else [8, 4],
        kernel_size=kernel_size,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
        activation=activation,
        learning_rate=learning_rate,
        batch_size=batch_size,
        random_state=random_state,
        val_fraction=val_fraction,
        patience=patience,
        deterministic=deterministic,
    )
    return SimpleNamespace(
        model=SimpleNamespace(autoencoder=cfg),
        train=SimpleNamespace(epochs=epochs, max_minutes=max_minutes),
        device=device,
    )


def _windows(n_windows: int, seed: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=(n_windows, WINDOW_LENGTH, N_CHANNELS)).astype(
        np.float32
    )


def _end_timestamps(n_windows: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_windows, freq="1min")


def _data(windows: np.ndarray) -> WindowedInput:
    return WindowedInput(
        windows=windows,
        end_timestamps=_end_timestamps(windows.shape[0]),
        channel_names=CHANNEL_NAMES,
        cycle_features=None,
    )


# --- 1. Protocol conformance -------------------------------------------


def test_protocol_conformance():
    model = AutoencoderModel(_settings())
    windows = _windows(60, seed=0)
    data = _data(windows)

    model.fit(data)
    assert isinstance(model, AnomalyModel)

    scores = model.score(data)
    contributions = model.contributions(data)

    assert scores.shape == (60,)
    assert contributions.shape == (60, N_CHANNELS)
    assert len(model.contributor_names) == N_CHANNELS


# --- 2. Score direction --------------------------------------------------


def test_score_direction_planted_outlier_scores_higher():
    model = AutoencoderModel(_settings(epochs=15))
    train_windows = _windows(300, seed=1)
    model.fit(_data(train_windows))

    normal_windows = _windows(30, seed=2)
    outlier_windows = normal_windows.copy()
    outlier_windows[0] += 50.0  # a single, sharply displaced window

    scores = model.score(_data(outlier_windows))
    assert scores[0] > np.median(scores[1:])


# --- 3. Per-channel attribution ------------------------------------------


def test_per_channel_attribution_ranks_corrupted_channel_top():
    model = AutoencoderModel(_settings(epochs=10))
    train_windows = _windows(300, seed=3)
    model.fit(_data(train_windows))

    probe = _windows(20, seed=4)
    probe[5, :, 1] += 40.0  # channel index 1 ("TP3"), row 5 only

    contributions = model.contributions(_data(probe))
    top_channel = model.contributor_names[np.argmax(contributions[5])]
    assert top_channel == "TP3"


# --- 4. Fit is train-only -------------------------------------------------


def test_fit_is_train_only():
    settings = _settings(epochs=10)
    train_windows = _windows(300, seed=5)

    contaminating_anomaly = _windows(20, seed=5, loc=100.0, scale=1.0)
    train_plus_test_windows = np.concatenate([train_windows, contaminating_anomaly], axis=0)

    clean_model = AutoencoderModel(settings)
    clean_model.fit(_data(train_windows))

    contaminated_model = AutoencoderModel(settings)
    contaminated_model.fit(
        _data(train_plus_test_windows)
    )  # fit sees the "test" anomaly -- contamination

    probe = _windows(30, seed=6)
    clean_scores = clean_model.score(_data(probe))
    contaminated_scores = contaminated_model.score(_data(probe))

    assert not np.allclose(clean_scores, contaminated_scores)


# --- 5. Validation split is time-based ------------------------------------


def test_validation_split_is_time_based():
    model = AutoencoderModel(_settings(val_fraction=0.2, epochs=1))
    windows = _windows(50, seed=7)
    model.fit(_data(windows))

    assert len(model.train_timestamps_) + len(model.val_timestamps_) == 50
    assert len(model.val_timestamps_) == round(50 * 0.2)
    assert model.train_timestamps_.max() < model.val_timestamps_.min()


# --- 6. Determinism --------------------------------------------------------


def test_determinism_same_seed_same_scores_different_seed_different():
    train_windows = _windows(100, seed=8)
    probe = _windows(20, seed=9)

    model_1 = AutoencoderModel(_settings(random_state=42, epochs=5))
    model_1.fit(_data(train_windows))
    model_2 = AutoencoderModel(_settings(random_state=42, epochs=5))
    model_2.fit(_data(train_windows))

    scores_1 = model_1.score(_data(probe))
    scores_2 = model_2.score(_data(probe))
    assert np.allclose(scores_1, scores_2, rtol=1e-5, atol=1e-6)

    model_3 = AutoencoderModel(_settings(random_state=7, epochs=5))
    model_3.fit(_data(train_windows))
    scores_3 = model_3.score(_data(probe))
    assert not np.allclose(scores_1, scores_3)


# --- 7. Constant channels ---------------------------------------------------


def test_constant_channel_near_zero_contribution():
    model = AutoencoderModel(_settings(epochs=30, learning_rate=0.03))
    rng = np.random.default_rng(10)
    windows = rng.normal(size=(200, WINDOW_LENGTH, N_CHANNELS)).astype(np.float32)
    windows[:, :, 2] = 0.0  # "Reservoirs" held constant, like a regime-inactive channel
    model.fit(_data(windows))

    probe = rng.normal(size=(20, WINDOW_LENGTH, N_CHANNELS)).astype(np.float32)
    probe[:, :, 2] = 0.0
    contributions = model.contributions(_data(probe))

    assert contributions[:, 2].mean() < 0.1 * contributions[:, :2].mean()


# --- 8. max_minutes respected -----------------------------------------------


def test_max_minutes_respected_aborts_and_reports(monkeypatch):
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        return 0.0 if calls["n"] <= 2 else 1e9

    monkeypatch.setattr(autoencoder_module.time, "monotonic", fake_monotonic)

    settings = _settings(epochs=20, max_minutes=1)
    model = AutoencoderModel(settings)
    train_windows = _windows(60, seed=11)
    model.fit(_data(train_windows))

    assert 0 < model.epochs_run_ < 20


def test_zero_arg_constructible():
    """models/base.py's AnomalyModel contract: every model must be
    constructible with zero arguments, falling back to config.load_config()
    -- tests/test_eval_contract.py parametrizes every model this way.
    """
    model = AutoencoderModel()
    assert isinstance(model, AnomalyModel)


# --- 9. Shape round-trip -----------------------------------------------------


def test_shape_round_trip_various_window_lengths_and_depths():
    """Encoder/decoder must return exactly the input shape, even when
    window_length doesn't divide evenly by 2**len(channels) -- the case a
    naive mirrored stride-2 stack gets subtly wrong (see
    _ConvAutoencoderNet's output_padding derivation).
    """
    import torch

    cases = [
        (13, [8, 4], 3),  # odd length, two stride-2 layers
        (30, [16, 8, 4], 3),  # three layers
        (7, [6], 3),  # single layer, small length
        (180, [32, 16], 5),  # real config.py default shape/kernel
    ]
    for window_length, channels, kernel_size in cases:
        net = autoencoder_module._ConvAutoencoderNet(
            n_channels=N_CHANNELS,
            window_length=window_length,
            channels=channels,
            kernel_size=kernel_size,
            bottleneck_dim=4,
            dropout=0.0,
            activation="relu",
        )
        x = torch.randn(2, window_length, N_CHANNELS)
        out = net(x)
        assert out.shape == x.shape, (window_length, channels, kernel_size)
