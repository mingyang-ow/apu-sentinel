"""1D Convolutional Autoencoder. Third and final stage of the model
progression in CLAUDE.md.

Rule-based and Isolation Forest both key on per-channel or per-feature
deviation. This model tests the one mechanism neither can see: whether
MULTIVARIATE RELATIONSHIPS between channels break down before failure (e.g.
pressure and motor current drifting apart while each stays individually in
range). Reconstruction error is naturally "higher = more anomalous" (no
score negation needed, unlike Isolation Forest's score_samples()), and
per-channel reconstruction error is exactly the diagnosis explain/ needs --
the first model in this project where attribution falls out of the
architecture for free, rather than needing a separate ablation pass
(models/isolation_forest.py's O(n_features) re-scoring).

Conv, not LSTM (pass 23 v2 -- replaces the pass 23 LSTM version, which hung
on CPU): convolutions parallelise across the time axis instead of stepping
through every timestep sequentially, so all CPU cores are actually used.
The cross-channel hypothesis above needs no sequential modelling either --
at a 30-minute window there is little long-range temporal dependency to
capture in the first place.

Input: the 15 scaled channels only (data/scaling.py's analog_columns +
passthrough_columns), no cycle features -- this tests the distinct
multivariate-relationship hypothesis, and it deliberately sidesteps the
NaN-imputation path documented in docs/RESULTS.md §22 Part A1
(gap-truncated cycle-feature runs produce NaN, silently imputed to a
training median by IsolationForestModel._fill_nan; this model never faces
that decision because it never reads cycle features at all).

`data` is `models.isolation_forest.WindowedInput` with `cycle_features`
always `None` -- reused rather than declaring a parallel dataclass, since
its shape (windows, end_timestamps, channel_names, cycle_features) is
already model-agnostic and its `.index` already aliases end_timestamps for
pipeline.py's model-agnostic evaluation helpers.

Regime-inactive channels are held at a constant 0.0 by
data/scaling.py's transform_by_regime (docs/ARCHITECTURE.md). The
autoencoder reconstructs a constant trivially, so those channels contribute
~0 reconstruction error -- intended, not a bug: an inactive channel can
never dominate the ranked diagnosis by construction.

Training discipline (one model per fold, fit on that fold's clean training
windows only):
- Internal early-stopping validation split is TIME-BASED: the TRAILING
  `model.autoencoder.val_fraction` of the (already time-ordered, per
  data/windows.py make_windows()'s own guarantee) training windows becomes
  validation -- never a random sample, which would leak windows temporally
  adjacent to training ones into "held out" data.
- Determinism: `torch.manual_seed(model.autoencoder.random_state)` is set at
  the start of every fit() call; minibatches are never shuffled (fixed,
  ascending order), so a fit is fully reproducible given the same seed and
  data on CPU without needing a seeded shuffle RNG. Full bit-for-bit GPU
  determinism additionally needs `torch.use_deterministic_algorithms(True)`
  plus `CUBLAS_WORKSPACE_CONFIG` -- both cost speed, so
  `model.autoencoder.deterministic` (default off) gates them; CPU runs are
  unaffected either way.
- `train.max_minutes` (pass 1's wall-clock budget) is checked once per
  epoch boundary: if exceeded, fit() logs a warning and returns whatever has
  already trained rather than raising or running over budget.

Public API:
- `AutoencoderModel(settings=None)` -- implements models/base.py's
  AnomalyModel. `settings=None` falls back to `config.load_config()`.
  After `fit()`, exposes fitted-attribute reporting (sklearn-style trailing
  underscore, matching data/scaling.py's FoldScaler convention):
  `epochs_run_`, `final_train_loss_`, `final_val_loss_`, `elapsed_seconds_`,
  `device_`, `train_timestamps_`, `val_timestamps_`.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from apu_sentinel.config import load_config
from apu_sentinel.models.isolation_forest import WindowedInput

logger = logging.getLogger(__name__)


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}


class _ConvAutoencoderNet(nn.Module):
    """Encoder: stacked stride-2 Conv1d, narrowing both channel width (per
    `channels`) and time length, then a linear bottleneck over the
    flattened result. Decoder: mirrored ConvTranspose1d, narrowing back to
    `n_channels`.

    Operates internally in (batch, channels, time) layout (conv1d's native
    layout); forward() transposes in and out so callers keep working in
    this project's (batch, time, channels) convention (data/windows.py
    make_windows()).

    Each decoder layer's `output_padding` is solved exactly from the
    corresponding encoder layer's own (input_length, output_length) pair,
    rather than assumed -- window_length rarely divides evenly by
    2**len(channels), so a naive mirrored stack would silently round to the
    wrong length. Solving `ConvTranspose1d`'s length formula for
    output_padding given the exact target guarantees the round trip lands
    on window_length exactly (see tests/test_autoencoder.py's shape
    round-trip test); raises loudly if a (window_length, channels,
    kernel_size) combination can't satisfy `output_padding`'s valid
    [0, stride) range instead of building a net that would silently
    mis-shape at forward() time.
    """

    _STRIDE = 2

    def __init__(
        self,
        n_channels: int,
        window_length: int,
        channels: list[int],
        kernel_size: int,
        bottleneck_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"model.autoencoder.kernel_size must be odd, got {kernel_size}")
        if not channels:
            raise ValueError("model.autoencoder.channels must list at least one layer width")

        act_cls = _ACTIVATIONS[activation]
        padding = kernel_size // 2
        stride = self._STRIDE

        def conv_out_length(length: int) -> int:
            return (length + 2 * padding - kernel_size) // stride + 1

        encoder_widths = [n_channels, *channels]
        lengths = [window_length]
        for _ in channels:
            length = conv_out_length(lengths[-1])
            if length < 1:
                raise ValueError(
                    f"model.autoencoder.channels ({channels}) narrows the "
                    f"{window_length}-sample window below 1 timestep -- reduce "
                    "channels' depth or kernel_size"
                )
            lengths.append(length)

        encoder_layers: list[nn.Module] = []
        for i in range(len(channels)):
            encoder_layers.append(
                nn.Conv1d(encoder_widths[i], encoder_widths[i + 1], kernel_size, stride, padding)
            )
            encoder_layers.append(act_cls())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        self._bottleneck_channels = encoder_widths[-1]
        self._bottleneck_length = lengths[-1]
        bottleneck_flat = self._bottleneck_channels * self._bottleneck_length
        self.to_bottleneck = nn.Linear(bottleneck_flat, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, bottleneck_flat)

        decoder_widths = list(reversed(encoder_widths))
        decoder_lengths = list(reversed(lengths))
        decoder_layers: list[nn.Module] = []
        for i in range(len(channels)):
            in_len, out_len = decoder_lengths[i], decoder_lengths[i + 1]
            output_padding = out_len - ((in_len - 1) * stride - 2 * padding + kernel_size)
            if not (0 <= output_padding < stride):
                raise ValueError(
                    f"model.autoencoder.channels ({channels}) with window_length="
                    f"{window_length}, kernel_size={kernel_size} cannot round-trip "
                    f"exactly (computed output_padding={output_padding} at decoder "
                    f"layer {i}, outside ConvTranspose1d's valid [0, {stride}) range)"
                )
            decoder_layers.append(
                nn.ConvTranspose1d(
                    decoder_widths[i],
                    decoder_widths[i + 1],
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )
            if i < len(channels) - 1:
                decoder_layers.append(act_cls())
                if dropout > 0:
                    decoder_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_chw = x.transpose(1, 2)  # (batch, time, channels) -> (batch, channels, time)
        encoded = self.encoder(x_chw)
        bottleneck = self.to_bottleneck(encoded.flatten(1))
        seed = self.from_bottleneck(bottleneck).view(
            -1, self._bottleneck_channels, self._bottleneck_length
        )
        decoded = self.decoder(seed)
        return decoded.transpose(1, 2)  # back to (batch, time, channels)


class AutoencoderModel:
    """Convolutional autoencoder AnomalyModel. See module docstring for the
    input shape, score direction, training discipline, and fitted-attribute
    reporting.

    `settings` defaults to the merged app config (load_config()) so the
    class remains zero-argument constructible (tests/test_eval_contract.py
    parametrizes MODEL_CLASSES this way); pass an explicit duck-typed
    settings object (exposing .model.autoencoder, .train, .device -- the
    shape of apu_sentinel.config.Settings) for tests against synthetic data.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = load_config()
        cfg = settings.model.autoencoder
        if cfg is None:
            raise ValueError(
                "AutoencoderModel requires settings.model.autoencoder to be configured"
            )
        self._settings = settings
        self._cfg = cfg
        self._device = torch.device(settings.device)
        self._net: _ConvAutoencoderNet | None = None
        self._channel_names: tuple[str, ...] = ()

        # Fitted-attribute reporting -- None until fit() runs.
        self.epochs_run_: int | None = None
        self.final_train_loss_: float | None = None
        self.final_val_loss_: float | None = None
        self.elapsed_seconds_: float | None = None
        self.device_: str | None = None
        self.train_timestamps_: pd.DatetimeIndex | None = None
        self.val_timestamps_: pd.DatetimeIndex | None = None

    @property
    def contributor_names(self) -> tuple[str, ...]:
        return self._channel_names

    def _time_based_split(
        self, data: WindowedInput
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
        """Trailing val_fraction of data.windows (already ascending by
        end_timestamp, per make_windows()'s own guarantee) becomes
        validation; everything before it is training. n<=1 windows gets no
        validation split at all (early stopping simply doesn't trigger).
        """
        n = data.windows.shape[0]
        if n == 0:
            raise ValueError("cannot fit AutoencoderModel on zero training windows")

        n_val = min(round(n * self._cfg.val_fraction), n - 1) if n > 1 else 0
        split = n - n_val

        train_windows = data.windows[:split]
        val_windows = data.windows[split:]
        train_ts = pd.DatetimeIndex(data.end_timestamps[:split])
        val_ts = pd.DatetimeIndex(data.end_timestamps[split:])
        return train_windows, val_windows, train_ts, val_ts

    def fit(self, train_data: WindowedInput) -> None:
        """Fit-on-train-only: `train_data` must already be restricted to a
        fold's clean training windows (caller's job, same contract as
        data/scaling.py fit_scaler) -- this function fits whatever it is
        given.
        """
        cfg = self._cfg
        torch.manual_seed(cfg.random_state)
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True)

        n_channels = train_data.windows.shape[2]
        if len(train_data.channel_names) != n_channels:
            raise ValueError(
                f"channel_names ({len(train_data.channel_names)}) does not match "
                f"windows' channel axis ({n_channels})"
            )
        self._channel_names = train_data.channel_names
        window_length = train_data.windows.shape[1]

        train_windows, val_windows, train_ts, val_ts = self._time_based_split(train_data)
        self.train_timestamps_ = train_ts
        self.val_timestamps_ = val_ts

        net = _ConvAutoencoderNet(
            n_channels,
            window_length,
            cfg.channels,
            cfg.kernel_size,
            cfg.bottleneck_dim,
            cfg.dropout,
            cfg.activation,
        )
        net.to(self._device)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
        loss_fn = nn.MSELoss()

        train_tensor = torch.from_numpy(train_windows).float().to(self._device)
        val_tensor = (
            torch.from_numpy(val_windows).float().to(self._device)
            if val_windows.shape[0] > 0
            else None
        )

        start = time.monotonic()
        deadline = start + self._settings.train.max_minutes * 60

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        final_train_loss = float("nan")
        final_val_loss: float | None = None
        epochs_run = 0

        for epoch in range(self._settings.train.epochs):
            if time.monotonic() >= deadline:
                logger.warning(
                    "AutoencoderModel.fit: aborting at epoch %d/%d -- exceeded "
                    "train.max_minutes=%s budget; reporting the model as trained so far",
                    epoch,
                    self._settings.train.epochs,
                    self._settings.train.max_minutes,
                )
                break

            net.train()
            batch_losses = []
            for batch_start in range(0, train_tensor.shape[0], cfg.batch_size):
                batch = train_tensor[batch_start : batch_start + cfg.batch_size]
                optimizer.zero_grad()
                reconstruction = net(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            final_train_loss = float(np.mean(batch_losses))
            epochs_run = epoch + 1

            if val_tensor is not None:
                net.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(net(val_tensor), val_tensor).item())
                final_val_loss = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= cfg.patience:
                        logger.info(
                            "AutoencoderModel.fit: early stopping at epoch %d "
                            "(no validation improvement for %d epochs)",
                            epochs_run,
                            cfg.patience,
                        )
                        break

            logger.info(
                "AutoencoderModel.fit: epoch %d/%d train_loss=%.6f val_loss=%s",
                epochs_run,
                self._settings.train.epochs,
                final_train_loss,
                f"{final_val_loss:.6f}" if final_val_loss is not None else "n/a",
            )

        if best_state is not None:
            net.load_state_dict(best_state)
            final_val_loss = best_val_loss

        self._net = net
        self.epochs_run_ = epochs_run
        self.final_train_loss_ = final_train_loss
        self.final_val_loss_ = final_val_loss
        self.elapsed_seconds_ = time.monotonic() - start
        self.device_ = str(self._device)

        logger.info(
            "AutoencoderModel.fit: done -- epochs_run=%d final_train_loss=%.6f "
            "final_val_loss=%s elapsed=%.1fs device=%s",
            epochs_run,
            final_train_loss,
            f"{final_val_loss:.6f}" if final_val_loss is not None else "n/a",
            self.elapsed_seconds_,
            self.device_,
        )

    def _batched_reduce(self, windows: np.ndarray, reduce_dims: tuple[int, ...]) -> np.ndarray:
        """Forward pass + squared-error reduction, one model.autoencoder.
        batch_size chunk at a time -- pooled_normal_stretches scored at
        windowing.score_stride (1min) can be hundreds of thousands of
        windows; a single unbatched forward pass would materialise the
        whole (n_windows, window_length, n_channels) tensor at once and
        risks exhausting memory regardless of device. Reduction happens
        per-chunk, so only the (small) per-window result accumulates.
        """
        if self._net is None:
            raise ValueError("AutoencoderModel.fit() must be called before score()/contributions()")
        batch_size = self._cfg.batch_size
        chunks = []
        self._net.eval()
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch_size):
                chunk = windows[start : start + batch_size]
                batch = torch.from_numpy(chunk).float().to(self._device)
                reconstruction = self._net(batch)
                sq_error = (reconstruction - batch) ** 2
                chunks.append(sq_error.mean(dim=reduce_dims).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def score(self, data: WindowedInput) -> np.ndarray:
        """Per-window anomaly score: mean squared reconstruction error
        across timesteps AND channels. Higher = more anomalous -- the
        natural direction for reconstruction error, no negation needed
        (unlike Isolation Forest's score_samples(), see
        models/isolation_forest.py).
        """
        if data.windows.shape[0] == 0:
            return np.empty(0)
        return self._batched_reduce(data.windows, (1, 2))

    def contributions(self, data: WindowedInput) -> np.ndarray:
        """Per-channel contribution: mean squared reconstruction error
        across timesteps, per channel -- falls straight out of the
        architecture, no ablation pass needed (contrast
        models/isolation_forest.py's contributions()).
        """
        if data.windows.shape[0] == 0:
            return np.empty((0, len(self._channel_names)))
        return self._batched_reduce(data.windows, (1,))
