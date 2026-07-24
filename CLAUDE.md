# apu-sentinel

Early-warning anomaly detection on MetroPT-3, the Air Production Unit
(compressor) telemetry from a metro train. Real analog + digital sensor
signals over months, with only a handful of true failure events.

Goal: raise an alert early enough before a failure to be useful, without
excessive false alarms on normal operation. This is an **anomaly /
early-warning** problem, NOT supervised failure classification — too few
failures to learn "failure vs not" without overfitting.

## Architecture

Installable package `apu_sentinel` (`uv sync` installs it editable). The SAME
code runs locally (CPU, small subset) and on Colab (GPU, full data);
behaviour is selected by CONFIG, never by editing code. Env & deps managed by
uv; Python 3.11.

- data/       loading, TIME-BASED splitting, windowing + scaling
- regimes/    operating-state segmentation (compressor on/off cycles)
- features/   feature engineering
- models/     all models implement the models/base.py interface
- evaluation/ failure-window labels + EPISODE-LEVEL metrics
- explain/    per-channel reconstruction-error attribution (CORE)
- pipeline.py ties stages together, config-driven

## Hard rules — DO NOT violate

1. Split by time only. No random/shuffle split. No training sample's
   timestamp may exceed the train/val boundary. Enforced by
   tests/test_split_no_leakage.py (blocking hook on data/ edits).
2. Fit scalers on the TRAINING window ONLY. Never fit on val/test, never
   compute normalization stats over the full series. Enforced by
   tests/test_scaler_train_only.py (blocking hook).
3. No information flows backward in time. No future-derived features.
4. Condition on operating regime. The compressor cycles on/off; most raw
   variance is mode-switching, not anomaly. Models must account for state.
5. Evaluate at the EPISODE level, never per-timestamp, never per-parameter.
   See metric below.
6. Labels are derived, documented, versioned. Pre-failure windows come from
   documented failure dates; window width is SWEPT, not a magic number.

## The metric (what every model is scored on)

Detection and false alarms are counted at the EPISODE level.

- Alert = one episode. A contiguous run of abnormal behaviour is ONE alert,
  not one-per-timestamp and not one-per-spiking-channel. Episode boundaries
  use a documented hold-time / hysteresis rule (in config, not hardcoded).
- Each alert carries a MIXED, RANKED diagnosis: per-channel contributions for
  that episode (explain/), presented as one alert's detail — NOT as separate
  parallel alarms per channel.
- Detection (per failure event): did an alert episode fire within the
  pre-failure window, and lead time.
- False alarms: counted per false EPISODE on normal operation, each also
  carrying its mixed diagnosis — this is MATERIAL FOR ERROR ANALYSIS.

Primary objective: RECALL — catch the real events. False-alarm rate is a
MONITORED SECONDARY under a documented tolerable ceiling (set AFTER seeing
baseline behaviour, not before; starts null in config). Alerting always is
not a win. False episodes are studied, not merely penalised.

A model is "better" if it improves recall / lead time while keeping
false-alarms/day under the ceiling. Point accuracy and point F1 are NOT
headline metrics.

## Pre-failure window width

NOT a fixed number — a SWEPT hyperparameter explored on train/val ONLY (never
tuned against the few test failures — that is leakage). The detection-vs-
lead-time tradeoff curve is a REPORTED RESULT of the project. Too short →
warning too late to act on; too long → pre-failure label dilutes, detection
degrades.

## Model progression (each stage complete + beats the last before the next)

rule-based baseline → isolation forest → autoencoder / temporal model →
window-width sweep → per-signal explanation (CORE, feeds error analysis).
Streaming sim, alert API, dashboard, drift detection are scoped EXTENSIONS,
not core promises.

## Commands

- make setup               uv sync (install pkg + dev deps + pre-commit)
- make baseline            run rule-based baseline
- make train CONFIG=local  train on CPU/subset (default)
- make train CONFIG=colab  train on GPU/full data (self-terminating)
- make evaluate            episode-level evaluation
- make test                full suite (includes leakage guards)
- make lint                ruff

## Config

configs/base.yaml = shared. local.yaml = cpu/subset/fast. colab.yaml =
cuda/full/real-budget. Select with CONFIG=. Never hardcode paths,
hyperparameters, device, or runtime budget — all live in config.

## Do NOT

- Do not use a shuffle/random train-test split anywhere.
- Do not fit scalers or compute stats outside the training window.
- Do not add supervised failure classification as the primary model.
- Do not report point-wise accuracy/F1 as the headline result.
- Do not count alarms per-timestamp or per-parameter — episodes only.
- Do not hardcode paths/hyperparameters/device — put them in configs.
- Do not commit anything under data/.
