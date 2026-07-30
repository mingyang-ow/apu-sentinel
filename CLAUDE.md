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

**Read `docs/ARCHITECTURE.md` first** for the actual pipeline order, a
one-line-per-module responsibility table, the core contracts (`AnomalyModel`,
`Fold`, `Episode`, `FoldEvaluation`, and pass 13's `ScoredTestData`/
`ChanceComparison`/`PooledEvaluation`), the invariant → guard-test mapping,
and non-obvious design decisions — it is written so a reader can answer "how
does data get from CSV to an evaluated alert?" without opening source. The
short list below is orientation only:

- data/       loading, TIME-BASED splitting (split.py), per-fold fit-on-train-only
              scaling (scaling.py), windowing (windows.py)
- regimes/    operating-state segmentation (compressor on/off cycles); produces
              PER-TIMESTAMP labels only. Regime-conditional handling (scaling,
              feature computation, etc.) must be applied per-timestamp BEFORE
              windowing, not at the window level — a typical ON/LOADED run
              (median ~99s) is far shorter than the 1800s (30min) window, so
              there is no such thing as a window that belongs to one regime.
- features/   feature engineering
- models/     all models implement the models/base.py interface
- evaluation/ failure-window labels + EPISODE-LEVEL metrics
- explain/    per-channel reconstruction-error attribution (CORE)
- pipeline.py ties stages together, config-driven

Empirical findings from the data are recorded in `docs/FINDINGS.md` —
consult it before making modelling decisions. It holds *what we learned*;
this file holds *rules for how to build*. `docs/FINDINGS.md` is now an
INDEX: it lists each topic with a one-line description and links into
`docs/findings/NN-topic.md`, where the actual content lives (verbatim,
unmerged, per its original section numbering). Baseline/evaluation
RESULTS are not in `docs/findings/` at all — see `docs/RESULTS.md`, which
reports current (null-comparison-corrected) figures first and keeps
superseded figures below, clearly labelled and never deleted.

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
7. `Motor_current` defines the STOPPED/OFFLOAD regime boundary
   (`regimes.offload_current_threshold`) and is therefore PARTIALLY
   SELF-REFERENTIAL if it is also scored for anomalies during OFF-derived
   states: a fault on that channel could simultaneously distort the regime
   label and the score. Mitigated (not eliminated) by the near-total
   separation between the two modes (only 0.015% of OFF samples fall in
   the 1-3A valley — see `docs/findings/07-regimes.md`), and optionally by
   `regimes.exclude_motor_current_when_off`. Whenever Motor_current is
   scored during an OFF-derived state, this must be documented at the call
   site.

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

The common sweep's cap (72h, `evaluation.window_widths`) is a
DATA-GEOMETRY CONSTRAINT — event 2 and event 3 are only 6 days 4 hours
apart, so a wider COMMON window would swallow event 2's post-repair
recovery period into event 3's pre-failure label (see `data/split.py` and
`docs/findings/04-split-design.md` / `docs/findings/08-cycle-timing.md`).
It is NOT a domain judgement that 72h is the right actionable lead time —
`docs/findings/08-cycle-timing.md` shows event 2's cycle-timing precursor
is visible ~10 days out, well past this cap.

Per-event window caps ARE implemented (`data/split.py`
`event_max_width_hours`, pass 13 Part C): each event's own maximum feasible
width, derived from its own prior-exclusion boundary, replacing the single
global cap for a SEPARATE sensitivity fold set. This does NOT resolve the
tradeoff above — it revealed a sharper one. `docs/RESULTS.md` §13 shows
that widening to an event's own maximum makes "detection" nearly automatic
regardless of model skill (`p_chance_permutation` → 1.0 once the window
approaches the width of the test period itself), which is exactly why the
null (chance) comparison (`evaluation/metrics.py` `evaluate_chance`) must
accompany ANY window-width result, common or per-event, before it is
reported as a finding.

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

## Code comment convention

Keep code comments short and sparse. Comment **why**, not **what** — a
non-obvious reason, a domain fact a reader could not infer, or a warning
about a non-obvious constraint. Do not narrate obvious steps or restate what
the code already plainly says. Docstrings are exempt — they are the
contracts (module/function purpose, parameters, raises) and stay as
thorough as they need to be; this rule is about inline `#` comments only.

## Working conventions for build-pass briefs

Text below was previously restated near-verbatim in every individual brief
(`docs/briefs/`); state it here once and have future briefs reference it
instead:

- Nothing gets committed unless the user explicitly asks for it. Every pass
  leaves its changes uncommitted for review, regardless of what the brief's
  scope was.
- Fail loud and early on anything unexpected — a bad config value, a
  disordered timestamp, a schema-drifted column. Surface the problem; never
  silently coerce, reorder, or "fix" it in the same pass that's supposed to
  only be loading/validating.
- A brief's stated scope is a hard boundary. If a pass reveals that a later
  stage's logic would need to be touched to finish cleanly, stop and say so
  rather than reaching ahead into the next pass.

## Do NOT

- Do not use a shuffle/random train-test split anywhere.
- Do not fit scalers or compute stats outside the training window.
- Do not add supervised failure classification as the primary model.
- Do not report point-wise accuracy/F1 as the headline result.
- Do not count alarms per-timestamp or per-parameter — episodes only.
- Do not hardcode paths/hyperparameters/device — put them in configs.
- Do not commit anything under data/.
