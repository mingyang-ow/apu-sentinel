# FINDINGS.md — Empirical results

This is a permanent, referenceable record of what we learned from the data.
It is distinct from `CLAUDE.md`: `CLAUDE.md` holds *rules for how to build*;
this file holds *what we found by looking*. Consult it before making
modelling decisions — several of the entries below directly constrain what a
later pass is allowed to assume.

---

## 1. Dataset characteristics

- Source: UCI ML Repository dataset 791, MetroPT-3 (Air Production Unit /
  compressor telemetry from a metro train).
- File: `MetroPT3(AirCompressor).csv`
- SHA256 (pinned, of the downloaded zip archive — see `data/download.py`'s
  checksum convention): `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`
- Span: 2020-02-01 → 2020-08-31, measured 213 days 03:59:50
- Rows: 1,516,948
- Columns: 15 usable — 7 analog + 8 digital (timestamp is the index)
  - Analog: `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`,
    `Oil_temperature`, `Motor_current`
  - Digital: `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`,
    `Pressure_switch`, `Oil_level`, `Caudal_impulses`
- **Sampling: 10 seconds nominal**, with clock jitter. Interval counts:
  10s → 1,337,521 | 9s → 128,277 | 12s → 38,321 | 13s → 7,988 | 11s → 4,471.
  This is NOT 1 Hz data.
- A `Unnamed: 0` column is present in the raw CSV — a pandas index
  serialisation artifact, not a sensor. Dropped at load time with a warning.

## 2. Ground truth — failure events

The dataset is **unlabeled**; UCI provides a company failure-report table for
evaluation. All four events are **Air leak / High stress**.

| # | Start | End | Maintenance |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | not recorded |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | 2020-05-30 12:00 ¹ |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | 2020-06-08 16:00 |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | 2020-07-16 00:00 |

Documented deviations from source (label-construction decisions):
- ¹ The source table prints "Maintenance on 30**Apr** at 12:00" for event 2,
  which predates the failure by a month and contradicts every other row (repair
  follows failure by hours). Treated as a typo for 30 May 12:00.
- The source table numbers the events **#1, #1, #3, #4** — the second entry
  should be #2. Ids here are sequential.
- Event 1 has no maintenance entry, so its repair time is unknown; a
  conservative fixed fallback margin is used instead.

## 3. Data quality — gaps

- **331 gaps exceeding 1 minute**, totalling **37 days 21:31:03** — roughly
  **18% of the timeline is missing**.
- Largest gaps (timestamp = end of gap): 2020-04-27 01:12:49 (2d 00:01:58) |
  2020-06-28 23:07:43 (1d 12:14:36) | 2020-03-01 04:00:09 (1d 04:03:01) |
  2020-08-05 08:23:01 (1d 00:40:33) | 2020-05-25 01:14:14 (1d 00:34:51).
- **Gaps are not random.** The 21h28m gap ending 2020-06-08 11:48 spans
  2020-06-07 14:19 → 2020-06-08 11:48, which is almost exactly event 3's
  out-of-service repair period (failure ended 7 Jun 14:30, maintenance 8 Jun
  16:00). Missing data carries information about machine state.
- **CORRECTION to an earlier claim:** the largest gap (25–27 April) does NOT
  coincide with any documented failure. Event 1 was a week earlier (18 April).
  An earlier pass summary asserted otherwise; that assertion was wrong.
- A 14h11m gap spanning 2020-07-14 07:16 → 21:28 sits **inside event 4's
  pre-failure window**.

### Pre-failure window coverage (at 72h width, measured)

| event | coverage |
|---|---|
| 1 | 0.878 |
| 2 | 0.886 |
| 3 | 0.843 |
| 4 | **0.704** |

Event 4 is materially worse: four gaps (14h11m, 5h36m, 1h45m, 5min ≈ 21.6h
total) fall inside its 72h window. If event 4 is the worst-detected event,
this is the likely explanation — it is a data limitation, not necessarily a
model failure. Note also that *score* coverage is slightly worse than *data*
coverage, because each gap additionally invalidates roughly one window-duration
of scoring positions on its trailing edge.

## 4. Split design

- Walk-forward (rolling-origin, expanding window), 4 folds, one per event.
  Chosen because 4 events cannot support a single holdout — recall on one test
  event is either 0% or 100%.
- Embargo: 24h between train end and test start, to prevent sliding windows
  straddling the boundary.
- **The event 2 → event 3 gap is 6 days 4 hours**, which caps how wide the
  pre-failure window sweep can go before event 3's window swallows event 2's
  post-repair recovery period. Cap is currently **72h** (tightened from 96h
  when the overlap check was extended to cover `test_start`, not just the label
  window).
- Fold 1's training slice: 529,973 rows over ~71 days — **86% of the 613,440
  expected at complete 10s sampling**.

## 5. Scaling findings

- **Robust (median/IQR) is the default**, because training data still contains
  *unreported* anomalies that only median/IQR resist.
- Fold 1 (Feb–mid-Apr) vs full series (Feb–Aug), robust stats:
  `Oil_temperature` center 58.275 → 62.700 (+7.6%), IQR 7.400 → 9.475 (+28%).
  This is **seasonal drift**, and it is the concrete demonstration that fitting
  a scaler on the full series would encode summer temperatures that had not yet
  happened when fold 1 makes its April prediction.
- Pathological spreads (global, fold 1): `TP2` IQR 0.004 (~250× amplification),
  `DV_pressure` IQR 0.006, `Motor_current` center 0.042 with IQR 3.750 (spread
  ~90× the center).
- `TP3` (center 8.992, IQR 0.964) and `Reservoirs` (8.994, 0.962) are
  near-duplicate — closely coupled parts of the same pneumatic circuit. This
  may make per-channel attribution ambiguous when both light up.
- `zero_scale_epsilon` (1e-8) guards **division by zero** on constant channels.
  It deliberately does NOT catch TP2's 0.004 — that is real amplification, not
  a numerical bug, and substituting 1.0 there would silently disable scaling.

## 6. Windowing

- 30-minute window = 180 samples at 10s; train stride 5min = 30 samples.
- Fold 1: 17,179 windows kept, **481 dropped (2.7%) for spanning gaps**,
  tensor 185.53 MB.
- Each gap invalidates roughly `window_duration / stride` windows, so longer
  windows discard more data around each gap — an argument for shorter windows
  given 331 gaps.

## 7. Operating regimes

### Flag polarity — verified empirically, not assumed

| flag | meaning | evidence (mean Motor_current) |
|---|---|---|
| `COMP` | 1 → OFF, 0 → ON | COMP=0 → 5.60; COMP=1 → 1.36 |
| `DV_eletric` | 1 → ON, 0 → OFF | mirror image of COMP |
| `MPG` | 1 → OFF, 0 → ON | MPG=1 → 1.34; MPG=0 → 5.56 |

COMP's polarity is the **inverse of the naive reading** — it is active when
there is *no* air intake. Pairwise agreement after normalising polarity:
COMP↔DV_eletric 98.9%, COMP↔MPG 99.6%, DV_eletric↔MPG 99.3%. COMP is the
deciding flag.

### Occupancy (original two-state segmentation, superseded by §7a below)

| state | occupancy | median run |
|---|---|---|
| OFF | 82.0% | 1080s (18 min) |
| ON | 13.8% | 99s |
| TRANSITION | 4.2% | 20s |

**Architecturally critical:** ON's median run is 99s ≈ 10 samples, far shorter
than the 180-sample window. **There is no such thing as an "ON window"** —
essentially every window spans multiple full cycles. Therefore regime handling
cannot be applied at window level; scaling must be applied **per-timestamp, by
that timestamp's regime, before windowing**.

### Within-regime variance (the key table)

| channel | global IQR | within-OFF | within-ON |
|---|---|---|---|
| `Motor_current` | 3.7675 | 3.675 | 0.630 |
| `TP2` | 0.0040 | 0.0040 | 2.3120 |
| `DV_pressure` | 0.0040 | 0.0040 | 0.8040 |

Interpretation:
- `Motor_current`: **fixed within ON** (3.77 → 0.63). Within-OFF barely moves
  because OFF is itself bimodal (below).
- `TP2` / `DV_pressure`: these are **LOADED-only channels**. They vent to ~0
  when the compressor stops, so their 0.004 OFF spread is sensor noise around
  zero carrying no information; during ON they have healthy spread (2.31 /
  0.80). Regime conditioning does not "fix" them — it reveals they are only
  alive in one state.

### OFF is bimodal — a third state

Of 1,269,620 OFF samples: **822,119 (64.8%) at Motor_current ≈ 0** and
**447,307 (35.2%) at ~3.5–4.3A sustained** (median run ~400s, so not a
transient).

**No digital flag separates them.** Separation scores
(|P(flag=1|low) − P(flag=1|high)|): Oil_level 0.042, Caudal_impulses 0.037,
DV_eletric 0.015, MPG 0.015, Pressure_switch 0.003, LPS 0.000, **Towers
0.000**. The flags are saturated during OFF. The tower-regeneration hypothesis
was tested and rejected.

Analog evidence that the two modes are physically distinct:

| channel | STOPPED median | OFFLOAD median | shift |
|---|---|---|---|
| `TP3` | 8.658 | 9.644 | 0.986 |
| `H1` | 8.646 | 9.630 | 0.984 |
| `Reservoirs` | 8.660 | 9.644 | 0.984 |
| `Oil_temperature` | 59.375 | 67.650 | 8.275 |
| `Motor_current` | 0.040 | 3.765 | 3.725 |
| `TP2` | -0.012 | -0.012 | **0.000** |
| `DV_pressure` | -0.020 | -0.020 | **0.000** |

Only **0.015% of OFF samples** fall in the 1–3A valley between the modes — the
separation is near-total, so a 2A threshold sits in genuine emptiness.

### The duty cycle

Putting it together: **LOADED** compresses the system up → **OFFLOAD** sits at
the top of the cycle (~9.64 bar, motor spinning, intake closed) → **STOPPED**
is the *decay phase*, pressure bleeding down from ~9.6 toward the 8.2 bar
threshold at which MPG restarts the compressor.

Consequence: STOPPED is **not a static state**. Its pressure IQR of 0.63
reflects the systematic sweep from 9.6 → 8.2, not variability around a stable
normal. A static per-state center is therefore the wrong model for STOPPED;
time-since-entering-STOPPED, or decay *rate* rather than level, may be needed.

### 7a. Third state implemented (pass 10)

The bimodal split above is now a real, config-driven fourth regime label.
Segmentation is four states: `LOADED` (COMP=0), `OFFLOAD` (COMP=1 and
Motor_current ≥ `regimes.offload_current_threshold`, default 2.0A), `STOPPED`
(COMP=1 and Motor_current below that threshold), and `TRANSITION` (within
`transition_settle` of any committed state change).

Measured four-state occupancy and run lengths (real dataset):

| state | occupancy | n_runs | median run | q25 | q75 |
|---|---|---|---|---|---|
| STOPPED | 52.48% | 10,293 | 674s | 416s | 981s |
| OFFLOAD | 27.46% | 10,694 | 376s | 367s | 377s |
| LOADED | 13.80% | 10,624 | 99s | 89s | 109s |
| TRANSITION | 6.25% | 31,611 | 20s | 20s | 20s |

LOADED's occupancy/run-length is unchanged from the two-state segmentation
above (it was never part of the OFF/COMP=1 split). TRANSITION occupancy rose
from 4.2% to 6.25% because the cycle now has up to three committed
transitions per cycle (LOADED→OFFLOAD→STOPPED→LOADED) instead of two.
OFFLOAD's run length is remarkably tight (q25=367s, q75=377s — a 10-second
spread) compared to STOPPED's wide spread (416–981s), consistent with §7's
"duty cycle" interpretation: OFFLOAD is a comparatively fixed-duration phase
at the top of the cycle, while STOPPED is a variable-length pressure-decay
phase, not a steady state.

## 8. Cycle timing — the strongest leak signal found

An air leak makes pressure decay faster → shorter STOPPED periods → more
frequent cycling. Median STOPPED duration (20-cycle rolling) was plotted
against the four events:

- **Event 2: clear precursor.** Duration collapses from ~1200s to ~280s around
  15–20 May and stays there until the 29 May failure — roughly **10 days of
  sustained warning**.
- **Event 4: sharp dip** to ~150s (the lowest point in the series) at the
  failure.
- **Events 1 and 3: no clear precursor.** Duration is ordinary (~1000–1400s)
  before event 1, and *elevated* going into event 3 (post-event-2 recovery).

So this single hand-built feature gives **2 of 4 events**, not 4 of 4.

### Unreported anomaly, empirically confirmed

Early March shows a ~1 week dip to ~200–250s — **as extreme as event 2's
precursor** — with no reported failure. The report table lists only high-stress
air leaks, so this may be an unreported event or a genuine near-miss. It is
**unresolvable from the available labels**, and any detector using this feature
will fire there. State this plainly rather than reclassifying it.

### Drift — breaks static thresholds

Median STOPPED duration trends from **~1400s (Feb) to ~500s (Aug)** — a **3×
shift, larger than most event-related dips**.

Consequences:
- Fold 4 trains back to February (normal ≈ 1400s) and tests in July (normal ≈
  500s). A static notion of normal cycle timing would alarm continuously
  through summer.
- Thresholds on this feature must be **relative to a trailing baseline**, not
  absolute.
- But a baseline that adapts too fast absorbs gradual degradation and loses the
  signal (boiling-frog). Event 2's step change would survive a ~7-day baseline;
  slow drift would not. The baseline window is a deliberate choice.
- **Confound:** `Oil_temperature` rose 58 → 63°C over the same period. Seasonal
  ambient change and cycle frequency move together — do not attribute the trend
  to wear without argument.

### Consequence for the window-width cap

Event 2's precursor is visible ~10 days out, but the sweep is capped at 72h by
the E2/E3 proximity. Events 1, 2 and 4 have no such constraint. Options:
per-event window widths, or **keep wide windows and mask the overlapping
region from false-alarm counting** (the evaluation harness already supports
masking). The latter is cleaner.

## 9. Open questions / unresolved

- **Is the STOPPED-duration drift partly artifact?** Run detection splits
  STOPPED runs on data gaps (>1min), and there are 331 such gaps. A gap inside
  a STOPPED period truncates it, making duration look shorter. If gap density
  increased over the recording period, some downward trend is artifact.

  **Checked in pass 10 — verdict: PHYSICAL, not artifact.**
  `analysis.monthly_gap_and_stopped_summary` on the real dataset:

  | month | n_gaps | total_gap_seconds | n_stopped_runs | median_stopped_seconds |
  |---|---|---|---|---|
  | 2020-02 | 36 | 390,648 | 1,188 | 1269 |
  | 2020-03 | 42 | 379,706 | 1,468 | 813 |
  | 2020-04 | 55 | 644,464 | 1,059 | 1021 |
  | 2020-05 | 52 | 499,183 | 1,407 | 565 |
  | 2020-06 | 37 | 394,735 | 1,524 | 555 |
  | 2020-07 | 64 | 462,660 | 1,880 | 555 |
  | 2020-08 | 44 | 495,963 | 1,925 | 595 |

  Gap density (both count and total missing seconds) is **noisy, not
  trending** — April has the single highest gap total (644,464s) of any
  month, yet a *higher* median STOPPED duration (1021s) than March (813s)
  and much higher than May–August (~555–595s). Correlation across the
  7 months: `corr(total_gap_seconds, median_stopped) = 0.047` (essentially
  zero), `corr(n_gaps, median_stopped) = -0.33` (weak, and in the artifact
  direction, but not remotely large enough to explain a ~2.3× decline from
  noise alone with n=7).

  Directly quantifying the gap-splitting mechanism's own effect: re-running
  the same run detection WITHOUT splitting on gaps (pure value-change runs,
  bridging over every gap) gives medians within **1–8% of the
  gap-splitting version, in every month** (e.g. Feb 1308s vs. 1269s, Aug
  605s vs. 595s) — an order of magnitude smaller than the observed ~2.3×
  decline, and the size of this small effect does not grow toward August
  (it is largest in May, at 8%). The mechanism being questioned is real but
  tiny, and does not track the drift.

  **Conclusion: the STOPPED-duration decline is physical**, not a
  by-product of increasing gap density biasing run detection.

- **Circularity in the third state:** Motor_current defines the STOPPED /
  OFFLOAD boundary but is also a scored channel. Mitigated by the 0.015% valley
  (a fault would need a ~2A electrical shift to move the assignment) and
  optionally by excluding Motor_current from scored channels during OFF.
  Documented, not eliminated. `regimes.exclude_motor_current_when_off`
  (default `false`) now records this intent in config, but it is deliberately
  NOT wired into scoring in this pass — scoring does not exist yet.
- Whether the 72h cap should be replaced by masking.
- How to normalise the pressure channels within STOPPED given it is a decay
  phase, not a steady state.

## 10. Process lessons

- Notebook CWD ≠ repo root; resolve data paths from the package location, not
  the working directory.
- After schema changes, restart the Jupyter kernel — `autoreload` does not
  reliably pick up changed class definitions.
- `extra="forbid"` on config models catches typo'd keys that would otherwise be
  silently ignored.
- Declare dependencies explicitly; matplotlib was present only transitively via
  mlflow, which is fragile.
- A guard test must be shown to **fail on bad input**, not merely pass on good
  input.
- Verify reported claims independently — one pass summary asserted a gap
  coincided with a failure; it did not (see §3). The zip-archive SHA256 in §1
  was independently re-verified against the file on disk before being recorded
  here, following the same principle.
- **Contract test caught a design gap before it became rework (pass 12).**
  `models/base.py`'s original `AnomalyModel` protocol promised
  `channel_contributions`: per-timestamp, per-CHANNEL attribution. That fits
  an autoencoder, where per-channel reconstruction error is free. It does
  NOT fit a rule-based model, whose interpretable unit of attribution is
  *which rule fired* (e.g. `short_stopped_duration`), not a raw channel —
  several rules read the same channel (`Reservoirs`), and one rule
  (`high_duty_ratio`) doesn't read a raw channel at all. Writing the first
  real model against the contract test (`tests/test_eval_contract.py`)
  surfaced this mismatch immediately, before a second model existed to make
  the fix harder. Resolution: renamed `channel_contributions` ->
  `contributions` and added a `contributor_names` property so each model
  declares its own attribution vocabulary; `evaluation/`'s harness and
  `explain/`'s ranking already took names as a parameter rather than reading
  config, so they needed no structural change — only the wiring (a future
  pipeline pass) must source contributor names from `model.contributor_names`,
  never from `scaling.analog_columns`. Lesson: a contract test earns its keep
  by being exercised against a real implementation as early as possible, not
  just against a mock.

## 11. Regime-conditional scaling and cycle-timing features (pass 11)

### Scaling: per-(fold, regime), active-channel policy

Rewired to `apply_fold -> assign_regimes -> fit_regime_scalers (train only)
-> transform_by_regime -> make_windows`, per-timestamp, before windowing
(§7's "no such thing as an ON/LOADED window" finding). Fold 1 fits 4 scalers
(one per regime), with training sample counts:

| regime | training samples | share of fold 1 train |
|---|---|---|
| STOPPED | 320,952 | 60.6% |
| OFFLOAD | 124,559 | 23.5% |
| LOADED | 55,734 | 10.5% |
| TRANSITION | 28,728 | 5.4% |

These proportions differ from the global four-state occupancy in §7a
(STOPPED 52.5% / OFFLOAD 27.5% / LOADED 13.8% / TRANSITION 6.25%) because
fold 1's training window is Feb-to-mid-April only, not the full Feb-Aug
span — consistent with §8's finding that cycle timing itself drifts over
the year.

`scaling.active_channels` zeroes TP2/DV_pressure in STOPPED/OFFLOAD (not
scaled — their spread there is sensor noise, §7). `TRANSITION` gets its own
scaler like any other regime (never gap-treated or excluded) since at
~6.25% occupancy with a state change every couple of minutes, excluding it
would leave no contiguous window anywhere — its statistics describe a
genuine mixture state, not a single clean one.

### Amplification warnings fired on the real dataset (fold 1)

`scaling.amplification_warn_factor` (100x) surfaced three cases NOT caught
by the TP2/DV_pressure exclusion above — none were substituted, all logged:

| fold | regime | channel | scale | amplification |
|---|---|---|---|---|
| 1 | LOADED | `H1` | 0.006 | 166.7x |
| 1 | STOPPED | `Motor_current` | 0.005 | 200.0x |
| 1 | TRANSITION | `DV_pressure` | 0.004 | 250.0x |

The `Motor_current`/STOPPED case is expected and connects directly to §9's
open circularity question: once STOPPED is restricted to samples where
Motor_current is BY DEFINITION below the OFFLOAD split threshold, its
within-STOPPED spread is mechanically narrow — this is a consequence of
the regime definition itself, not a new anomaly-detection concern, but it
sharpens the case for `regimes.exclude_motor_current_when_off` when
scoring is implemented. The `H1`/LOADED and `DV_pressure`/TRANSITION cases
are new findings, not previously characterised, and are left as open
questions for the modelling pass (are these genuinely narrow, or another
instance of a channel that's only "alive" in a different regime?).

### Cycle-timing features: gap-truncation on the real dataset

Of 10,456 STOPPED runs detected by `features/cycles.py` (using its own
1-minute `gap_threshold`, split identically to `analysis.
monthly_gap_and_stopped_summary`'s convention), **166 (1.6%) were
gap-truncated** — their duration is correctly NaN, not the truncated
value. 163 of those 166 have ≥2 observed samples and therefore a valid
`decay_rate_last`; the remaining 3 have only a single observed sample
before the gap, too few for any slope. 5,327 of 1,516,948 rows (0.35%)
carry `run_gap_truncated = True` at any given time (forward-filled between
a truncated completion and the next genuine one) — confirming gap
truncation is a real but small effect on the feature set as a whole, not a
dominant one.

## 12. Baseline results (pass 12) — first model scored end-to-end

`models/rule_based.py`'s four rules (`short_stopped_duration`,
`fast_pressure_decay`, `low_peak_pressure`, `high_duty_ratio`; MAX-aggregated
score, per-rule fit-on-train-only calibration) run through the full
`data -> regimes -> model -> evaluation` pipeline (`pipeline.run_pipeline`),
on the real dataset, `CONFIG=local` (device=cpu; `data.subset` is NOT applied
— see `pipeline.py` module docstring — since walk-forward folds need the
full Feb–Aug span). Threshold: 99.5th percentile of each fold's own training
scores. Full run (4 folds × 5 widths): ~70s on CPU.

### Per-fold results across the swept window widths

| event | width | detected | lead time | false episodes | false alarms/day | coverage | top rule (detecting episode) |
|---|---|---|---|---|---|---|---|
| 1 | 6h  | no  | —        | 3 | 0.855 | 0.829 | — |
| 1 | 12h | no  | —        | 3 | 0.855 | 0.829 | — |
| 1 | 24h | **yes** | 0d19h20m | 2 | 0.570 | 0.729 | `low_peak_pressure` (0.999) |
| 1 | 48h | **yes** | 1d21h31m | 1 | 0.285 | 0.829 | `low_peak_pressure` (0.998) |
| 1 | 72h | **yes** | **2d22h57m** | 0 | 0.000 | 0.878 | `low_peak_pressure` (0.997) |
| 2 | 6h–72h | no | — | 1 | 0.277 | 0.86–1.00 | — |
| 3 | 6h–72h | no | — | 0 | 0.000 | 0.84–1.00 | — |
| 4 | 6h  | no  | —        | 3 | 1.100 | 1.000 | — |
| 4 | 12h | no  | —        | 3 | 1.100 | 1.000 | — |
| 4 | 24h | **yes** | **0d17h01m** | 2 | 0.733 | 0.716 | `low_peak_pressure` (0.997) |
| 4 | 48h | **yes** | 0d17h01m | 2 | 0.733 | 0.702 | `low_peak_pressure` (0.997) |
| 4 | 72h | **yes** | 0d17h01m | 2 | 0.733 | 0.704 | `low_peak_pressure` (0.997) |

Event 4's coverage (~0.70–0.72) matches §3's independently-predicted value
almost exactly, cross-validating both the coverage calculation and the gap
accounting.

### Cross-fold summary

- **2 of 4 events detected** at width ≥ 24h: **event 1** (lead time up to
  2d22h57m at 72h) and **event 4** (lead time steady at ~17h from 24h width
  onward — the detecting episode is the same one at every width ≥ 24h,
  since 2020-07-14's episode is the only qualifying candidate in range).
- **False-alarm rate range** (at 72h, the primary width): **0.000–0.733/day**
  across the four folds; pooling all widths, 0.000–1.100/day.
- This is a legitimate "2 of 4" baseline result, matching CLAUDE.md's
  stated expectation — but via a **different pair of events, and a
  different dominant rule, than §8's duration-only analysis anticipated**.
  §8 analyzed ONLY `stopped_duration_last`, which is clear for events 2 and
  4 and absent for 1 and 3. The full 4-rule ensemble instead detects
  **events 1 and 4**, both driven by `low_peak_pressure` — a signal §8 never
  examined (it is new in this pass). Two independent, verified explanations
  for the divergence, not a bug:
  - **Event 2 is a real miss caused by the documented 72h cap, not an
    absent signal.** Directly inspecting `stopped_duration_last_rel_baseline`
    in the real data confirms §8's collapse (daily mean ratio falls to
    0.14–0.42 during 2020-05-17 to 05-21, `short_stopped_duration` severity
    peaking at 0.98–0.999 on several of those days) — but the ratio has
    already recovered to ≥1.0 by 2020-05-24, three days before the 72h
    pre-failure window (2020-05-26 23:30 onward) even opens. This is
    exactly the gap CLAUDE.md's "Pre-failure window width" section warns
    about: the precursor is real and ~10 days out, well past the swept cap.
  - **Event 1's detection is a genuinely new, physically-verified signal.**
    The detecting episode (2020-04-15 01:02–01:16) corresponds to a real
    OFFLOAD-run peak of **8.49 bar** against a nominal ~9.9–10.0 bar
    (confirmed by direct inspection of `last_completed_run_peak`'s raw,
    pre-baseline-relative output) — a genuine ~1.4 bar shortfall, not an
    artifact. `short_stopped_duration` is ordinary for event 1 (§8), so this
    finding is additive, not contradictory.

### Unreported-anomaly episode (early March), confirmed again

Per §8/§9, running the fitted fold-1 model's own threshold back over its
training period (2020-02-01 to 2020-04-13, i.e. purely descriptive — no
evaluation harness involvement, since no fold's TEST period ever includes
March) surfaces a **dense cluster of 8 above-threshold episodes between
2020-03-03 and 2020-03-12**, versus 1–2 episodes per ~week elsewhere in
Feb/late-March/April:

| start | top rule (severity) | 2nd rule (severity) |
|---|---|---|
| 2020-03-03 02:52 | `low_peak_pressure` (0.999) | `high_duty_ratio` (0.891) |
| 2020-03-03 12:23 | `short_stopped_duration` (0.998) | `fast_pressure_decay` (0.985) |
| 2020-03-03 21:57 | `short_stopped_duration` (0.999) | `low_peak_pressure` (0.872) |
| 2020-03-04 04:58 | `low_peak_pressure` (0.997) | `short_stopped_duration` (0.964) |
| 2020-03-09 06:14 | `short_stopped_duration` (0.999) | `high_duty_ratio` (0.826) |
| 2020-03-10 00:37 | `low_peak_pressure` (0.998) | `high_duty_ratio` (0.846) |
| 2020-03-11 08:31 | `short_stopped_duration` (0.999) | `high_duty_ratio` (0.860) |
| 2020-03-11 18:12 | `low_peak_pressure` (0.998) | `fast_pressure_decay` (0.994) |
| 2020-03-12 15:29 | `low_peak_pressure` (1.000) | `high_duty_ratio` (0.910) |

This is reported as-is, per the brief: **not** reclassified as a success
(no documented failure exists here) and **not** buried. It corroborates
§8's independent early-March observation with a second, differently-derived
signal (`low_peak_pressure`, not just duration), strengthening rather than
merely repeating the earlier finding. It remains unresolvable from the
available labels — an unreported event or near-miss, not a labelling
target.

### Process note

`models/rule_based.py`'s max-of-4-rules score, thresholded at each fold's
OWN 99.5th percentile, reproduces roughly 0.5% exceedance on that same
training data by construction (a threshold fit against the empirical CDF of
the score it will be applied to) — the ~26 training-period episodes over
fold 1's 71-day training window (~1 every 2–3 days) is consistent with this,
not evidence of a bug. This is expected of any percentile-threshold
approach and is the reason false-alarm rate is a MONITORED secondary
(CLAUDE.md), not a pass/fail gate.
