# 9. Open questions / unresolved

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
- **Boiling-frog — RESOLVED WITH FIX (pass 17); fix does not produce skill.**
  Confirmed in pass 16 (`findings/12-event2-error-analysis.md` Part A):
  `model.rule_based.baseline_window` (7 days) is exactly the interval between
  event 2's STOPPED-duration collapse (17 May) and its ratio-based "recovery"
  (24 May) — the rule scores `value / trailing_median(value)`, and a value
  that drops and *stays* low pulls its own 7-day trailing median down to meet
  it. Pass 17 added a `lagged` baseline mode (`baseline_mode`, `baseline_lag`
  — `features/cycles.py` `baseline_relative_lagged`, `models/rule_based.py`)
  anchoring the reference 14 days back: **verified working at the feature
  level** — event 2's window-open ratio moves from 1.195 (trailing, reads as
  "better than normal") to 0.274 (lagged, strongly abnormal), matching the
  brief's target almost exactly with `baseline_window` left unchanged.

  **But the fix does not produce skill** (`docs/RESULTS.md` §17, full sweep
  re-run under both modes): event 2 remains undetected, and the aggregate
  skill statistic is unchanged to three decimal places at every quantile and
  width tested. Two separate, independently verified reasons, each pointing
  at a different next step rather than at the lagged-baseline mechanism
  itself (which behaves exactly as designed):

  1. **Calibration absorbs the fix.** `split.training_exclusion.pre_margin_hours`
     (24h, a fixed margin) purges training data only from 24h before failure
     onset — far short of the ~10-day precursor — so fold 2's training window
     (2020-02-01 → 2020-05-24 23:30) includes the entire mid-May collapse
     itself as "normal" data. Fold 2's fitted calibration for
     `short_stopped_duration` (lagged) has its own 5th percentile at ratio
     0.151 — already below event 2's window-open ratio of 0.274 — so severity
     there is only 0.775, nowhere near the 0.9964 alert threshold. **Not
     addressed in pass 17** (out of that pass's stated scope — widening
     `pre_margin_hours` is a distinct config change needing its own
     justification/sweep, not a silent side-effect of a feature fix). Open
     follow-up: does detection improve if `pre_margin_hours` is widened to
     match the precursor's own horizon?
  2. **Sustained depression is not shown to be failure-specific.** Re-scoring
     the early-March cluster under lagged mode fires 14 episodes (vs.
     trailing's 8) — a period with no documented failure. Independent of the
     calibration issue above, this directly instantiates the brief's
     anticipated "detections rise, but the signal is not failure-specific"
     outcome.

  `baseline_window` sweeping itself (varying 7D) remains a candidate
  follow-up, not carried out in pass 17 (scope was the lagged-mode addition,
  a comparison arm at a fixed, verified-against-real-data window/lag pair —
  see `docs/RESULTS.md` §17), pass 16 (analysis-only), or pass 15 (deferred,
  more expensive than a threshold sweep since it changes the model's own
  features).

- **Training-exclusion margin sweep — NEGATIVE, for a verified structural
  reason (pass 18, `docs/RESULTS.md` §18).**

  > **Correction (pass 20, `docs/RESULTS.md` §20):** the "verified structural
  > reason" below was itself a bug, not a property of `pre_margin_hours`.
  > `make_folds()`'s exclusion loop selected regions by event IDENTITY (`if
  > other.id == event.id: continue`), which is what made a fold's own event
  > unreachable — not anything inherent to the config parameter. Fixed by
  > selecting by OVERLAP with the training span instead. Re-running the
  > identical sweep with the fix: the **ultimate verdict is unchanged** (no
  > skill gained, `pre_margin_hours` stays at 24h) but for the corrected
  > reason — a *second*, independent, un-anchored contamination source (the
  > early-March cluster) sits at comparably extreme values and takes over the
  > calibration floor the moment the actually-reachable contamination (event
  > 2's own precursor) is removed. The "open follow-up" at the end of this
  > bullet (excluding an event's own precursor from its own fold) is answered:
  > done, verified structurally correct, and still not sufficient on its own.
  > This bullet's own body below is kept unedited as the historical record of
  > what pass 18 believed and why it was wrong — not deleted.

  Pass 17 traced event 2's
  non-detection to fold 2's own calibration being contaminated by its own
  mid-May collapse, sitting inside fold 2's own training window because
  `training_exclusion.pre_margin_hours` (24h) purges only 24h before onset —
  far short of the ~12-day precursor. Pass 18 swept `pre_margin_hours` over
  `[24h, 7d, 14d, 21d]` under `baseline_mode: lagged`, all four folds, on the
  real dataset, expecting to test whether widening it exposes the
  precursor.

  **It cannot, at any margin, for a reason verified directly against the
  code before being recorded**: `data/split.py`'s `make_folds()` builds a
  fold's `train_exclusions` only from **other** documented events (`if
  other.id == event.id: continue`) — a fold never excludes its own event's
  own precursor from its own training, regardless of `pre_margin_hours`'s
  value. That setting protects a **later** fold from an **earlier** event's
  ramp-up; it was never the mechanism that could remove an event's own
  precursor from its own fold. Directly confirmed: fold 1 (earliest event,
  no earlier event to exclude) shows byte-identical calibration and
  detection numbers at every margin swept — nothing in this parameter ever
  touches its training slice. Fold 2's `short_stopped_duration` calibration
  5th percentile stays at 0.14–0.15 (lagged mode) across every margin,
  *falling* slightly as margin widens rather than rising toward the 0.274
  needed — never approaching what would be required for event 2's
  window-open ratio to read as high severity.

  **Design decision, recorded per CLAUDE.md's documented-label-construction
  discipline: `training_exclusion.pre_margin_hours` stays at 24h.** Widening
  it was motivated by event 2's observed ~12-day precursor — data-derived
  knowledge, permitted for a label-construction decision provided it is
  recorded as a decision and not presented as a neutral default (this is
  that record). The sweep found no benefit to offset widening's real costs:
  training data shrinks monotonically and materially (event 3 down to 61.4%
  of its span at 21d), and at close event pairs (events 2/3, ~6 days apart)
  widening past ~7d collapses event 2's own false-alarm evaluation window
  from 7.01 to 3.61 days via a second bug this pass found and fixed
  (`extend_test_end_for_false_alarms` previously assumed the next event's
  exclusion always starts safely after the current fold's own test period —
  false once a wide-enough margin reaches backward past it; now clamped to
  never move `test_end` earlier than the fold's own un-extended value). The
  aggregate skill statistic's small drift with margin (e.g. 72h: p=0.737 →
  0.636) is unrelated to event 2 — it comes from events 1 and 3's own
  false-alarm rates tightening slightly, not from any progress on the
  event this pass targeted.

  Fixing fold 2's own contamination would need a different mechanism
  entirely: excluding an event's own precursor from its own fold's
  training. `make_folds()` deliberately does not do this today, and doing so
  safely is non-trivial — an event's own pre-failure ramp-up is exactly the
  region a detection is supposed to credit as `early_warning`, not discard
  from training as "abnormal" a priori, so a naive version of this change
  risks trading one contamination problem for a different circularity.
  **Open follow-up, not decided here**: is there a safe way to exclude (or
  down-weight) an event's own precursor from its own fold's training without
  compromising the early_warning credit that same period earns at
  evaluation time?

  Separately, `event_max_width_hours()`'s per-event maximum feasible
  width was checked and found **invariant** to `pre_margin_hours` (identical
  at every margin: 1800.0h / 911.5h / 94.0h / 838.5h for events 1–4) — its
  two constraints read an earlier event's exclusion window's END
  (`post_settle_hours`/`fallback_post_hours`) and the lead-in-vs-`data_start`
  bound, never the START that `pre_margin_hours` moves. Recorded as a
  verified fact, not an assumption carried forward unchecked.

- **Standing limitation on training purity — applies to every subsequent
  model (pass 18 Part E; corrected pass 20).** Pass 18 recorded TWO reasons
  no sweep of `pre_margin_hours` restores training purity. **Pass 20 fixed
  one of them**: (2) below ("a fold's OWN event's precursor is structurally
  un-excludable") was a `make_folds()` bug (event-identity exclusion
  selection), not a structural property — fixed by selecting exclusions by
  overlap with the training span instead, verified directly
  (`tests/test_split_no_leakage.py`). Reason (1) remains and is now the
  ENTIRE standing limitation: degraded-looking operation recurs OUTSIDE
  every documented failure event — the early-March cluster
  (`findings/12-event2-error-analysis.md`) is comparably severe to mid-May
  with no documented failure, and no failure-event-anchored exclusion
  mechanism can ever reach it, since it isn't anchored to any event at all.
  Directly measured (pass 20, `docs/RESULTS.md` §20): even with (2) fixed
  and both event 1's and event 2's own precursors excluded from fold 2's
  training, 74.6% of its calibration's bottom-5% tail is still March data.
  Isolation Forest and the autoencoder (CLAUDE.md's next model-progression
  stages), trained on these same per-fold slices, inherit this one remaining
  limitation identically — a property of the walk-forward split
  construction and the absence of an undocumented-anomaly label, not of the
  rule-based model being evaluated against it.

