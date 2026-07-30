# RESULTS.md

Baseline and evaluation results, moved out of `docs/FINDINGS.md` (which now
holds only empirical data findings, not model results — see
`docs/findings/`). **Current results are reported first; superseded results
are kept below, clearly labelled and unedited** — the correction is part of
the record, the same principle as the April-gap correction in
`findings/03-data-quality.md`. Nothing here is deleted.

## Contents

**Current**
- [18. Training-exclusion margin sweep — the sweep cannot fix what pass 17 found (pass 18)](#18-training-exclusion-margin-sweep--the-sweep-cannot-fix-what-pass-17-found-pass-18)
- [17. Lagged baseline — fixes the mechanism, does not change the verdict (pass 17)](#17-lagged-baseline--fixes-the-mechanism-does-not-change-the-verdict-pass-17)
- [15. Threshold sweep diagnostic (pass 15)](#15-threshold-sweep-diagnostic-pass-15)
- [13. Null comparison and honest false-alarm estimation (pass 13)](#13-null-comparison-and-honest-false-alarm-estimation-pass-13)

**Superseded**
- [12. Baseline results (pass 12) — first model scored end-to-end](#12-baseline-results-pass-12--first-model-scored-end-to-end)

---

## Current results

### 18. Training-exclusion margin sweep — the sweep cannot fix what pass 17 found (pass 18)

Pass 17 traced event 2's non-detection to calibration contamination: fold 2's
training window (2020-02-01 → 2020-05-24 23:30) includes the entire mid-May
collapse as "normal" data, because `training_exclusion.pre_margin_hours` (24h)
purges only 24h before onset — far short of the ~12-day precursor. This
section sweeps `pre_margin_hours` over `[24h (reference), 7d, 14d, 21d]` under
`baseline_mode: lagged` across all four folds, on the real dataset
(`CONFIG=local`; `data.subset` not applied, per `pipeline.py`'s own
docstring — full Feb–Aug span needed for walk-forward folds).

**Headline finding, verified directly against the code before being recorded
(`findings/10-process-lessons.md` discipline): widening `pre_margin_hours`
cannot fix fold 2's contamination, at any margin, because it was never the
right lever.** `data/split.py`'s `make_folds()` builds a fold's
`train_exclusions` only from **other** events (`if other.id == event.id:
continue`) — a fold never excludes its own event's precursor from its own
training. Fold 2's mid-May collapse is event 2's **own** precursor, sitting in
event 2's **own** fold — structurally un-excludable by
`training_exclusion.pre_margin_hours` regardless of its value, because that
setting only ever protects a **later** fold from an **earlier** event's
ramp-up, not an event from itself. This is confirmed directly: fold 1 (the
earliest event, with no earlier event to exclude at all) shows **byte-identical**
calibration and detection numbers at every margin tested — its own remaining
training days (72.00) and `short_stopped_duration` calibration 5th percentile
(0.1223, lagged) never move, because nothing in this sweep ever touches its
training slice.

#### Part A — Remaining training days per fold, per margin

| margin | event 1 | event 2 | event 3 | event 4 |
|---|---|---|---|---|
| 24h (reference/lagged) | 72.00 / 72.00 (100.0%) | 109.98 / 113.98 (96.5%) | 113.98 / 120.42 (94.7%) | 148.83 / 160.60 (92.7%) |
| 7d | 72.00 / 72.00 (100.0%) | 103.98 / 113.98 (91.2%) | 101.98 / 120.42 (84.7%) | 132.92 / 160.60 (82.8%) |
| 14d | 72.00 / 72.00 (100.0%) | 94.42 / 113.98 (82.8%) | 87.98 / 120.42 (73.1%) | 118.92 / 160.60 (74.0%) |
| 21d | 72.00 / 72.00 (100.0%) | 80.42 / 113.98 (70.6%) | 73.98 / 120.42 (61.4%) | 104.92 / 160.60 (65.3%) |

Training shrinks materially and monotonically as the margin widens — at 21d,
event 3 loses nearly 39% of its training span, close to the brief's
"roughly a quarter of fold 4" estimate (fold 4 loses 34.7% — same order).
Event 1 is flat by construction (see above). Every value stays comfortably
above `split.min_training_days` (30 days, adopted below) — no fold in this
sweep collapses toward pass 13's near-empty-training failure mode, though the
trend shows it is not an implausible future concern at even wider margins.

#### Part A — Calibration 5th percentile (`short_stopped_duration`, the rule pass 17 implicated)

| margin/mode | event 1 | event 2 | event 3 | event 4 |
|---|---|---|---|---|
| trailing@24h (reference) | 0.2384 | 0.2146 | 0.2183 | 0.2453 |
| lagged@24h | 0.1223 | 0.1514 | 0.1496 | 0.1771 |
| lagged@7d | 0.1223 | 0.1408 | 0.1371 | 0.1794 |
| lagged@14d | 0.1223 | 0.1362 | 0.1408 | 0.1324 |
| lagged@21d | 0.1223 | 0.1319 | 0.1215 | 0.1447 |

Event 2's window-open ratio (lagged mode, pass 17) is **0.274**. For
detection, fold 2's calibration 5th percentile needs to rise **above** 0.274
so that ratio reads as a high severity rather than "unremarkably low, 5th
percentile or worse." It never does — and it does not even move in the
helpful direction: 0.1514 → 0.1408 → 0.1362 → 0.1319 **falls** as the margin
widens, because widening only removes more of event 1's era (Feb–March) from
fold 2's training (an unrelated, earlier event), concentrating the *remaining*
training distribution more heavily around whatever low values (May's own
collapse, and the March cluster — neither excludable by this lever) are
already in it. Event 1's own 0.1223 is exactly flat across every margin,
direct confirmation of the mechanism above.

#### Part A/C — Detection, lead time, false-alarm rates (width=72h, the primary common width)

| event | margin | detected | lead time | fa/day (infold) | eval_days (infold) | fa/day (pooled) | p_poisson | p_perm | flagged |
|---|---|---|---|---|---|---|---|---|---|
| 1 | trailing@24h | **yes** | 2d22h57m | 0.683 | 32.22 | 0.504 | 0.871 | 0.781 | yes |
| 1 | lagged@24h | **yes** | 2d22h57m | 0.404 | 32.22 | 0.305 | 0.702 | 0.708 | yes |
| 1 | lagged@7d | **yes** | 2d22h57m | 0.431 | 27.84 | 0.305 | 0.726 | 0.696 | yes |
| 1 | lagged@14d | **yes** | 2d22h57m | 0.362 | 22.08 | 0.305 | 0.663 | 0.625 | yes |
| 1 | lagged@21d | **yes** | 2d22h57m | 0.361 | 16.63 | 0.305 | 0.661 | 0.680 | yes |
| 2 | trailing@24h | no | — | 0.285 | 7.01 | 0.385 | 0.575 | 0.381 | — |
| 2 | lagged@24h | no | — | 0.285 | 7.01 | 0.278 | 0.575 | 0.381 | — |
| 2 | lagged@7d | no | — | 0.277 | 3.61 | 0.305 | 0.564 | 0.185 | — |
| 2 | lagged@14d | no | — | 0.277 | 3.61 | 0.292 | 0.564 | 0.185 | — |
| 2 | lagged@21d | no | — | 0.277 | 3.61 | 0.338 | 0.564 | 0.185 | — |
| 3 | trailing@24h | no | — | 0.426 | 32.83 | 0.371 | 0.722 | 0.503 | — |
| 3 | lagged@24h | no | — | 0.548 | 32.83 | 0.278 | 0.807 | 0.548 | — |
| 3 | lagged@7d | no | — | 0.276 | 29.03 | 0.298 | 0.562 | 0.444 | — |
| 3 | lagged@14d | no | — | 0.310 | 22.58 | 0.298 | 0.606 | 0.540 | — |
| 3 | lagged@21d | no | — | 0.350 | 17.16 | 0.331 | 0.650 | 0.524 | — |
| 4 | trailing@24h | **yes** | 0d17h01m | 0.242 | 41.25 | 0.358 | 0.517 | 0.440 | yes |
| 4 | lagged@24h | **yes** | 0d17h01m | 0.242 | 41.25 | 0.239 | 0.517 | 0.440 | yes |
| 4 | lagged@7d | **yes** | 1d16h51m | 0.315 | 41.25 | 0.259 | 0.611 | 0.553 | yes |
| 4 | lagged@14d | **yes** | 1d16h51m | 0.315 | 41.25 | 0.245 | 0.611 | 0.553 | yes |
| 4 | lagged@21d | **yes** | 0d17h01m | 0.242 | 41.25 | 0.232 | 0.517 | 0.440 | yes |

Event 2 remains undetected at **every** margin tested — the aggregate
statistic below is the headline, but this row-by-row view already shows there
is no margin at which fold 2 crosses its own threshold. Event 4's lead time
shifts between the original 0d17h01m and 1d16h51m at 7d/14d margins (a
*different* episode becomes the earliest `early_warning` one as fold 4's own
calibration shifts with its own training exclusions) then reverts at 21d —
noted as a real, non-monotonic side effect of recalibration, not a
progression toward a longer lead time.

Event 2's evaluated_days (infold) **drops from 7.01 to 3.61** at 7d+ margins.
This is a second, distinct consequence of widening surfaced by this sweep
(Part B below): `extend_test_end_for_false_alarms` (pass 13) pushes a fold's
false-alarm-counting test_end out to just before the *next* event's exclusion
window begins — but at a 7d+ margin, event 3's exclusion window now starts
*before* event 2's own un-extended test_end, so no extension is possible and
the false-alarm denominator collapses back to its tight, ~3.6-day original
span. Reported here as a real cost of widening, not folded into "no effect."

#### Aggregate skill statistic — the headline comparison

`expected = Σ p_chance_permutation`; `observed` = detection count (of 4
folds); `p(X≥observed)` is the exact Poisson-binomial survival probability.

| quantile 0.995 | width | trailing@24h | lagged@24h | lagged@7d | lagged@14d | lagged@21d |
|---|---|---|---|---|---|---|
| obs/exp/p | 24h | 2 / 1.032 / 0.271 | 2 / 1.026 / 0.270 | 2 / 0.935 / 0.228 | 2 / 0.903 / 0.217 | 2 / 0.914 / 0.218 |
| obs/exp/p | 48h | 2 / 1.650 / 0.549 | 2 / 1.646 / 0.547 | 2 / 1.446 / 0.462 | 2 / 1.414 / 0.448 | 2 / 1.374 / 0.427 |
| obs/exp/p | 72h | 2 / 2.106 / 0.737 | 2 / 2.078 / 0.724 | 2 / 1.878 / 0.658 | 2 / 1.902 / 0.666 | 2 / 1.829 / 0.636 |
| obs/exp/p | 6h/12h | 0 / (falls with quantile) / 1.000 | same | same | same | same |

The p-value drifts modestly downward as the margin widens (e.g. 72h: 0.737 →
0.724 → 0.658 → 0.666 → 0.636) — but **this drift is not driven by event 2**
(still undetected everywhere, contributing nothing to `observed`); it comes
entirely from events 1 and 3's own false-alarm rates tightening slightly as
their own earlier-event exclusions widen, lowering their individual
`p_chance_permutation` inputs to the sum. Observed stays at 2 of 4 across
every margin tested. **This is not the improvement the brief's first
anticipated outcome describes** ("detects more, and the aggregate p-value
falls" because of *event 2*) — it is a small, unrelated, second-order effect
on the *other* two detecting folds' own calibration. Per the brief's explicit
requirement: increased margin does **not** buy skill on event 2 at any width
or margin tested.

#### Part B — Exclusion-region overlap, the union fix, and a second bug it surfaced

At wide margins, exclusion regions from adjacent events overlap (e.g. events
1/2's own windows would overlap each other were they close enough in time,
matching the brief's own worked example for events 2/3 at a 14d margin).
`make_folds()` now takes the union of overlapping exclusion regions before
storing them on a `Fold` (`_merge_exclusion_windows`), so `train_exclusions`
is always disjoint — verified directly (`test_overlapping_exclusions_merge_into_one_region`)
and exercised on real event geometry via `event_max_width_hours`'s own
pre_margin-invariance below.

**A second, distinct bug surfaced by actually running the sweep, not by
inspection alone**: `extend_test_end_for_false_alarms` (pass 13) assumed the
*next* event's exclusion window always starts safely after the *current*
fold's own test period. At `lagged@14d`, event 3's exclusion window now
starts 2020-05-22 10:00 — **before** event 2's own `test_start`
(2020-05-25 23:30) — because event 3 and event 2 are only ~6 days apart and
14 days now reaches back past event 2's own onset entirely. Unclamped, this
silently produced `test_end < test_start`, which crashed
`p_chance_permutation` downstream with an opaque "no candidate placement
fits" error rather than surfacing the real problem. Fixed by clamping the
extension to never move `test_end` before the fold's own (un-extended)
`test_end` — if the next event's exclusion has already swallowed the entire
would-be extension, no extra false-alarm-counting time is available, full
stop. Covered by `test_extend_test_end_never_moves_before_folds_own_test_end`.
This is exactly the "surface it, never silently absorb it" instruction in the
brief's Part B, just for a mechanism the brief didn't name explicitly — the
same root cause (a widened `pre_margin_hours` reaching backward far enough to
collide with an adjacent event) breaks two different functions, not one.

`split.min_training_days` (guard against pass 13's near-empty-training
collapse) is set to **30 days** — comfortably below fold 1's fixed 72.00
days (the tightest value seen anywhere in this sweep, at any margin), high
enough to catch a genuinely starved config. No fold in this sweep tripped it;
see `test_min_training_days_guard_raises_naming_fold_and_remaining_days` for
the guard itself.

#### Part B — Per-event maximum-width cap: verified INVARIANT to `pre_margin_hours`, not tightening

| margin | event 1 | event 2 | event 3 | event 4 |
|---|---|---|---|---|
| every margin (24h/7d/14d/21d) | 1800.0h | 911.5h | 94.0h | 838.5h |

**Identical at every margin tested — verified directly, not assumed.**
`event_max_width_hours()`'s two constraints (an earlier event's exclusion
window **END**, anchored on `post_settle_hours`/`fallback_post_hours`; and
the lead-in-vs-`data_start` bound) never read `pre_margin_hours`, which only
ever moves an exclusion window's **START** further into the past. A
window/test_start reaching backward from a later event always meets that
END first (it is closer), so the START moving further back changes nothing
this guard checks. The brief anticipated "per-event caps will tighten" —
**this sweep shows they do not, for a specific, verified, structural
reason**, not a lack of effort. `test_per_event_caps_invariant_to_pre_margin_and_infeasible_width_still_raises`
locks this in as a regression test, alongside confirming the pre-existing
infeasible-width raise still fires once a widened `pre_margin_hours` is
layered on top of it (the union/merge changes above do not defeat it).

#### Part D — The honesty requirement: pre_margin_hours stays at 24h; here is why

**Adopted: `training_exclusion.pre_margin_hours` stays at 24h.** Widening it
was the entire premise of this pass, motivated by event 2's ~12-day observed
precursor — an instance of using data-derived knowledge for a
label-construction decision, which CLAUDE.md permits provided it is recorded
as a decision, not a neutral default (the brief's explicit requirement). The
sweep shows widening delivers **no benefit that offsets its costs**:

- Event 2 is unfixable by this lever at any margin (see the headline finding
  above) — the mechanism the brief hypothesized would expose it (removing
  contamination so the calibration's 5th percentile rises past 0.274) does
  not apply, because `pre_margin_hours` never touches a fold's own event's
  training data in the first place.
- Training data shrinks monotonically and materially (Part A) with nothing
  gained in return.
- A second, real cost surfaces at close event pairs: widening past ~7d
  collapses event 2's own false-alarm evaluation window from 7.01 to 3.61
  days (Part A/C above) via `extend_test_end_for_false_alarms`'s clamp — a
  worse-supported false-alarm-rate estimate for a fold that was already the
  thinnest (7.01 days pre-existing, from pass 13).
- The aggregate skill statistic's small drift with margin (0.737 → 0.636 at
  72h) is unrelated to event 2 and too small to read as skill gained.

**24h remains the adopted value** — not because it was never questioned, but
because this pass's sweep is the record of having questioned it and found
nothing that justifies changing it. Fixing fold 2's own calibration
contamination would require a genuinely different mechanism: excluding an
event's own precursor from its own fold's training, which
`make_folds()` currently and deliberately does not do (`if other.id ==
event.id: continue` is there for a reason — event 2's own OWN pre-failure
ramp-up is exactly the region a detection is supposed to credit as
`early_warning`, not exclude from training as "abnormal" a priori; excluding
it would risk substituting one form of leakage/circularity for another).
Whether and how to address that safely is a distinct design question, out of
this pass's scope — recorded as a follow-up in
`docs/findings/09-open-questions.md`, not decided here.

#### Part E — Standing limitation: training purity cannot be fully restored by any margin

Recorded in full in `docs/findings/09-open-questions.md` (Part E of this
pass): degraded-looking operation recurs outside every documented failure
event (the March cluster, `findings/12-event2-error-analysis.md`) and, as
this pass additionally shows, a fold's **own** event's precursor is
structurally un-excludable from its own training by this configuration lever
regardless of value. No sweep of `pre_margin_hours` removes either source of
contamination. This applies to every subsequent model (Isolation Forest,
autoencoder) trained the same way, not just the rule-based baseline.

#### Bottom line

**The sweep is negative, for a well-understood and now-documented reason, not
an unexplained one.** Widening `pre_margin_hours` cannot fix event 2's
detection at any margin tested, because it was never the mechanism
responsible for the contamination pass 17 found — that contamination comes
from a fold's own event sitting inside its own training window, a case this
config value was never designed to reach. The aggregate skill statistic
stays essentially flat (all p-values 0.2–0.7, no margin/width combination
remotely significant); event 2 is undetected at every margin; per-event
caps are verified invariant to this parameter, not tightened as anticipated;
and a second bug (the false-alarm-extension collision at close event pairs)
was found and fixed along the way. `training_exclusion.pre_margin_hours`
stays at 24h. The path to testing whether sustained cycle-duration
depression is failure-specific — the actual open question pass 17 leaves
behind — runs through a different mechanism than this one, not through this
parameter.

---

### 17. Lagged baseline — fixes the mechanism, does not change the verdict (pass 17)

Pass 16 confirmed a structural flaw: a ratio to a TRAILING median can only see
the *rate of change* of a signal, never its *level* — event 2's
STOPPED-duration collapse read as ratio 1.195 ("better than normal") right
when its 72h window opened, because the 7-day trailing baseline had adapted
to the sustained degradation. This section adds a `lagged` baseline mode
(`models.rule_based.baseline_mode`, `baseline_lag`; see `features/cycles.py`
`baseline_relative_lagged` and `models/rule_based.py`) that anchors the
reference `baseline_lag` (14 days, `baseline_window` unchanged at 7 days)
back in time, so a sustained degradation shorter than the lag cannot
contaminate its own reference. **The mechanism fix works exactly as
predicted** — the same threshold sweep re-run under both modes shows it does
**not** change the bottom line, for a documented, verified reason.

#### The fix works at the feature level

At event 2's window-open (2020-05-26 23:30), lagged mode's reference (spanning
2020-04-28 → 2020-05-05, entirely before the 17 May collapse) gives ratio
**0.274** — strongly abnormal — versus trailing mode's 1.195. Verified
directly against the real data before choosing defaults (not merely the
brief's illustrative arithmetic).

#### But event 2 is still not detected, for a verified reason

The fold's own calibration absorbs the ratio fix. `training_exclusion.pre_margin_hours`
(24h, a **fixed, generous margin** per `split.py`'s own docstring) purges
training data only from 24h before failure onset — far short of the
~10-day precursor — so fold 2's training window (2020-02-01 → 2020-05-24
23:30) **includes the entire mid-May collapse itself** as "normal" training
data. Directly measured: fold 2's fitted calibration array for
`short_stopped_duration` (lagged) has its 5th percentile at ratio 0.151 —
already lower than event 2's window-open ratio of 0.274. Severity at
window-open is therefore only **0.775** (percentile-rank against a
distribution that already contains comparably extreme values), nowhere near
the 0.9964 alert threshold. Fold 2's two officially-evaluated episodes are
**byte-identical between modes** (same two timestamps, same driving rule
`low_peak_pressure`, peak scores agreeing to 3 decimal places) — the fix
changed the *number* the rule reads, but not enough to cross a threshold that
recalibrated around the same fix.

This is a THIRD outcome the brief's two anticipated cases didn't quite
name: not "signal revealed, skill improves" and not simply "no
failure-specific signal" — the feature-level fix is real and verified, but a
**separate, fixed-margin config value** (`training_exclusion.pre_margin_hours`)
prevents the model's own calibration from ever seeing the fix take effect.
Widening `pre_margin_hours` to match the fix's own horizon is a distinct
config change, **out of this pass's scope** (not requested by the brief, and
would itself need its own sweep/justification) — recorded as a follow-up in
`docs/findings/09-open-questions.md`, not silently made here.

#### Side-by-side: common widths, q=0.995 (both modes reproduce §15's
`trailing` row exactly, confirming no regression)

| event | mode | n_ep | width | detected | lead time | fa/day (infold) | p_poisson | p_permutation | flagged |
|---|---|---|---|---|---|---|---|---|---|
| 1 | trailing | 25 | 6h–12h | no | — | 0.776 | 0.176–0.322 | 0.127–0.235 | — |
| 1 | trailing | 25 | 24h–72h | **yes** | 0d19h20m–**2d22h57m** | 0.683–0.745 | 0.525–0.871 | 0.426–0.781 | **yes** |
| 1 | lagged   | 17 | 6h–12h | no | — | 0.497 | 0.117–0.220 | 0.103–0.198 | — |
| 1 | lagged   | 17 | 24h–72h | **yes** | 0d19h20m–**2d22h57m** (identical) | 0.404–0.466 | 0.372–0.702 | 0.373–0.708 | **yes** |
| 2 | trailing | 2  | 6h–72h | no | — | 0.285 | 0.069–0.575 | 0.056–0.381 | — |
| 2 | lagged   | 2  | 6h–72h | no (identical episodes) | — | 0.285 | 0.069–0.575 | 0.056–0.381 | — |
| 3 | trailing | 14 | 6h–72h | no | — | 0.426 | 0.101–0.722 | 0.083–0.503 | — |
| 3 | lagged   | 18 | 6h–72h | no | — | 0.548 | 0.128–0.807 | 0.095–0.548 | — |
| 4 | trailing | 11 | 6h–12h | no | — | 0.267 | 0.065–0.125 | 0.054–0.103 | — |
| 4 | trailing | 11 | 24h–72h | **yes** | **0d17h01m** | 0.242 | 0.215–0.517 | 0.192–0.440 | **yes** |
| 4 | lagged   | 11 | 6h–12h | no (identical) | — | 0.267 | 0.065–0.125 | 0.054–0.103 | — |
| 4 | lagged   | 11 | 24h–72h | **yes** (identical) | **0d17h01m** | 0.242 | 0.215–0.517 | 0.192–0.440 | **yes** |

Events 1 and 4's detecting episodes are entirely unaffected by baseline mode
— both are driven by `low_peak_pressure` (pass 16), which never showed
sustained depression, so it has nothing for lagged mode to fix. Event 3 gets
*more* episodes under lagged mode (14 → 18) without ever detecting — see
the March comparison below for why this is not surprising.

#### Pooled false-alarm rate at q=0.995 (whole-series, 150.84 days both modes)

| event | trailing (fa/day) | lagged (fa/day) | change |
|---|---|---|---|
| 1 | 0.504 | 0.305 | −39% |
| 2 | 0.385 | 0.278 | −28% |
| 3 | 0.371 | 0.278 | −25% |
| 4 | 0.358 | 0.239 | −33% |

Pooled rate falls 25–39% under lagged mode across every fold — fewer
episodes fire across the whole series — while fold 3's own **in-fold** rate
rose (0.426→0.548). These measure different things (contemporaneous vs.
whole-series, pass 13 §Part B2) and are not expected to move together; both
are reported, never merged.

#### Aggregate skill statistic — the headline comparison

`expected = Σ p_chance_permutation` over the 4 folds; `observed` = detection
count; `p(X≥observed)` is the exact Poisson-binomial survival probability.

| quantile | width | mode | observed | expected | p(X≥observed) |
|---|---|---|---|---|---|
| 0.995 | 24h | trailing | 2 | 1.033 | 0.271 |
| 0.995 | 24h | lagged | 2 | 1.026 | 0.270 |
| 0.995 | 48h | trailing | 2 | 1.650 | 0.549 |
| 0.995 | 48h | lagged | 2 | 1.646 | 0.547 |
| 0.995 | 72h | trailing | 2 | 2.106 | 0.737 |
| 0.995 | 72h | lagged | 2 | 2.078 | 0.724 |
| 0.999 / 0.9995 / 0.9999 | any | both | 0 | (falls with quantile) | 1.000 (mechanical, §15) |

**Essentially unchanged at every (quantile, width) tested** — differences are
in the third decimal place. Trailing mode's best (§15) was observed 2 vs
expected 1.033, p=0.271 — no skill; lagged mode's best is observed 2 vs
expected 1.026, p=0.270 — also no skill. The fix does not move this number.

#### Does sustained depression also fire on non-failure periods? Yes — more so

Re-scoring the early-March cluster (`docs/findings/12-event2-error-analysis.md`)
with fold 1's model under both modes:

| mode | March episodes (03-03→03-12) |
|---|---|
| trailing | 8 |
| lagged | 14 |

Lagged mode fires **75% more** in a period with no documented failure — this
is the brief's second anticipated outcome ("if detections rise but the
false-alarm rate rises proportionally... the feature is not failure-specific")
playing out directly, independent of the calibration-contamination story
above: even where the fix's reference computation is working exactly as
designed, sustained cycle-duration depression is not a specific signature of
impending failure — it also occurs, comparably often and comparably severely,
during ordinary (if unusual) operation.

#### Warm-up (Part A requirement)

`baseline_lag` (14D) + `baseline_window` (7D) = 21 days of history required
before a lagged rule reports anything other than NaN. Measured directly per
fold (each rule counted independently; `high_duty_ratio` is never lagged, 0
NaN always):

| event | timestamps (train_start→extended test_end) | NaN per lagged rule | % |
|---|---|---|---|
| 1 | 833,274 | 160,013 | 19.2% |
| 2 | 882,094 | 160,013 | 18.1% |
| 3 | 1,164,908 | 160,013 | 13.7% |
| 4 | 1,516,948 | 160,013 | 10.5% |

The absolute count (160,013 rows ≈ 21 days at the ~10s modal sampling rate,
minus data gaps) is identical across folds since every fold's training
starts at the same `data_start` (2020-02-01) — only the percentage shrinks
as each successive fold's total scored span grows. NaN severity never
silently substitutes a partial baseline (verified by
`test_warm_up_yields_nan_not_partial_baseline`), and never blots out another
rule's already-valid severity for the same timestamp
(`score()`'s NaN-aware `nanmax`, high_duty_ratio rescues every row here since
it is never NaN) — the aggregate `score()` itself is NaN 0 times in every
fold.

**Process note**: an early implementation propagated NaN for *any* invalid
raw quantity in lagged mode (not just the warm-up gap), which surfaced a
pre-existing, unrelated NaN source (`decay_rate_last` is undefined for a
degenerate single-sample STOPPED run, always present but previously silently
read as severity 0 in both modes) as a NaN-ranked-first entry in one
episode's `explain/` diagnosis. Caught by direct inspection before this
section was written, not by a test — narrowed the NaN policy to the genuine
warm-up window only (`_BASELINE_RELATIVE_RULES`, `models/rule_based.py`).
Recorded per this project's verify-before-recording discipline
(`findings/10-process-lessons.md`), not because it changed any reported
number here (it didn't — the affected cell was rare and never the deciding
rule for a threshold-crossing episode).

#### Bottom line

**The lagged-baseline fix is verified correct at the feature level and does
not change this baseline's skill.** Event 2 remains undetected; the
aggregate skill statistic is unchanged to three decimal places; false-alarm
rates fall modestly (pooled) while one fold's in-fold rate rises. The reason
is now understood and documented, not merely re-confirmed: a **separate**
fixed training-exclusion margin (24h) lets the fold's own calibration absorb
the very degradation the fix was built to expose, and independently,
sustained cycle-duration depression is shown (via the March comparison) to
recur in ordinary operation, not only before failures. Both findings point
at different next steps (widen `pre_margin_hours`, or accept the feature is
not failure-specific) rather than at the lagged-baseline mechanism itself,
which behaves exactly as designed.

---

### 15. Threshold sweep diagnostic (pass 15)

§13 flagged both of the rule-based baseline's "detections" (events 1 and 4,
common widths ≥24h) as not distinguishable from chance at `threshold_quantile:
0.995`. The open question: does the model genuinely rank pre-failure periods
higher and merely operate at too loose a threshold (in which case tightening
the threshold should make the chance p-values fall while the detections
persist), or is there no ranking signal at any operating point (in which case
detections and chance collapse together as the threshold tightens)? This
section sweeps `threshold_quantile` over `[0.995, 0.999, 0.9995, 0.9999]`
across all four folds at the common widths only (per-event maxima are already
known degenerate, §13). **No model logic changed — model scores were computed
once per fold and reused across every quantile/width combination**, exactly as
run by pass 13's `run_pipeline`; the q=0.995 row below reproduces §13's
common-widths table number-for-number, confirming this sweep's fold-fitting
and scoring path is identical to the one already in `pipeline.py`.

#### Per-fold sweep

`n_ep` is the total episode count for that (event, quantile) pair — invariant
across width, since width only changes categorisation of the same episode set,
never how many episodes exist. `fa/day` and the two chance estimates DO vary
slightly by width (a wider pre-failure window reclassifies some `false_alarm`
episodes as `early_warning`), shown as a range across the five widths
6/12/24/48/72h when the values change and as narrower spans when they flip
detected/not-detected partway through, matching this doc's existing table
convention (§13/§12).

| event | quantile | n_ep | eval_days | width(s) | detected | lead time | fa/day (infold) | p_poisson | p_permutation | flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.995  | 25 | 32.22 | 6h–12h  | no  | — | 0.776 | 0.176–0.322 | 0.127–0.235 | — |
| 1 | 0.995  | 25 | 32.22 | 24h     | **yes** | 0d19h20m | 0.745 | 0.525 | 0.426 | **yes** |
| 1 | 0.995  | 25 | 32.22 | 48h     | **yes** | 1d21h31m | 0.714 | 0.760 | 0.625 | **yes** |
| 1 | 0.995  | 25 | 32.22 | 72h     | **yes** | **2d22h57m** | 0.683 | 0.871 | 0.781 | **yes** |
| 1 | 0.999  | 3  | 32.22 | 6h–72h  | no  | — | 0.093 | 0.023–0.244 | 0.018–0.148 | — |
| 1 | 0.9995 | 2  | 32.22 | 6h–72h  | no  | — | 0.062 | 0.015–0.170 | 0.012–0.137 | — |
| 1 | 0.9999 | **0** | 32.22 | 6h–72h | no | — | undefined (0 ep) | 0.000 | 0.000 | — |
| 2 | 0.995  | 2  | 7.01  | 6h–72h | no | — | 0.285 | 0.069–0.575 | 0.056–0.381 | — |
| 2 | 0.999  | 1  | 7.01  | 6h–72h | no | — | 0.143 | 0.035–0.348 | 0.029–0.044 | — |
| 2 | 0.9995 | 1  | 7.01  | 6h–72h | no | — | 0.143 | 0.035–0.348 | 0.029–0.044 | — |
| 2 | 0.9999 | **0** | 7.01  | 6h–72h | no | — | undefined (0 ep) | 0.000 | 0.000 | — |
| 3 | 0.995  | 14 | 32.83 | 6h–72h | no | — | 0.426 | 0.101–0.722 | 0.083–0.503 | — |
| 3 | 0.999  | 5  | 32.83 | 6h–72h | no | — | 0.152 | 0.037–0.367 | 0.030–0.269 | — |
| 3 | 0.9995 | 5  | 32.83 | 6h–72h | no | — | 0.152 | 0.037–0.367 | 0.030–0.269 | — |
| 3 | 0.9999 | **0** | 32.83 | 6h–72h | no | — | undefined (0 ep) | 0.000 | 0.000 | — |
| 4 | 0.995  | 11 | 41.25 | 6h–12h  | no  | — | 0.267 | 0.065–0.125 | 0.054–0.103 | — |
| 4 | 0.995  | 11 | 41.25 | 24h–72h | **yes** | **0d17h01m** | 0.242 | 0.215–0.517 | 0.192–0.440 | **yes** |
| 4 | 0.999  | 3  | 41.25 | 6h–72h | no | — | 0.073 | 0.018–0.196 | 0.016–0.187 | — |
| 4 | 0.9995 | **0** | 41.25 | 6h–72h | no | — | undefined (0 ep) | 0.000 | 0.000 | — |
| 4 | 0.9999 | **0** | 41.25 | 6h–72h | no | — | undefined (0 ep) | 0.000 | 0.000 | — |

**Zero-episode guard applied as specified**: wherever `n_ep = 0` (event 1 at
0.9999; event 2 at 0.9999; event 3 at 0.9999; event 4 at 0.9995 and 0.9999),
`false_alarms_per_day` is reported as **undefined (0 ep)**, not `0.0` — the
model raised no episodes at all in that fold's test period, so there is no
meaningful rate to report, only an absence of alerts. `p_chance_permutation`
is `0.000` in every one of these rows **by construction** (no episodes exist
to overlap any randomly-placed candidate window) and must not be read as
evidence of skill — it reflects the absence of alerts, not a well-separated
score distribution.

#### Aggregate skill statistic (per quantile × width)

`expected_detections_by_chance = Σ p_chance_permutation` over the 4 folds;
`observed_detections` = count of folds with `detected=True`; `p(X≥observed)`
is the exact Poisson-binomial survival probability of observing at least
`observed_detections` given each fold's own `p_chance_permutation` as its
independent per-fold chance-of-detection probability.

| quantile | width | observed | expected (Σ p_perm) | p(X≥observed) | zero-episode folds |
|---|---|---|---|---|---|
| 0.995  | 6h  | 0 | 0.319 | 1.000 | 0 |
| 0.995  | 12h | 0 | 0.572 | 1.000 | 0 |
| 0.995  | 24h | 2 | 1.033 | 0.271 | 0 |
| 0.995  | 48h | 2 | 1.650 | 0.549 | 0 |
| 0.995  | 72h | 2 | 2.106 | 0.737 | 0 |
| 0.999  | 6h  | 0 | 0.093 | 1.000 | 0 |
| 0.999  | 12h | 0 | 0.147 | 1.000 | 0 |
| 0.999  | 24h | 0 | 0.246 | 1.000 | 0 |
| 0.999  | 48h | 0 | 0.468 | 1.000 | 0 |
| 0.999  | 72h | 0 | 0.648 | 1.000 | 0 |
| 0.9995 | 6h  | 0 | 0.071 | 1.000 | 1 |
| 0.9995 | 12h | 0 | 0.105 | 1.000 | 1 |
| 0.9995 | 24h | 0 | 0.177 | 1.000 | 1 |
| 0.9995 | 48h | 0 | 0.334 | 1.000 | 1 |
| 0.9995 | 72h | 0 | 0.450 | 1.000 | 1 |
| 0.9999 | 6h  | 0 | 0.000 | 1.000 | 4 |
| 0.9999 | 12h | 0 | 0.000 | 1.000 | 4 |
| 0.9999 | 24h | 0 | 0.000 | 1.000 | 4 |
| 0.9999 | 48h | 0 | 0.000 | 1.000 | 4 |
| 0.9999 | 72h | 0 | 0.000 | 1.000 | 4 |

At the current 0.995 setting, 24h (the width closest to the brief's
illustrative "~1.25 expected vs 2 observed, p≈0.35"): observed=2,
expected=1.033, p=0.271 — unremarkable, as the brief anticipated. Widening to
48h/72h makes it *more* unremarkable, not less (p climbs to 0.55–0.74),
because the false-alarm rate the permutation null is calibrated against grows
with width too.

The `p(X≥observed)=1.000` rows at 0.999/0.9995/0.9999 are a **mechanical
consequence of observed=0**, not a second, independent piece of evidence — a
Poisson-binomial "at least zero" probability is trivially 1 regardless of the
per-fold p's. The real evidence at those quantiles is the disappearance of
`observed_detections` itself (2 → 0), not this column.

#### Interpretation: No ranking signal

Tightening the threshold from 0.995 to 0.999 (roughly cutting the alert rate
by 8–10×: event 1 goes from 25 episodes/32.22 days to 3; event 4 from 11 to 3)
does not shrink events 1 and 4's chance p-values while the detections persist
— **both detections vanish outright**, at every common width, and stay gone
through 0.9995 and 0.9999. If the rules genuinely ranked pre-failure periods
higher than ordinary operation, the episodes overlapping events 1's and 4's
pre-failure windows should be among the *last* to survive a tightening
threshold (they'd be higher-severity than the noise), producing a curve where
detections hold at fewer, more significant folds as chance collapses. Instead
the specific episodes that overlapped the pre-failure windows at 0.995 are
just as quick to disappear as everything else once the cut tightens — they
were unremarkable members of the broader alert population, not
disproportionately severe. By 0.9999 three of four folds (events 1, 2, 3) and
by two quantiles earlier event 4 fire **zero** episodes anywhere in their
entire multi-week test period, meaning the rules' max-severity score rarely
if ever reaches the extreme tail at all outside the training window it was
calibrated against.

**Conclusion: no ranking signal.** This four-rule, max-aggregated rule-based
baseline does not separate pre-failure operation from ordinary noisy
operation at any operating point tested. §13's flagged "detections" at 0.995
were an artefact of a loose operating point (roughly one alert every 1.3
days) catching almost anything within a multi-day pre-failure window, not
evidence the rules were discriminating anything. This is the correct, if
unwelcome, reading for a first baseline, and it raises the bar Isolation
Forest must clear: it needs to show detections that *survive* a tightening
threshold, not merely detections that exist at one loose quantile.

#### Follow-up not done in this pass

Per the brief, `model.rule_based.baseline_window` was **not** swept —
changing it alters the model's own features (a rescoring, not a
reclassification), which is a materially more expensive experiment than a
threshold sweep and is only worth running if this pass had shown signal worth
chasing. It didn't, so this is left as a candidate follow-up, not carried out
here.

---

### 13. Null comparison and honest false-alarm estimation (pass 13)

§12's two "detections" (events 1 and 4) turned out to be measured against a
false-alarm denominator of 2.7–3.6 days — far too short to trust a rate from
2-3 episodes, and, checked against that same rate, statistically
indistinguishable from chance. This section adds a null (chance) comparison,
a dual (adequate-denominator) false-alarm estimate, and per-event window
caps, then re-runs the exact same model. **No new model was implemented.**

#### What changed

- **Null comparison** (`evaluation/metrics.py`): `p_chance_poisson` (rate ×
  width, Poisson survival) and `p_chance_permutation` (a deterministic,
  evenly-spaced grid of `evaluation.permutation_samples` candidate failure
  times across the test period, asking what fraction the model's actual
  episodes would have "caught" by pure placement luck) are computed for
  every result via `evaluate_chance`. A detection is flagged **"not
  distinguishable from chance"** when either exceeds
  `evaluation.chance_threshold` (0.10).
- **Dual false-alarm estimation**: `data/split.py`'s
  `extend_test_end_for_false_alarms` pushes each fold's `test_end` forward
  to just before the next event's exclusion region (or `data_end` for the
  last fold) — detection/lead-time logic is untouched (`categorise_episode`
  never reads `test_end`), only the false-alarm denominator grows. Reported
  as `false_alarms_per_day` with `evaluated_days` in the "common widths"
  table below (what the brief calls the **infold** rate). Separately,
  `evaluation/events.py`'s `pooled_normal_stretches` carves out
  normal-operation stretches across the WHOLE series (outside every event's
  pre-failure window, failure/settle period, and a configurable buffer),
  and `evaluate_pooled_stretches` pools the false-alarm rate across them —
  reported alongside, never merged (Feb vs. August operating conditions
  genuinely differ, `findings/08-cycle-timing.md`).
- **Per-event window caps**: `data/split.py`'s `event_max_width_hours`
  replaces the single global 72h cap with each event's own maximum feasible
  width, derived from its own prior-exclusion boundary. Fed back into
  `make_folds(..., width_hours_by_event=...)` as a SEPARATE fold set —
  never substituted for the common, cross-model-comparable sweep.

Full run (4 folds × 5 common widths + per-event max + pooled stretches over
the whole series, 4 times): ~207s on CPU, `CONFIG=local` (same caveats as
§12: `data.subset` not applied).

#### Corrected per-fold results — common widths, both false-alarm rates

`false_alarms_per_day (infold)` uses the EXTENDED test period (`evaluated_days`
shown); `pooled` is one number per fold (shared `evaluated_days` — the same
150.84 days of pooled normal stretches for every fold, since the stretches
are global, only the fitted model scoring them differs per fold).

| event | width | detected | lead time | fa/day (infold) | evaluated_days (infold) | fa/day (pooled) | p\_poisson | p\_permutation | flagged |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6h  | no  | — | 0.776 | 32.22 | 0.504 | 0.176 | 0.127 | — |
| 1 | 12h | no  | — | 0.776 | 32.22 | 0.504 | 0.322 | 0.235 | — |
| 1 | 24h | **yes** | 0d19h20m | 0.745 | 32.22 | 0.504 | 0.525 | 0.426 | **yes** |
| 1 | 48h | **yes** | 1d21h31m | 0.714 | 32.22 | 0.504 | 0.760 | 0.625 | **yes** |
| 1 | 72h | **yes** | **2d22h57m** | 0.683 | 32.22 | 0.504 | 0.871 | 0.781 | **yes** |
| 2 | 6h–72h | no | — | 0.285 | 7.01 | 0.385 | 0.069–0.575 | 0.056–0.381 | — |
| 3 | 6h–72h | no | — | 0.426 | 32.83 | 0.371 | 0.101–0.722 | 0.083–0.503 | — |
| 4 | 6h  | no  | — | 0.267 | 41.25 | 0.358 | 0.065 | 0.054 | — |
| 4 | 12h | no  | — | 0.267 | 41.25 | 0.358 | 0.125 | 0.103 | — |
| 4 | 24h | **yes** | **0d17h01m** | 0.242 | 41.25 | 0.358 | 0.215 | 0.192 | **yes** |
| 4 | 48h | **yes** | 0d17h01m | 0.242 | 41.25 | 0.358 | 0.384 | 0.321 | **yes** |
| 4 | 72h | **yes** | 0d17h01m | 0.242 | 41.25 | 0.358 | 0.517 | 0.440 | **yes** |

Detection and lead time are IDENTICAL to §12's original figures at every
width (as designed — `extend_test_end_for_false_alarms` never touches
detection logic). What changed is everything downstream of it: with 9–15×
more normal-operation time to measure against (32.2/7.0/32.8/41.3 evaluated
days vs. §12's 3.5/3.6/3.5/2.7), `false_alarms_per_day` is now a real
estimate rather than a 2-3-episode guess, and it is high enough that **both
of §12's "detections" (events 1 and 4) are flagged "not distinguishable from
chance" at every width they fire** (24h through 72h). The pooled rate
(0.36–0.50/day, measured over 150.84 days spanning the whole series)
corroborates the infold rate is not an outlier of the short original
measurement — both estimates now agree the model fires roughly every 2-3
days regardless of which stretch of the year is used.

#### Per-event window caps — sensitivity results at each event's own maximum

| event | own max width | detected | lead time | false episodes | p\_poisson | p\_permutation | flagged | note |
|---|---|---|---|---|---|---|---|---|
| 1 | 1800.0h (75.0d) | yes | 75d19h26m | 0 | 0.000 | **1.000** | yes | **degenerate — see below** |
| 2 | 911.5h (38.0d) | yes | 36d19h08m | 0 | 0.000 | **1.000** | yes | window ≈ test period width |
| 3 | 94.0h (3.9d) | no | — | 0 | — | 0.000 | no | 0 episodes at all |
| 4 | 838.5h (34.9d) | yes | 27d17h14m | 0 | 0.000 | **1.000** | yes | window ≈ test period width |

**None of these are usable results, and the null comparison is exactly what
catches it** — precisely the failure mode CLAUDE.md's window-width section
and this pass's own brief warned about ("wider windows make detection
nearly automatic while meaning less"):

- **Event 1's own maximum is a broken measurement, not a finding.** Its
  derived 75-day width consumes essentially all its lead-in time (CLAUDE.md
  rule: event 1 is the earliest event, so its cap is bounded by
  `data_start`, not a prior exclusion — see `event_max_width_hours`'s
  docstring). The resulting fold has almost no training data, so
  `fit_threshold` returns **0.0** — every timestamp scores `>= 0` by
  construction, so the ENTIRE ~75-day test period becomes one continuous
  run of "above-threshold" episodes (61 of them, split only by internal
  data gaps, ALL categorised `early_warning` because the pre-failure window
  is nearly as wide as the test period itself). This is not a detection; it
  is a threshold that cannot fail to fire. Reported as broken, not as a
  result.
- **Events 2 and 4's "detections" are the textbook case the brief
  predicted.** Their training data is fine (thresholds ≈0.996, same order
  as the common sweep) and multiple distinct episodes exist (20 and 15
  respectively) — but the pre-failure window (38 / 34.9 days) is now nearly
  as wide as the ENTIRE test period, so almost every one of those episodes
  falls inside it by construction, and a randomly-placed failure would
  "catch" one of them with probability ≈1 (`p_chance_permutation` = 1.000
  exactly for both). **This is exactly what Part A exists to catch**: a
  9-day-plus window against a ~0.4/day firing rate makes detection nearly
  automatic, and reporting these lead times (36.8 / 27.7 days) as
  improvements over the common sweep would have been fabricating skill
  purely from window geometry.
- **Event 3 fires nothing even at its own ~94h maximum** (0 episodes,
  matching every common-width result for this event) — the one event whose
  own cap is genuinely tight (`findings/09-open-questions.md` / this
  section's motivating geometry), and where the answer is unambiguous: no
  signal, at any width this event can support.
- **`p_chance_poisson` vs. `p_chance_permutation` diverge sharply for events
  2 and 4's own-maximum results** (0.000 vs. 1.000) — exactly the
  divergence CLAUDE.md flags as a clustering signal, not something to
  resolve to one number. Poisson sees zero false EPISODES (the window
  swallowed them all into early_warning) and reports zero chance risk;
  permutation sees the actual episode layout and correctly reports it as
  saturated. Reporting both, as designed, is what surfaces this.

#### Is event 2 detectable at its own cap?

**Only in the crude "did an early_warning episode exist" sense — not in any
sense that supports a skill claim.** At its own derived maximum (911.5h /
38.0 days), event 2 shows `detected=True` with lead time ~36 days 19 hours —
but `p_chance_permutation=1.000` (flagged not-distinguishable-from-chance),
for the structural reason above: a 38-day window against a ~38-day test
period will contain almost any episode the model ever raises, by
construction, independent of whether the model has any real skill. This is
NOT the same finding as §8/§12's observation that event 2's duration
precursor is real and recovers before the 72h common-sweep window opens —
that finding stands (§12, confirmed again by direct inspection of the raw
feature). What pass 13 shows is that naively re-running detection at event
2's theoretical maximum width does not honestly demonstrate it, because the
window itself is wide enough to manufacture a "detection" regardless of
skill. Concluding "event 2 is detectable given enough lead time" would
require a narrower, purpose-chosen width and a permutation test that
survives it — not simply maximising the width until something fires.

#### Cross-fold summary (corrected)

- **Same "2 of 4 events" headline count as §12** (events 1 and 4, common
  sweep, widths ≥24h) — but **every one of those detections is now flagged
  "not distinguishable from chance"** once measured against an adequate
  false-alarm denominator. Events 2 and 3 are never detected at any common
  width, so they have nothing to flag.
- **False-alarm rate**, properly estimated: infold 0.24–0.78/day
  (evaluated over 7–41 days depending on fold), pooled 0.36–0.50/day
  (evaluated over the same 150.84 days for every fold) — the two estimates
  agree within a factor of ~1.5, which is reassuring (the in-fold measurement
  is not a fluke of its particular stretch of the year) even though they
  answer different questions (contemporaneous vs. whole-series).
- **Bottom line: this baseline has not yet demonstrated recall better than
  its own noise level.** That is the correct, if unwelcome, conclusion for a
  FIRST baseline to reach, and it is precisely what CLAUDE.md's model
  progression is for — Isolation Forest (the next stage) must now be
  compared against this same null-comparison bar, not against §12's
  uncorrected numbers.

---

## Superseded results

### 12. Baseline results (pass 12) — first model scored end-to-end

> **SUPERSEDED by "13. Null comparison and honest false-alarm estimation"
> above.** This section's per-fold false-alarm rates were measured over
> evaluated_days as short as 2.7–3.6 days (2-3 episodes) — not enough to
> support a rate estimate — and, worse, checking those two "detections"
> against the model's own firing rate shows neither is distinguishable from
> chance (event 1 @72h: P(≥1 by chance) ≈ 82%; event 4 @24h: ≈ 52%). §13
> adds a null (chance) comparison, dual false-alarm estimation with
> adequate denominators, and per-event window caps, then re-runs this same
> model. Kept below, unedited, as the historical record of what pass 12
> actually reported and why it wasn't sufficient — not deleted.

`models/rule_based.py`'s four rules (`short_stopped_duration`,
`fast_pressure_decay`, `low_peak_pressure`, `high_duty_ratio`; MAX-aggregated
score, per-rule fit-on-train-only calibration) run through the full
`data -> regimes -> model -> evaluation` pipeline (`pipeline.run_pipeline`),
on the real dataset, `CONFIG=local` (device=cpu; `data.subset` is NOT applied
— see `pipeline.py` module docstring — since walk-forward folds need the
full Feb–Aug span). Threshold: 99.5th percentile of each fold's own training
scores. Full run (4 folds × 5 widths): ~70s on CPU.

#### Per-fold results across the swept window widths

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

Event 4's coverage (~0.70–0.72) matches `findings/03-data-quality.md`'s
independently-predicted value almost exactly, cross-validating both the
coverage calculation and the gap accounting.

#### Cross-fold summary

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

#### Unreported-anomaly episode (early March), confirmed again

Per `findings/08-cycle-timing.md` / `findings/09-open-questions.md`, running
the fitted fold-1 model's own threshold back over its training period
(2020-02-01 to 2020-04-13, i.e. purely descriptive — no evaluation harness
involvement, since no fold's TEST period ever includes March) surfaces a
**dense cluster of 8 above-threshold episodes between 2020-03-03 and
2020-03-12**, versus 1–2 episodes per ~week elsewhere in Feb/late-March/
April:

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
`findings/08-cycle-timing.md`'s independent early-March observation with a
second, differently-derived signal (`low_peak_pressure`, not just
duration), strengthening rather than merely repeating the earlier finding.
It remains unresolvable from the available labels — an unreported event or
near-miss, not a labelling target.

#### Process note

`models/rule_based.py`'s max-of-4-rules score, thresholded at each fold's
OWN 99.5th percentile, reproduces roughly 0.5% exceedance on that same
training data by construction (a threshold fit against the empirical CDF of
the score it will be applied to) — the ~26 training-period episodes over
fold 1's 71-day training window (~1 every 2–3 days) is consistent with this,
not evidence of a bug. This is expected of any percentile-threshold
approach and is the reason false-alarm rate is a MONITORED secondary
(CLAUDE.md), not a pass/fail gate.
