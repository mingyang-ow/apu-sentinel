# 12. Event 2's out-of-window signal — error analysis (pass 16)

> **Forward pointer (pass 17, `docs/RESULTS.md` §17,
> `findings/09-open-questions.md`):** the boiling-frog mechanism confirmed in
> Part A below was fixed with a `lagged` baseline mode, verified working at
> the feature level (event 2's window-open ratio moves from 1.195 to 0.274).
> **It did not change event 2's detection outcome or the aggregate skill
> statistic** — fold 2's own training window absorbs the fix (the ~24h
> `pre_margin_hours` training-exclusion margin is far shorter than the
> ~10-day precursor, so the collapse itself sits inside "normal" training
> data), and lagged mode also fires *more* on the March cluster below (14 vs.
> 8 episodes), reinforcing Part E's "indistinguishable" finding rather than
> resolving it. This section's original analysis stands unedited below — the
> mechanism it identified was real, its fix is verified, and the fix alone
> does not change the bottom line, for reasons now separately documented.
>
> **Second forward pointer (pass 18, `docs/RESULTS.md` §18,
> `findings/09-open-questions.md`):** pass 18 swept the `pre_margin_hours`
> margin cited above over `[24h, 7d, 14d, 21d]`, expecting widening it might
> expose the precursor pass 17's fix left uncalibrated-against. It cannot, at
> any margin: `make_folds()` only ever excludes an **earlier** event's
> precursor from a **later** fold's training, never an event's own precursor
> from its own fold — fold 2's mid-May collapse is event 2's own precursor,
> structurally un-excludable by this setting regardless of its value.
> `pre_margin_hours` stays at 24h (recorded design decision, pass 18 Part D).

`docs/RESULTS.md` (pass 15) concludes the rule-based baseline has **no
ranking signal**: tightening the alert threshold makes both of pass 13's
"detections" vanish rather than persist. But `findings/08-cycle-timing.md`
records event 2's STOPPED-duration collapse (2020-05-17 → 05-21, recovering
by 05-24) as real and verified — a genuine precursor sitting ~9 days before
event 2's 72h pre-failure window opens (2020-05-26 23:30), so it was scored
as false alarms, never detection. This is the mis-scored genuine precursor
CLAUDE.md calls out as material for error analysis, not for minimising.
Full evidence and plots: `notebooks/exploratory/02_event2_error_analysis.ipynb`.

## Part A — Boiling-frog hypothesis: CONFIRMED

`model.rule_based.baseline_window` = 7 days, and the collapse-to-"recovery"
interval (17 → 24 May) is exactly 7 days — not a coincidence. Every rule
scores `value / trailing_median(value)`; a value that drops and *stays* low
pulls its own 7-day trailing median down to meet it.

| | pre-collapse (05-10→05-16) | collapse (05-17→05-21) | at window-open (05-24→05-26) |
|---|---|---|---|
| `stopped_duration_last` (raw, s) | 1261.6 | 271.7 | 352.8 |
| trailing baseline (7d median, s) | 1352.2 | 960.3 → falling | 306.8 |
| ratio (what the rule scores) | 0.934 | 0.390 | **1.150** |

The absolute duration never recovers — it stays in the 250-450s range all the
way through the 29 May failure and beyond (05-30: 732s, 05-31: 701s, still
well below the ~1300s pre-collapse level). But the ratio the rule actually
scores on is back **above 1.0** by the time the 72h window opens, because the
baseline itself collapsed to match the depressed value. The baseline's daily
value falls off a cliff between 05-20 (843.9s) and 05-21 (311.9s) — a rolling
*median* flips abruptly once more than half its trailing window is occupied
by the new, low regime, rather than descending smoothly.

Consequence, directly observed in the model's own attribution: of the five
mid-May/late-May episodes inside event 2's 21-day pre-onset horizon (Part D),
`short_stopped_duration` is the *top-ranked* rule in only one (05-21 17:18);
everywhere else `low_peak_pressure` dominates, and by the closest episode to
onset (05-26 06:01, 3d17h before failure) `short_stopped_duration`'s severity
has dropped to **0.0**. The rule that carries the strongest, most
independently-verified physical signal for this event goes quiet exactly when
it would matter most, for a mechanism-level reason (baseline adaptation), not
because the underlying degradation stopped.

`fast_pressure_decay` and `low_peak_pressure` were checked the same way and do
**not** show the same effect in this window: neither channel's absolute value
was persistently depressed to begin with here (decay stays small and noisy
throughout; OFFLOAD peak pressure stays within ~1-3% of its own baseline the
whole time), so there was nothing for their baselines to "catch up to." The
boiling-frog failure mode is a property of *this specific feature having a
sustained step-change*, not of the baseline-relative mechanism in the
abstract — any rule whose signal degrades gradually and durably is exposed to
it.

**This directly extends pass 15's "no ranking signal" finding**: `baseline_window`
sweeping was explicitly deferred there. This pass shows *why* it matters —
not as a hypothetical, but as the demonstrated mechanism suppressing the one
rule (`short_stopped_duration`) that findings/08 independently verified as
physically real for this event.

## Part B — Inside event 2's own 72h window: an elevated near-miss, not nothing

| | value |
|---|---|
| max score in window (2020-05-26 23:30 → 05-29 23:30) | 0.9899, at 2020-05-28 03:28:31 |
| percentile against fold 2's training score distribution | 98.77% |
| q=0.995 threshold | 0.9964 |
| margin below threshold | 0.0064 |
| timestamps ≥ q0.995 threshold in window | 0 / 22,965 |
| timestamps ≥ q0.999 threshold in window | 0 / 22,965 |

Not "ordinary levels" — the max score sits at the 98.8th percentile of the
fold's own training distribution, meaningfully elevated, but never crosses the
99.5th-percentile alert threshold at any point in the window. **A genuine,
measurable near-miss**, not an absence of signal. This is consistent with, but
distinct from, pass 15's "no ranking signal" verdict: pass 15 showed the
alerts that *did* fire near events 1 and 4 weren't disproportionately severe;
this shows that even where the model came closest to firing on the actual
failure (event 2), it still fell short of its own operating threshold rather
than firing and being swamped by unrelated false alarms.

## Part C — The mid-May episodes: which fold actually scored them

Fold 2's own test period starts 2020-05-25 23:30 (`test_start = event2.start
- 72h - embargo`) — mid-May is entirely inside **fold 2's training window**,
never its test period. The false alarms `docs/RESULTS.md` attributes near
event 2 come from **fold 1** instead: `extend_test_end_for_false_alarms`
pushes fold 1's evaluated test period out to 2020-05-28 23:30 (Part B1,
pass 13), which happens to swallow all of mid-May. This matters for Part E:
fold 1's training data (Feb 1 → Apr 13) already contains the March cluster,
so its rule calibration had already seen equally extreme
`stopped_duration`-ratio values *before* mid-May happened.

Seven episodes fall in 2020-05-15 → 05-25 under fold 1's evaluation (all
`false_alarm`, since none overlap event 1's own pre-failure/masked regions):

| start | duration | peak | top rule | rules > 0.5 severity |
|---|---|---|---|---|
| 05-15 00:49:34 | 34m11s | 0.9966 | fast_pressure_decay | 2/4 |
| 05-15 20:45:20 | 14m32s | 0.9964 | low_peak_pressure | 2/4 |
| 05-17 03:53:13 | 12m24s | 0.9987 | low_peak_pressure | 4/4 |
| 05-18 19:56:44 | 11m44s | 0.9966 | fast_pressure_decay | 4/4 |
| 05-19 00:30:51 | 9m05s | 0.9964 | low_peak_pressure | 4/4 |
| 05-21 05:56:14 | 14m52s | 0.9997 | low_peak_pressure | 3/4 |
| 05-21 17:18:53 | 15m51s | 0.9994 | short_stopped_duration | 3/4 |

All seven are short (9-34 minutes), all peak in the 0.996-0.9997 range, and
no single rule dominates — `low_peak_pressure` tops four, `fast_pressure_decay`
two, `short_stopped_duration` one. `promoting explain/ to core` (CLAUDE.md)
does its job here: the mixed, ranked diagnosis shows this is not one rule
crying wolf repeatedly, it's several different rules each independently
crossing their own calibrated threshold at different moments through the
collapse.

## Part D — Symmetric check: event 2 is not uniquely out-of-window

21-day pre-onset horizons, all four events, each scored by its own fitted
fold model (full detail and plots in the notebook):

| event | onset | episodes in 21d horizon | inside 72h window | closest out-of-window episode |
|---|---|---|---|---|
| 1 | 2020-04-18 | 11 | 3 (early_warning, 24h-72h) | 04-12 04:45 (5d19h before onset) |
| 2 | 2020-05-29 23:30 | 5 | 0 | 05-26 06:01 (3d17h before onset) |
| 3 | 2020-06-05 10:00 | 4 | 0 | 05-26 06:01 (10d04h before onset) |
| 4 | 2020-07-15 14:30 | 9 | 1 (early_warning, 24h-72h) | 07-12 04:38 (3d09h before onset) |

Every event has multiple above-threshold episodes in its own 21-day run-up,
and **every event except 1 and 4** has zero of them inside the scored 72h
window — event 2 is not a special case, it's the general pattern. Event 1's
closest out-of-window episode (5d19h before onset) is itself not far outside
the 72h cap. `low_peak_pressure` is the single most common top-ranked rule
across every event's pre-onset episodes, which is itself worth flagging: a
rule that tops the ranking almost everywhere, regardless of which event is
approaching, is not discriminating between events either — consistent with
pass 15's "no ranking signal" conclusion, now shown to hold event-by-event,
not just in aggregate.

## Part E — Mid-May vs. early March: indistinguishable

Both clusters scored with fold 1's model for a fair, single-calibration
comparison:

| | March (03-03→03-12) | Mid-May (05-15→05-25) |
|---|---|---|
| episodes | 8 | 7 |
| span | 9 days | 10 days |
| density | 0.89/day | 0.70/day |
| peak scores | 0.9973 – 0.9994 | 0.9964 – 0.9997 |
| rule agreement (>0.5 severity) per episode | 2-4 of 4 | 2-4 of 4 |
| top-rule mix | low_peak_pressure (4), short_stopped_duration (4) | low_peak_pressure (4), fast_pressure_decay (2), short_stopped_duration (1) |
| `stopped_duration_last` median (absolute, s) | 525 | 307 |
| `decay_rate_last` median (signed) | -0.00162 | -0.00190 |
| OFFLOAD peak (Reservoirs) median | 9.988 | 9.868 |

Episode count, density, peak-score range, rule-agreement distribution, and
absolute feature depression are all the same order of magnitude between the
two clusters — mid-May is, if anything, slightly *more* depressed on
`stopped_duration_last` and slightly lower on OFFLOAD peak pressure than
March, not less. Neither cluster has a signature the other lacks.

**Stated plainly, per the brief, without resolving by assertion**: this cuts
both ways. Either early March hides an unreported failure or genuine
near-miss (plausible — the machine is not documented as having failed then,
but under-reporting a near-miss is exactly the kind of gap a source table
compiled after the fact could have), **or** mid-May's signal — and by
extension whatever role it might have played in "detecting" event 2 — is not
failure-specific at all, i.e. the model produces comparably-severe multi-day
clusters during ordinary operation with no distinguishing feature from a real
precursor. The data available cannot adjudicate between these two
readings; both remain open (`findings/09-open-questions.md`).

## Bottom line

Pass 15 asked whether the baseline has any ranking signal; the answer was no.
This pass asks *why* the one physically-verified precursor (event 2's
duration collapse) still failed to produce a detection, and finds a
demonstrated mechanism (baseline adaptation absorbing a sustained step
change) plus confirmation the pattern is general (Part D) and that the
closest comparable cluster (March) is not distinguishable from it (Part E).
None of this overturns pass 15's verdict — it explains one concrete way a
real signal gets thrown away by the current rule design, which is exactly
the kind of lead a first baseline's error analysis is supposed to produce for
the next model stage (Isolation Forest) to either avoid repeating or to
explicitly test against.
