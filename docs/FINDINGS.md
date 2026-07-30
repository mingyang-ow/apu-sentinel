# FINDINGS.md — index

This is a permanent, referenceable record of what we learned from the data.
It is distinct from `CLAUDE.md`: `CLAUDE.md` holds *rules for how to build*;
these findings hold *what we found by looking*. Consult them before making
modelling decisions — several entries directly constrain what a later pass is
allowed to assume.

Content lives in `docs/findings/*.md`, one file per topic, listed below in
reading order. **Baseline/evaluation results are NOT here** — they live in
`docs/RESULTS.md` (current results, plus superseded ones kept for the
record), since they are read for different reasons and grow fastest.

| # | File | What's in it |
|---|---|---|
| 1 | [`findings/01-dataset.md`](findings/01-dataset.md) | Source, span (Feb–Aug 2020), 15 columns, **10s nominal sampling** (not 1Hz), the `Unnamed: 0` load-time artifact. |
| 2 | [`findings/02-ground-truth.md`](findings/02-ground-truth.md) | The 4 documented failure events (all Air leak / High stress) and the source-table typos/corrections applied to them. |
| 3 | [`findings/03-data-quality.md`](findings/03-data-quality.md) | **331 gaps, ~18% of the timeline missing**; gaps are not random (one aligns with event 3's repair); a corrected earlier claim; per-event pre-failure window data coverage (event 4 worst, ~0.70). |
| 4 | [`findings/04-split-design.md`](findings/04-split-design.md) | Walk-forward 4-fold scheme, embargo rationale, the event-2/event-3 proximity that originally capped the window sweep at 72h (superseded by per-event caps, see `RESULTS.md` §13). |
| 5 | [`findings/05-scaling.md`](findings/05-scaling.md) | Why robust (median/IQR) scaling is the default; seasonal drift in `Oil_temperature`; pathological global spreads (`TP2`, `DV_pressure`, `Motor_current`) that motivate regime-conditional scaling. |
| 6 | [`findings/06-windowing.md`](findings/06-windowing.md) | 30-minute window sizing vs. the measured 10s cadence; gap-dropped window counts for fold 1. |
| 7 | [`findings/07-regimes.md`](findings/07-regimes.md) | Flag-polarity verification (COMP is inverted from the naive reading); the two-state → four-state (LOADED/OFFLOAD/STOPPED/TRANSITION) segmentation; OFF's bimodality; the pressure duty-cycle interpretation. |
| 8 | [`findings/08-cycle-timing.md`](findings/08-cycle-timing.md) | **STOPPED-duration is the strongest single leak signal found** — clear precursor for events 2 & 4, absent for 1 & 3; a **3× seasonal drift** that breaks static thresholds; an **unreported March anomaly** as extreme as event 2's precursor. |
| 9 | [`findings/09-open-questions.md`](findings/09-open-questions.md) | Whether the STOPPED-duration drift is a gap-splitting artifact (checked — **physical, not artifact**); the `Motor_current` regime-definition circularity; the 72h window cap question; the boiling-frog baseline bug — **resolved with a fix that doesn't produce skill** (pass 17); the training-exclusion margin sweep — **negative, for a verified structural reason** (pass 18), plus the standing training-purity limitation that applies to every subsequent model. |
| 10 | [`findings/10-process-lessons.md`](findings/10-process-lessons.md) | Working-practice lessons (notebook CWD, kernel restarts, config strictness, verifying claims independently, a contract test catching a design gap before a second model made it costly). |
| 11 | [`findings/11-regime-scaling-and-cycle-features.md`](findings/11-regime-scaling-and-cycle-features.md) | Per-(fold, regime) scaler wiring and sample counts; amplification warnings fired on the real dataset; cycle-timing feature gap-truncation stats. |
| 12 | [`findings/12-event2-error-analysis.md`](findings/12-event2-error-analysis.md) | Error analysis on event 2's out-of-window precursor (pass 16): **boiling-frog mechanism confirmed** (the 7-day trailing baseline absorbs the sustained STOPPED-duration collapse); event 2's own window shows an elevated 98.8th-percentile near-miss, not nothing; the pattern is general across all four events, not unique to event 2; mid-May is **indistinguishable** from the unreported March cluster. |

## Results

See [`docs/RESULTS.md`](RESULTS.md) for baseline/evaluation results:
current (null-comparison-corrected) figures first, superseded figures kept
below them and clearly labelled — never deleted, per the same
verify-before-recording principle as the April-gap correction in
`findings/03-data-quality.md`.
