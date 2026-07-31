# PROJECT-STORY.md

The narrative companion to `ARCHITECTURE.md`. ARCHITECTURE says *how it
works*; this says *what happened and why*.

## What the project does

Watch a train compressor's sensors. Warn before an air leak causes failure.
Don't cry wolf.

Four real failures in seven months of data. That is the core constraint: too
few examples to learn "what failure looks like", so the approach is to learn
"what normal looks like" and flag departures.

## The story so far

- Built a pipeline that cannot cheat (no future information reaches the
  model).
- Built the scoring system **before** any model, so it could not be tuned to.
- Found the compressor has distinct operating states that had to be handled
  first.
- Built a simple rule-based model. It reported 2 of 4 events caught.
- Checked whether chance alone would do that well. It would. Honest score:
  **0 of 4 with skill**.
- Investigated why. Found the model's own normalisation erased the signal:
  when degradation lasted longer than its 7-day reference window, the
  reference slid down to meet it and the machine looked normal again
  ("boiling frog").
- Fixed the feature. The ratio now moves correctly (1.195 → 0.274 at event
  2's window-open).
- The **training calibration** then absorbed the fix the same way, one level
  up — training data contained a week of degraded operation, so the model
  learned degradation was normal.
- Pass 18 addressed that at the split level. It couldn't be fixed there
  either: the parameter that would need widening only ever excludes an
  *earlier* event's precursor from a *later* fold, never an event's own
  precursor from its own fold. The margin stays where it was.
- Built Isolation Forest, the second model in the progression. A wide sweep
  looked promising for event 4 (p as low as 0.004) — but a 160-combination
  grid finds a maximum, not a p-value. Reading the result off ONE
  pre-registered operating point instead (chosen on false-alarm grounds
  only, before looking at detections) gave an unremarkable p≈0.37–0.83,
  matching the rule-based baseline's own honest score.
- Built a convolutional autoencoder, the third and final model — testing
  whether relationships BETWEEN channels break down before failure, the one
  mechanism neither prior model could see. Trained successfully on Colab.
  No swept detection threshold kept false alarms under the ceiling, at any
  quantile tested. Traced why: the model's reconstruction error tracks
  *how far the test period sits from the training window* (the system
  drifts ~3× over the recording span), not whether the data is anomalous —
  a fault-detection failure mode distinct from the first two models' own
  (chance-level rather than drift-dominated).
- Three models, three honest negative results, one consistent metric. That
  consistency — not a working detector — is this project's actual output.

## The passes

| # | What it built |
|---|---|
| 1 | Repo structure, CLAUDE.md rules, test stubs |
| 2 | Download + checksum + loader |
| 3 | Config loading |
| 4 | EDA notebook, failure identification |
| **5** | **Walk-forward split + leakage guard** |
| **6** | **Per-fold scaling + fit-on-train-only guard** |
| 7 | Gap-aware windowing |
| **8** | **Evaluation harness (built before any model)** |
| 9 | Operating regimes — found 3 states, not 2 |
| 10 | Findings record + third state |
| 11 | Regime-conditional scaling + cycle features |
| **12** | **Rule-based baseline → "2 of 4"** |
| **13** | **Null comparison → "2 of 4" was chance** |
| 14 | Documentation reorganisation |
| 15 | Threshold sweep → no ranking signal |
| **16** | **Error analysis → boiling-frog bug found** |
| 17 | Lagged baseline → feature fixed, calibration absorbed it |
| 18 | Training-margin sweep → negative, for a verified structural reason |
| 19 | Housekeeping pass |
| **20-22** | **Isolation Forest → one pre-registered operating point → no skill (p≈0.37-0.83)** |
| 23 | Convolutional autoencoder (replacing an LSTM that hung on CPU) |
| **24** | **No operating point at any threshold → traced to drift, not faults** |

Bold = load-bearing. The rest support them.

## The four ideas everything rests on

1. **Never let the model see the future.** Split by time; fit everything on
   training data only. Two automated guards block edits that break this.
2. **Score by event, not by data point.** How many failures caught, how
   early, how often crying wolf — not per-timestamp accuracy, which would
   look good and mean nothing.
3. **Build the scoring before the model.** Otherwise you invent a metric
   that flatters what you built.
4. **Compare against chance.** A model firing every 1.5 days "catches" a
   failure within 3 days about 80% of the time by luck. This check turned 2
   of 4 into 0 of 4.

## Status

Pipeline: complete and guarded. Evaluation: complete, with chance
comparison. Model progression complete — three models, one consistent
metric, none with skill at an honest single operating point:

| model | aggregate skill result |
|---|---|
| rule-based (lagged, 24h margin) | p = 0.737 |
| Isolation Forest, arm A | p = 0.375 |
| Isolation Forest, arm B | p = 0.833 |
| conv autoencoder | no operating point ≤ 0.3/day |

Optional extensions from here: live-data simulation, alert API, dashboard —
or drift-adaptive scoring, the autoencoder's own most direct follow-up
(`docs/findings/13-autoencoder-drift.md`).
