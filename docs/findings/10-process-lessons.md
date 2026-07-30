# 10. Process lessons

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
- **A correct-looking implementation of a wrong specification produced a
  misleading negative result, and was only caught by asking why (pass 20).**
  Pass 5's brief specified excluding "all earlier events'" training-exclusion
  regions from each fold. `make_folds()` implemented that literally: `if
  other.id == event.id: continue` -- skip the fold's own event, keep every
  other one. This passed every test written against it (including pass 18's
  own new tests) for 15 passes, because "earlier event" was the only case
  anyone had reason to construct a fixture for. The actual bug: a fold's own
  target event starts AFTER that fold's own `train_end` by construction (that
  is what makes it a future event to predict), so its own precursor was never
  excluded from its own training, at ANY `pre_margin_hours` value -- the
  parameter was reaching for data it could never touch. Pass 18's margin
  sweep came back completely flat as a direct, deterministic consequence, and
  read at face value as "this lever doesn't work." It was only caught by
  refusing to accept that reading and asking why a mechanism that should
  provably work at some margin showed zero response at any margin tested --
  the flatness itself was the tell, not a shrug-worthy null result. Lesson: a
  specification can be implemented faithfully and still be wrong; a negative
  sweep result is itself evidence to interrogate, not just report, especially
  when the direction of the non-effect is total rather than merely weak.
- **A 160-combination sweep produced an apparently significant result that
  required a single pre-chosen operating point to interpret honestly (pass
  22).** Pass 21's Isolation Forest sweep (4 quantiles × 5 widths × 4 folds ×
  2 arms) reported event 4 detecting at `p_chance_permutation` as low as
  0.004 — read naively, a striking result. At α=0.02 over 160 cells, a
  handful of such hits are expected by chance alone; "the best cell in a
  large sweep" is a maximum, not a p-value, no different in kind from
  picking the test-optimal threshold quantile after seeing results (already
  forbidden by `fit_threshold_sweep`'s own docstring). Selecting ONE
  operating point by a rule that depends only on false-alarm rate (never on
  detection outcomes), then reporting the aggregate statistic there,
  produced a materially different, unremarkable picture (p≈0.37–0.83,
  matching the rule-based baseline) — even though the underlying event-4
  signal itself survived independent scrutiny (a gap-artifact check) at the
  narrower cells where it was strong. Lesson: a model's headline result must
  be read off a point chosen before the sweep, or by a rule stated and
  applied independently of the sweep's own outcomes; the single most
  extreme cell in a wide grid is not that, however small its p-value looks.

