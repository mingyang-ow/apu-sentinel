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

