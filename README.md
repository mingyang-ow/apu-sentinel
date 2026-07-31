# apu-sentinel

Early-warning anomaly detection on MetroPT-3 — Air Production Unit
(compressor) telemetry from a metro train, Feb–Aug 2020. Four documented
air-leak failures in seven months of data. Too few to learn "what failure
looks like" without overfitting, so the approach is to learn what normal
operation looks like and flag departures — anomaly detection, not
supervised classification.

## Headline result

Three models were built, in increasing order of capability. **None shows
skill at an honest, pre-registered operating point:**

| model | result |
|---|---|
| rule-based | p = 0.737 |
| Isolation Forest (arm A / arm B) | p = 0.375 / 0.833 |
| conv autoencoder | no operating point ≤ 0.3 false alarms/day |

"p" is the probability chance alone would produce the same detection —
higher means less distinguishable from noise. Nothing here beats chance.

## Why the negative result is the point

Three times this project produced something that looked like a win, and
three times a check built *before* the model dismantled it:

- The rule-based baseline reported **2 of 4 events detected**. Comparing
  against the model's own firing rate showed both detections were exactly
  what chance would produce.
- Isolation Forest showed `p < 0.02` on one event. That was the best cell
  of a 160-combination sweep — a maximum, not a p-value. At a single
  operating point chosen before looking at detections, it was p = 0.375.
- A confirmed feature fix (the "boiling frog" bug: a 7-day trailing
  baseline absorbing a sustained degradation until it read as normal again)
  moved the underlying signal correctly and still produced no detection —
  the training calibration had absorbed the same problem one level up.

The value here isn't a working detector. It's a harness rigorous enough
that when a result looks like success, you can trust the check that says
it isn't.

## Method — what makes the result trustworthy

- Time-based walk-forward splits with an embargo gap; two automated guards
  block any edit that would let training data see the future.
- The evaluation harness was built **before any model existed**, so no
  metric was invented after the fact to flatter a result.
- Every detection is compared against chance, two ways (Poisson process +
  empirical permutation).
- Metrics are episode-level (one alert = one contiguous abnormal stretch),
  never per-timestamp accuracy or point F1.
- The operating point (alert threshold) is chosen on false-alarm grounds
  only, before detection outcomes are ever looked at.
- Findings and corrections are recorded as they happened, including where
  an earlier reading turned out to be wrong.

## What was found about the data

- An undocumented, unlabelled period in early March is comparably extreme
  to real pre-failure behaviour and accounts for ~75% of the extreme tail
  in training data — no failure-anchored mechanism can exclude it.
- The compressor has **three** operating states, not two (LOADED / OFFLOAD
  / STOPPED, plus TRANSITION) — and 18% of the seven-month timeline is
  missing (331 gaps).
- Cycle behaviour drifts roughly 3× across the recording period, which is
  exactly what defeats reconstruction-based anomaly detection (see the
  autoencoder result above).

## Repo map

| Doc | What's in it |
|---|---|
| [`docs/PROJECT-STORY.md`](docs/PROJECT-STORY.md) | What happened, pass by pass, and why |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How it works: pipeline order, module responsibilities, contracts |
| [`docs/RESULTS.md`](docs/RESULTS.md) | The numbers: full evaluation results, current first, superseded kept below |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | What the data showed (index into `docs/findings/`) |
| [`docs/logs/`](docs/logs/) | Raw evidence — e.g. the Colab training log behind the autoencoder result |

## Running it

```bash
uv sync                     # install package + dev deps + pre-commit
make test                   # full suite, incl. leakage guards
make baseline                # rule-based baseline
make train CONFIG=local      # train on CPU / small subset — a code-path check
make train CONFIG=colab      # real training on GPU (run in an actual Colab
                              # notebook; self-terminates when done)
make evaluate                 # episode-level evaluation
make lint                    # ruff
```

Same code runs locally (CPU, subset) or on Colab (GPU, full data) — behaviour
is selected entirely by `configs/{local,colab}.yaml`, never by editing code.

## Limitations and what was not attempted

- Four documented failures is too few to support strong statistical
  conclusions about any model's true skill, positive or negative.
- The March period's status is unresolved: comparably severe to a real
  precursor, but un-anchored to any documented event — neither confirmed
  nor ruled out as a real (unreported) failure precursor.
- Identified but not attempted: drift-adaptive normalisation of
  reconstruction error, shorter training windows positioned nearer the test
  period, and differencing the model's input so it reconstructs change
  rather than level.
- Scoped out entirely: streaming simulation, an alert API, a dashboard.

## What I would do differently

- Build the chance comparison *before* the first model, not after — it
  would have caught the rule-based baseline's "2 of 4" the same week it
  was reported, not a pass later.
- Test shell scripts, not just Python: a Colab bootstrap script carried a
  syntax error for 22 passes before anyone actually ran it end to end.
- Make diagnostics permanent from the start rather than patching them in
  when something fails — the autoencoder's failure mode was only
  understood because a print statement was added by hand, after a
  completed GPU run had already crashed on the way to reporting it.

---

## Appendix: downloading MetroPT-3 (establish-then-pin checksum)

The correct SHA256 for MetroPT-3 is unknown until first download, so
`data/download.py` runs in one of two modes, selected by `data.checksum` in
`configs/base.yaml`:

1. **Establish mode** (`checksum: null`, the default): run
   `uv run python scripts/download.py`. It downloads the file, prints
   `Computed SHA256: <hash>`, and does not fail — copy that hash into
   `configs/base.yaml` (`data.checksum`).
2. **Verify mode** (`checksum` set to that hash): every subsequent run
   re-verifies the file against the pinned hash and raises if it doesn't
   match (corruption / re-published dataset / wrong file).

Re-running is idempotent — an existing file in `data/raw/` is verified
rather than re-downloaded; pass `--force` to re-download.
