# 13. Autoencoder reconstruction error tracks drift, not faults

Pass 23/24, full detail and both data tables in `docs/RESULTS.md` §23.

- Reconstruction-based anomaly detection implicitly assumes the training
  distribution stays stationary through the test period. This system does
  not: the compressor's cycle timing drifts ~3× over the seven-month
  recording span (`findings/08-cycle-timing.md`), and the conv
  autoencoder's reconstruction error drifts with it. No swept quantile
  (up to 0.99995) keeps the worst-case pooled false-alarm rate under
  0.3/day.
- The per-fold gradient is direct evidence for the mechanism, not just a
  correlation: fold 1 tests one week past its training window and barely
  shifts (+7% median reconstruction error); folds 2-4 test weeks to months
  later and shift +39-88%. Excess false-alarm rate over the calibrated
  expectation scales the same way (2.2× at the loosest quantile up to 32×
  at the tightest) — the model's error tracks *distance from its own
  training window*, not anomalousness of the data it's scoring.
- Train/test maxima barely differ (within ~1 unit, in both directions,
  across all four folds) while medians shift by up to 88% — a fault
  producing a distinctive reconstruction failure would move the tail, not
  the bulk of the distribution. The exceedance fraction above the train
  p99 is only 1.3-3.7% (close to the 1% a stationary distribution would
  give), yet false-alarm rate is 6-32× the calibrated expectation, because
  the drift-driven exceedances cluster into sustained episodes rather than
  scattering — episode-level counting (this project's own metric,
  correctly applied) turns a mild sustained shift into many alerted
  episodes.
- Possible mitigations, **not attempted here**:
  - Shorter training windows positioned nearer the test period (reduces
    exposure to drift, at the cost of less training data per fold).
  - Drift-adaptive normalisation of reconstruction error (e.g. a trailing
    baseline on the score itself, the same class of fix pass 17 applied to
    the rule-based model's own features — and the same class of fix that
    pass 17/18 showed only partially works, see `findings/09-open-questions.md`).
  - Differencing the input so the model reconstructs change rather than
    level, which would be insensitive to a slowly drifting baseline by
    construction.
