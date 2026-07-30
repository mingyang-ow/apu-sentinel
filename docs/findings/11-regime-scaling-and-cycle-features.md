# 11. Regime-conditional scaling and cycle-timing features (pass 11)

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

