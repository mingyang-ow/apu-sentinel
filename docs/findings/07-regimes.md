# 7. Operating regimes

### Flag polarity — verified empirically, not assumed

| flag | meaning | evidence (mean Motor_current) |
|---|---|---|
| `COMP` | 1 → OFF, 0 → ON | COMP=0 → 5.60; COMP=1 → 1.36 |
| `DV_eletric` | 1 → ON, 0 → OFF | mirror image of COMP |
| `MPG` | 1 → OFF, 0 → ON | MPG=1 → 1.34; MPG=0 → 5.56 |

COMP's polarity is the **inverse of the naive reading** — it is active when
there is *no* air intake. Pairwise agreement after normalising polarity:
COMP↔DV_eletric 98.9%, COMP↔MPG 99.6%, DV_eletric↔MPG 99.3%. COMP is the
deciding flag.

### Occupancy (original two-state segmentation, superseded by §7a below)

| state | occupancy | median run |
|---|---|---|
| OFF | 82.0% | 1080s (18 min) |
| ON | 13.8% | 99s |
| TRANSITION | 4.2% | 20s |

**Architecturally critical:** ON's median run is 99s ≈ 10 samples, far shorter
than the 180-sample window. **There is no such thing as an "ON window"** —
essentially every window spans multiple full cycles. Therefore regime handling
cannot be applied at window level; scaling must be applied **per-timestamp, by
that timestamp's regime, before windowing**.

### Within-regime variance (the key table)

| channel | global IQR | within-OFF | within-ON |
|---|---|---|---|
| `Motor_current` | 3.7675 | 3.675 | 0.630 |
| `TP2` | 0.0040 | 0.0040 | 2.3120 |
| `DV_pressure` | 0.0040 | 0.0040 | 0.8040 |

Interpretation:
- `Motor_current`: **fixed within ON** (3.77 → 0.63). Within-OFF barely moves
  because OFF is itself bimodal (below).
- `TP2` / `DV_pressure`: these are **LOADED-only channels**. They vent to ~0
  when the compressor stops, so their 0.004 OFF spread is sensor noise around
  zero carrying no information; during ON they have healthy spread (2.31 /
  0.80). Regime conditioning does not "fix" them — it reveals they are only
  alive in one state.

### OFF is bimodal — a third state

Of 1,269,620 OFF samples: **822,119 (64.8%) at Motor_current ≈ 0** and
**447,307 (35.2%) at ~3.5–4.3A sustained** (median run ~400s, so not a
transient).

**No digital flag separates them.** Separation scores
(|P(flag=1|low) − P(flag=1|high)|): Oil_level 0.042, Caudal_impulses 0.037,
DV_eletric 0.015, MPG 0.015, Pressure_switch 0.003, LPS 0.000, **Towers
0.000**. The flags are saturated during OFF. The tower-regeneration hypothesis
was tested and rejected.

Analog evidence that the two modes are physically distinct:

| channel | STOPPED median | OFFLOAD median | shift |
|---|---|---|---|
| `TP3` | 8.658 | 9.644 | 0.986 |
| `H1` | 8.646 | 9.630 | 0.984 |
| `Reservoirs` | 8.660 | 9.644 | 0.984 |
| `Oil_temperature` | 59.375 | 67.650 | 8.275 |
| `Motor_current` | 0.040 | 3.765 | 3.725 |
| `TP2` | -0.012 | -0.012 | **0.000** |
| `DV_pressure` | -0.020 | -0.020 | **0.000** |

Only **0.015% of OFF samples** fall in the 1–3A valley between the modes — the
separation is near-total, so a 2A threshold sits in genuine emptiness.

### The duty cycle

Putting it together: **LOADED** compresses the system up → **OFFLOAD** sits at
the top of the cycle (~9.64 bar, motor spinning, intake closed) → **STOPPED**
is the *decay phase*, pressure bleeding down from ~9.6 toward the 8.2 bar
threshold at which MPG restarts the compressor.

Consequence: STOPPED is **not a static state**. Its pressure IQR of 0.63
reflects the systematic sweep from 9.6 → 8.2, not variability around a stable
normal. A static per-state center is therefore the wrong model for STOPPED;
time-since-entering-STOPPED, or decay *rate* rather than level, may be needed.

### 7a. Third state implemented (pass 10)

The bimodal split above is now a real, config-driven fourth regime label.
Segmentation is four states: `LOADED` (COMP=0), `OFFLOAD` (COMP=1 and
Motor_current ≥ `regimes.offload_current_threshold`, default 2.0A), `STOPPED`
(COMP=1 and Motor_current below that threshold), and `TRANSITION` (within
`transition_settle` of any committed state change).

Measured four-state occupancy and run lengths (real dataset):

| state | occupancy | n_runs | median run | q25 | q75 |
|---|---|---|---|---|---|
| STOPPED | 52.48% | 10,293 | 674s | 416s | 981s |
| OFFLOAD | 27.46% | 10,694 | 376s | 367s | 377s |
| LOADED | 13.80% | 10,624 | 99s | 89s | 109s |
| TRANSITION | 6.25% | 31,611 | 20s | 20s | 20s |

LOADED's occupancy/run-length is unchanged from the two-state segmentation
above (it was never part of the OFF/COMP=1 split). TRANSITION occupancy rose
from 4.2% to 6.25% because the cycle now has up to three committed
transitions per cycle (LOADED→OFFLOAD→STOPPED→LOADED) instead of two.
OFFLOAD's run length is remarkably tight (q25=367s, q75=377s — a 10-second
spread) compared to STOPPED's wide spread (416–981s), consistent with §7's
"duty cycle" interpretation: OFFLOAD is a comparatively fixed-duration phase
at the top of the cycle, while STOPPED is a variable-length pressure-decay
phase, not a steady state.

