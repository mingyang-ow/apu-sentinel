# 5. Scaling findings

- **Robust (median/IQR) is the default**, because training data still contains
  *unreported* anomalies that only median/IQR resist.
- Fold 1 (Feb–mid-Apr) vs full series (Feb–Aug), robust stats:
  `Oil_temperature` center 58.275 → 62.700 (+7.6%), IQR 7.400 → 9.475 (+28%).
  This is **seasonal drift**, and it is the concrete demonstration that fitting
  a scaler on the full series would encode summer temperatures that had not yet
  happened when fold 1 makes its April prediction.
- Pathological spreads (global, fold 1): `TP2` IQR 0.004 (~250× amplification),
  `DV_pressure` IQR 0.006, `Motor_current` center 0.042 with IQR 3.750 (spread
  ~90× the center).
- `TP3` (center 8.992, IQR 0.964) and `Reservoirs` (8.994, 0.962) are
  near-duplicate — closely coupled parts of the same pneumatic circuit. This
  may make per-channel attribution ambiguous when both light up.
- `zero_scale_epsilon` (1e-8) guards **division by zero** on constant channels.
  It deliberately does NOT catch TP2's 0.004 — that is real amplification, not
  a numerical bug, and substituting 1.0 there would silently disable scaling.

