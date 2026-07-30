# 8. Cycle timing — the strongest leak signal found

An air leak makes pressure decay faster → shorter STOPPED periods → more
frequent cycling. Median STOPPED duration (20-cycle rolling) was plotted
against the four events:

- **Event 2: clear precursor.** Duration collapses from ~1200s to ~280s around
  15–20 May and stays there until the 29 May failure — roughly **10 days of
  sustained warning**.
- **Event 4: sharp dip** to ~150s (the lowest point in the series) at the
  failure.
- **Events 1 and 3: no clear precursor.** Duration is ordinary (~1000–1400s)
  before event 1, and *elevated* going into event 3 (post-event-2 recovery).

So this single hand-built feature gives **2 of 4 events**, not 4 of 4.

### Unreported anomaly, empirically confirmed

Early March shows a ~1 week dip to ~200–250s — **as extreme as event 2's
precursor** — with no reported failure. The report table lists only high-stress
air leaks, so this may be an unreported event or a genuine near-miss. It is
**unresolvable from the available labels**, and any detector using this feature
will fire there. State this plainly rather than reclassifying it.

### Drift — breaks static thresholds

Median STOPPED duration trends from **~1400s (Feb) to ~500s (Aug)** — a **3×
shift, larger than most event-related dips**.

Consequences:
- Fold 4 trains back to February (normal ≈ 1400s) and tests in July (normal ≈
  500s). A static notion of normal cycle timing would alarm continuously
  through summer.
- Thresholds on this feature must be **relative to a trailing baseline**, not
  absolute.
- But a baseline that adapts too fast absorbs gradual degradation and loses the
  signal (boiling-frog). Event 2's step change would survive a ~7-day baseline;
  slow drift would not. The baseline window is a deliberate choice.
- **Confound:** `Oil_temperature` rose 58 → 63°C over the same period. Seasonal
  ambient change and cycle frequency move together — do not attribute the trend
  to wear without argument.

### Consequence for the window-width cap

Event 2's precursor is visible ~10 days out, but the sweep is capped at 72h by
the E2/E3 proximity. Events 1, 2 and 4 have no such constraint. Options:
per-event window widths, or **keep wide windows and mask the overlapping
region from false-alarm counting** (the evaluation harness already supports
masking). The latter is cleaner.

