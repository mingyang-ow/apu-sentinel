# 3. Data quality — gaps

- **331 gaps exceeding 1 minute**, totalling **37 days 21:31:03** — roughly
  **18% of the timeline is missing**.
- Largest gaps (timestamp = end of gap): 2020-04-27 01:12:49 (2d 00:01:58) |
  2020-06-28 23:07:43 (1d 12:14:36) | 2020-03-01 04:00:09 (1d 04:03:01) |
  2020-08-05 08:23:01 (1d 00:40:33) | 2020-05-25 01:14:14 (1d 00:34:51).
- **Gaps are not random.** The 21h28m gap ending 2020-06-08 11:48 spans
  2020-06-07 14:19 → 2020-06-08 11:48, which is almost exactly event 3's
  out-of-service repair period (failure ended 7 Jun 14:30, maintenance 8 Jun
  16:00). Missing data carries information about machine state.
- **CORRECTION to an earlier claim:** the largest gap (25–27 April) does NOT
  coincide with any documented failure. Event 1 was a week earlier (18 April).
  An earlier pass summary asserted otherwise; that assertion was wrong.
- A 14h11m gap spanning 2020-07-14 07:16 → 21:28 sits **inside event 4's
  pre-failure window**.

### Pre-failure window coverage (at 72h width, measured)

| event | coverage |
|---|---|
| 1 | 0.878 |
| 2 | 0.886 |
| 3 | 0.843 |
| 4 | **0.704** |

Event 4 is materially worse: four gaps (14h11m, 5h36m, 1h45m, 5min ≈ 21.6h
total) fall inside its 72h window. If event 4 is the worst-detected event,
this is the likely explanation — it is a data limitation, not necessarily a
model failure. Note also that *score* coverage is slightly worse than *data*
coverage, because each gap additionally invalidates roughly one window-duration
of scoring positions on its trailing edge.

