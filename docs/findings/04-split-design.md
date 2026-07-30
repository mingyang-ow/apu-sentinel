# 4. Split design

- Walk-forward (rolling-origin, expanding window), 4 folds, one per event.
  Chosen because 4 events cannot support a single holdout — recall on one test
  event is either 0% or 100%.
- Embargo: 24h between train end and test start, to prevent sliding windows
  straddling the boundary.
- **The event 2 → event 3 gap is 6 days 4 hours**, which caps how wide the
  pre-failure window sweep can go before event 3's window swallows event 2's
  post-repair recovery period. Cap is currently **72h** (tightened from 96h
  when the overlap check was extended to cover `test_start`, not just the label
  window).
- Fold 1's training slice: 529,973 rows over ~71 days — **86% of the 613,440
  expected at complete 10s sampling**.

