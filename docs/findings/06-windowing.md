# 6. Windowing

- 30-minute window = 180 samples at 10s; train stride 5min = 30 samples.
- Fold 1: 17,179 windows kept, **481 dropped (2.7%) for spanning gaps**,
  tensor 185.53 MB.
- Each gap invalidates roughly `window_duration / stride` windows, so longer
  windows discard more data around each gap — an argument for shorter windows
  given 331 gaps.

