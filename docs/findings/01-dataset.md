# 1. Dataset characteristics

- Source: UCI ML Repository dataset 791, MetroPT-3 (Air Production Unit /
  compressor telemetry from a metro train).
- File: `MetroPT3(AirCompressor).csv`
- SHA256 (pinned, of the downloaded zip archive — see `data/download.py`'s
  checksum convention): `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`
- Span: 2020-02-01 → 2020-08-31, measured 213 days 03:59:50
- Rows: 1,516,948
- Columns: 15 usable — 7 analog + 8 digital (timestamp is the index)
  - Analog: `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`,
    `Oil_temperature`, `Motor_current`
  - Digital: `COMP`, `DV_eletric`, `Towers`, `MPG`, `LPS`,
    `Pressure_switch`, `Oil_level`, `Caudal_impulses`
- **Sampling: 10 seconds nominal**, with clock jitter. Interval counts:
  10s → 1,337,521 | 9s → 128,277 | 12s → 38,321 | 13s → 7,988 | 11s → 4,471.
  This is NOT 1 Hz data.
- A `Unnamed: 0` column is present in the raw CSV — a pandas index
  serialisation artifact, not a sensor. Dropped at load time with a warning.

