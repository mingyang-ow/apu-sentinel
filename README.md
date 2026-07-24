# apu-sentinel

Early-warning anomaly detection on MetroPT-3 (Air Production Unit /
compressor telemetry from a metro train). Anomaly / early-warning framing,
not supervised failure classification — see `CLAUDE.md` for the full project
contract and hard rules.

## Quickstart

```bash
make setup              # uv sync + pre-commit install
make baseline           # rule-based baseline
make train CONFIG=local # train on CPU / data subset
make evaluate           # episode-level evaluation
make test                # full suite, incl. leakage guards
```

Runs locally (WSL2, CPU, small subset) or on Colab (GPU, full data) from the
same code — behaviour is selected entirely by `configs/{local,colab}.yaml`.
