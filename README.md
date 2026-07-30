# apu-sentinel

Early-warning anomaly detection on MetroPT-3 (Air Production Unit /
compressor telemetry from a metro train). Anomaly / early-warning framing,
not supervised failure classification — see `CLAUDE.md` for the full project
contract and hard rules.

## Documentation map

- **[`docs/PROJECT-STORY.md`](docs/PROJECT-STORY.md)** — what this project
  does and why, and the story of how it got here, pass by pass.
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — how it works: pipeline
  order, module responsibilities, contracts, invariants.
- **[`docs/FINDINGS.md`](docs/FINDINGS.md)** — what we learned from the data
  (index into `docs/findings/`).
- **[`docs/RESULTS.md`](docs/RESULTS.md)** — the numbers: baseline/evaluation
  results, current first, superseded kept below.

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

## Downloading MetroPT-3 (establish-then-pin checksum)

The correct SHA256 for MetroPT-3 is unknown until first download, so
`data/download.py` runs in one of two modes, selected by `data.checksum` in
`configs/base.yaml`:

1. **Establish mode** (`checksum: null`, the default): run
   `uv run python scripts/download.py`. It downloads the file, prints
   `Computed SHA256: <hash>`, and does not fail — copy that hash into
   `configs/base.yaml` (`data.checksum`).
2. **Verify mode** (`checksum` set to that hash): every subsequent run
   re-verifies the file against the pinned hash and raises if it doesn't
   match (corruption / re-published dataset / wrong file).

Re-running is idempotent — an existing file in `data/raw/` is verified
rather than re-downloaded; pass `--force` to re-download.
