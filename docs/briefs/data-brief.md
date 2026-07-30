# apu-sentinel — Build Pass 2: Data Download, Checksum Verify & Minimal Load

**Read this whole brief first. This pass implements the FIRST real logic:
downloading MetroPT-3, verifying its integrity by checksum, and loading it
into a clean time-sorted DataFrame. It STOPS THERE. Do NOT implement splitting,
scaling, windowing, feature engineering, regime segmentation, or any model
logic — those are later passes and are protected by leakage guards that don't
belong in this pass. Respect the hard rules in CLAUDE.md at all times.**

Repo: `/home/bren/Projects/apu-sentinel` (already scaffolded, on branch main).

## Scope of THIS pass (and its hard boundary)

IN scope:
1. `src/apu_sentinel/data/download.py` — fetch MetroPT-3 into `data/raw/`, with
   SHA256 checksum verification (two modes — see below).
2. `src/apu_sentinel/data/load.py` — minimal loader: raw file → clean,
   time-sorted pandas DataFrame with parsed datetime index/column, plus light
   structural sanity checks. NOTHING beyond loading.
3. Tests for both, including the checksum-rejection test.
4. Thin `scripts/download.py` wrapper (already stubbed) wired to the function.

OUT of scope (do NOT touch in this pass):
- Time-based splitting (next pass — has its own leakage guard).
- Scaling / normalization (next pass — fit-on-train-only guard).
- Windowing, features, regimes, models, evaluation.
The moment logic would split, scale, or window the data, STOP — that is the
next pass.

## The checksum flow (two-mode, deliberate)

The correct SHA256 for MetroPT-3 is UNKNOWN until first download — there is no
value to hardcode. So `download.py` operates in two modes driven by
`config.data.checksum`:

- **`checksum is null` (establish mode):** download the file, compute its
  SHA256, PRINT it clearly (e.g. `Computed SHA256: <hash>`), and DO NOT fail on
  mismatch (there is nothing to match yet). Save the file to `data/raw/`.
- **`checksum is a string` (verify mode):** download (or use existing file),
  compute SHA256, and RAISE a clear error if it does not match the configured
  value. Only proceed if it matches.

The user will run once in establish mode, copy the printed hash into
`configs/base.yaml` (`data.checksum`), and thereafter the function runs in
verify mode. Document this two-step flow in the function's docstring and in a
short note in the README.

Implementation notes:
- Compute SHA256 by streaming the file in chunks (do not read a large file
  fully into memory).
- If the file already exists in `data/raw/`, verify it rather than
  re-downloading (idempotent); provide a `force` option to re-download.
- MetroPT-3 source: UCI Machine Learning Repository, dataset 791. Put the
  actual download URL in config (`data.url`), not hardcoded in code — the user
  will confirm/fill it. If the exact URL is uncertain, leave a clearly-marked
  config placeholder and a docstring note rather than guessing a wrong URL.
- Print the downloaded file size after fetching, so the user can sanity-check
  the download completed (a few KB = likely an error page, not data).
- The download runs in the USER's environment (WSL2 / Colab), never assume it
  runs in CI or this environment. Network access is the user's.

## The loader (minimal, safe)

`load.py` — a function that:
- Reads the raw MetroPT-3 file (CSV) into a pandas DataFrame.
- Parses the timestamp column into real datetimes.
- Sorts by time ascending and verifies monotonic non-decreasing timestamps
  (report if not — do NOT silently reorder-and-hide problems; surface it).
- Runs light structural checks: expected columns present, reports shape,
  time span (min/max timestamp), and any all-null columns.
- Returns the DataFrame. That's it. No transformation, no derived columns,
  no filtering beyond what's needed to load cleanly.

Do NOT: split, scale, resample, window, drop rows based on values, or engineer
features. If a check reveals a data-quality issue, REPORT it (log/print), do
not fix it here — fixing belongs in a later, deliberate cleaning step.

## Tests to implement (real, not stubs, for this pass)

- `test_download` (or add to an existing data test file):
  - Checksum VERIFY mode REJECTS a corrupted/mismatched file — construct a
    small temp file with a known hash, configure a DIFFERENT expected hash,
    assert the verify function raises. This is the crown test of this pass.
  - Checksum VERIFY mode ACCEPTS a matching file — same file, correct hash,
    asserts no raise.
  - Establish mode (null checksum) computes and returns/prints a hash without
    raising.
  - Use small synthetic temp files — do NOT download the real dataset in tests.
- `test_load`:
  - Given a tiny synthetic CSV fixture (a handful of rows with a timestamp
    column + a couple of numeric columns), the loader returns a DataFrame with
    parsed datetimes, sorted ascending, correct shape.
  - Given an out-of-order fixture, the loader surfaces/reports the disorder
    (per the "surface, don't hide" rule).
- Keep the existing scaffold tests green. The leakage-guard tests remain
  skipped/placeholder — this pass does NOT implement split or scaling, so do
  NOT un-skip them.

Put shared synthetic fixtures in `tests/conftest.py`.

## Config additions (configs/*.yaml)

Add under `data:` in `base.yaml`:
```yaml
data:
  url: "<MetroPT-3 UCI download URL — confirm/fill>"
  raw_dir: data/raw
  raw_filename: "<expected filename>"
  checksum: null          # establish first, then pin — see download.py
```
Keep `local.yaml` / `colab.yaml` pointing at the same raw data (the subset
logic that differs between them is a LATER pass — do not add it here).

## Constraints reminder (from CLAUDE.md)

- No hardcoded paths/URLs/hyperparameters — all in config.
- Do not commit anything under `data/` (the download output is gitignored).
- Loader surfaces data issues, does not silently "fix" them.
- This pass touches NO logic protected by the leakage guards.

## Definition of done for THIS pass

- `download.py` implements two-mode checksum flow, streams the hash, idempotent,
  size reported, URL/filename/checksum in config.
- `load.py` loads → parses datetimes → sorts → sanity-checks → returns df, and
  nothing more.
- `scripts/download.py` wired as a thin wrapper.
- New tests pass (`make test` green), INCLUDING the checksum-rejection test.
  Leakage-guard tests remain skipped (not implemented this pass).
- `make lint` clean.
- README updated with the establish-then-pin checksum instructions.
- Nothing committed — leave it for the user to review and commit.

When done: print (a) a summary of the two files and what each does, (b) the
exact commands the user should run to establish the checksum
(`uv run python scripts/download.py` → copy hash → paste into base.yaml), and
(c) confirmation that no split/scale/window logic was added. Then STOP.