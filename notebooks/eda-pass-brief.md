# apu-sentinel — Build Pass 4: EDA & Failure-Window Identification (notebook)

**Read this whole brief first. This pass scaffolds an EXPLORATORY notebook in
`notebooks/exploratory/` whose PURPOSE is to help the USER identify the
MetroPT-3 failure windows by eye from the signals. The notebook is a tool for
human judgment, NOT an automated labeller. Do NOT algorithmically declare
failure periods, do NOT split/scale/window/transform the data, do NOT touch
model or pipeline code. Read-only exploration. Respect CLAUDE.md.**

Repo: `/home/bren/Projects/apu-sentinel` (branch main; data download +
checksum + config loading already committed; real CSV is in `data/raw/`).

## Why this pass exists, and its philosophy

The split (next pass) needs the failure event date-ranges. Those are the only
things evaluation can score against, so they must be correct. We are deriving
them EMPIRICALLY from the signals first (then the user cross-checks against the
dataset documentation separately).

CRITICAL framing: the notebook must NOT use an anomaly-detection algorithm to
decide the failure windows — that would be circular (using an anomaly detector
to label the data we will train an anomaly detector on). The notebook's job is
to make the signals VISIBLE and let the USER decide where failures are. The
final failure windows are a human judgment the user records with reasoning —
this is the label-construction step CLAUDE.md says the user owns.

## Scope

IN scope: one notebook, `notebooks/exploratory/01_failure_identification.ipynb`
(or a `.py` percent-format notebook if that's cleaner to generate — user
preference is a real notebook they can run cell-by-cell). It uses the EXISTING
`apu_sentinel.data.load` loader and `apu_sentinel.config` to get the data.

OUT of scope: any split/scale/window/feature/regime/model logic; anything that
writes to `data/`; anything that auto-labels failures.

## What the notebook should contain (cells, in order)

1. **Setup:** load config (`load_config`), load the full raw DataFrame via the
   existing `load_raw` loader. Print shape, column list, and the full time span
   (min/max timestamp). Do not re-implement loading — reuse the package.

2. **Signal inventory:** briefly describe each column (analog vs digital where
   known — MetroPT-3 has analog sensors like pressures, oil temperature, motor
   current, and digital/status signals). A short markdown cell noting which are
   the physically meaningful continuous signals for spotting anomalies.

3. **Full-timeline overview plots:** plot each key ANALOG channel across the
   ENTIRE time span (downsample for plotting if needed for performance — e.g.
   resample to a coarse interval FOR THE PLOT ONLY, never mutating the source
   df). The goal: the user can see, at a glance, where behaviour departs from
   the normal pattern over months. Multiple stacked subplots sharing the time
   axis is ideal so anomalies line up across channels.

4. **Operating-regime visibility:** plot the digital/status signals (and/or
   the motor-current/COMP behaviour) over time so the compressor's ON/OFF
   cycling is visible. This is the core challenge flagged in CLAUDE.md (most
   raw variance is mode-switching, not anomaly) — seeing it now informs later
   regime handling. Do NOT segment or model regimes here — just show them.

5. **Zoom-in scaffolding:** provide a reusable helper cell that plots all key
   channels over a USER-SPECIFIED date range, so the user can zoom into
   candidate failure periods and inspect them closely. Leave example date
   ranges as clearly-marked placeholders the user edits.

6. **Distributions / summary stats:** per-channel summary statistics and simple
   distribution plots, to help characterise "normal" — again descriptive only.

7. **Failure-window recording cell (the deliverable):** a clearly-marked
   markdown + code cell where the USER writes down the failure windows they
   identified, as a list of (start, end, notes) — with a prompt reminding them
   to record their REASONING (which signals, what they saw) and to CROSS-CHECK
   against the dataset documentation. This list is the output that feeds the
   split pass. Do NOT fill it in with guessed dates — leave it as a template
   for the user, with a comment that these must be the user's judgment,
   verified against docs, NOT auto-generated.

## Hard constraints

- Read-only: the notebook never writes to `data/`, never mutates the loaded df
  in place in a way that would mislead (downsampling for a plot must be to a
  temporary local variable).
- No anomaly algorithm decides failures. No `IsolationForest`, no thresholding
  that outputs "these are failures". Visualisation and human judgment only.
- No split, scale, window, feature-engineering, or regime-segmentation logic.
- Reuse the existing package (`load`, `config`) — do not duplicate loading.
- Plotting library: matplotlib is fine; keep it dependency-light and add any
  new dep to pyproject via uv if truly needed (prefer not to).

## Definition of done

- The notebook exists in `notebooks/exploratory/`, runs top-to-bottom against
  the real downloaded data without errors (assuming data/raw is populated),
  and produces clear full-timeline and regime plots plus the zoom helper.
- The failure-window recording cell is present as a TEMPLATE for the user (not
  auto-filled).
- No pipeline/model/split/scale logic added anywhere. Existing `make test` and
  `make lint` remain green (the notebook is not part of the test suite, but
  nothing else should have changed).
- Nothing committed — left for user review. (Note: notebooks can carry heavy
  output; the user may want to clear outputs before committing — mention this.)

When done: print (a) how to open/run the notebook, (b) a reminder that the
user identifies the failure windows by eye and records them with reasoning +
doc cross-check, and (c) confirmation that no auto-labelling and no
split/scale/model logic was added. Then STOP.
```