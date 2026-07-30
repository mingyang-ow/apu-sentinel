# Build briefs — the project's decision record

These are the original pass-by-pass briefs given to build this project, in
order. Each one scoped a single pass tightly (with an explicit "do NOT
implement X yet" boundary) so later passes could build on stable, reviewed
foundations. Kept verbatim as the record of what was asked for and why, at
each stage — not maintained or updated after the fact.

Later passes (5 onward: walk-forward split, gap-aware windowing,
episode-level evaluation, regime segmentation, regime-conditional scaling +
cycle-timing features, the rule-based baseline, null-comparison/honest
false-alarm estimation, this documentation pass) were not saved as separate
brief files — their outcomes are recorded in `docs/FINDINGS.md` /
`docs/RESULTS.md` and in git history instead.

1. **[`scaffolding-brief.md`](scaffolding-brief.md)** — Pass 1: repository
   structure and stubs only (directories, `CLAUDE.md`, config skeletons,
   the model contract interface, Makefile, test stubs) — no modelling
   logic, no algorithms.
2. **[`data-brief.md`](data-brief.md)** — Pass 2: MetroPT-3 download with
   two-mode (establish/verify) SHA256 checksum handling, and a minimal
   loader (parse, sort, sanity-check, nothing more).
3. **[`config-pass-brief.md`](config-pass-brief.md)** — Pass 3: implement
   `load_config` itself — layered YAML loading (base → env → optional
   experiment overlay), deep merge, `CONFIG` selection, pydantic
   validation at load time.
4. **[`eda-pass-brief.md`](eda-pass-brief.md)** — Pass 4: an exploratory
   notebook for the USER to identify MetroPT-3's failure windows by eye
   (never an algorithmic labeller — that would be circular), recording
   the resulting dates + reasoning that later split/evaluation passes
   depend on.
