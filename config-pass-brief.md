# apu-sentinel — Build Pass 3: Implement Config Loading (`load_config`)

**Read this whole brief first. This pass makes `src/apu_sentinel/config.py`'s
`load_config` actually work: layered YAML loading, CONFIG selection, deep
merge, and pydantic validation. It is small and foundational — every future
pass (download script, train, evaluate) depends on it. Respect CLAUDE.md.
Do NOT implement any split/scale/window/model/evaluation logic.**

Repo: `/home/bren/Projects/apu-sentinel` (branch main; download+loader pass
already committed).

## Why this pass exists

The scaffold left `config.py` as a schema stub (pydantic classes, no loading
logic). `scripts/download.py` is wired to call config loading but currently
hits `NotImplementedError`. This pass implements the loader so the script — and
all future `make` commands — work as designed.

## Scope

IN scope:
1. Implement `load_config` in `src/apu_sentinel/config.py`.
2. Ensure the pydantic schema classes are complete enough to validate the
   current `base/local/colab.yaml` (add missing fields ONLY as needed to load
   what already exists in those files — do not invent new config surface).
3. Tests for the loader (merge order, deep merge, CONFIG selection, validation
   failure).
4. Confirm `scripts/download.py` now runs through the loader.

OUT of scope: any data/model/eval logic; new config fields beyond what the
existing YAMLs and the download pass already require.

## Behaviour `load_config` must implement

1. **Layered load, in this precedence (later overrides earlier):**
   `base.yaml`  →  the environment file (`local.yaml` or `colab.yaml`)  →
   optional experiment overlay (`configs/experiment/<name>.yaml`) if one is
   requested. Base is the foundation; env overrides base; experiment overrides
   env.

2. **Environment selection via `CONFIG`:** read which environment to load from
   a `CONFIG` value (env var and/or function argument). Default to `local`
   when unset (safe default — CPU/subset, per CLAUDE.md). An unknown CONFIG
   value must raise a clear error naming the valid options.

3. **DEEP merge, not shallow.** Nested dicts merge key-by-key. Example: if
   `base.yaml` has
   ```yaml
   data: {url: ..., raw_dir: ..., checksum: null}
   ```
   and `local.yaml` has
   ```yaml
   data: {subset: 0.05}
   ```
   the result must contain url, raw_dir, checksum AND subset — NOT just subset.
   A shallow merge that drops base's `data` keys is a BUG and must be
   prevented. Cover this specific case with a test.

4. **Validate through the pydantic schema at load time.** After merging, the
   dict is parsed into the typed config model. A wrong type or unknown key
   fails HERE, loudly, with a message identifying the bad field — do not
   silently coerce or ignore. This is the entire reason pydantic-settings was
   chosen; the failure must be at load time, not deep in a training loop later.

5. **Return the typed config object** (not a raw dict), so callers get
   attribute access and IDE/autocomplete benefits.

Implementation notes:
- Paths to configs are relative to the repo/configs dir — resolve robustly
  (do not assume the current working directory). No hardcoded absolute paths.
- Keep it dependency-light — PyYAML + pydantic(-settings) already in the stack.
- If the experiment overlay mechanism adds complexity, a minimal version
  (support it but it's optional and defaults to none) is fine for this pass.

## Tests to implement (real)

`tests/test_config.py`:
- **Deep-merge test (the important one):** base defines a nested block with
  several keys, env file overrides ONE key in that block; assert the merged
  result retains ALL base keys plus the override. This guards the most common
  loader bug.
- **Precedence test:** a key present in base, env, and an experiment overlay
  resolves to the experiment value; a key in base+env resolves to env; a
  base-only key survives.
- **CONFIG selection test:** CONFIG=local vs CONFIG=colab load the respective
  files; unset defaults to local; an invalid CONFIG raises a clear error.
- **Validation-failure test:** a config with a wrong-typed field (e.g. a
  string where a number is required) raises a pydantic validation error at
  load time.
- Use small temp YAML files / fixtures where practical rather than depending
  on the real configs, so tests are stable if real config values change.

Keep all existing tests green. Leakage-guard tests remain skipped (this pass
does not implement split or scaling).

## Verification

After implementing, confirm `scripts/download.py` runs THROUGH the loader
without `NotImplementedError` (it may still stop for lack of a real network in
some environments, but it must get PAST config loading). Note in the summary
whether the download script now reaches the download step.

## Constraints reminder (CLAUDE.md)

- No hardcoded paths/values — config is the single source.
- Default environment is `local` (safe: CPU/subset).
- Fail loud and early on bad config — never silently coerce.

## Definition of done

- `load_config` implements layered load + CONFIG selection + DEEP merge +
  pydantic validation, returns a typed object.
- Schema complete enough to validate existing base/local/colab.yaml.
- `tests/test_config.py` passes, including the deep-merge and
  validation-failure tests; `make test` and `make lint` green.
- `scripts/download.py` gets past config loading.
- Nothing committed — left for user review.

When done: print (a) what `load_config` now does in 3-4 lines, (b) confirmation
the deep-merge test exists and passes, (c) whether `scripts/download.py` now
reaches the download step, and (d) the command to establish the checksum
(`uv run python scripts/download.py` → copy the printed hash → paste into
configs/base.yaml). Then STOP.
