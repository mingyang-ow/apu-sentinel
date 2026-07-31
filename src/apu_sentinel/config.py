"""Typed config schema (pydantic-settings) for apu_sentinel.

Loading model: configs/base.yaml merged with one of configs/local.yaml or
configs/colab.yaml (selected by CONFIG=), optionally overlaid with a
configs/experiment/*.yaml file. Validation is strict -- unknown keys or
wrong types must fail loud at load time, never silently coerce.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/apu_sentinel/config.py -> src/apu_sentinel -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

VALID_CONFIG_NAMES = ("local", "colab")

# data.* fields holding filesystem paths -- resolved to absolute (against
# REPO_ROOT, not CWD) at load time so callers work regardless of where the
# process was launched from (notebooks, Colab, scripts run from subdirs).
_DATA_PATH_FIELDS = ("raw_dir", "interim_dir", "processed_dir")

_STRICT = ConfigDict(extra="forbid")


class DataConfig(BaseModel):
    model_config = _STRICT

    url: str
    raw_dir: Path
    raw_filename: str
    interim_dir: Path
    processed_dir: Path
    # null = establish mode, string = verify mode -- see data/download.py.
    checksum: str | None = None
    subset: float = 1.0


class AdditionalExclusionRegion(BaseModel):
    """One extra training-exclusion region not tied to a documented failure
    event -- pass 21's March sensitivity arm (docs/RESULTS.md §21): the
    early-March cluster (`findings/12-event2-error-analysis.md`) is
    comparably severe to a real precursor but un-anchored to any event, so
    no event-margin mechanism can ever exclude it (pass 20's standing
    limitation). Training-only, same as event-derived exclusions -- never
    affects test periods or detection.
    """

    model_config = _STRICT

    start: str
    end: str
    reason: str | None = None


class TrainingExclusionConfig(BaseModel):
    """Fixed, generous margins purging training data around each failure.

    Deliberately kept separate from evaluation.window_widths (the SWEPT
    pre-failure label width) -- one must never be derived from the other.
    All values in hours.
    """

    model_config = _STRICT

    pre_margin_hours: float
    post_settle_hours: float
    # Used instead of post_settle_hours when an event's maintenance
    # timestamp is unrecorded (null) -- see evaluation.failure_events.
    fallback_post_hours: float
    # Sensitivity-only, empty by default -- see AdditionalExclusionRegion.
    additional_regions: list[AdditionalExclusionRegion] = Field(default_factory=list)


class SplitConfig(BaseModel):
    model_config = _STRICT

    # Gap enforced between a fold's train_end and test_start. Must be >=
    # the longest sequence window any later windowing pass will use.
    embargo_hours: float
    training_exclusion: TrainingExclusionConfig
    # Pass 18 (docs/RESULTS.md §18, docs/findings/09-open-questions.md):
    # guards against the failure mode pass 13 already caught once -- a
    # fold squeezed to near-empty training data fits a threshold of ~0.0
    # and turns its entire test period into one continuous episode.
    # data/split.py make_folds() raises, naming the offending fold and its
    # actual remaining training days (training_days_remaining()), if this
    # falls below the configured minimum. 0.0 = off (the schema default,
    # matching this codebase's convention for opt-in guards -- e.g.
    # min_episode_duration -- so small synthetic test fixtures that never
    # set this explicitly aren't affected); configs/base.yaml sets the
    # REAL, informed value (30 days) for actual runs -- see its comment and
    # docs/RESULTS.md §18 for the real remaining-days numbers behind it.
    min_training_days: float = 0.0


class FailureEvent(BaseModel):
    """One documented MetroPT-3 failure event. Timestamps are kept as
    strings here (parsed to pd.Timestamp downstream in data/split.py),
    matching the existing "timestamps as strings in config" convention.
    """

    model_config = _STRICT

    id: int
    start: str
    end: str
    # null when the source table has no maintenance/repair entry for this
    # event (see event 1's note) -- callers must fall back to
    # training_exclusion.fallback_post_hours in that case.
    maintenance: str | None = None
    note: str | None = None


class MaskedRegionConfig(BaseModel):
    """An explicitly configured region excluded from false-alarm counting,
    in addition to the automatic per-event masked regions -- see
    evaluation/events.py masked_regions().
    """

    model_config = _STRICT

    start: str
    end: str
    note: str | None = None


class EvaluationConfig(BaseModel):
    model_config = _STRICT

    # SWEPT candidate pre-failure window widths (hours) -- a reported
    # result, not a single magic number. The widest value must respect the
    # event-2/event-3 proximity cap enforced by data/split.py.
    window_widths: list[float] = Field(default_factory=list)

    # Threshold is fit on TRAINING scores ONLY (evaluation/metrics.py
    # fit_threshold) -- a high quantile, never a hardcoded score value.
    threshold_quantile: float = 0.995
    # Optional sweep: fit_threshold_sweep() fits ALL of these on train
    # scores and returns the whole curve -- picking the test-optimal one
    # and reporting it as "the" result is forbidden (see fit_threshold_sweep
    # docstring).
    threshold_quantiles: list[float] = Field(default_factory=list)

    # Hysteresis / hold-time: a below-threshold stretch shorter than this
    # does not end an episode. Pandas duration string (e.g. "10min").
    episode_hold_time: str = "10min"
    # A gap in the score timeline (from dropped windows) longer than this
    # ENDS an episode outright, regardless of episode_hold_time -- absence
    # of evidence is not evidence of continuation. Pandas duration string.
    score_gap_threshold: str = "30min"
    # Episodes shorter than this are dropped. "0min" (default) = OFF --
    # enabling this is a knowing choice, not an automatic default.
    min_episode_duration: str = "0min"
    # How per-timestamp channel contributions are aggregated across an
    # episode's timestamps into its ranked diagnosis (explain/).
    contribution_aggregation: Literal["mean", "max"] = "mean"

    # Deferred until baseline behaviour has been observed.
    false_alarm_ceiling: float | None = None
    failure_events: list[FailureEvent] = Field(default_factory=list)
    # Extra masked regions beyond the automatic per-event ones (e.g. a
    # known sensor outage unrelated to a documented failure).
    additional_masked_regions: list[MaskedRegionConfig] = Field(default_factory=list)

    # --- pass 13: null comparison + honest false-alarm estimation --------

    # Number of candidate placement times evaluated by the empirical
    # permutation null (evaluation/metrics.py p_chance_permutation) -- a
    # deterministic, evenly-spaced grid across the test period, not a
    # random sample (see that function's docstring for why).
    permutation_samples: int = 500
    # A detection is flagged "not distinguishable from chance" when EITHER
    # null estimate (Poisson or permutation) exceeds this. 0.10 is a
    # starting point, not a validated significance level -- tune only
    # after seeing baseline chance-comparison behaviour, same spirit as
    # false_alarm_ceiling.
    chance_threshold: float = 0.10
    # Symmetric padding (hours) applied around every excluded region
    # (pre-failure window + failure/settle period) before taking the
    # complement to build pooled_normal_stretches (evaluation/events.py) --
    # keeps a stretch from starting/ending right at the edge of a real
    # precursor or repair period.
    pooled_buffer_hours: float = 24.0


class TrainConfig(BaseModel):
    model_config = _STRICT

    epochs: int
    max_minutes: int


class ScalingConfig(BaseModel):
    """Column selection is explicit and config-driven -- see
    data/scaling.py. A DataFrame column absent from BOTH analog_columns and
    passthrough_columns must raise, never be silently guessed at.
    """

    model_config = _STRICT

    method: Literal["robust", "standard", "minmax"] = "robust"
    analog_columns: list[str]
    passthrough_columns: list[str]
    # A computed scale below this is treated as zero (constant/near-constant
    # channel) -- substituted with 1.0 rather than dividing by ~0. See
    # data/scaling.py fit_scaler().
    zero_scale_epsilon: float = 1e-8
    # Regime-conditional scaling (data/scaling.py fit_regime_scalers /
    # transform_by_regime): a (fold, regime) pair with fewer training
    # samples than this raises rather than fitting a near-meaningless
    # scaler. Empty dict default keeps this field opt-in for callers that
    # don't do regime-conditional scaling at all.
    min_samples_per_regime: int = 100
    # regime name -> analog channels that carry information in that state.
    # Every regime assign_regimes() can produce (LOADED/OFFLOAD/STOPPED/
    # TRANSITION) must have an explicit entry when regime-conditional
    # scaling is used -- a channel NOT listed for a regime is set to a
    # constant 0.0 after transform (docs/FINDINGS.md §7: TP2/DV_pressure
    # vent to ~0 when the compressor stops; their OFF-state spread is
    # sensor noise, not signal -- scaling it would raise ~250x
    # amplification to ~500x for no benefit). Empty by default -- opt-in.
    active_channels: dict[str, list[str]] = Field(default_factory=dict)
    # Warn-only (never substitutes/clamps): after fitting a regime scaler,
    # log a WARNING naming any ACTIVE channel whose within-regime scale
    # implies amplification (1/scale) above this factor -- surfaces future
    # instances of the TP2-style pathology instead of hiding them.
    amplification_warn_factor: float = 100.0


class ResampleConfig(BaseModel):
    """Resampling to a regular grid -- OFF by default. Enabling it is a
    modeling decision the user should make knowingly after reviewing
    data/windows.py characterise_sampling()'s output, not an automatic
    default.
    """

    model_config = _STRICT

    enabled: bool = False
    # Pandas duration string (e.g. "1min"), parsed downstream in
    # data/windows.py -- never hardcoded there.
    interval: str = "1min"


class WindowingConfig(BaseModel):
    """Durations are pandas duration strings (e.g. "30min"), parsed to
    pd.Timedelta in data/windows.py and converted to a sample count using
    the empirically measured expected_interval -- never a hardcoded sample
    count here.
    """

    model_config = _STRICT

    window_duration: str
    train_stride: str
    score_stride: str
    # Fractional slack on a window's expected wall-clock span before it is
    # dropped for spanning a gap.
    gap_tolerance: float = 0.1
    # What counts as a "gap" when characterise_sampling() reports on the
    # raw sampling (a separate concern from gap_tolerance, which governs
    # per-window drop decisions).
    gap_threshold: str
    resample: ResampleConfig = Field(default_factory=ResampleConfig)


class RegimesConfig(BaseModel):
    """Operating-regime segmentation, derived from the compressor's own
    digital control signals -- NOT by clustering the analog channels (see
    regimes/__init__.py). No hardcoded flag names, polarities, or durations
    in code; all of it lives here.
    """

    model_config = _STRICT

    # Cross-referenced against each other in regimes.verify_flag_semantics
    # (their pairwise agreement rate, after normalising via `polarity`, is
    # logged as evidence for which one to trust as the deciding flag).
    control_columns: list[str]
    # Empirically-derived (regimes.verify_flag_semantics): raw flag value ->
    # normalised OFF/ON reading. Filled from evidence against Motor_current,
    # NEVER assumed -- a naive reading can be exactly backwards (see the
    # pass-9 write-up on COMP).
    polarity: dict[str, dict[int, str]]
    # Explicit condition -> state name: {state_name: {flag_name: required
    # raw value}}. Checked in this dict's order; the first state whose
    # listed flag(s) all match a row wins. Rows matching none of these
    # (i.e. "no air intake") are further split into OFFLOAD/STOPPED below --
    # see regimes/__init__.py.
    states: dict[str, dict[str, int]]
    # Channel + threshold used to split whatever doesn't match `states`
    # into OFFLOAD (>= threshold) vs STOPPED (< threshold). 2.0A is the
    # empirically-justified default -- see docs/FINDINGS.md §7: only
    # 0.015% of OFF samples fall in the 1-3A valley between the two modes,
    # so 2.0A sits in genuine emptiness, not an arbitrary split point.
    offload_split_channel: str = "Motor_current"
    offload_current_threshold: float = 2.0
    # A newly-changed raw state is committed only once it has persisted (a
    # backward-looking check) for at least this long -- shorter blips are
    # absorbed into the preceding committed state, never emitted as their
    # own regime. Pandas duration string.
    min_duration: str
    # Samples within this long of a committed state change are labelled
    # TRANSITION rather than the new state, since analog channels are
    # still settling. Pandas duration string.
    transition_settle: str
    # Records the INTENT to optionally drop offload_split_channel
    # (Motor_current) from scored channels while in an OFFLOAD/STOPPED
    # state, to remove the circularity of using it both to DEFINE those
    # states and to SCORE them for anomalies (docs/FINDINGS.md §9). Default
    # false so enabling it is a knowing choice. NOT wired into scoring --
    # scoring does not exist yet.
    exclude_motor_current_when_off: bool = False


class FeaturesConfig(BaseModel):
    """Causal cycle-timing features (features/cycles.py) -- duration and
    pressure-decay families, both built so a later baseline pass can
    choose between them on evidence (docs/FINDINGS.md §8).
    """

    model_config = _STRICT

    # Pressure channel the decay-rate family reads. Reservoirs is the
    # default; TP3 is a near-duplicate (docs/FINDINGS.md §5) and is an
    # equally valid alternative, never hardcoded.
    decay_source_channel: str = "Reservoirs"
    # decay_rate_running is NaN until this many samples have accumulated
    # within the current run -- a slope from 1-2 points is noise.
    decay_min_samples: int = 3
    # What counts as a data gap for run-boundary purposes here -- matches
    # the convention established in analysis.monthly_gap_and_stopped_summary
    # (docs/FINDINGS.md §9): a run split by a gap this long or longer gets
    # an invalid (NaN) duration but keeps a valid decay rate over its
    # observed samples. Pandas duration string.
    gap_threshold: str = "1min"
    # Trailing window for baseline-relative (value / trailing_median)
    # variants. Pandas duration string. 7 days is a deliberate choice, not
    # a default to take for granted -- see features/cycles.py docstring:
    # short enough that event 2's ~3x step change survives it, long enough
    # that gradual seasonal drift (docs/FINDINGS.md §8) isn't itself
    # absorbed into "normal".
    baseline_window: str = "7D"
    # Trailing window for duty_ratio_trailing (LOADED fraction of time) --
    # deliberately much shorter than baseline_window: this is a
    # current-state indicator, not a drift baseline.
    duty_ratio_window: str = "1h"


class RuleConfig(BaseModel):
    """One rule's enable flag, plus an optional channel override.

    Every rule shares this same {enabled, source_channel} shape even though
    only fast_pressure_decay currently reads source_channel -- a uniform
    per-rule shape is simpler than a one-off schema per rule.
    """

    model_config = _STRICT

    enabled: bool = True
    # Only read by fast_pressure_decay; ignored by every other rule. None
    # means "use that rule's own hardcoded default channel".
    source_channel: str | None = None


class RuleBasedModelConfig(BaseModel):
    """models/rule_based.py: calibration window + rule selection.

    Rule-based "fitting" calibrates comparability (each enabled rule's
    training-distribution quantiles), it does not learn parameters -- see
    models/rule_based.py module docstring. baseline_window is the trailing
    window each ratio-based rule (short_stopped_duration,
    fast_pressure_decay, low_peak_pressure -- NOT high_duty_ratio, which
    reads cycle_features.duty_ratio_trailing directly and has never gone
    through a baseline ratio) divides its raw quantity by, deliberately a
    model-level setting rather than reusing features.baseline_window, so
    the model's own drift-robustness tradeoff (docs/FINDINGS.md §8) can be
    tuned independently of feature engineering.
    """

    model_config = _STRICT

    baseline_window: str = "7D"
    # trailing (default): reference over [t-window, t] -- baseline_relative().
    # lagged: reference over [t-lag-window, t-lag] -- baseline_relative_lagged().
    # Both modes are retained as valid, config-selectable comparison arms
    # (pass 17) -- trailing is NOT superseded, only shown to have a
    # structural blind spot for degradation sustained longer than
    # baseline_window (docs/findings/12-event2-error-analysis.md).
    baseline_mode: Literal["trailing", "lagged"] = "trailing"
    # lagged mode only: gap between "now" and the end of the reference
    # window, so a sustained degradation lasting less than this cannot
    # contaminate its own baseline. 14 days verified against the real event
    # 2 precursor (features/cycles.py baseline_relative_lagged docstring):
    # with baseline_window's own default (7D) unchanged, the reference at
    # event 2's window-open (2020-05-26 23:30) spans 2020-04-28 -> 2020-05-05
    # -- entirely before the 17 May collapse -- giving ratio 0.274 (strongly
    # abnormal) versus trailing mode's 1.195 (reads as *better* than normal).
    # Ignored when baseline_mode is "trailing".
    baseline_lag: str = "14D"
    # Keyed by rule name (short_stopped_duration, fast_pressure_decay,
    # low_peak_pressure, high_duty_ratio). A rule absent from this dict is
    # treated as disabled, same as an explicit {enabled: false}.
    rules: dict[str, RuleConfig] = Field(default_factory=dict)


class IsolationForestContributionsConfig(BaseModel):
    """Ablation attribution (models/isolation_forest.py): for each feature,
    re-score with that feature replaced by its training median, and take
    the score drop as its contribution -- interpretable and model-agnostic,
    the only per-feature attribution sklearn's IsolationForest doesn't give
    natively. `enabled: false` returns zeros (and logs) instead of paying
    the O(n_features) re-scoring cost -- for speed during sweeps.
    """

    model_config = _STRICT

    method: Literal["ablation"] = "ablation"
    # Pass 21: defaults OFF -- per-timestamp ablation over a full fold's
    # sweep (every width x quantile) is what exhausted the CPU. Turn on
    # only for a final confirmed run; use IsolationForestModel.explain_episode
    # (models/isolation_forest.py) for the handful of flagged detections
    # instead of paying the full-fold cost.
    enabled: bool = False


class IsolationForestModelConfig(BaseModel):
    """models/isolation_forest.py: sklearn.ensemble.IsolationForest wrapped
    in the AnomalyModel contract, one instance per fold.

    `contamination` is deliberately NOT exposed here: it only shifts
    sklearn's own `predict()` decision offset, never `score_samples()` (what
    this model actually calls, negated -- see models/isolation_forest.py),
    and the harness fits its own threshold from training scores
    (evaluation/metrics.py fit_threshold). Exposing it would invite
    "tuning" a knob that provably does nothing here.
    """

    model_config = _STRICT

    n_estimators: int = 200
    # sklearn accepts "auto", an int, or a float -- passed through as-is.
    max_samples: str | int | float = "auto"
    random_state: int = 42
    # Passed straight to sklearn's IsolationForest -- -1 uses all cores.
    n_jobs: int = -1
    # Per-channel window summary stats -- config-listed so the feature set
    # is a documented, swept choice, not hardcoded. Channel order matches
    # data/windows.py make_windows()'s own (analog_columns + passthrough_columns).
    window_stats: list[str] = Field(default_factory=lambda: ["mean", "std", "min", "max", "slope"])
    # Whether to append features/cycles.py compute_cycle_features()'s
    # columns, sampled at each window's end timestamp (the existing
    # score/label convention) -- same causal features the rule-based model
    # reads, now available to a windowed model too.
    include_cycle_features: bool = True
    contributions: IsolationForestContributionsConfig = Field(
        default_factory=IsolationForestContributionsConfig
    )
    # Pass 22 diagnostic: drop scored (never training) windows whose end
    # timestamp falls within one window_duration of a data gap boundary --
    # off by default so this stays a deliberate probe, never a silent
    # change to a real run's results (docs/RESULTS.md pass 22).
    exclude_gap_adjacent_windows: bool = False


class AutoencoderModelConfig(BaseModel):
    """models/autoencoder.py: 1D convolutional autoencoder, one instance per
    fold.

    Third and final stage of the model progression (CLAUDE.md). Reads
    WindowedInput with cycle_features=None (docs/RESULTS.md §22's NaN
    imputation path is deliberately not exercised by this model) -- input
    is the scaled channels only, so there is no include_cycle_features
    toggle here unlike IsolationForestModelConfig.

    Conv, not LSTM (pass 23 v2): convolutions parallelise across the time
    axis, so training actually uses all CPU cores instead of stepping
    through 180 timesteps sequentially per window -- the LSTM version hung
    on CPU for exactly this reason. The cross-channel-relationship
    hypothesis this model tests does not need sequential modelling anyway.
    """

    model_config = _STRICT

    # Encoder layer widths, e.g. [32, 16] -- each is a stride-2 Conv1d
    # narrowing both channel width and time length; the decoder mirrors
    # this list in reverse with ConvTranspose1d. Local smoke config uses a
    # much smaller list (see configs/local.yaml).
    channels: list[int] = Field(default_factory=lambda: [32, 16])
    # Must be odd -- padding=kernel_size//2 relies on symmetric "same"
    # padding at stride 1; see models/autoencoder.py _ConvAutoencoderNet.
    kernel_size: int = 5
    bottleneck_dim: int = 8
    dropout: float = 0.0
    activation: Literal["relu", "gelu", "leaky_relu"] = "relu"
    learning_rate: float = 1e-3
    batch_size: int = 64
    random_state: int = 42
    # Trailing fraction of a fold's (time-ordered) training windows held out
    # for early-stopping validation -- a TRAILING block, never a random
    # sample, so validation never sees windows chronologically interleaved
    # with training ones. See models/autoencoder.py fit()'s docstring.
    val_fraction: float = 0.15
    # Early-stopping patience, in epochs of no validation-loss improvement.
    patience: int = 5
    # Full bit-for-bit GPU determinism needs torch.use_deterministic_algorithms
    # plus CUBLAS_WORKSPACE_CONFIG, both of which cost speed -- OFF by default.
    # Only matters on CUDA; CPU training is already deterministic without it
    # for this architecture (no shuffling, no nondeterministic CUDA kernels).
    deterministic: bool = False


class ModelConfig(BaseModel):
    """Container for the currently-selected model's own config subtree.

    Exactly one of these is populated per the model progression in
    CLAUDE.md (rule-based -> isolation forest -> autoencoder); later passes
    add their own optional field here as each model is implemented, never
    requiring all of them at once.
    """

    model_config = _STRICT

    rule_based: RuleBasedModelConfig | None = None
    isolation_forest: IsolationForestModelConfig | None = None
    autoencoder: AutoencoderModelConfig | None = None


class Settings(BaseSettings):
    """Top-level, merged config for a single run."""

    model_config = SettingsConfigDict(extra="forbid")

    device: str
    data: DataConfig
    split: SplitConfig
    evaluation: EvaluationConfig
    scaling: ScalingConfig
    windowing: WindowingConfig
    regimes: RegimesConfig
    features: FeaturesConfig
    train: TrainConfig
    model: ModelConfig = Field(default_factory=ModelConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_data_paths(data: dict[str, Any]) -> None:
    """Resolve data.* path fields to absolute, in place, against REPO_ROOT.

    Relative paths in configs/*.yaml (e.g. "data/raw") are meant relative to
    the repo root, not whatever directory the process happens to be running
    from. An already-absolute value (e.g. an experiment override) is left
    untouched.
    """
    for field in _DATA_PATH_FIELDS:
        value = data.get(field)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            data[field] = str((REPO_ROOT / path).resolve())


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay onto base; overlay wins on conflicts, but
    keys base has that overlay doesn't are kept (never a shallow replace of
    nested dicts).
    """
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_name: str | None = None,
    experiment: str | None = None,
    config_dir: Path | None = None,
) -> Settings:
    """Load configs/base.yaml, merge configs/{config_name}.yaml on top, then
    optionally overlay configs/experiment/{experiment}.yaml. Later layers
    override earlier ones; nested dicts are merged key-by-key (deep merge),
    not replaced wholesale.

    config_name resolution: explicit argument > CONFIG env var > "local"
    (the safe CPU/subset default). Raises ValueError for an unrecognized
    name.

    data.* path fields (raw_dir, interim_dir, processed_dir) are resolved to
    absolute paths against the repo root before validation, so callers work
    regardless of the process's current working directory.

    The merged dict is validated through the Settings schema -- a wrong
    type or unknown key fails here, loudly, rather than later in a training
    loop.

    config_dir defaults to the repo's configs/ directory (resolved from
    this file's location, not the working directory); overridable for tests.
    """
    if config_name is None:
        config_name = os.environ.get("CONFIG", "local")
    if config_name not in VALID_CONFIG_NAMES:
        raise ValueError(
            f"Unknown CONFIG={config_name!r} -- valid options: {', '.join(VALID_CONFIG_NAMES)}"
        )

    if config_dir is None:
        config_dir = CONFIG_DIR

    merged = _load_yaml(config_dir / "base.yaml")
    merged = _deep_merge(merged, _load_yaml(config_dir / f"{config_name}.yaml"))

    if experiment is not None:
        merged = _deep_merge(merged, _load_yaml(config_dir / "experiment" / f"{experiment}.yaml"))

    if "data" in merged:
        _resolve_data_paths(merged["data"])

    return Settings(**merged)
