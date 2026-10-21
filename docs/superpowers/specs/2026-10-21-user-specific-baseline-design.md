# User-Specific Baseline Removal (`per_user` mode)

**Date:** 2026-10-21
**Status:** Proposed

## Context

`BaselineRemoval` currently subtracts a single **global** per-channel mean rest
level (estimated from all of the train partition's `noGesture` trials) from
every trial. This averages away per-subject differences in resting EMG offset.

The user wants an optional **user-specific** baseline: each subject's baseline
is estimated from **that subject's own** rest trials, and each trial is
corrected by its own subject's baseline. This is only valid for **trial-level
splits** (`random`, `stratified`, `trial_wise`), where a subject appears in both
train and test — under a subject-level split (`subject_wise`, `session_wise`,
`predefined`) the test subjects are unseen in train, so their baseline cannot be
estimated.

## Goals

- Add `baseline.mode: global | per_user` (default `global`, preserving current
  behavior).
- `per_user`: `fit(train)` builds a `{subject_id → per-channel-mean}` map from
  each subject's rest trials; `transform` subtracts each trial's own subject's
  baseline.
- **Fail loudly** when a subject has no rest trials in train (fit) or no
  baseline at all (transform) — this "shouldn't happen" in trial-level splits.
- Validate at config load: `per_user` requires a trial-level split.
- Keep everything config-driven and deterministic.

## Non-goals

- No change to `global` mode behavior.
- No automatic fallback to global for unseen subjects (fail loudly instead).
- No per-session or per-channel-more-granular baselines.

## Design

### 1. Config

`src/hand_gesture_recognition/configuration/schema.py`:

```python
@dataclass
class BaselineConfig:
    enabled: bool = False
    mode: str = "global"  # "global" | "per_user"
```

`configs/baseline/default.yaml` adds `mode: global`; a new
`configs/baseline/per_user.yaml` preset sets `enabled: true` + `mode: per_user`.

### 2. `BaselineRemoval(mode)`

`src/hand_gesture_recognition/preprocessing/baseline.py` gains a `mode` param:

- `__init__(self, mode: str = "global")` — stores `mode`; `baseline_` is `None`
  until fit.
- `fit(train)`:
  - `global`: unchanged (global mean of train rest trials; raise if none).
  - `per_user`: for each distinct `subject_id`, compute the per-channel mean of
    that subject's rest trials; **raise** if a subject has no rest trials. Store
    `{subject_id → (n_channels,)}`.
- `transform(dataset)`:
  - `global`: unchanged.
  - `per_user`: subtract each trial's own subject's baseline; **raise** if a
    subject has no entry in `baseline_`.

### 3. Runner

Both baseline call sites pass `mode=self.cfg.baseline.mode`:
- `_run_partition` (training baseline, fit on `split.train`).
- `run()` (visualization-only baseline, fit on the full cropped dataset).

### 4. Validation

In `validate_experiment_config`:

```python
if cfg.baseline.mode not in ("global", "per_user"):
    raise ConfigValidationError(...)
if (
    cfg.baseline.enabled
    and cfg.baseline.mode == "per_user"
    and cfg.split.strategy in ("subject_wise", "session_wise", "predefined")
):
    raise ConfigValidationError(
        "baseline.mode=per_user requires a trial-level split "
        "(random/stratified/trial_wise)"
    )
```

## Edge cases

- **Subject with no rest trials in train** → `fit` raises (per-user).
- **Val/test subject absent from train** → `transform` raises (per-user).
- **`per_user` + subject-level split** → config validation error.
- **Synthetic data** (no rest trials) → baseline removal raises for both modes,
  unchanged (baseline already requires rest trials).
- **`include_no_gesture=false`** → baseline is fit *before* class selection in
  `_run_partition`, so rest trials are still available; unchanged.

## Testing

- **Unit `BaselineRemoval`** (new `tests/unit/test_baseline.py`):
  - global mode unchanged (same as today).
  - per-user: builds one baseline per subject; subtracts each subject's own
    baseline; raises on missing rest trials (fit) and on unseen subject
    (transform).
- **Unit config**: `per_user` + subject-level split raises; unknown `mode`
  raises; default `mode` is `global`.
- **Integration**: a synthetic-free trial-level run with `baseline=per_user`
  is not possible (synthetic has no rest); rely on unit tests + a config-load
  test for the validation path.

## Out of scope

- Auto-fallback to global baseline.
- Per-session baselines.
- Baseline estimation from non-rest trials.
