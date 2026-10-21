# Task Toggle — Classification vs Recognition

**Date:** 2026-10-21
**Status:** Proposed

## Context

The framework currently supports exactly one problem formulation: **recognition**
of gestures on a continuous signal stream. `window_dataset` slides a fixed-size
window with a stride over each continuous trial, assigning a per-window label
(`0` = rest, `1..N` = gesture) from the gesture ground-truth bounds. A separate
`gesture_crop` transform can crop trials to the gesture region + context margin,
but windowing still slides over the (cropped) trial afterward — so every run is
effectively recognition with an optional pre-crop.

The user needs a **second problem formulation — classification** — where the
dataset consists of pre-cut clips and each sample contains exactly one gesture:

| Formulation | Input | Sample building |
|---|---|---|
| **classification** | pre-cut clip, one gesture per trial | one sample per gesture (`window_gestures`), rest dropped |
| **recognition** | continuous timeline | sliding window (`window_dataset`), rest is a class |

Both formulations must emit the **same `WindowedDataset` shape**
`(n_windows, window_size, channels)`, so the classical/neural backends, the
`Prediction` adapters, evaluation, tracking, and reporting are reused unchanged.
The toggle only changes *how samples are built*.

Two-level (big-window → small-window) windowing is a **separate follow-up**
(Spec B) and is explicitly out of scope here.

## Goals

- Add a `task` configuration switch with values `classification | recognition`
  (default `recognition`, preserving today's behavior).
- `recognition`: unchanged sliding-window behavior.
- `classification`: build **one sample per gesture trial** by slicing
  `[gesture_start, gesture_end]`, pad/truncating to `window_size`, labeling it
  with the gesture class, and dropping rest trials.
- Keep everything config-driven, deterministic, backend-agnostic, and
  **dataset-agnostic** (a pre-cut-clip dataset is simply
  `gesture_start=0, gesture_end=length`, the same shape the loader already
  produces for synthetic data).
- Keep the single shared windowing implementation philosophy: the toggle is a
  first-class, swappable sample-building concept.

## Non-goals

- Two-level (hierarchical) windowing — Spec B.
- Online / two-stage activity detection.
- Changes to split logic, leakage checks, `evaluate()`, the `Prediction`
  schema, or the backend adapters.
- Arbitrary "task" plugins beyond the two defined formulations.

## Design

### 1. Config — `TaskConfig` + `configs/task/`

`src/hand_gesture_recognition/configuration/schema.py`:

```python
SUPPORTED_TASKS = ("classification", "recognition")

@dataclass
class TaskConfig:
    name: str = "recognition"   # "classification" | "recognition"
```

- Add `task: TaskConfig = field(default_factory=TaskConfig)` to
  `ExperimentConfig`.
- Add validation in `validate_experiment_config`:

```python
if cfg.task.name not in SUPPORTED_TASKS:
    raise ConfigValidationError(
        f"task.name={cfg.task.name!r} not in {SUPPORTED_TASKS}"
    )
if cfg.task.name == "classification" and cfg.dataset.include_no_gesture:
    warnings.warn(
        "task=classification always excludes the no-gesture class; "
        "dataset.include_no_gesture is ignored in classification mode."
    )
```

New Hydra group `configs/task/`:

`configs/task/recognition.yaml`:

```yaml
# @package task
name: recognition
```

`configs/task/classification.yaml`:

```yaml
# @package task
name: classification
```

Register in `configs/config.yaml`:

```yaml
defaults:
  ...
  - task: recognition
  ...
```

Wire into `configuration/loader.py::_coerce`:

```python
task=TaskConfig(**d.get("task", {})),
```

Replay of a previously-resolved config without a `task` key defaults to
`recognition` (backward compatible).

### 2. Sample builder — `window_gestures`

New module `src/hand_gesture_recognition/data/windowing/gesture.py`:

```python
def window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    ...
```

Logic, mirroring `window_dataset`'s structure and output shape:

1. `ws, _ = config.resolve(sampling_rate=dataset.metadata.sampling_rate)`.
2. Compute `n_channels` the same way `window_dataset` does.
3. For each trial with `gesture_start >= 0` (a gesture trial):
   - `seg = trial[gesture_start:gesture_end]` — the whole gesture, no margin.
   - If `len(seg) < ws`: zero-pad at the **end** to `ws`.
   - If `len(seg) > ws`: truncate to the **first `ws`** samples.
   - Append `y=int(label)`, `window_start=gesture_start`,
     `window_end=gesture_end` (original trial coordinates), `subject_id`,
     `trial_id`, and `window_id=f"{subj}_{tid}_gesture"`.
4. **Skip rest trials** (`gesture_start < 0`) — rest is dropped.
5. Return `WindowedDataset` (empty `X` with shape `(0, ws, n_channels)` if no
   gesture trials, mirroring `window_dataset`'s guard).

Labels are already in the final class space when called (the runner applies
class-selection before windowing), exactly as `window_dataset` assumes.

### 3. Dispatcher — `build_windowed_dataset`

New module `src/hand_gesture_recognition/data/windowing/dispatch.py` (avoids a
`base` ↔ `gesture` import cycle):

```python
def build_windowed_dataset(
    dataset: Dataset, config: WindowingConfig, task: str
) -> WindowedDataset:
    if task == "classification":
        return window_gestures(dataset, config)
    return window_dataset(dataset, config)
```

This is the single place that maps `task` → sample builder, so the runner and
the prediction-overlay visualizations stay consistent.

### 4. Runner wiring

`src/hand_gesture_recognition/runner/run.py`:

- `_window(split)` builds each partition via
  `build_windowed_dataset(split.<part>, self.cfg.windowing, self.cfg.task.name)`.
- `_gesture_crop` is **skipped** when `cfg.task.name == "classification"`
  (classification extracts the gesture exactly; the margin crop is a
  recognition pre-step):

```python
with stage(self.logger, "gesture_crop"):
    if cfg.task.name == "recognition":
        cropped = self._gesture_crop(raw_dataset)
    else:
        cropped = raw_dataset
```

### 5. Rest / no-gesture semantics

Classification always excludes rest, regardless of `dataset.include_no_gesture`:

- Add a helper `_should_drop_no_gesture() -> bool` returning
  `(not self.cfg.dataset.include_no_gesture) or (self.cfg.task.name == "classification")`.
- In `run()`: gate `build_no_gesture_remap` on that helper.
- In `_run_partition()`: gate `drop_no_gesture` on that helper.

`window_gestures` also skips rest trials as a defensive backstop, so rest can
never leak into classification samples even if called directly on an
untransformed dataset. `include_no_gesture` is therefore ignored in
classification mode (warning emitted per §1).

### 6. Evaluation & downstream

**No changes.** Both modes emit an identical `WindowedDataset`, so `Prediction`
(sample IDs = window IDs), `evaluate`, the confusion matrix, tracking, and
reporting all work as-is. In classification mode `sample_ids` are
`{subj}_{tid}_gesture` and `window_start`/`window_end` hold the true gesture
bounds.

### 7. Visualization — task-aware re-windowing

`reporting/visualization.py`'s prediction-overlay functions re-window with
`window_dataset` to recover window positions and align `prediction.y_pred`:

- `plot_predictions_on_signal`
- `plot_signal_data_flow`

Both must switch to `build_windowed_dataset(dataset, windowing, task)` so their
re-windowed grid matches the pipeline's. Add a `task: str = "recognition"`
parameter to each; `runner._visualize` passes `self.cfg.task.name`.

`plot_feature_over_windows` and `plot_fft_spectrum` do **not** align
predictions (they only compute features/FFT from a sliding-window view); they
stay on `window_dataset` in Spec A and are noted for possible task-awareness in
Spec B. They do not crash in classification mode.

### 8. Dataset-agnosticism

Nothing in this change references EPN612. The `Dataset`/loader abstraction
already supports both formulations: pre-cut-clip datasets set
`gesture_start=0, gesture_end=length` per trial (as `make_synthetic_dataset`
does), while continuous datasets carry real bounds. `window_gestures` and
`window_dataset` both key off `gesture_start`/`gesture_end`, so new datasets
require only a loader, not code changes.

## Data flow — classification mode

```
loader (continuous or pre-cut) ──> Dataset (trials + gesture bounds)
  ├─ analytics (unchanged)
  ├─ gesture_crop            SKIPPED
  ├─ build_no_gesture_remap  (forced) ──> remap labels 0..N-1, class_names w/o "noGesture"
  ├─ split (unchanged)
  ├─ _run_partition:
  │    ├─ baseline removal (unchanged)
  │    ├─ drop_no_gesture    (forced) ──> rest trials removed
  │    ├─ signal processing  (per trial, unchanged)
  │    └─ _window ──> window_gestures ──> one padded/truncated window per gesture
  └─ train → predict → evaluate (unchanged; N-class, no rest)
```

## Interaction with `gesture_crop`

`gesture_crop` is skipped in classification mode. The two are conceptually
opposed: `gesture_crop` keeps a context *margin* around the gesture for sliding
recognition; classification wants exactly `[gesture_start, gesture_end]`. No
validation error is raised if a user enables `gesture_crop` with
`task=classification` — it is simply not applied (the runner guards the stage,
mirroring the existing `enabled` guard).

## Edge cases

- **Gesture longer than `ws`**: truncated to the first `ws` samples (documented;
  the end of the gesture is dropped). Rare given the typical window length.
- **Gesture shorter than `ws`**: zero-padded at the end.
- **Rest trial** (`gesture_start < 0`): skipped entirely.
- **Empty partition** (a subject-wise partition with no gesture trials):
  `window_gestures` returns an empty `WindowedDataset`; the existing 0-window
  warning in `_window` fires; downstream behaves as it does today for empty
  recognition partitions.
- **Synthetic dataset**: `gesture_start=0`, `gesture_end=length` → each trial
  yields one sample equal to the whole (padded/truncated) trial; no rest class
  exists, so the remap is effectively an identity.
- **Replay of a pre-`task` config**: `task` defaults to `recognition`.

## Testing

- **Unit `window_gestures`** (new `tests/unit/test_gesture_windowing.py`):
  - gesture shorter than `ws` → padded to `ws`, correct label;
  - gesture longer than `ws` → truncated to `ws`;
  - exact-length gesture → unchanged;
  - rest trials dropped; one sample per gesture trial;
  - `window_start`/`window_end`/`subject_ids`/`trial_ids`/`window_ids` correct;
  - empty input → empty `WindowedDataset` with correct shape.
- **Unit dispatcher**: `build_windowed_dataset(..., "classification")` equals
  `window_gestures(...)`; `"recognition"` equals `window_dataset(...)`.
- **Unit config**: invalid `task.name` raises `ConfigValidationError`;
  `task=classification` + `include_no_gesture=true` emits a warning; replay
  without `task` defaults to `recognition`.
- **Unit visualization**: `plot_predictions_on_signal` with `task="classification"`
  does not mis-index `y_pred` (uses `window_gestures` internally).
- **Integration**: synthetic `task=classification` end-to-end for **both**
  backends (classical + neural) completes and reports N-class (no rest) metrics;
  a synthetic `task=recognition` run is byte-identical in behavior to today.
- **Existing tests stay green** (default `task=recognition` path unchanged).

## Out of scope

- Two-level (hierarchical) windowing, per-big-window filtering, big-window
  feature extraction, and small→big voting — **Spec B**.
- New dataset loaders (only ensuring the design remains dataset-agnostic).
