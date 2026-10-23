# Pipeline Refactor — Design Spec

Date: 2026-10-23
Status: Draft (awaiting review)
Scope: `hand_gesture_recognition` — data model, windowing, splitting, preprocessing, training, runner orchestration, evaluation grouping.

## 1. Motivation

`src/hand_gesture_recognition/runner/run.py` is a 759-line god class that mixes
orchestration, dataset loading, analytics, class remapping, CV/split logic,
baseline removal, signal processing, windowing, classical + neural training
(each duplicated for single- vs two-level windowing), visualization, and
reporting. Branching logic is scattered as string-enum `if` chains
(`split.strategy`, `task.name`, `windowing.mode`, `backend`). Two-level windowing
duplicates ~90% of the single-level windowing loop in
`data/windowing/two_level.py`, and the neural two-level path requires a parallel
datamodule/dataset pair (`TwoLevelEMGDataModule`/`TwoLevelEMGDataset`).

The goal is a smaller, extensible, behavior-preserving codebase: strategies are
first-class and swappable, the runner orchestrates rather than implements, and
the data contracts are explicit enough that adding a new split/windowing/
preprocessing/backend strategy — or grouping results by user/gesture — is a
local change, not a rewrite of `run.py`.

## 2. Goals

- Replace the flat parallel-array `Dataset` with a nested `Dataset → User → Trial → emg` model.
- Make windowing a pure, re-applyable transform over `emg` so two-level windowing is "window, then window the windows" (no duplicated loop).
- Extract splitting, windowing, and backend training/prediction into Strategy classes, dispatching by registry instead of `if` chains.
- Formalize preprocessing as an ordered `fit`/`transform` pipeline.
- Reduce `ExperimentRunner` to orchestration via a `Pipeline` (Template Method) and a `PipelineBuilder`.
- Standardize the five data contracts (`Dataset`, `DataSplit`, `WindowedDataset`/`WindowedPartitions`, `Prediction`/`PartitionPredictions`, `EvaluationResult`) and enforce them with types + tests, not preemptive ABCs.
- Preserve current behavior, verified by regression tests at each step.

## 3. Non-goals

- No registries for leaf components Hydra already instantiates cleanly (models, optimizers, filters, feature extractors, datamodules).
- No new public CLI; the existing `python -m hand_gesture_recognition` and `compare.py` entry points remain.
- No new metrics; grouping is applied to existing metrics.
- No behavioral changes to the ML pipeline itself (same windowing semantics, same signal-processing order, same evaluation).

## 4. Design principles (agreed)

1. **Standardize pipeline data contracts.** Partition, windowed-dataset, prediction, and evaluation-result structures are the stable interfaces everything passes around.
2. **Extract existing branching logic** (splitting, windowing, backend training/prediction), preserving behavior with regression tests.
3. **Keep Hydra for leaf components.** Do not add registries for models, optimizers, filters, or feature extractors.
4. **Formalize preprocessing** as an ordered transformation pipeline with explicit fit/transform behavior.
5. **Reduce the executor to orchestration.** Remove task/backend conditionals; move them into strategies.
6. **Add interfaces only when needed.** Enforce contracts via type checking and tests.

## 5. Target architecture

### 5.1 Data model (nested)

Replace `Dataset`'s flat parallel arrays (`trials`, `labels`, `subject_ids`,
`trial_ids`, `gesture_start`, `gesture_end`) and the `subject_info` DataFrame
with a nested object graph:

```python
@dataclass
class Dataset:
    metadata: DatasetMetadata
    users: List[User]          # authoritative container

@dataclass
class User:
    user_id: str
    trials: List[Trial]
    demographics: Dict[str, Any] | None = None

@dataclass
class Trial:
    trial_id: str
    user_id: str               # denormalized back-ref for flatten/split
    gesture_name: str          # == metadata.no_gesture_name -> rest
    start: int                 # sample index; -1 for rest trials
    end: int                   # sample index; -1 for rest trials
    emg: np.ndarray            # (time_steps, channels); ONLY thing windowing touches
```

Convenience accessors (derived, cached, never stored):
`Dataset.trials` (flattened), `Dataset.labels` (int, via
`class_names.index(gesture_name)`), `Dataset.n_trials`,
`Dataset.subject_ids` / `Dataset.trial_ids` (for compatibility / grouping).

`DatasetMetadata` gains `no_gesture_name: str = "noGesture"`. Label/rest
convention (decided): **labels are derived from `class_names` ordering** (index 0
is always the no-gesture class), **rest trials are `gesture_name ==
no_gesture_name` with `start == end == -1`**. This preserves today's
`w_label = 0 if is_rest` convention and makes no-gesture-drop a trial filter +
re-derived `class_names`.

The `drop_rest_windows` metadata flag (consulted by windowing, set by
`drop_no_gesture`) moves into `WindowingConfig` as an `include_rest_windows`
option; its value-setting by the no-gesture-drop path is preserved so windowing
behavior is unchanged.

The source data already fits this model: `EPN612Loader` reads `gestureName`
per sample and per-user `userInfo` demographics; `make_synthetic_dataset`
generates it directly.

**Performance note (validated):** iteration cost is unchanged. The pipeline
already loops over trials in Python everywhere (`window_dataset`,
`_apply_signal_steps`, baseline fitting, `_subset`). The heavy math (`filtfilt`,
FFT, sklearn, Lightning) is already bound to per-trial `emg` arrays or the
post-windowing `WindowedDataset.X`. Windowing is the single conversion point from
the "tall" object graph to the "wide" flat ndarray contract; everything
downstream of windowing is ndarray/tensor. The flat `trials` view is cached once
at load to avoid repeated flattening.

### 5.2 Canonical contracts

Five dataclasses are the stable interfaces. Immutability is by convention via
`dataclasses.replace` (not enforced with `frozen=True`); type hints + targeted
tests enforce the contracts. No new `Protocol`/`ABC` until a second
implementation demands one.

| Contract | Shape | Notes |
|---|---|---|
| `Dataset` | nested (5.1) | input to splitting |
| `DataSplit` | `train/val/test: Dataset` | unchanged concept |
| `WindowedDataset` | `X (n_windows, time, ch)`, `y`, `WindowMetadata` | flat; model input |
| `WindowedPartitions` | `train/val/test: WindowedDataset` | unchanged concept |
| `Prediction` | `run_id, sample_ids, subject_ids, trial_ids, y_true, y_pred, y_score, duration, backend` | **add `subject_ids`, `trial_ids`** (see 5.6) |
| `EvaluationResult` | `run_id, dataset_id, split, metrics, per_class, confusion_matrix` | unchanged concept |

`WindowMetadata` stays as the per-window lineage record (`window_start`,
`window_end`, `subject_ids`, `trial_ids`, `window_ids`), extended in two-level
mode with `big_window_id` (see 5.5).

### 5.3 Preprocessing pipeline

Preprocessing is formalized as a `PreprocessingPipeline` of ordered steps, each
with `fit(X)` / `transform(X)` (scikit-learn estimator convention). To preserve
the exact current execution order it is split into **two stages**:

1. **Pre-split** (applied once to the full `Dataset`, no fit): **gesture crop**
   (recognition task only) — absorbs `crop_to_gesture`. Must run on the raw
   signal before filtering (it indexes raw `start`/`end`).
2. **Post-split** (applied per `DataSplit` partition): **baseline removal**
   (stateful; fit on train rest trials only) then **signal processing** steps
   (rectify, butterworth, envelope; deterministic, `fit` is a no-op).

`fit(train_split)` fits stateful steps on the train partition;
`transform(split)` applies all steps to every partition with correct
fit-on-train-only semantics. This is the Template Method seam for data
preparation and removes the scattered `_signal_processing` / `_gesture_crop` /
baseline branches in `run.py`.

### 5.4 Strategies + minimal registry

A tiny registry (`@register("name")` decorating a strategy class + a
`get_strategy(cfg)` lookup) replaces the string-enum `if` chains. Leaf
components stay Hydra `_target_`.

- **`SplitStrategy`** — `random`, `stratified`, `group_wise`, `session_wise`,
  `trial_wise`, `predefined` (from `split_dataset`). Each implements
  `split(dataset, cfg) -> DataSplit`.
- **`CVStrategy`** — `kfold`, `loso` (from `generate_folds`), plus the
  group- vs trial-level validation-carve logic.
- **`WindowingStrategy`** — `recognition` (sliding `window_segments`),
  `classification` (`window_gestures`), `two_level` (two chained
  `window_segments`). **Absorbs the `task.name` and `windowing.mode` conditionals.**
- **`TrainingStrategy`** — `classical`, `neural`. Each encapsulates
  preprocessing→feature→fit→predict for its backend and consumes a
  `WindowedPartitions` (or two-level equivalent) to emit `PartitionPredictions`.
  **Absorbs the `backend` conditional and the two-level flatten-vs-sequence choice.**

Feature extractors remain Hydra `_target_` leaves; only their *application*
(FeatureUnion-style concatenation, and per-small-window application in two-level
mode) is factored into a shared `FeatureExtractor` helper used by both
`TrainingStrategy`s.

### 5.5 Two-level windowing (decided: Option A — compositional)

Extract a single primitive:

```python
window_segments(segments, labels, metadata, config, signal_steps) -> WindowedDataset
```

- Single-level recognition = `window_segments(dataset.trials, ...)`.
- Classification = one window per gesture segment (`window_gestures`).
- Two-level = two chained calls: window each trial's `emg` into big windows
  (with per-big-window signal processing, preserving today's semantics), then
  window each big window into small windows.

Two-level output keeps two artifacts because two backends consume different ones:

- **Classical** trains on the **big windows** (flat `WindowedDataset`).
- **Neural** trains on **sequences of small-window features grouped per big window**.

`TwoLevelWindowedDataset` is simplified/renamed to a `GroupedWindowedDataset`
(big `WindowedDataset` + grouped small windows tagged with `big_window_id`);
`TwoLevelEMGDataset` is **deleted** — its feature-extraction-and-padding logic
folds into the shared feature application + a single grouped neural datamodule.

### 5.6 Runner → Pipeline (Template Method) + Builder

`ExperimentRunner` shrinks to: resolve strategies from `ExperimentConfig`, build
a `Pipeline`, run it. The `Pipeline` executes ordered phases via Template Method:

```
load → analytics → preprocess → split/CV → window → train/predict → evaluate → track/report
```

A `PipelineBuilder` composes strategies/phases into a runnable pipeline
(fluent API), so experiments can be assembled and re-run with different phase
combinations. `run.py` becomes a thin facade for backward compatibility with the
existing `__main__.py` / tests.

### 5.7 Grouped evaluation (per-user / per-gesture)

Because `Prediction` now carries `subject_ids`/`trial_ids` (and `y_true` per
sample), grouping is a pure function over the lineage columns:

```python
evaluate_grouped(prediction, by="subject_id") -> dict[str, EvaluationResult]
```

- `by="subject_id"` → per-user metrics (subject-independent generalization view).
- `by="y_true"` → per-gesture metrics (uniform with the existing `per_class`).
- Empty groups are skipped with a warning and their size logged; per-group results
  aggregate via the same `aggregate_folds` machinery (mean/std).

Config-driven via `evaluation.breakdown_by: List[str]` (default `[]` = current
behavior). Reporting renders per-user/per-gesture tables and plots when set.

## 6. Migration plan

Each step is independently mergeable and verified by regression tests before
proceeding. Step 0 is the largest and must land first; it is a mechanical
data-model swap with no behavioral change.

| Step | Scope | Removes / introduces |
|---|---|---|
| **0** | Nested `Dataset/User/Trial/emg` model; rewrite `EPN612Loader`, `fixtures`, `splits/base.py`, all call sites, tests | parallel arrays, `subject_info` DataFrame; adds `no_gesture_name`, derived `labels`/`trials` |
| **1** | `window_segments` primitive + `WindowingStrategy` (recognition/classification/two-level) | `window_dataset_two_level` duplication; `TwoLevelWindowedDataset` → `GroupedWindowedDataset` |
| **2** | `SplitStrategy` + `CVStrategy` registries | `if strategy in (...)` chains in `splits/base.py`, `folds.py` |
| **3** | `PreprocessingPipeline` (baseline + signal + gesture-crop, fit/transform) | scattered `_signal_processing`/`_gesture_crop`/baseline branches |
| **4** | `TrainingStrategy` (classical/neural) + shared `FeatureExtractor` application | backend/mode conditionals; `TwoLevelEMGDataset` |
| **5** | `Pipeline` + `PipelineBuilder`; `ExperimentRunner` reduced to facade | the god-class `run()` |
| **6** | Contract hardening: `Prediction` lineage + grouped evaluation; type checks, docs, dead-code removal | leftover aliases; adds `evaluation.breakdown_by` |

## 7. Testing strategy

- **Behavioral spec**: integration/smoke tests
  (`test_smoke.py`, `test_two_level_neural.py`, `test_two_level_classical.py`,
  `test_cross_backend_parity.py`, `test_task_classification.py`,
  `test_partition_evaluation.py`, `test_cv.py`, `test_replay.py`) must stay green
  or be updated only to reflect intentional contract renames.
- **Unit tests**: rewritten to target the new public APIs (contracts, strategies,
  pipeline, grouping); internals-coupled tests (e.g. `test_two_level_windowing.py`,
  `test_two_level_datamodule.py`) are replaced.
- **New regression tests** per step: windowing equivalence (single vs two-level
  counts/labels/metadata), split equivalence (same partitions as before for a
  fixed seed), preprocessing order equivalence, and grouped-evaluation correctness.
- **Config contract**: Hydra configs (`config.yaml`, `windowing/*.yaml`, etc.)
  remain the composition surface; strategy names stay valid overrides.

## 8. Risks & decisions

- **Step 0 blast radius** is large (every module touches `Dataset`). Mitigated by
  doing it first in isolation with the full test suite as the net, and by keeping
  derived accessors (`Dataset.labels`, `.trials`) so call sites migrate
  incrementally where possible.
- **Signal-processing order in two-level mode** (per-big-window filtering) must be
  preserved exactly — this is the one place the two-level path intentionally
  diverges from single-level.
- **Per-user metrics validity**: groups with few/no windows (after class selection
  or windowing) produce noisy or missing scores; grouped evaluation skips empty
  groups and reports sizes.
- **Decided defaults**: `evaluation.breakdown_by` defaults to `[]` (no breakdown,
  current behavior); `DatasetMetadata.no_gesture_name = "noGesture"`;
  two-level keeps both big-window and grouped-small-window artifacts (Option A).
