# Cross-Validation, Split Documentation, and Configurable Visualization

**Date:** 2026-10-08
**Status:** Draft (awaiting user review)

## Context

The framework runs a single experiment end-to-end (`ExperimentRunner.run`):
load → analytics → gesture crop → split → baseline removal → class selection →
signal processing → windowing → train/predict → evaluate → track/report →
visualize. This spec adds four capabilities:

1. **Cross-validation** (k-fold or leave-one-subject-out), configurable.
2. **Documentation** of trial-level vs subject-wise splitting (trial-level
   already exists as `random`/`stratified`; the deliverable is docs + a
   discoverable alias).
3. **Consistent visualization scale** within a figure so channels are
   comparable.
4. **Configurable visualization** — per-gesture channel selection that drives
   a "signal at each preprocessing stage" plot and a "feature-over-windows"
   plot.

## Goals / Non-goals

**Goals**

- `cv` config group: `kfold` and `loso` methods, group-aware or trial-level
  folds, optional val carve-out, aggregate + per-fold reporting.
- `visualization` config group: consistent scale, gesture→channel selection
  (single source of truth), stage-progression plot, feature-over-windows plot.
- A `docs/experiments/splitting.md` page explaining trial-level vs
  subject-wise splitting.

**Non-goals**

- No change to the *training* pipeline's signal-processing/baseline order in
  this spec (see "Risks & notes").
- No new feature extractors; only named, discoverable access to the existing
  time-domain and frequency-domain features.
- No hyperparameter search; CV reports held-out metrics, it does not select
  hyperparameters.

## 1. Configuration schema

### 1.1 `CrossValidationConfig` (new `cv` group)

```python
@dataclass
class CrossValidationConfig:
    enabled: bool = False
    method: str = "kfold"        # kfold | loso
    n_splits: int = 5            # kfold only
    group_folds: bool = True     # kfold only: subject-wise (GroupKFold) vs trial-level (KFold)
    include_val: bool = True     # carve a val subset out of each training fold
    val_ratio: float = 0.2       # fraction of the *training fold* carved into val
    seed: int = 42
    report_per_fold: bool = True # write per-fold metrics + confusion matrices
```

Validation: `method in ("kfold", "loso")`; `n_splits >= 2` (when kfold);
`0 < val_ratio < 1`; when `include_val` is false the fold's `val` partition is
empty and the runner skips val predict/evaluate.

Config files:
- `configs/cv/none.yaml` (enabled: false, the default)
- `configs/cv/kfold.yaml`
- `configs/cv/loso.yaml`

### 1.2 `VisualizationConfig` (new `visualization` group)

```python
@dataclass
class VisualizationSelectionConfig:
    gestures: List[str] = field(default_factory=list)   # [] = all classes
    channels: List[int] = field(default_factory=list)   # [] = all channels (1-based)
    by_gesture: Dict[str, List[int]] = field(default_factory=dict)  # per-gesture override

@dataclass
class FeatureOverWindowsConfig:
    features: List[str] = field(default_factory=list)   # feature names; [] = all
    max_windows: int = 0                                # 0 = all windows

@dataclass
class VisualizationConfig:
    enabled: bool = True
    consistent_scale: bool = True
    selection: VisualizationSelectionConfig = field(default_factory=VisualizationSelectionConfig)
    stage_progression: bool = True
    feature_over_windows: FeatureOverWindowsConfig = field(default_factory=FeatureOverWindowsConfig)
```

Config file: `configs/visualization/default.yaml`.

### 1.3 Rename `reporting.visualize` → `visualization.enabled`

`ReportingConfig` keeps `formats` only; the master plot toggle moves to
`visualization.enabled`. `ExperimentConfig` gains `cv: CrossValidationConfig`
and `visualization: VisualizationConfig`. `configuration/loader.py::_coerce`
must construct both. Existing `configs/experiment/default.yaml` (and any test
fixtures referencing `reporting.visualize`) are updated.

## 2. Cross-validation

### 2.1 Fold generation — `data/splits/folds.py`

```python
def generate_folds(dataset: Dataset, cv: CrossValidationConfig,
                   split: SplitConfig) -> List[DataSplit]:
```

- Group identity: `dataset.subject_ids` when `split.split_on == "subject_ids"`,
  else `dataset.trial_ids`.
- `loso`: `LeaveOneGroupOut` over groups — each group is one test fold, the
  rest train. When `include_val`, carve a subject-wise val subset out of the
  training groups by `val_ratio`.
- `kfold`, `group_folds=True`: `GroupKFold(n_splits=cv.n_splits)` over groups;
  test = held-out groups, train = rest; val carved group-wise from train.
- `kfold`, `group_folds=False`: `KFold(n_splits=cv.n_splits, shuffle=True,
  random_state=cv.seed)` over trial indices; val carved trial-wise from train.
- Each fold → `DataSplit(train=_subset(train_idx), val=_subset(val_idx),
  test=_subset(test_idx))`, reusing the existing `_subset` helper.
- For group-based folds, assert `assert_no_group_leakage` on each fold.

### 2.2 Runner refactor — unified per-partition core

Extract the post-split core into:

```python
def _run_partition(self, split: DataSplit, class_names, remap):
    # baseline removal (fit on split.train), class selection (drop no-gesture
    # via global remap), signal processing, windowing, train/predict, evaluate
    return predictions, {"train": r_train, "val": r_val, "test": r_test}
```

`run()` then:

```python
class_names = cropped.metadata.class_names
remap = None
if not cfg.dataset.include_no_gesture:
    remap, class_names = build_no_gesture_remap(cropped)   # computed ONCE, global

if cfg.cv.enabled:
    folds = generate_folds(cropped, cfg.cv, cfg.split)
    fold_test_results = []
    for i, fold in enumerate(folds):
        _, results = self._run_partition(fold, class_names, remap)
        fold_test_results.append(results["test"])
    summary = aggregate_folds(fold_test_results)          # {metric}_mean / {metric}_std
else:
    split = split_dataset(cropped, cfg.split)
    predictions, results = self._run_partition(split, class_names, remap)
```

Per-fold correctness: class selection/remap is computed once (it depends on the
global class set); baseline removal is **fit on each fold's train** inside
`_run_partition`.

### 2.3 Aggregation and reporting

New helper (in `evaluation/`) `aggregate_folds(results: List[EvaluationResult])
-> Dict[str, float]` producing `{f"{m}_mean": ... , f"{m}_std": ...}` for every
metric in `evaluation.metrics`.

Reporting in CV mode:

- MLflow: log `{m}_mean` / `{m}_std` as primary metrics. When
  `report_per_fold`, also log `fold{i}_{m}` and per-fold confusion matrices
  (`confusion_matrix_fold{i}_test.png`).
- JSON/HTML report: include a `cv_summary` block (per-fold metrics + mean/std).
- `test_accuracy`, `test_macro_f1`, etc. remain the headline keys for
  downstream tooling; their values are the fold means.

Both backends are supported; neural is slower (N fits) but follows the same
loop.

## 3. Split documentation + `trial_wise` alias

- Add `docs/experiments/splitting.md`: trial-level (`random`, `stratified`,
  `trial_wise`) vs subject-wise (`subject_wise`, `session_wise`, `predefined`),
  with leakage implications and when to use each. Cross-link from
  `docs/architecture/overview.md` and `README.md`.
- Add `trial_wise` to `SplitConfig.validate` allowlist and to `split_dataset`
  (alias of `random`: trial-level shuffle). Explicit name for discoverability.

## 4. Visualization

### 4.1 Consistent scale (within-figure)

Add a helper that computes the global min/max across the plotted channels of a
figure and applies it to every channel axis. Used by the channel-stacked plots
(raw signal, per-class, stage-progression, predictions-on-signal) when
`visualization.consistent_scale` is true. Per-figure, not cross-figure.

### 4.2 Selection as single source of truth

```python
def resolve_selection(selection, class_names, n_channels) -> Dict[str, List[int]]
```

Returns gesture name → 0-based channel indices. `gestures: []` → all present
classes; `channels: []` → all channels; `by_gesture` overrides per gesture.
Plots A, B, C all iterate this mapping.

### 4.3 Plot A — per-gesture channel selection

`plot_raw_signal_by_class` plots only the selected gestures and only their
selected channels (subplots), with consistent scale.

### 4.4 Plot B — stage progression

Stages = `["raw"] + [label per signal_processing step] + ["baseline_removed"
if baseline enabled]`. Step labels derive from the transformer (e.g.
`bandpass(10-95Hz)`, `notch(60Hz)`, `rectified`, `envelope`).

Layout: **one figure per (stage, gesture) combination**. Each figure shows all
selected channels for that gesture (one subplot per channel, shared y-scale).
Filenames: `stage_progression_{stage}_{gesture}.png`.

Requires snapshotting intermediate signals — new helper
`compute_stages(dataset, signal_processing_steps, baseline_remover) ->
List[(label, Dataset)]`. The baseline remover used for visualization is fitted
on the full (cropped) dataset for diagnostics.

### 4.5 Plot C — feature-over-windows

For each selected gesture: one figure, rows = selected features, columns =
selected channels; each cell plots that feature's value over windows (x =
window index). Within a feature row, channels share y-scale (features have
different units, so no cross-feature scale).

Computation: window the selected (processed) trial → apply both the
time-domain and frequency-domain extractors → `(n_windows, 11 * n_channels)`
matrix → select by (feature name, channel).

Feature naming (add to extractors as a class attribute):

- `TimeDomainFeatures.FEATURE_NAMES = ["rms", "std", "min", "max", "mean",
  "max_over_mean", "min_over_max"]`
- `FrequencyDomainFeatures.FEATURE_NAMES = ["total_power", "peak_frequency",
  "mean_frequency", "median_frequency"]`

A helper builds the flattened names and resolves `features: [...]` to column
indices; `features: []` = all 11.

### 4.6 CV-mode visualization behavior

In CV mode the split-dependent plots (`predictions_on_signal`,
`signal_data_flow`) are skipped; the data-diagnostic plots (A, B, C, FFT,
signal-processing comparison) run once on the full dataset.

## Data flow (summary)

```
cropped ──► (single) split_dataset   OR   generate_folds (cv)
                 │                              │
                 └────────── _run_partition ─────┘  (baseline→classsel→signal→window→train→eval)
                                     │
                              results / fold_results
                                     │
                          aggregate (cv) / report / visualize
```

## Testing

- `tests/unit/test_cv.py`: fold generation (kfold group/trial, loso), leakage
  assertions, empty-val handling, aggregation mean/std.
- `tests/unit/test_visualization.py` (extend): consistent-scale ylim, selection
  resolution (gestures/channels/by_gesture), feature-name resolution, stage
  labels.
- `tests/unit/test_config_schema.py` (extend): `cv`/`visualization` validation,
  `trial_wise` allowlist, `reporting.visualize` removal.
- `tests/integration/test_cv.py`: end-to-end CV on synthetic (classical),
  per-fold artifact presence, aggregate metrics.
- Update existing tests referencing `reporting.visualize`.

## Risks & notes

- **Baseline/signal-processing order inconsistency (pre-existing):** the
  training path applies baseline removal *before* signal processing, while the
  visualization `processed_dataset` applies signal processing *then* baseline.
  This spec uses the user-stated order (filter → rectify → baseline) for the
  stage-progression plot and feature-over-windows, and does **not** change the
  training pipeline. If the training order should be reconciled, that is a
  separate follow-up.
- **Neural CV cost:** N fits per run; accepted as inherent.
- **Empty val fold** (include_val=false): runner skips val predict/evaluate and
  the val report entry.
