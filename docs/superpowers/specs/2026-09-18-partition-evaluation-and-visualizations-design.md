# Partition-wise Evaluation and Richer Visualizations

**Date:** 2026-09-18
**Status:** Approved (pending implementation plan)

## Context

The framework currently evaluates a run only on the **test** partition. The
runner's train step returns a single `Prediction` (test only), and the
classical backend drops the `val` partition entirely. Reporting emits a small
set of figures (`raw_signal.png`, `predictions_on_signal.png`,
`signal_data_flow.png`) plus a single test confusion matrix.

We want richer diagnostics and visuals:

1. Confusion matrices for **train, val, and test** (to visualize the
   generalization gap / overfitting).
2. `predictions_on_signal` figures for **val and test**.
3. Raw-signal figures **one per class** (all 6 classes).
4. A **before/after signal-processing comparison** figure.

## Goals

- Produce a `Prediction` per partition (train / val / test) and evaluate each
  separately.
- Log three confusion matrices (raw + row-normalized) and per-partition metrics
  to MLflow.
- Emit the four visualization families above into
  `output/evaluation/<run_id>/`.
- Keep `run()` returning the **test** `EvaluationResult` so the `compare` CLI,
  reporters, and existing tests remain unchanged.

## Non-goals

- No change to the split logic or leakage checks.
- No hyperparameter tuning (the classical backend will now compute val
  predictions, but val is still not used to *select* the classical model).
- No change to the `Prediction` / `EvaluationResult` public schemas.
- No new report formats.

## Design

### 1. Partition-aware predictions

New dataclass in `prediction/base.py` (co-located with `Prediction`):

```python
@dataclass
class PartitionPredictions:
    train: Prediction
    val: Prediction
    test: Prediction
```

The train step returns `PartitionPredictions` instead of a single `Prediction`.

**Classical backend** (`_train_and_predict_classical(split)` — signature changes
from `(train, test)` to `(split)`):

- Window `split.train` (fit) and `split.val` / `split.test` (transform).
- Learned preprocessing and feature extraction: `fit` on train, `transform` on
  all three partitions.
- Model: `fit` on train; `predict` + `predict_proba` on all three partitions.
- Build three `Prediction`s via `SklearnPredictionAdapter`.

**Neural backend** (`_train_and_predict_neural(split)`):

- After `trainer.fit`, run `trainer.predict(model, dataloaders=...)` for
  `dm.train_dataloader()`, `dm.val_dataloader()`, and `dm.test_dataloader()`.
- All dataloaders use `shuffle=False` (framework default), so window order
  matches `window_dataset` output — the same ordering guarantee the current
  test path already relies on.
- Window each partition via `window_dataset` to recover metadata + `y_true`;
  build three `Prediction`s via `LightningPredictionAdapter`.

### 2. Runner flow

```python
with stage(..., "train"):
    predictions = self._train_and_predict_classical(split)  # or neural
with stage(..., "evaluate"):
    results = {
        p: evaluate(getattr(predictions, p), cfg.evaluation,
                    dataset_id=..., split_info=..., class_names=...)
        for p in ("train", "val", "test")
    }
    result = results["test"]          # primary; returned by run()
with stage(..., "track_and_report"):
    self.tracker.log_dataset(dataset)
    self.tracker.log_params(...)
    for p, r in results.items():
        self.tracker.log_results(r, partition=(None if p == "test" else p))
        self.tracker.log_confusion_matrix(r, class_names, partition=p)
    self._visualize(raw_dataset, dataset, split, predictions)
    # `dataset` above is the signal-processed dataset (see `_signal_processing`).
    self._persist_artifacts(dataset, result)
return result
```

### 3. MLflow logging

- `MlflowTracker.log_results(result, partition=None)`: when `partition` is
  given, prefix metric keys with `{partition}_` (e.g. `val_accuracy`).
- `MlflowTracker.log_confusion_matrix(result, class_names, partition)`: artifact
  names become `confusion_matrix_{partition}.png` and
  `confusion_matrix_{partition}_normalized.png`.
- Metrics: **test** stays unprefixed (`accuracy`, `balanced_accuracy`,
  `macro_f1`, per-class `precision_X`/`recall_X`/`f1_X`/`tp_X`/...); **train**
  and **val** log top-level metrics only (`{partition}_accuracy`,
  `{partition}_balanced_accuracy`, `{partition}_macro_f1`) — per-class detail is
  conveyed by the CMs, avoiding ~126 noisy metrics.

### 4. Visualizations

1. **Raw signal per class** — new `plot_raw_signal_by_class(dataset, run_dir,
   class_names)` in `reporting/visualization.py`. For each class present in the
   dataset, select the first trial of that class and reuse `plot_raw_signal`
   with `example_tid` → `raw_signal_{class_name}.png` (6 files:
   `noGesture`, `fist`, `waveIn`, `waveOut`, `open`, `pinch`). Skip classes
   with no trials (log a warning). Supersedes the old single `raw_signal.png`.

2. **Predictions on signal (val + test)** — call `plot_predictions_on_signal`
   for `split.val` and `split.test` → `predictions_on_signal_val.png` and
   `predictions_on_signal_test.png`.

3. **Signal-processing comparison** — new
   `plot_signal_processing_comparison(raw_dataset, processed_dataset, path)`:
   one row per channel, two columns (raw | processed), for a representative
   non-noGesture trial → `signal_processing_comparison.png`. Skipped when
   `cfg.signal_processing` is empty (nothing to compare).

4. `signal_data_flow.png` unchanged.

`_visualize` signature changes to
`(self, raw_dataset, processed_dataset, split, predictions)`.

### 5. Edge cases / error handling

- `_log_cm_figure` skips when `cm.size == 0` (e.g. `exclude_classes` removes all
  samples of a partition).
- `plot_raw_signal_by_class` skips classes with no trials.
- `plot_signal_processing_comparison` is only called when signal processing was
  actually applied.

## Testing

- Unit: `plot_raw_signal_by_class` emits one file per present class;
  `plot_signal_processing_comparison` emits a figure; `PartitionPredictions`
  has the expected three fields.
- Integration: extend a smoke test to assert the new artifacts exist
  (`confusion_matrix_test.png`, `raw_signal_fist.png`,
  `predictions_on_signal_val.png`, `signal_processing_comparison.png`).
- Existing `test_cross_backend_parity.py` (uses local helpers),
  `test_replay.py`, and `test_smoke.py` (assert on the returned test result)
  remain compatible.

## Out of scope

- Model selection / hyperparameter tuning on val.
- Per-class metrics for train/val partitions.
- Per-class signal-processing comparison figures.
