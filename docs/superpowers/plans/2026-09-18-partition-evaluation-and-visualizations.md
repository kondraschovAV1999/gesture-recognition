# Partition-wise Evaluation and Richer Visualizations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate a run on train/val/test (three confusion matrices + prefixed metrics) and emit richer figures (raw signal per class, predictions on val+test, signal-processing before/after).

**Architecture:** Introduce a `PartitionPredictions` container (train/val/test `Prediction`s). The runner's train step produces predictions for all three partitions; each is evaluated and logged separately; `run()` still returns the test `EvaluationResult`. Two new plotting helpers are added to `reporting/visualization.py`.

**Tech Stack:** Python 3.10, scikit-learn, PyTorch Lightning, MLflow, matplotlib (Agg), Hydra/OmegaConf.

## Global Constraints

- Package name: `hand_gesture_recognition`; source under `src/hand_gesture_recognition/`.
- Windows shell: run tests with `python -m pytest` (never bare `pytest`); commit with single-token `git commit -m <token>`; never use `python -c "..."` (quote-stripping) — write a `.py` file instead.
- MLflow backend is `sqlite:///mlflow.db`; tests must override `tracking.tracking_uri` to a `tmp_path`.
- Neural trainer uses `accelerator: auto` (GPU when available), `num_workers: 0`, `shuffle: false` — window order is deterministic and must match `window_dataset`.
- `Prediction` and `EvaluationResult` public schemas must NOT change (cross-backend parity test depends on them).
- Do NOT modify the `mlflow.sklearn.autolog()/mlflow.pytorch.autolog()` block the user added in `run.py` (lines 52–55).

---

### Task 1: `PartitionPredictions` dataclass

**Files:**
- Modify: `src/hand_gesture_recognition/prediction/base.py`
- Test: `tests/unit/test_prediction.py`

**Interfaces:**
- Consumes: existing `Prediction` dataclass (same module).
- Produces: `PartitionPredictions(train: Prediction, val: Prediction, test: Prediction)` — used by Tasks 2 and 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_prediction.py` (add `Prediction` and `PartitionPredictions` to the imports at the top):

```python
from hand_gesture_recognition.prediction.base import PartitionPredictions, Prediction


def test_partition_predictions_dataclass():
    def make_pred(tag: str) -> Prediction:
        return Prediction(
            run_id="r",
            sample_ids=np.array([tag]),
            y_true=np.array([0]),
            y_pred=np.array([0]),
            y_score=np.array([[1.0]]),
            prediction_duration_s=0.0,
            backend="classical",
        )

    pp = PartitionPredictions(
        train=make_pred("train"), val=make_pred("val"), test=make_pred("test")
    )
    assert pp.train.sample_ids.tolist() == ["train"]
    assert pp.val.sample_ids.tolist() == ["val"]
    assert pp.test.sample_ids.tolist() == ["test"]
    assert pp.train.backend == "classical"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_prediction.py::test_partition_predictions_dataclass -q`
Expected: FAIL with `ImportError: cannot import name 'PartitionPredictions'`.

- [ ] **Step 3: Write minimal implementation**

In `src/hand_gesture_recognition/prediction/base.py`, add after the `Prediction` dataclass (line 18):

```python
@dataclass
class PartitionPredictions:
    """Predictions for the train/val/test partitions of a single run."""

    train: Prediction
    val: Prediction
    test: Prediction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_prediction.py::test_partition_predictions_dataclass -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/prediction/base.py tests/unit/test_prediction.py
git commit -m feat-partition-predictions
```

---

### Task 2: Tracker partition-aware logging + empty-CM guard

**Files:**
- Modify: `src/hand_gesture_recognition/tracking/mlflow_tracker.py`
- Test: `tests/unit/test_tracking.py`

**Interfaces:**
- Consumes: `EvaluationResult` (from `evaluation.base`).
- Produces:
  - `log_results(result: EvaluationResult, partition: Optional[str] = None) -> None` — prefixes metric keys with `{partition}_` when `partition` is given; per-class metrics only when `partition is None`.
  - `log_confusion_matrix(result, class_names=None, partition: str = "test") -> None` — artifacts named `confusion_matrix_{partition}.png` / `confusion_matrix_{partition}_normalized.png`; skips when the CM is empty.
  - `_log_cm_figure(...)` returns early when `cm.size == 0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tracking.py` (add `import numpy as np` and `from hand_gesture_recognition.evaluation.base import EvaluationResult` at the top):

```python
import numpy as np

from hand_gesture_recognition.evaluation.base import EvaluationResult


def test_log_results_partition_prefix(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-3")
    res = EvaluationResult(run_id="r", dataset_id="d", metrics={"accuracy": 0.8})
    t.log_results(res, partition="val")
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    assert client.get_run(t.run_id).data.metrics["val_accuracy"] == 0.8


def test_log_results_no_partition_logs_per_class(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-3b")
    res = EvaluationResult(
        run_id="r",
        dataset_id="d",
        metrics={"accuracy": 0.9},
        per_class={"fist": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "tp": 1, "fp": 1, "tn": 1, "fn": 1}},
    )
    t.log_results(res, partition=None)
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    metrics = client.get_run(t.run_id).data.metrics
    assert metrics["accuracy"] == 0.9
    assert metrics["precision_fist"] == 0.5


def test_log_confusion_matrix_empty_skips(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-4")
    res = EvaluationResult(run_id="r", dataset_id="d", confusion_matrix=np.zeros((0, 0)))
    t.log_confusion_matrix(res, class_names=["a", "b"], partition="train")
    t.finish("FINISHED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    artifacts = [a.path for a in client.list_artifacts(t.run_id)]
    assert "confusion_matrix_train.png" not in artifacts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_tracking.py -q`
Expected: FAIL — `TypeError: log_results() got an unexpected keyword argument 'partition'` (and `log_confusion_matrix` similarly).

- [ ] **Step 3: Write minimal implementation**

Replace `log_results` (lines 57–71) in `src/hand_gesture_recognition/tracking/mlflow_tracker.py`:

```python
    def log_results(self, result: EvaluationResult, partition: Optional[str] = None) -> None:
        prefix = f"{partition}_" if partition else ""
        if result.metrics:
            mlflow.log_metrics({f"{prefix}{k}": v for k, v in result.metrics.items()})
        # Per-class precision/recall/F1 + confusion-matrix components are only
        # logged for the primary (test) result; train/val rely on their CMs.
        if partition is None:
            for cls, cls_metrics in result.per_class.items():
                mlflow.log_metrics(
                    {
                        f"precision_{cls}": cls_metrics.get("precision", 0.0),
                        f"recall_{cls}": cls_metrics.get("recall", 0.0),
                        f"f1_{cls}": cls_metrics.get("f1", 0.0),
                        f"tp_{cls}": cls_metrics.get("tp", 0),
                        f"fp_{cls}": cls_metrics.get("fp", 0),
                        f"tn_{cls}": cls_metrics.get("tn", 0),
                        f"fn_{cls}": cls_metrics.get("fn", 0),
                    }
                )
```

Replace `log_confusion_matrix` (lines 73–81):

```python
    def log_confusion_matrix(
        self,
        result: EvaluationResult,
        class_names: Optional[List[str]] = None,
        partition: str = "test",
    ) -> None:
        """Log the confusion matrix as heatmap figures (raw counts + normalized)."""
        cm = result.confusion_matrix
        base = f"confusion_matrix_{partition}"
        self._log_cm_figure(cm, class_names, f"{base}.png", normalize=False)
        self._log_cm_figure(cm, class_names, f"{base}_normalized.png", normalize=True)
```

Add the empty-CM guard at the top of `_log_cm_figure` (line 84, after the docstring/imports):

```python
    @staticmethod
    def _log_cm_figure(cm, class_names, artifact_name: str, normalize: bool) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if cm.size == 0:
            return

        n = cm.shape[0]
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_tracking.py -q`
Expected: PASS (3 new tests + 2 existing).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/tracking/mlflow_tracker.py tests/unit/test_tracking.py
git commit -m feat-partition-tracker
```

---

### Task 3: `plot_raw_signal_by_class`

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Test: `tests/unit/test_visualization.py` (new file)

**Interfaces:**
- Consumes: `plot_raw_signal(dataset, example_tid, path)` (same module) and `Dataset` (`labels`, `trial_ids`, `metadata.class_names`, `n_trials`).
- Produces: `plot_raw_signal_by_class(dataset, run_dir, class_names=None) -> List[str]` — writes `raw_signal_{class_name}.png` per class present, returns the list of written paths.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_visualization.py`:

```python
import os

from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.reporting.visualization import plot_raw_signal_by_class


def test_plot_raw_signal_by_class(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=3, n_classes=3, length=100, n_channels=4, seed=0
    )
    paths = plot_raw_signal_by_class(ds, str(tmp_path), class_names=ds.metadata.class_names)
    assert len(paths) == 3  # classes 0,1,2 all present
    for p in paths:
        assert os.path.exists(p)
    assert os.path.exists(str(tmp_path / "raw_signal_class_0.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: FAIL with `ImportError: cannot import name 'plot_raw_signal_by_class'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/hand_gesture_recognition/reporting/visualization.py` (after `plot_raw_signal`, around line 66):

```python
def plot_raw_signal_by_class(dataset: Dataset, run_dir: str, class_names: Optional[List[str]] = None):
    """Write one raw-signal figure per class present in the dataset.

    Returns the list of file paths written (for artifact logging). Classes with
    no trials in this dataset are naturally absent (skipped by iterating over
    the unique labels actually present).
    """
    paths: List[str] = []
    if dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else dataset.metadata.class_names
    for cls in np.unique(dataset.labels):
        cls = int(cls)
        idx = np.where(dataset.labels == cls)[0]
        tid = str(dataset.trial_ids[idx[0]])
        name = names[cls] if cls < len(names) else str(cls)
        path = os.path.join(run_dir, f"raw_signal_{name}.png")
        plot_raw_signal(dataset, example_tid=tid, path=path)
        paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py tests/unit/test_visualization.py
git commit -m feat-raw-signal-per-class
```

---

### Task 4: `plot_signal_processing_comparison`

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Test: `tests/unit/test_visualization.py` (append)

**Interfaces:**
- Consumes: `_trial_index`, `_channel_axes` pattern (same module); `Dataset`.
- Produces: `plot_signal_processing_comparison(raw_dataset, processed_dataset, example_tid=None, path=None)` — 2-column × n_channel figure (raw | processed) for one representative non-noGesture trial; saves to `path` when given; returns the `Figure`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_visualization.py`:

```python
from dataclasses import replace

import numpy as np

from hand_gesture_recognition.reporting.visualization import (
    plot_signal_processing_comparison,
)


def test_plot_signal_processing_comparison(tmp_path):
    raw = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=3, n_classes=3, length=100, n_channels=4, seed=0
    )
    # a trivially different "processed" signal (same trial structure/id)
    proc = replace(raw, trials=[np.abs(t) for t in raw.trials])
    path = str(tmp_path / "signal_processing_comparison.png")
    fig = plot_signal_processing_comparison(raw, proc, path=path)
    assert fig is not None
    assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py::test_plot_signal_processing_comparison -q`
Expected: FAIL with `ImportError: cannot import name 'plot_signal_processing_comparison'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/hand_gesture_recognition/reporting/visualization.py` (after `plot_signal_data_flow`, at end of file):

```python
def plot_signal_processing_comparison(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """Before/after comparison of signal processing: raw vs processed signal,
    one row per channel (2 columns), for a representative non-noGesture trial."""
    if raw_dataset.n_trials == 0:
        return None
    tid = example_tid
    if tid is None:
        for i in range(raw_dataset.n_trials):
            if int(raw_dataset.labels[i]) != 0:
                tid = str(raw_dataset.trial_ids[i])
                break
        tid = tid or str(raw_dataset.trial_ids[0])

    i = _trial_index(raw_dataset, tid)
    raw_trial = raw_dataset.trials[i]
    proc_trial = processed_dataset.trials[_trial_index(processed_dataset, tid)]
    fs = raw_dataset.metadata.sampling_rate
    n_ch = raw_trial.shape[1]

    fig, axes = plt.subplots(n_ch, 2, figsize=(12, 1.8 * n_ch), squeeze=False)
    t_raw = np.arange(raw_trial.shape[0]) / fs
    t_proc = np.arange(proc_trial.shape[0]) / fs
    for c in range(n_ch):
        axes[c, 0].plot(t_raw, raw_trial[:, c], lw=0.6)
        axes[c, 1].plot(t_proc, proc_trial[:, c], lw=0.6, color="tab:orange")
        axes[c, 0].set_ylabel(f"ch{c + 1}")
    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("signal-processed")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"signal processing — {tid}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py tests/unit/test_visualization.py
git commit -m feat-signal-processing-comparison
```

---

### Task 5: Runner — partition predictions, evaluation, and visualization wiring

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/integration/test_partition_evaluation.py` (new file)

**Interfaces:**
- Consumes: `PartitionPredictions` (Task 1), `MlflowTracker.log_results(..., partition=...)` and `log_confusion_matrix(..., partition=...)` (Task 2), `plot_raw_signal_by_class` / `plot_signal_processing_comparison` (Tasks 3–4), `split_dataset` / `DataSplit`.
- Produces (internal): `_train_and_predict_classical(split) -> PartitionPredictions`, `_train_and_predict_neural(split) -> PartitionPredictions`, and `run() -> EvaluationResult` (unchanged return type).

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_partition_evaluation.py`:

```python
import os

import mlflow

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _artifacts(runner):
    client = mlflow.tracking.MlflowClient(tracking_uri=runner.cfg.tracking.tracking_uri)
    return [a.path for a in client.list_artifacts(runner.tracker.run_id)]


def test_classical_partition_evaluation(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics  # returned result is the test result
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "raw_signal_class_0.png").exists()
    assert (out / "predictions_on_signal_val.png").exists()
    assert (out / "predictions_on_signal_test.png").exists()
    assert (out / "signal_processing_comparison.png").exists()

    artifacts = _artifacts(runner)
    for name in (
        "confusion_matrix_train.png",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
        "confusion_matrix_train_normalized.png",
        "confusion_matrix_val_normalized.png",
        "confusion_matrix_test_normalized.png",
    ):
        assert name in artifacts, name


def test_neural_partition_evaluation(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=neural",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    artifacts = _artifacts(runner)
    assert "confusion_matrix_train.png" in artifacts
    assert "confusion_matrix_val.png" in artifacts
    assert "confusion_matrix_test.png" in artifacts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_partition_evaluation.py -q`
Expected: FAIL — the runner still returns a single `Prediction` (not `PartitionPredictions`), so `run()` raises an `AttributeError` (`.train` missing) or the artifacts are absent.

- [ ] **Step 3: Wire the runner**

Edit `src/hand_gesture_recognition/runner/run.py` in four places (do NOT touch the autolog block at lines 52–55):

**(a)** Add the import near the top (after line 12):

```python
from hand_gesture_recognition.prediction.base import PartitionPredictions, Prediction
```

**(b)** Replace `_train_and_predict_classical` (lines 139–175) entirely:

```python
    def _train_and_predict_classical(self, split) -> PartitionPredictions:
        from hand_gesture_recognition.classical.pipelines.windowing import (
            WindowingTransformer,
        )

        # shared windowing via the spec-mandated sklearn wrapper (Section 6.2)
        windower = WindowingTransformer(self.cfg.windowing)
        train_wd = windower.fit_transform(split.train)
        val_wd = windower.transform(split.val)
        test_wd = windower.transform(split.test)

        # Learned preprocessing (3D, fit on train only) then feature extraction
        # (3D -> 2D) — both via hydra.utils.instantiate (ADR #18).
        X_train, X_val, X_test = train_wd.X, val_wd.X, test_wd.X
        for step_cfg in self.cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)
        for step_cfg in self.cfg.classical.features:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)

        model = instantiate(self.cfg.classical.model)
        t0 = time.perf_counter()
        model.fit(X_train, train_wd.y)

        adapter = SklearnPredictionAdapter()

        def build(wd, X) -> Prediction:
            return adapter.to_prediction(
                windowed=wd,
                y_pred=model.predict(X),
                y_score=model.predict_proba(X),
                run_id=self.cfg.run_id,
                duration_s=0.0,
            )

        train_p = build(train_wd, X_train)
        val_p = build(val_wd, X_val)
        test_p = build(test_wd, X_test)
        duration = time.perf_counter() - t0
        train_p.prediction_duration_s = duration
        val_p.prediction_duration_s = duration
        test_p.prediction_duration_s = duration
        return PartitionPredictions(train=train_p, val=val_p, test=test_p)
```

**(c)** Replace the tail of `_train_and_predict_neural` (from the `t0 = time.perf_counter()` line, ~219, through the final `return adapter.to_prediction(...)`, ~234) with:

```python
        t0 = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        train_raw = trainer.predict(model, dataloaders=datamodule.train_dataloader())
        val_raw = trainer.predict(model, dataloaders=datamodule.val_dataloader())
        test_raw = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        duration = time.perf_counter() - t0

        # Window each partition (same shared windowing) to recover per-window
        # metadata + y_true for the Prediction.
        train_wd = window_dataset(split.train, self.cfg.windowing)
        val_wd = window_dataset(split.val, self.cfg.windowing)
        test_wd = window_dataset(split.test, self.cfg.windowing)

        adapter = LightningPredictionAdapter()
        return PartitionPredictions(
            train=adapter.to_prediction(
                windowed=train_wd, raw_predictions=train_raw,
                run_id=self.cfg.run_id, duration_s=duration,
            ),
            val=adapter.to_prediction(
                windowed=val_wd, raw_predictions=val_raw,
                run_id=self.cfg.run_id, duration_s=duration,
            ),
            test=adapter.to_prediction(
                windowed=test_wd, raw_predictions=test_raw,
                run_id=self.cfg.run_id, duration_s=duration,
            ),
        )
```

Update the method signature line to `def _train_and_predict_neural(self, split) -> PartitionPredictions:`.

**(d)** Replace the `train`, `evaluate`, and `track_and_report` stage bodies in `run()` (lines 71–93) with:

```python
            with stage(self.logger, "train"):
                if cfg.backend == "classical":
                    predictions = self._train_and_predict_classical(split)
                else:
                    predictions = self._train_and_predict_neural(split)
            with stage(self.logger, "evaluate"):
                results = {}
                for part in ("train", "val", "test"):
                    results[part] = evaluate(
                        getattr(predictions, part),
                        cfg.evaluation,
                        dataset_id=dataset.metadata.name,
                        split_info={"strategy": cfg.split.strategy},
                        class_names=dataset.metadata.class_names,
                    )
                result = results["test"]
                self.logger.info(f"Evaluation metrics: {result.metrics}")
            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(dataset)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                for part, r in results.items():
                    self.tracker.log_results(
                        r, partition=(None if part == "test" else part)
                    )
                    self.tracker.log_confusion_matrix(
                        r, dataset.metadata.class_names, partition=part
                    )
                self._visualize(raw_dataset, dataset, split, predictions)
                self._persist_artifacts(dataset, result)
            self.tracker.finish("FINISHED")
            return result
```

- [ ] **Step 4: Replace `_visualize`**

Replace `_visualize` (lines 236–271) with:

```python
    def _visualize(
        self, raw_dataset, processed_dataset, split, predictions
    ) -> None:
        if not self.cfg.reporting.visualize:
            return
        from hand_gesture_recognition.reporting.visualization import (
            plot_predictions_on_signal,
            plot_raw_signal_by_class,
            plot_signal_data_flow,
            plot_signal_processing_comparison,
        )

        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        class_names = processed_dataset.metadata.class_names

        # raw signal per class (6 for EPN612; N for synthetic)
        for p in plot_raw_signal_by_class(raw_dataset, run_dir, class_names):
            self.tracker.log_artifact(p)

        # predictions on signal: validation + test
        for part, ds, pred in (
            ("val", split.val, predictions.val),
            ("test", split.test, predictions.test),
        ):
            p = os.path.join(run_dir, f"predictions_on_signal_{part}.png")
            plot_predictions_on_signal(ds, pred, self.cfg.windowing, class_names, path=p)
            self.tracker.log_artifact(p)

        # before/after signal processing (only when processing was applied)
        if self.cfg.signal_processing:
            p = os.path.join(run_dir, "signal_processing_comparison.png")
            plot_signal_processing_comparison(raw_dataset, processed_dataset, path=p)
            self.tracker.log_artifact(p)

        # existing 4-panel data flow (test partition)
        p = os.path.join(run_dir, "signal_data_flow.png")
        plot_signal_data_flow(
            raw_dataset,
            processed_dataset,
            split.test,
            predictions.test,
            self.cfg.windowing,
            class_names,
            path=p,
        )
        self.tracker.log_artifact(p)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/integration/test_partition_evaluation.py -q`
Expected: PASS (both classical and neural).

Then run the full suite: `python -m pytest -q`
Expected: all existing tests still pass (66 prior + new ones). If `mlflow.pytorch.autolog()` (user's line 55) interferes with the neural `trainer.predict` calls, surface it as a note rather than editing that line.

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_partition_evaluation.py
git commit -m feat-partition-evaluation-visualizations
```

---

## Self-Review Notes (for the implementer)

- Spec coverage: confusion matrices (Task 2 + 5), `predictions_on_signal` val/test (Task 5 `_visualize`), raw signal per class (Task 3 + 5), signal-processing before/after (Task 4 + 5), empty-CM guard (Task 2), tests (Tasks 1–5). All spec sections covered.
- The returned `run()` value stays the test `EvaluationResult` (Task 5d `result = results["test"]`), preserving `compare`/reporters/smoke tests.
- `_visualize` keeps `signal_data_flow.png` (test partition) as specified.
