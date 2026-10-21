# Task Toggle (Classification vs Recognition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `task` switch (`classification | recognition`) so the pipeline can build **one sample per gesture** (classification) or keep the existing **sliding window** (recognition), both emitting the same `WindowedDataset` shape.

**Architecture:** A new `data/windowing/gesture.py::window_gestures` builds one padded/truncated window per gesture trial and drops rest trials. A `data/windowing/dispatch.py::build_windowed_dataset(dataset, config, task)` is the single place that maps a `task` name to a sample builder, shared by the runner and the prediction-overlay visualizations. A `TaskConfig` schema field + Hydra `task/` group select the mode; the runner branches on it, skips `gesture_crop` for classification, and always excludes the no-gesture class in classification mode.

**Tech Stack:** Python 3.10, Hydra (config), NumPy, scikit-learn, PyTorch Lightning, pytest.

## Global Constraints

- Default `task: recognition` preserves current behavior; all existing tests must stay green.
- `task.name` restricted to `("classification", "recognition")` via schema `SUPPORTED_TASKS`.
- Classification sample = `trial[gesture_start:gesture_end]`; zero-pad at the **end** if shorter than `window_size`; truncate to the **first** `window_size` samples if longer; **rest trials dropped**.
- Classification always excludes the no-gesture class; `dataset.include_no_gesture` is ignored in classification mode (a `UserWarning` is emitted, not an error).
- Rest is identified **structurally** by `gesture_start < 0`, never by name or integer value.
- Both modes return the same `WindowedDataset`; **no** changes to `Prediction`, evaluation, adapters, or split logic.
- Fail loudly on ambiguous config with `ConfigValidationError`.
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: `window_gestures` sample builder

**Files:**
- Create: `src/hand_gesture_recognition/data/windowing/gesture.py`
- Test: `tests/unit/test_gesture_windowing.py`

**Interfaces:**
- Produces: `window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset` (consumed by Task 2's dispatcher).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_gesture_windowing.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.gesture import window_gestures


def _dataset():
    # trial 0: gesture [10,30] in a 40-sample trial (label 1)
    # trial 1: gesture [0,60] in a 60-sample trial (label 2)
    # trial 2: rest (label 0)
    return Dataset(
        trials=[
            np.arange(40, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.arange(60, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.arange(30, dtype=np.float32)[:, None].repeat(2, axis=1),
        ],
        labels=np.array([1, 2, 0]),
        subject_ids=np.array(["s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        gesture_start=np.array([10, 0, -1]),
        gesture_end=np.array([30, 60, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist", "waveIn"],
            content_hash="h",
        ),
    )


def test_one_sample_per_gesture_trial():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    # rest trial dropped; two gesture trials -> two samples
    assert wd.X.shape == (2, 25, 2)
    assert wd.y.tolist() == [1, 2]
    assert wd.metadata.subject_ids.tolist() == ["s1", "s1"]
    assert wd.metadata.trial_ids.tolist() == ["t1", "t2"]


def test_short_gesture_is_zero_padded():
    # trial 0 gesture [10,30] = 20 samples < ws 25 -> padded to 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[0].shape[0] == 25
    assert np.all(wd.X[0, 20:] == 0)  # trailing padding is zeros


def test_long_gesture_is_truncated():
    # trial 1 gesture [0,60] = 60 samples > ws 25 -> truncated to first 25
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 25
    assert np.array_equal(wd.X[1], _dataset().trials[1][:25])


def test_exact_length_gesture_unchanged():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=60, stride_samples=10)
    )
    assert wd.X[1].shape[0] == 60
    assert np.array_equal(wd.X[1], _dataset().trials[1])


def test_metadata_carries_gesture_bounds_and_ids():
    wd = window_gestures(
        _dataset(), WindowingConfig(window_size_samples=25, stride_samples=10)
    )
    assert wd.metadata.window_start.tolist() == [10, 0]
    assert wd.metadata.window_end.tolist() == [30, 60]
    assert wd.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]


def test_all_rest_trials_returns_empty_dataset():
    ds = _dataset()
    ds.gesture_start = np.array([-1, -1, -1])
    ds.gesture_end = np.array([-1, -1, -1])
    wd = window_gestures(ds, WindowingConfig(window_size_samples=25, stride_samples=10))
    assert wd.X.shape == (0, 25, 2)
    assert wd.y.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_gesture_windowing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.windowing.gesture'`

- [ ] **Step 3: Write the implementation**

Create `src/hand_gesture_recognition/data/windowing/gesture.py`:

```python
from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowMetadata,
)


def window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    """Build one sample per gesture trial (classification formulation).

    Each gesture trial (``gesture_start >= 0``) contributes exactly one window:
    the segment ``trial[gesture_start:gesture_end]`` (the whole gesture, no
    context margin), zero-padded at the end if shorter than the window size and
    truncated to the first ``window_size`` samples if longer. Rest trials
    (``gesture_start < 0``) are dropped entirely. Returns the same
    ``WindowedDataset`` shape as ``window_dataset`` so all downstream stages
    (classical/neural backends, evaluation, reporting) are unchanged.
    """
    ws, _ = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    n_channels = (
        len(dataset.metadata.channels)
        if dataset.metadata.channels
        else (dataset.trials[0].shape[1] if dataset.trials else 0)
    )
    xs: List[np.ndarray] = []
    ys: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    subjects: List[str] = []
    trials: List[str] = []
    wins: List[str] = []

    for trial, label, subj, tid, g_start, g_end in zip(
        dataset.trials,
        dataset.labels,
        dataset.subject_ids,
        dataset.trial_ids,
        dataset.gesture_start,
        dataset.gesture_end,
    ):
        g_start = int(g_start)
        g_end = int(g_end)
        if g_start < 0:
            continue  # rest trial: dropped in classification mode
        seg = trial[g_start:g_end]
        if seg.shape[0] < ws:
            pad = np.zeros((ws - seg.shape[0], seg.shape[1]), dtype=seg.dtype)
            seg = np.vstack([seg, pad])
        elif seg.shape[0] > ws:
            seg = seg[:ws]
        xs.append(seg)
        ys.append(int(label))
        starts.append(g_start)
        ends.append(g_end)
        subjects.append(str(subj))
        trials.append(str(tid))
        wins.append(f"{subj}_{tid}_gesture")

    return WindowedDataset(
        X=np.stack(xs, axis=0)
        if xs
        else np.zeros((0, ws, n_channels), dtype=np.float32),
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
            window_ids=np.asarray(wins),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_gesture_windowing.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/gesture.py tests/unit/test_gesture_windowing.py
git commit -m "feat: window_gestures one-sample-per-gesture builder"
```

---

### Task 2: `build_windowed_dataset` dispatcher

**Files:**
- Create: `src/hand_gesture_recognition/data/windowing/dispatch.py`
- Test: `tests/unit/test_gesture_windowing.py` (append one test)

**Interfaces:**
- Consumes: `window_gestures` (Task 1), `window_dataset` (existing).
- Produces: `build_windowed_dataset(dataset: Dataset, config: WindowingConfig, task: str) -> WindowedDataset` (consumed by Tasks 4 and 5).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_gesture_windowing.py`:

```python
from hand_gesture_recognition.data.windowing.dispatch import build_windowed_dataset


def test_dispatcher_routes_by_task():
    ds = _dataset()
    cfg = WindowingConfig(window_size_samples=25, stride_samples=10)
    cls = build_windowed_dataset(ds, cfg, "classification")
    rec = build_windowed_dataset(ds, cfg, "recognition")
    # classification: one window per gesture trial (2), rest dropped
    assert cls.X.shape[0] == 2
    assert cls.metadata.window_ids.tolist() == ["s1_t1_gesture", "s1_t2_gesture"]
    # recognition: sliding windows over all trials (2 + 4 + 1 = 7)
    assert rec.X.shape[0] == 7
    assert rec.metadata.window_ids[0].endswith("_window0")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_gesture_windowing.py::test_dispatcher_routes_by_task -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.windowing.dispatch'`

- [ ] **Step 3: Write the implementation**

Create `src/hand_gesture_recognition/data/windowing/dispatch.py`:

```python
from __future__ import annotations

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    window_dataset,
)
from hand_gesture_recognition.data.windowing.gesture import window_gestures


def build_windowed_dataset(
    dataset: Dataset, config: WindowingConfig, task: str
) -> WindowedDataset:
    """Map a ``task`` name to the matching sample builder.

    ``classification`` segments each gesture into one window (``window_gestures``);
    anything else (``recognition``) slides windows over the continuous signal
    (``window_dataset``). This is the single source of truth shared by the runner
    and the prediction-overlay visualizations, so both re-window the same way the
    training pipeline did.
    """
    if task == "classification":
        return window_gestures(dataset, config)
    return window_dataset(dataset, config)
```

- [ ] **Step 4: Run the full windowing test file**

Run: `pytest tests/unit/test_gesture_windowing.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/dispatch.py tests/unit/test_gesture_windowing.py
git commit -m "feat: task-to-sample-builder dispatcher"
```

---

### Task 3: `TaskConfig` schema, Hydra group, validation, loader wiring

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/configuration/loader.py`
- Create: `configs/task/recognition.yaml`
- Create: `configs/task/classification.yaml`
- Modify: `configs/config.yaml`
- Test: `tests/unit/test_config_schema.py` (add three tests), `tests/unit/test_config_loader.py` (add two tests)

**Interfaces:**
- Produces: `TaskConfig(name: str = "recognition")` and `ExperimentConfig.task: TaskConfig` (consumed by Task 4).
- Produces: `SUPPORTED_TASKS = ("classification", "recognition")`.
- Produces: validation error on unknown `task.name`; `UserWarning` when `task=classification` and `include_no_gesture=true`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_schema.py`:

```python
def test_task_defaults_to_recognition():
    cfg = _base_experiment()
    out = validate_experiment_config(cfg)
    assert out.task.name == "recognition"


def test_task_rejects_unknown_value():
    cfg = _base_experiment()
    cfg.task.name = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_task_classification_warns_when_include_no_gesture():
    cfg = _base_experiment()
    cfg.task.name = "classification"
    cfg.dataset.include_no_gesture = True
    with pytest.warns(UserWarning):
        validate_experiment_config(cfg)
```

Append to `tests/unit/test_config_loader.py`:

```python
def test_compose_defaults_to_recognition_task():
    cfg = load_config(mode="compose")
    assert cfg.task.name == "recognition"


def test_compose_loads_classification_task():
    cfg = load_config(mode="compose", overrides=["task=classification"])
    assert cfg.task.name == "classification"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: FAIL — `AttributeError: 'ExperimentConfig' object has no attribute 'task'` (the `_base_experiment` tests), and `Hydra` `ConfigCompositionException` for `task=classification` (no `configs/task/` group).

- [ ] **Step 3: Add `SUPPORTED_TASKS` and `TaskConfig` to the schema**

In `src/hand_gesture_recognition/configuration/schema.py`, change the top import block and constants:

```python
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hand_gesture_recognition.configuration.errors import ConfigValidationError

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("classical", "neural")
SUPPORTED_TASKS = ("classification", "recognition")
```

Then add `TaskConfig` immediately before `ExperimentConfig` (after the `ReportingConfig` dataclass):

```python
@dataclass
class TaskConfig:
    name: str = "recognition"  # "classification" | "recognition"
```

- [ ] **Step 4: Add `task` field to `ExperimentConfig`**

In `ExperimentConfig`, add the field after the `windowing` line:

```python
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    signal_processing: List[Dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 5: Add validation**

In `validate_experiment_config`, immediately after the `cfg.backend` check, add:

```python
    if cfg.task.name not in SUPPORTED_TASKS:
        raise ConfigValidationError(
            f"task.name={cfg.task.name!r} not in {SUPPORTED_TASKS}"
        )
    if cfg.task.name == "classification" and cfg.dataset.include_no_gesture:
        warnings.warn(
            "task=classification always excludes the no-gesture class; "
            "dataset.include_no_gesture is ignored in classification mode.",
            UserWarning,
            stacklevel=2,
        )
```

- [ ] **Step 6: Wire `_coerce` in the loader**

In `src/hand_gesture_recognition/configuration/loader.py`, add `TaskConfig` to the import list (after `SplitConfig`):

```python
    SplitConfig,
    TaskConfig,
    TrackingConfig,
```

Then, in `_coerce`, add the `task` argument to the `ExperimentConfig(...)` constructor (after the `windowing=` line):

```python
        windowing=WindowingConfig(**d.get("windowing", {})),
        task=TaskConfig(**d.get("task", {})),
        signal_processing=list(d.get("signal_processing", [])),
```

- [ ] **Step 7: Create the Hydra group files**

Create `configs/task/recognition.yaml`:

```yaml
# @package task
name: recognition
```

Create `configs/task/classification.yaml`:

```yaml
# @package task
name: classification
```

- [ ] **Step 8: Register the group in `configs/config.yaml`**

In `configs/config.yaml`, add `- task: recognition` to the `defaults:` list (after the `- windowing: default` line):

```yaml
defaults:
  - dataset: epn612
  - split: subject_wise
  - windowing: default
  - task: recognition
  - signal_processing: emg
  ...
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS (all tests, including the 5 new ones)

- [ ] **Step 10: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py configs/task/recognition.yaml configs/task/classification.yaml configs/config.yaml tests/unit/test_config_schema.py tests/unit/test_config_loader.py
git commit -m "feat: task config group, schema, validation, and loader wiring"
```

---

### Task 4: Runner wiring + integration tests

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Create: `tests/integration/test_task_classification.py`

**Interfaces:**
- Consumes: `build_windowed_dataset` (Task 2), `ExperimentConfig.task` (Task 3), existing `build_no_gesture_remap` / `drop_no_gesture`.
- Produces: runner behavior — `_window` dispatches on `cfg.task.name`; `gesture_crop` skipped for classification; no-gesture drop forced for classification.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/integration/test_task_classification.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _classification_cfg(tmp_path, monkeypatch, backend):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "task=classification",
            "windowing.window_size_samples=40",
            f"backend={backend}",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    return cfg


def test_classical_classification_smoke(tmp_path, monkeypatch):
    runner = ExperimentRunner(_classification_cfg(tmp_path, monkeypatch, "classical"))
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)


def test_neural_classification_smoke(tmp_path, monkeypatch):
    cfg = _classification_cfg(tmp_path, monkeypatch, "neural")
    cfg.neural.trainer.max_epochs = 1
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_task_classification.py -v`
Expected: FAIL — the pipeline still uses `window_dataset` for every run, so `task=classification` has no effect (and the visualization overlay likely crashes on the misaligned prediction indexing, or the test fails its `(3, 3)` assertion).

- [ ] **Step 3: Skip `gesture_crop` for classification**

In `src/hand_gesture_recognition/runner/run.py::run`, replace:

```python
            with stage(self.logger, "gesture_crop"):
                cropped = self._gesture_crop(raw_dataset)
```

with:

```python
            with stage(self.logger, "gesture_crop"):
                if cfg.task.name == "recognition":
                    cropped = self._gesture_crop(raw_dataset)
                else:
                    cropped = raw_dataset
```

- [ ] **Step 4: Force no-gesture drop for classification (part 1 — remap)**

In `run`, replace:

```python
            class_names = cropped.metadata.class_names
            remap = None
            if not cfg.dataset.include_no_gesture:
                from hand_gesture_recognition.data.class_selection import (
                    build_no_gesture_remap,
                )

                remap, class_names = build_no_gesture_remap(cropped)
```

with:

```python
            class_names = cropped.metadata.class_names
            remap = None
            if self._should_drop_no_gesture():
                from hand_gesture_recognition.data.class_selection import (
                    build_no_gesture_remap,
                )

                remap, class_names = build_no_gesture_remap(cropped)
```

- [ ] **Step 5: Force no-gesture drop for classification (part 2 — partition transform)**

In `_run_partition`, replace:

```python
        if not self.cfg.dataset.include_no_gesture:
            from hand_gesture_recognition.data.class_selection import drop_no_gesture

            split = DataSplit(
                train=drop_no_gesture(split.train, remap, class_names),
                val=drop_no_gesture(split.val, remap, class_names),
                test=drop_no_gesture(split.test, remap, class_names),
            )
```

with:

```python
        if self._should_drop_no_gesture():
            from hand_gesture_recognition.data.class_selection import drop_no_gesture

            split = DataSplit(
                train=drop_no_gesture(split.train, remap, class_names),
                val=drop_no_gesture(split.val, remap, class_names),
                test=drop_no_gesture(split.test, remap, class_names),
            )
```

- [ ] **Step 6: Add the `_should_drop_no_gesture` helper**

Add this method to the `ExperimentRunner` class (directly after `_gesture_crop`):

```python
    def _should_drop_no_gesture(self) -> bool:
        return (not self.cfg.dataset.include_no_gesture) or (
            self.cfg.task.name == "classification"
        )
```

- [ ] **Step 7: Dispatch windowing on `task` in `_window`**

Replace the `_window` method body:

```python
    def _window(self, split) -> WindowedPartitions:
        from hand_gesture_recognition.data.windowing.base import (
            WindowedPartitions,
            window_dataset,
        )

        windowed = WindowedPartitions(
            train=window_dataset(split.train, self.cfg.windowing),
            val=window_dataset(split.val, self.cfg.windowing),
            test=window_dataset(split.test, self.cfg.windowing),
        )
        for part in ("train", "val", "test"):
            if getattr(windowed, part).X.shape[0] == 0:
                self.logger.warning(f"partition '{part}' has 0 windows after windowing")
        return windowed
```

with:

```python
    def _window(self, split) -> WindowedPartitions:
        from hand_gesture_recognition.data.windowing.base import WindowedPartitions
        from hand_gesture_recognition.data.windowing.dispatch import (
            build_windowed_dataset,
        )

        windowed = WindowedPartitions(
            train=build_windowed_dataset(
                split.train, self.cfg.windowing, self.cfg.task.name
            ),
            val=build_windowed_dataset(
                split.val, self.cfg.windowing, self.cfg.task.name
            ),
            test=build_windowed_dataset(
                split.test, self.cfg.windowing, self.cfg.task.name
            ),
        )
        for part in ("train", "val", "test"):
            if getattr(windowed, part).X.shape[0] == 0:
                self.logger.warning(f"partition '{part}' has 0 windows after windowing")
        return windowed
```

- [ ] **Step 8: Run the integration tests**

Run: `pytest tests/integration/test_task_classification.py -v`
Expected: PASS (2 tests)

- [ ] **Step 9: Run the full suite to confirm no regressions**

Run: `pytest -q`
Expected: PASS (all existing tests green — default `task=recognition` path unchanged)

- [ ] **Step 10: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_task_classification.py
git commit -m "feat: wire task switch into runner (classification vs recognition)"
```

---

### Task 5: Task-aware visualization re-windowing

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Modify: `src/hand_gesture_recognition/runner/run.py` (`_visualize` call sites)
- Test: `tests/unit/test_visualization.py` (add two tests)

**Interfaces:**
- Consumes: `build_windowed_dataset` (Task 2).
- Produces: `plot_predictions_on_signal(..., task="recognition")` and `plot_signal_data_flow(..., task="recognition")` — both now re-window with the same builder the pipeline used.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_visualization.py`:

```python
import numpy as np

from hand_gesture_recognition.classical.adapters.sklearn import (
    SklearnPredictionAdapter,
)
from hand_gesture_recognition.data.windowing.gesture import window_gestures
from hand_gesture_recognition.reporting.visualization import (
    plot_predictions_on_signal,
    plot_signal_data_flow,
)


def test_plot_predictions_on_signal_classification_task(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=2, n_classes=2, length=50, n_channels=2, seed=0
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_gestures(ds, windowing)
    pred = SklearnPredictionAdapter().to_prediction(
        windowed=wd,
        y_pred=wd.y.copy(),
        y_score=None,
        run_id="test",
        duration_s=0.0,
    )
    path = str(tmp_path / "pred_on_signal.png")
    fig = plot_predictions_on_signal(
        ds, pred, windowing, ds.metadata.class_names, path=path, task="classification"
    )
    assert fig is not None
    assert os.path.exists(path)


def test_plot_signal_data_flow_classification_task(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=2, n_classes=2, length=50, n_channels=2, seed=0
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_gestures(ds, windowing)
    pred = SklearnPredictionAdapter().to_prediction(
        windowed=wd,
        y_pred=wd.y.copy(),
        y_score=np.eye(2)[wd.y],
        run_id="test",
        duration_s=0.0,
    )
    path = str(tmp_path / "data_flow.png")
    fig = plot_signal_data_flow(
        ds, ds, ds, pred, windowing, ds.metadata.class_names,
        path=path, task="classification",
    )
    assert fig is not None
    assert os.path.exists(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_visualization.py::test_plot_predictions_on_signal_classification_task tests/unit/test_visualization.py::test_plot_signal_data_flow_classification_task -v`
Expected: FAIL with an indexing error (`boolean index did not match indexed array`) — the functions still re-window with the sliding `window_dataset`, so `prediction.y_pred[mask]` (length 4) is indexed by a mask of length 8.

- [ ] **Step 3: Add the dispatcher import**

In `src/hand_gesture_recognition/reporting/visualization.py`, replace:

```python
from hand_gesture_recognition.data.windowing.base import window_dataset
```

with:

```python
from hand_gesture_recognition.data.windowing.base import window_dataset
from hand_gesture_recognition.data.windowing.dispatch import build_windowed_dataset
```

- [ ] **Step 4: Make `plot_predictions_on_signal` task-aware**

Change its signature (append `task` after `path`):

```python
def plot_predictions_on_signal(
    dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
```

and change the re-windowing line from:

```python
    test_wd = window_dataset(dataset, windowing)
```

to:

```python
    test_wd = build_windowed_dataset(dataset, windowing, task)
```

- [ ] **Step 5: Make `plot_signal_data_flow` task-aware**

Change its signature (append `task` after `path`):

```python
def plot_signal_data_flow(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    test_dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
```

and change the re-windowing line from:

```python
    test_wd = window_dataset(test_dataset, windowing)
```

to:

```python
    test_wd = build_windowed_dataset(test_dataset, windowing, task)
```

- [ ] **Step 6: Pass `task` from the runner**

In `src/hand_gesture_recognition/runner/run.py::_visualize`, update the two call sites.

First (`plot_predictions_on_signal`):

```python
                plot_predictions_on_signal(
                    ds, pred, self.cfg.windowing, class_names, path=p
                )
```

becomes:

```python
                plot_predictions_on_signal(
                    ds,
                    pred,
                    self.cfg.windowing,
                    class_names,
                    path=p,
                    task=self.cfg.task.name,
                )
```

Second (`plot_signal_data_flow`):

```python
            plot_signal_data_flow(
                raw_dataset,
                processed_dataset,
                split.test,
                predictions.test,
                self.cfg.windowing,
                class_names,
                path=p,
            )
```

becomes:

```python
            plot_signal_data_flow(
                raw_dataset,
                processed_dataset,
                split.test,
                predictions.test,
                self.cfg.windowing,
                class_names,
                path=p,
                task=self.cfg.task.name,
            )
```

- [ ] **Step 7: Run the visualization tests**

Run: `pytest tests/unit/test_visualization.py -v`
Expected: PASS (existing tests + the 2 new ones)

- [ ] **Step 8: Run the integration tests again**

Run: `pytest tests/integration/test_task_classification.py -v`
Expected: PASS (now that the overlay re-windowing matches, the classification runs complete without crashing)

- [ ] **Step 9: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py src/hand_gesture_recognition/runner/run.py tests/unit/test_visualization.py
git commit -m "feat: task-aware re-windowing in prediction overlay visualizations"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/experiments/configuration.md`
- Modify: `docs/experiments/running.md`
- Modify: `docs/architecture/overview.md`

**Interfaces:**
- None (docs only). Documents the `task` config group and its two modes.

- [ ] **Step 1: Document the `task` group in `configuration.md`**

In `docs/experiments/configuration.md`, add `- task: recognition` to the defaults list (after the `- dataset: epn612` line):

```yaml
defaults:
  - dataset: epn612
  - task: recognition
  - split: subject_wise
  ...
```

Then add a new section immediately before `### split`:

```markdown
### `task`

| Field | Values | Default |
|---|---|---|
| `name` | `classification` \| `recognition` | `recognition` |

Selects the problem formulation:

- **`recognition`** — the continuous-stream, sliding-window pipeline
  (`window_dataset`): windows slide over each trial, labeled `0` (rest) or the
  gesture class. This is the default and matches historical behavior.
- **`classification`** — the dataset is treated as pre-cut clips with one
  gesture per trial: `window_gestures` slices `[gesture_start, gesture_end]` into
  exactly one sample per trial, pad/truncates it to `window_size`, and drops rest
  trials. In this mode the no-gesture class is always excluded (so
  `dataset.include_no_gesture` is ignored), and `gesture_crop` is not applied.

```bash
# classification: one sample per gesture
python -m hand_gesture_recognition --mode compose task=classification dataset.name=synthetic
```
```

- [ ] **Step 2: Add the override + example to `running.md`**

In `docs/experiments/running.md`, add a row to the "Common overrides" table (after the `dataset.include_no_gesture` row):

```markdown
| `task=classification` / `task=recognition` | one-sample-per-gesture vs sliding-window pipeline |
```

and add an example to the Examples block:

```bash
# Classification: each sample is one whole gesture (rest dropped)
python -m hand_gesture_recognition --mode compose task=classification dataset.name=synthetic
```

- [ ] **Step 3: Update the architecture overview**

In `docs/architecture/overview.md`, update step 5 ("Windowing") of the data-flow list to mention the dispatch:

```markdown
5. **Windowing** — a single shared sample-builder dispatcher
   (`build_windowed_dataset`) selects the strategy from `cfg.task.name`:
   `recognition` slides windows over each partition (assigning per-window labels
   `0` = rest, `1..N` = gesture from the gesture bounds); `classification` emits
   one padded/truncated sample per gesture trial (`window_gestures`) and drops
   rest. Windows stay in **temporal order** (no window shuffling).
```

and add a bullet to "Key design decisions":

```markdown
6. **`task` toggle** — `classification` (one sample per gesture) vs `recognition`
   (sliding window) is a first-class config switch; both modes emit the same
   `WindowedDataset`, so every downstream stage is shared.
```

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/configuration.md docs/experiments/running.md docs/architecture/overview.md
git commit -m "docs: document the task (classification vs recognition) toggle"
```

---

## Self-Review

- **Spec coverage:** §1 config (`TaskConfig`, group, validation, `_coerce`) → Task 3; §2 `window_gestures` → Task 1; §3 dispatcher → Task 2; §4 runner wiring + `gesture_crop` skip → Task 4; §5 rest/no-gesture semantics → Task 4 (steps 4–6); §6 evaluation/downstream unchanged → enforced by the shared `WindowedDataset` shape across Tasks 1–5 (no edits needed); §7 task-aware re-windowing → Task 5; §8 dataset-agnosticism → inherent (rest identified structurally; no EPN612 references) and asserted via the synthetic integration test in Task 4.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test commands include expected outcomes.
- **Type consistency:** `window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset` is defined in Task 1 and consumed in Tasks 2 and 5 with the same signature. `build_windowed_dataset(dataset, config, task) -> WindowedDataset` is defined in Task 2 and consumed in Tasks 4 and 5 with the same signature. `TaskConfig.name` and `ExperimentConfig.task` are defined in Task 3 and consumed in Task 4 with the same names. `SUPPORTED_TASKS` is defined in Task 3 and used only there. `_should_drop_no_gesture()` is defined in Task 4 step 6 and called in Task 4 steps 4–5. `task="recognition"` keyword default is used consistently in Tasks 5's two plot signatures and matches `TaskConfig`'s default value.
