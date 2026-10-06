# No-Gesture Class Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dataset.include_no_gesture` so the pipeline can run as a 5-class (gesture-only) problem, removing the `noGesture`/rest class from training, windowing, and evaluation.

**Architecture:** A post-load pure transform (`remove_no_gesture`) drops rest trials, remaps labels to contiguous `0..4`, and sets a `drop_rest_windows` flag on the dataset metadata. `window_dataset` reads that flag to *skip* rest windows instead of labeling them `0`, so every downstream stage (train, predict, confusion matrix, plots) operates only on gesture windows. A single config field drives it, defaulting to the current 6-class behavior.

**Tech Stack:** Python 3.10, Hydra (config), NumPy, scikit-learn, PyTorch Lightning, pytest.

## Global Constraints

- Default `include_no_gesture: true` preserves current 6-class behavior; all existing tests must stay green.
- Remapped labels are contiguous `0..4` (`fist→0, waveIn→1, waveOut→2, open→3, pinch→4`).
- Rest is identified **structurally** by `gesture_start < 0`, never by the string `"noGesture"` or the integer value `0`.
- The confusion matrix and every metric must be computed on the selected (reduced) data only — never the raw 6-class set.
- Fail loudly on ambiguous config (framework `ConfigValidationError`).
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: `remove_no_gesture` transform + `drop_rest_windows` metadata field

**Files:**
- Create: `src/hand_gesture_recognition/data/class_selection.py`
- Modify: `src/hand_gesture_recognition/data/datasets/base.py:10-18` (add field to `DatasetMetadata`)
- Test: `tests/unit/test_class_selection.py`

**Interfaces:**
- Produces: `remove_no_gesture(dataset: Dataset) -> Dataset` (pure; returns the same object unchanged when there are no rest trials).
- Produces: `DatasetMetadata.drop_rest_windows: bool = False` (consumed by Task 2's windowing).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_class_selection.py`:

```python
import numpy as np
import pandas as pd

from hand_gesture_recognition.data.class_selection import remove_no_gesture
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def _dataset():
    # 4 trials: fist(1), waveIn(2), rest(0), open(4)
    return Dataset(
        trials=[
            np.zeros((50, 2), dtype=np.float32),
            np.zeros((60, 2), dtype=np.float32),
            np.zeros((40, 2), dtype=np.float32),
            np.zeros((70, 2), dtype=np.float32),
        ],
        labels=np.array([1, 2, 0, 4]),
        subject_ids=np.array(["s1", "s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3", "t4"]),
        gesture_start=np.array([10, 20, -1, 30]),
        gesture_end=np.array([50, 60, -1, 70]),
        metadata=DatasetMetadata(
            name="epn612",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"],
            content_hash="h",
        ),
    )


def test_removes_rest_trials_and_remaps_labels():
    ds = remove_no_gesture(_dataset())
    assert ds.n_trials == 3
    assert ds.labels.tolist() == [0, 1, 2]  # fist->0, waveIn->1, open->2
    assert ds.subject_ids.tolist() == ["s1", "s1", "s2"]
    assert ds.trial_ids.tolist() == ["t1", "t2", "t4"]
    assert ds.gesture_start.tolist() == [10, 20, 30]
    assert ds.gesture_end.tolist() == [50, 60, 70]


def test_updates_class_names_and_signals_windowing():
    ds = remove_no_gesture(_dataset())
    assert ds.metadata.class_names == ["fist", "waveIn", "open"]
    assert ds.metadata.drop_rest_windows is True


def test_no_op_when_no_rest_trials():
    ds = _dataset()
    ds.labels = np.array([1, 2, 4, 4])
    ds.gesture_start = np.array([10, 20, 30, 30])
    ds.gesture_end = np.array([50, 60, 70, 70])
    out = remove_no_gesture(ds)
    assert out is ds


def test_filters_subject_info_to_remaining_subjects():
    ds = _dataset()
    ds.labels = np.array([1, 2, 0, 0])  # s2's only trial becomes rest
    ds.gesture_start = np.array([10, 20, -1, -1])
    ds.gesture_end = np.array([50, 60, -1, -1])
    ds.subject_info = pd.DataFrame({"age": [20, 21]}, index=pd.Index(["s1", "s2"]))
    out = remove_no_gesture(ds)
    assert list(out.subject_info.index) == ["s1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_class_selection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.class_selection'`

- [ ] **Step 3: Add the metadata field**

In `src/hand_gesture_recognition/data/datasets/base.py`, change `DatasetMetadata`:

```python
@dataclass
class DatasetMetadata:
    name: str
    version: str
    sampling_rate: float
    channels: List[str]
    class_names: List[str]
    content_hash: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    drop_rest_windows: bool = False  # windowing drops (not labels) rest windows
```

- [ ] **Step 4: Implement the transform**

Create `src/hand_gesture_recognition/data/class_selection.py`:

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def remove_no_gesture(dataset: Dataset) -> Dataset:
    """Remove the rest (no-gesture) class from a dataset.

    Rest trials are exactly those with ``gesture_start < 0``. They are dropped,
    the remaining labels are remapped to a contiguous ``0..N-1`` range (in
    ascending order of the original labels), ``metadata.class_names`` is reduced
    to the kept classes, and ``metadata.drop_rest_windows`` is set so windowing
    later *skips* (rather than labels) the rest windows outside each gesture's
    bounds. If there are no rest trials (e.g. the synthetic dataset), the
    dataset is returned unchanged.
    """
    keep = dataset.gesture_start >= 0
    if keep.all():
        return dataset

    kept_labels = np.unique(dataset.labels[keep])
    old_to_new = {int(old): new for new, old in enumerate(kept_labels)}
    new_labels = np.asarray(
        [old_to_new[int(l)] for l in dataset.labels[keep]], dtype=int
    )
    new_class_names = [dataset.metadata.class_names[int(o)] for o in kept_labels]

    remaining_subjects = set(str(s) for s in dataset.subject_ids[keep])
    subject_info = dataset.subject_info
    if subject_info is not None:
        subject_info = subject_info.loc[subject_info.index.isin(remaining_subjects)]

    return replace(
        dataset,
        trials=[t for t, k in zip(dataset.trials, keep) if k],
        labels=new_labels,
        subject_ids=dataset.subject_ids[keep],
        trial_ids=dataset.trial_ids[keep],
        gesture_start=dataset.gesture_start[keep],
        gesture_end=dataset.gesture_end[keep],
        subject_info=subject_info,
        metadata=replace(
            dataset.metadata,
            class_names=new_class_names,
            drop_rest_windows=True,
        ),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_class_selection.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/data/class_selection.py src/hand_gesture_recognition/data/datasets/base.py tests/unit/test_class_selection.py
git commit -m "feat: no-gesture removal transform and drop_rest_windows signal"
```

---

### Task 2: Windowing drops rest windows when signaled

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/base.py:37-101`
- Test: `tests/unit/test_windowing.py` (add one test)

**Interfaces:**
- Consumes: `DatasetMetadata.drop_rest_windows` (from Task 1).
- Produces: unchanged public API `window_dataset(dataset, config) -> WindowedDataset`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_windowing.py`:

```python
def test_drop_rest_windows_skips_rest():
    ds = Dataset(
        trials=[
            np.zeros((50, 2), dtype=np.float32),
            np.zeros((50, 2), dtype=np.float32),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([10, -1]),
        gesture_end=np.array([30, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["fist", "waveIn"],
            content_hash="h",
            drop_rest_windows=True,
        ),
    )
    wd = window_dataset(ds, WindowingConfig(window_size_samples=10, stride_samples=10))
    # trial 0 (gesture [10,30]): only the two overlapping windows survive
    # trial 1 (rest): fully dropped
    assert wd.y.tolist() == [1, 1]
    assert wd.X.shape == (2, 10, 2)
    assert wd.metadata.window_ids.tolist() == ["s1_t1_window1", "s1_t1_window2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_windowing.py::test_drop_rest_windows_skips_rest -v`
Expected: FAIL (rest windows are still emitted; `wd.y.tolist()` is 10 values, not `[1, 1]`)

- [ ] **Step 3: Implement the drop behavior**

In `src/hand_gesture_recognition/data/windowing/base.py`, after
`ws, stride = config.resolve(...)`, add the flag read:

```python
    ws, stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    drop_rest = dataset.metadata.drop_rest_windows
```

Then replace the per-window label block:

```python
        for i in range(n_windows):
            s = i * stride
            e = s + ws
            # A window is "rest" if it does not overlap the gesture segment
            # [g_start, g_end] (or the trial itself is a rest trial, g_start < 0).
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
            if drop_rest and is_rest:
                continue  # no-gesture class removed: skip rest windows entirely
            w_label = 0 if is_rest else int(label)
            xs.append(trial[s:e])
            ys.append(w_label)
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))
            wins.append(f"{subj}_{tid}_window{i}")
```

(Delete the old `if g_start < 0: ... elif ... elif ... else:` chain in favor of the `is_rest` form above.)

- [ ] **Step 4: Run the full windowing suite**

Run: `pytest tests/unit/test_windowing.py tests/unit/test_windowing_alignment.py -v`
Expected: PASS (new test + all existing tests, which exercise `drop_rest_windows=False` default behavior unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/base.py tests/unit/test_windowing.py
git commit -m "feat: windowing drops rest windows when drop_rest_windows is set"
```

---

### Task 3: `include_no_gesture` config field + validation

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py:74-83` (`DatasetConfig`), `:262-277` (`validate_experiment_config`)
- Modify: `configs/dataset/epn612.yaml`
- Test: `tests/unit/test_config_schema.py` (add two tests)

**Interfaces:**
- Produces: `DatasetConfig.include_no_gesture: bool = True` (consumed by Task 4's runner).
- Produces: validation error when `include_no_gesture=False` and `0 in evaluation.exclude_classes`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_schema.py`:

```python
def test_include_no_gesture_defaults_true():
    out = validate_experiment_config(_base_experiment())
    assert out.dataset.include_no_gesture is True


def test_include_no_gesture_false_rejects_exclude_classes_zero():
    cfg = _base_experiment()
    cfg.dataset.include_no_gesture = False
    cfg.evaluation.exclude_classes = [0]
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py::test_include_no_gesture_defaults_true tests/unit/test_config_schema.py::test_include_no_gesture_false_rejects_exclude_classes_zero -v`
Expected: FAIL (`DatasetConfig` has no `include_no_gesture`; second test does not raise)

- [ ] **Step 3: Add the config field**

In `src/hand_gesture_recognition/configuration/schema.py`, change `DatasetConfig`:

```python
@dataclass
class DatasetConfig:
    name: str = "epn612"
    path: str = "resources/data/EPN612"
    sampling_rate: float = 200.0
    include_no_gesture: bool = True
    loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.data.loaders.epn612.EPN612Loader"
        }
    )
```

- [ ] **Step 4: Add the validation**

In `validate_experiment_config` (before `return cfg`), add:

```python
    if not cfg.dataset.include_no_gesture and 0 in cfg.evaluation.exclude_classes:
        raise ConfigValidationError(
            "evaluation.exclude_classes=[0] is ambiguous in 5-class mode "
            "(label 0 is now 'fist'); remove it or set dataset.include_no_gesture=true"
        )
```

- [ ] **Step 5: Wire the YAML**

In `configs/dataset/epn612.yaml`, add one line after `sampling_rate: 200`:

```yaml
name: epn612
path: resources/data/EPN612
sampling_rate: 200
include_no_gesture: true   # false => 5-class (rest removed); pair with gesture_crop=none
loader:
  _target_: hand_gesture_recognition.data.loaders.epn612.EPN612Loader
  subjects: null
  max_train_users: 20
  max_test_users: 20
  user_seed: 42
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py configs/dataset/epn612.yaml tests/unit/test_config_schema.py
git commit -m "feat: include_no_gesture config field and validation"
```

---

### Task 4: Runner wiring, visualization fix, integration test, docs

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py:60-65` (apply transform), `:161-171` (`_window` warning)
- Modify: `src/hand_gesture_recognition/reporting/visualization.py:110-117` and `:307-313` (rest-trial heuristic)
- Create: `tests/integration/test_no_gesture_integration.py`
- Modify: `docs/experiments/running.md`, `docs/datasets/epn612.md`

**Interfaces:**
- Consumes: `DatasetConfig.include_no_gesture` (Task 3), `remove_no_gesture` (Task 1).

- [ ] **Step 1: Wire the transform into the runner**

In `src/hand_gesture_recognition/runner/run.py`, replace the load/analytics block:

```python
            with stage(self.logger, "load_dataset"):
                raw_dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
```

with:

```python
            with stage(self.logger, "load_dataset"):
                raw_dataset = self._load_dataset()
            if not self.cfg.dataset.include_no_gesture:
                from hand_gesture_recognition.data.class_selection import (
                    remove_no_gesture,
                )

                raw_dataset = remove_no_gesture(raw_dataset)
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
```

- [ ] **Step 2: Add the empty-partition warning**

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

- [ ] **Step 3: Fix the rest-trial heuristic in visualization**

In `src/hand_gesture_recognition/reporting/visualization.py`, two spots use
`int(...labels[i]) != 0` to "pick the first non-noGesture trial". Under 5-class
mode `0` is `fist`, so switch to the structural check `gesture_start >= 0`.

Spot 1 (`plot_predictions_on_signal`, ~line 112):

```python
        # pick the first gesture trial as a meaningful example (rest trials
        # have gesture_start < 0)
        for i in range(dataset.n_trials):
            if int(dataset.gesture_start[i]) >= 0:
                tid = str(dataset.trial_ids[i])
                break
        tid = tid or str(dataset.trial_ids[0])
```

Spot 2 (`plot_signal_processing_comparison`, ~line 309):

```python
        for i in range(raw_dataset.n_trials):
            if int(raw_dataset.gesture_start[i]) >= 0:
                tid = str(raw_dataset.trial_ids[i])
                break
        tid = tid or str(raw_dataset.trial_ids[0])
```

- [ ] **Step 4: Write the integration test**

Create `tests/integration/test_no_gesture_integration.py`:

```python
import json

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _write_user(root, subdir, user_id, n_samples=6, length=200):
    """Minimal EMG-EPN612-like user JSON with gesture segments and rest trials."""
    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "waveIn", "waveOut", "open", "pinch"]

    def make_samples(n):
        samples = {}
        for i in range(n):
            g = gestures[i % len(gestures)]
            s = {
                "emg": {
                    f"ch{c}": [float(c + j % 5) for j in range(length)]
                    for c in range(1, 9)
                },
                "gestureName": g,
            }
            if g != "noGesture":
                s["groundTruthIndex"] = [60, 140]
            samples[f"idx_{i + 1}"] = s
        return samples

    doc = {
        "generalInfo": {
            "samplingFrequencyInHertz": 200,
            "myoPredictionLabel": {
                "noGesture": 0, "fist": 1, "waveIn": 2,
                "waveOut": 3, "open": 4, "pinch": 5,
            },
        },
        "userInfo": {"name": f"user{user_id}", "age": 20 + user_id},
        "trainingSamples": make_samples(n_samples),
        "testingSamples": make_samples(n_samples),
    }
    (d / f"user{user_id}.json").write_text(json.dumps(doc))


def test_no_gesture_excluded_5_class_run(tmp_path, monkeypatch):
    for uid in range(1, 4):
        _write_user(tmp_path, "trainingJSON", uid)
    for uid in range(1, 4):
        _write_user(tmp_path, "testingJSON", uid)

    cfg = load_config(
        mode="compose",
        overrides=[
            "backend=classical",
            "split.strategy=random",
            "reporting.formats=[json]",
            "dataset.include_no_gesture=false",
            "gesture_crop=none",
        ],
    )
    cfg.dataset.path = str(tmp_path)
    cfg.dataset.loader["max_train_users"] = 3
    cfg.dataset.loader["max_test_users"] = 3
    cfg.evaluation.exclude_classes = []  # 5-class mode: 0 is now 'fist'
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))

    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (5, 5)
    assert "noGesture" not in result.per_class
    assert set(result.per_class.keys()) <= {"fist", "waveIn", "waveOut", "open", "pinch"}
```

- [ ] **Step 5: Run the integration test**

Run: `pytest tests/integration/test_no_gesture_integration.py -v`
Expected: PASS (completes; 5×5 confusion matrix; no `noGesture` in per-class)

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `pytest -q`
Expected: PASS (all tests green)

- [ ] **Step 7: Update docs**

In `docs/experiments/running.md`, add a row to the "Common overrides" table:

```markdown
| `dataset.include_no_gesture=false` | drop the `noGesture`/rest class: 5-class (gesture-only) pipeline |
```

and add an example to the Examples block:

```bash
# 5-class (no noGesture/rest class): classify only gesture-active windows
python -m hand_gesture_recognition --mode compose \
  dataset.include_no_gesture=false gesture_crop=none
```

In `docs/datasets/epn612.md`, after the "Gestures" paragraph (line ~69), add:

```markdown
To run a **5-class** (gesture-only) experiment that excludes `noGesture`, set
`dataset.include_no_gesture=false` (optionally with `gesture_crop=none`): the
rest class is removed before windowing, and the model is trained and scored only
on gesture-active windows.
```

- [ ] **Step 8: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py src/hand_gesture_recognition/reporting/visualization.py tests/integration/test_no_gesture_integration.py docs/experiments/running.md docs/datasets/epn612.md
git commit -m "feat: wire no-gesture removal into runner and visualization"
```

---

## Self-Review

- **Spec coverage:** config field (§1) → Task 3; metadata signal (§2) → Task 1; transform (§3) → Task 1; windowing drop (§4) → Task 2; runner ordering (§5) → Task 4; evaluation selected-data guarantee (§6) → invariant established by Tasks 1–2 + asserted in Task 4's integration test; validation (§7) → Task 3; visualization fix (§8) → Task 4.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test commands include expected outcomes.
- **Type consistency:** `drop_rest_windows` is introduced in Task 1 (`DatasetMetadata`) and consumed in Task 2 (`window_dataset`) with the same name; `include_no_gesture` is introduced in Task 3 and consumed in Task 4 with the same name; `remove_no_gesture(dataset: Dataset) -> Dataset` signature is consistent across Tasks 1 and 4.
