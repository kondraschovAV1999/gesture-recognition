# Global Baseline Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-free global baseline removal: subtract the per-channel mean of the train partition's `noGesture` trials from all partitions, as the first preprocessing step before `signal_processing`.

**Architecture:** A learned `BaselineRemoval` transformer is fit on the train split's `noGesture` trials only and applied to train/val/test. To make baseline "first" while staying fit-on-train, the runner reorders to `split → baseline → class_selection → signal_processing → windowing`, and `remove_no_gesture` is refactored into a global remap (`build_no_gesture_remap`) + per-partition apply (`drop_no_gesture`).

**Tech Stack:** Python 3.10, Hydra, NumPy, scikit-learn (`BaseEstimator`), PyTorch Lightning, pytest.

## Global Constraints

- Baseline is estimated from the **train partition's `noGesture` trials only** (`gesture_start < 0`); val/test are never used in `fit`.
- Baseline is a **global per-channel mean** (shape `(channels,)`), subtracted from every trial of train/val/test.
- Baseline runs **first**, on the raw signal, before `signal_processing` (bandpass → notch → rectify).
- `noGesture` trials are usable for baseline estimation **regardless of `include_no_gesture`**; they are dropped from the model only afterwards when `include_no_gesture=false`.
- **No validation couples `baseline.enabled` to `include_no_gesture`.**
- Default `baseline.enabled=false` preserves current behavior; the existing 111-test suite stays green.
- Rest is identified structurally by `gesture_start < 0`, never by label value or name.

---

### Task 1: `baseline` config plumbing

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py` (add `BaselineConfig` + field)
- Modify: `src/hand_gesture_recognition/configuration/loader.py:34-58` (`_coerce`)
- Create: `configs/baseline/none.yaml`, `configs/baseline/default.yaml`
- Modify: `configs/config.yaml:1-12` (defaults)
- Test: `tests/unit/test_config_schema.py`

**Interfaces:**
- Produces: `BaselineConfig(enabled: bool = False)` and `ExperimentConfig.baseline: BaselineConfig`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_schema.py`:

```python
def test_baseline_defaults_disabled():
    out = validate_experiment_config(_base_experiment())
    assert out.baseline.enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config_schema.py::test_baseline_defaults_disabled -v`
Expected: FAIL (`ExperimentConfig` has no attribute `baseline`)

- [ ] **Step 3: Add `BaselineConfig` and the field**

In `src/hand_gesture_recognition/configuration/schema.py`, add after `GestureCropConfig` (line ~71):

```python
@dataclass
class BaselineConfig:
    enabled: bool = False
```

Add to `ExperimentConfig` (after the `gesture_crop` field, ~line 253):

```python
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
```

- [ ] **Step 4: Wire `_coerce`**

In `src/hand_gesture_recognition/configuration/loader.py`, add to the `_coerce` return (after the `gesture_crop=` line) and to the import list:

```python
from hand_gesture_recognition.configuration.schema import (
    AnalyticsConfig,
    BaselineConfig,
    ...
)
```

and in the `return ExperimentConfig(...)`:

```python
        gesture_crop=GestureCropConfig(**(d.get("gesture_crop") or {})),
        baseline=BaselineConfig(**(d.get("baseline") or {})),
```

- [ ] **Step 5: Create the config group**

Create `configs/baseline/none.yaml`:

```yaml
# @package baseline
enabled: false
```

Create `configs/baseline/default.yaml`:

```yaml
# @package baseline
enabled: true
```

In `configs/config.yaml`, add `- baseline: none` to the `defaults` list (after `- gesture_crop: none`):

```yaml
defaults:
  - dataset: epn612
  - split: subject_wise
  - windowing: default
  - signal_processing: emg
  - gesture_crop: none
  - baseline: none
  - classical: random_forest
  - neural: lstm
  - evaluation: basic
  - analytics: basic
  - experiment: default
  - _self_
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py configs/baseline configs/config.yaml tests/unit/test_config_schema.py
git commit -m "feat: baseline config group and BaselineConfig"
```

---

### Task 2: `BaselineRemoval` transformer

**Files:**
- Create: `src/hand_gesture_recognition/preprocessing/baseline.py`
- Test: `tests/unit/test_baseline.py`

**Interfaces:**
- Produces: `BaselineRemoval().fit(train: Dataset) -> BaselineRemoval` (sets `baseline_`, shape `(channels,)`); `transform(dataset: Dataset) -> Dataset` (subtracts `baseline_` from every trial).
- Consumes: `Dataset` (from `hand_gesture_recognition.data.datasets.base`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_baseline.py`:

```python
import numpy as np
import pytest

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval


def _ds():
    # 3 trials: two gesture, one rest (gesture_start < 0)
    return Dataset(
        trials=[
            np.ones((50, 2), dtype=np.float32) * 10.0,   # gesture
            np.ones((60, 2), dtype=np.float32) * 30.0,   # gesture
            np.ones((40, 2), dtype=np.float32) * 20.0,   # rest (baseline source)
        ],
        labels=np.array([1, 2, 0]),
        subject_ids=np.array(["s1", "s1", "s1"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        gesture_start=np.array([10, 20, -1]),
        gesture_end=np.array([50, 60, -1]),
        metadata=DatasetMetadata(
            name="x", version="0", sampling_rate=200.0,
            channels=["c1", "c2"], class_names=["noGesture", "fist", "open"],
            content_hash="h",
        ),
    )


def test_fit_uses_only_rest_trials():
    r = BaselineRemoval().fit(_ds())
    # rest trial is a constant 20.0 per channel -> baseline_ == [20, 20]
    assert r.baseline_.tolist() == [20.0, 20.0]


def test_transform_subtracts_baseline():
    r = BaselineRemoval().fit(_ds())
    out = r.transform(_ds())
    # gesture trial 1 (constant 10) becomes -10
    np.testing.assert_allclose(out.trials[0], -10.0)
    # rest trial (constant 20) becomes 0
    np.testing.assert_allclose(out.trials[2], 0.0)
    # non-signal fields preserved
    assert out.labels.tolist() == [1, 2, 0]
    assert out.trial_ids.tolist() == ["t1", "t2", "t3"]


def test_fit_requires_rest_trials():
    ds = _ds()
    ds.gesture_start = np.array([10, 20, 30])  # no rest trials
    with pytest.raises(ValueError):
        BaselineRemoval().fit(ds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.preprocessing.baseline'`

- [ ] **Step 3: Implement the transformer**

Create `src/hand_gesture_recognition/preprocessing/baseline.py`:

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator

from hand_gesture_recognition.data.datasets.base import Dataset


class BaselineRemoval(BaseEstimator):
    """Subtract the global per-channel mean rest level.

    ``fit`` estimates the baseline from the *train partition's* noGesture
    trials only (``gesture_start < 0``); ``transform`` subtracts the frozen
    per-channel mean from every trial of any partition.
    """

    def __init__(self):
        self.baseline_ = None

    def fit(self, train: Dataset) -> "BaselineRemoval":
        rest = [t for t, gs in zip(train.trials, train.gesture_start) if int(gs) < 0]
        if not rest:
            raise ValueError(
                "baseline removal requires noGesture (rest) trials in the train partition"
            )
        self.baseline_ = np.mean(np.concatenate(rest, axis=0), axis=0).astype(
            np.float32
        )
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if self.baseline_ is None:
            raise ValueError("BaselineRemoval must be fit before transform")
        trials = [t.astype(np.float32) - self.baseline_ for t in dataset.trials]
        return replace(dataset, trials=trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_baseline.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/preprocessing/baseline.py tests/unit/test_baseline.py
git commit -m "feat: BaselineRemoval transformer (fit on train rest, global per-channel mean)"
```

---

### Task 3: `remove_no_gesture` refactor (global remap + per-partition apply)

**Files:**
- Modify: `src/hand_gesture_recognition/data/class_selection.py`
- Test: `tests/unit/test_class_selection.py`

**Interfaces:**
- Produces: `build_no_gesture_remap(dataset: Dataset) -> tuple[dict[int, int], list[str]]` (global remap + reduced class_names, computed once).
- Produces: `drop_no_gesture(dataset: Dataset, remap: dict, new_class_names: list[str]) -> Dataset` (per-partition: drop rest trials, remap labels via the global map).
- Produces: `remove_no_gesture(dataset: Dataset) -> Dataset` (convenience wrapper: build + drop on one dataset).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_class_selection.py`:

```python
def test_build_remap_and_drop_apply_globally():
    full = _dataset()  # class_names: ["noGesture","fist","waveIn","waveOut","open","pinch"]
    remap, names = build_no_gesture_remap(full)
    assert names == ["fist", "waveIn", "open"]
    # a partition MISSING a class (no 'open') must still map consistently
    part = _dataset()
    part.labels = np.array([1, 2, 0, 0])           # only fist/waveIn/rest present
    part.gesture_start = np.array([10, 20, -1, -1])
    part.gesture_end = np.array([50, 60, -1, -1])
    out = drop_no_gesture(part, remap, names)
    assert out.labels.tolist() == [0, 1]           # fist->0, waveIn->1 (global map)
    assert out.metadata.class_names == names
    assert out.metadata.drop_rest_windows is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_class_selection.py::test_build_remap_and_drop_apply_globally -v`
Expected: FAIL (`build_no_gesture_remap` not defined)

- [ ] **Step 3: Refactor `class_selection.py`**

Replace the body of `src/hand_gesture_recognition/data/class_selection.py` with:

```python
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def build_no_gesture_remap(dataset: Dataset) -> Tuple[Dict[int, int], List[str]]:
    """Compute the global label remap and reduced class_names (no noGesture).

    Computed ONCE from the full (pre-split) dataset so every partition applies
    the same mapping, even if a partition is missing a class.
    """
    keep = dataset.gesture_start >= 0
    kept_labels = np.unique(dataset.labels[keep])
    remap = {int(old): new for new, old in enumerate(kept_labels)}
    new_class_names = [dataset.metadata.class_names[int(o)] for o in kept_labels]
    return remap, new_class_names


def drop_no_gesture(
    dataset: Dataset, remap: Dict[int, int], new_class_names: List[str]
) -> Dataset:
    """Drop rest trials and remap labels using a GLOBAL remap (per-partition)."""
    keep = dataset.gesture_start >= 0
    new_labels = np.asarray(
        [remap[int(l)] for l in dataset.labels[keep]], dtype=int
    )
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


def remove_no_gesture(dataset: Dataset) -> Dataset:
    """Convenience: build the global remap and apply it to a single dataset."""
    remap, new_class_names = build_no_gesture_remap(dataset)
    return drop_no_gesture(dataset, remap, new_class_names)
```

- [ ] **Step 4: Run the full class-selection suite**

Run: `pytest tests/unit/test_class_selection.py -v`
Expected: PASS (existing tests + new test; `remove_no_gesture` wrapper keeps the old behavior)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/class_selection.py tests/unit/test_class_selection.py
git commit -m "feat: split class selection into global remap + per-partition drop"
```

---

### Task 4: Runner restructure

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/integration/test_baseline_integration.py` (create)

**Interfaces:**
- Consumes: `BaselineRemoval` (Task 2), `build_no_gesture_remap`/`drop_no_gesture` (Task 3), `DataSplit` (from `hand_gesture_recognition.data.splits.base`), `cfg.baseline.enabled` (Task 1).

- [ ] **Step 1: Update imports**

In `src/hand_gesture_recognition/runner/run.py`, change the splits import (line 16-19):

```python
from hand_gesture_recognition.data.splits.base import (
    DataSplit,
    assert_no_group_leakage,
    split_dataset,
)
```

- [ ] **Step 2: Rewrite the `run()` pipeline body**

Replace lines 61-114 (from `try:` through the `track_and_report` stage, keeping the `except` block) with:

```python
        try:
            with stage(self.logger, "load_dataset"):
                raw_dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
            with stage(self.logger, "gesture_crop"):
                cropped = self._gesture_crop(raw_dataset)
            with stage(self.logger, "split"):
                split = split_dataset(cropped, cfg.split)
                if cfg.split.strategy in ("subject_wise", "session_wise", "predefined"):
                    assert_no_group_leakage(split)
            # class set is tracked explicitly: class selection may reduce it.
            class_names = cropped.metadata.class_names
            baseline_remover = None
            if cfg.baseline.enabled:
                with stage(self.logger, "baseline"):
                    from hand_gesture_recognition.preprocessing.baseline import (
                        BaselineRemoval,
                    )

                    baseline_remover = BaselineRemoval().fit(split.train)
                    split = DataSplit(
                        train=baseline_remover.transform(split.train),
                        val=baseline_remover.transform(split.val),
                        test=baseline_remover.transform(split.test),
                    )
            if not cfg.dataset.include_no_gesture:
                with stage(self.logger, "class_selection"):
                    from hand_gesture_recognition.data.class_selection import (
                        build_no_gesture_remap,
                        drop_no_gesture,
                    )

                    remap, class_names = build_no_gesture_remap(cropped)
                    split = DataSplit(
                        train=drop_no_gesture(split.train, remap, class_names),
                        val=drop_no_gesture(split.val, remap, class_names),
                        test=drop_no_gesture(split.test, remap, class_names),
                    )
            with stage(self.logger, "signal_processing"):
                split = self._signal_processing(split)
            # reconstruct a processed dataset (frozen baseline + signal processing)
            # for visualization/tracking display only.
            processed_dataset = self._apply_signal_processing(cropped)
            if baseline_remover is not None:
                processed_dataset = baseline_remover.transform(processed_dataset)
            with stage(self.logger, "windowing"):
                windowed = self._window(split)
            with stage(self.logger, "train"):
                if cfg.backend == "classical":
                    predictions = self._train_and_predict_classical(windowed)
                else:
                    predictions = self._train_and_predict_neural(
                        windowed, n_classes=len(class_names)
                    )
            with stage(self.logger, "evaluate"):
                results = {}
                for part in ("train", "val", "test"):
                    results[part] = evaluate(
                        getattr(predictions, part),
                        cfg.evaluation,
                        dataset_id=cropped.metadata.name,
                        split_info={"strategy": cfg.split.strategy},
                        class_names=class_names,
                    )
                result = results["test"]
                self.logger.info(f"Evaluation metrics: {result.metrics}")
            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(cropped)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                for part, r in results.items():
                    self.tracker.log_results(
                        r, partition=(None if part == "test" else part)
                    )
                    self.tracker.log_confusion_matrix(
                        r, class_names, partition=part
                    )
                self._visualize(
                    cropped, processed_dataset, split, predictions, class_names
                )
                self._persist_artifacts(cropped, result)
            self.tracker.finish("FINISHED")
            return result
        except Exception:
            self.tracker.finish("FAILED")
            raise
```

- [ ] **Step 3: Replace `_signal_processing` with per-partition + display variants**

Replace the current `_signal_processing` method (lines 130-140) with:

```python
    def _apply_signal_steps(self, trials):
        if not self.cfg.signal_processing:
            return trials
        steps = [instantiate(step_cfg) for step_cfg in self.cfg.signal_processing]
        for step in steps:
            trials = [step.transform(t) for t in trials]
        return trials

    def _apply_signal_processing(self, dataset):
        return replace(dataset, trials=self._apply_signal_steps(dataset.trials))

    def _signal_processing(self, split):
        # Deterministic per-trial steps; applied per-partition after baseline /
        # class selection. Equivalent to the old pre-split application.
        return DataSplit(
            train=replace(
                split.train, trials=self._apply_signal_steps(split.train.trials)
            ),
            val=replace(split.val, trials=self._apply_signal_steps(split.val.trials)),
            test=replace(split.test, trials=self._apply_signal_steps(split.test.trials)),
        )
```

- [ ] **Step 4: Update `_visualize` to take `class_names`**

Change the `_visualize` signature (line 331) and its internal `class_names` line (line 344):

```python
    def _visualize(
        self, raw_dataset, processed_dataset, split, predictions, class_names
    ) -> None:
        if not self.cfg.reporting.visualize:
            return
        ...
        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
```

and change the `plot_raw_signal_by_class` call (line 347) to use the raw dataset's own class names (so `noGesture` is not mislabeled in 5-class mode):

```python
        for p in plot_raw_signal_by_class(raw_dataset, run_dir, None):
            self.tracker.log_artifact(p)
```

(Leave the `predictions_on_signal`, `signal_processing_comparison`, `signal_data_flow`, and `fft_spectrum` calls unchanged — they already receive `class_names`/`processed_dataset` correctly.)

- [ ] **Step 5: Write the integration test**

Create `tests/integration/test_baseline_integration.py`:

```python
import json

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def _write_user(root, subdir, user_id, n_samples=6, length=200):
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


def test_baseline_5_class_run(tmp_path, monkeypatch):
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
            "baseline=default",
            "dataset.include_no_gesture=false",
            "gesture_crop=none",
        ],
    )
    cfg.dataset.path = str(tmp_path)
    cfg.dataset.loader["max_train_users"] = 3
    cfg.dataset.loader["max_test_users"] = 3
    cfg.evaluation.exclude_classes = []
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))

    result = ExperimentRunner(cfg).run()

    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (5, 5)
    assert "noGesture" not in result.per_class
```

- [ ] **Step 6: Run the integration test + full suite**

Run: `pytest tests/integration/test_baseline_integration.py -v` then `pytest -q`
Expected: PASS (new integration test + all 111 existing tests green)

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_baseline_integration.py
git commit -m "feat: reorder runner to split→baseline→class_selection→signal_processing"
```

---

## Self-Review

- **Spec coverage:** config group (§1) → Task 1; `BaselineRemoval` (§2) → Task 2; runner restructure (§3) → Task 4; `remove_no_gesture` refactor (§4) → Task 3; orthogonality (§5) → Task 4 (no guard added); edge cases (no rest → error; global remap consistency) → Tasks 2 & 3.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test commands include expected outcomes.
- **Type consistency:** `build_no_gesture_remap -> tuple[dict[int,int], list[str]]` and `drop_no_gesture(dataset, remap, new_class_names)` are defined in Task 3 and consumed identically in Task 4; `BaselineRemoval().fit(train) -> self` / `transform(dataset) -> Dataset` defined in Task 2 and consumed in Task 4; `BaselineConfig.enabled` defined in Task 1 and consumed in Task 4.
