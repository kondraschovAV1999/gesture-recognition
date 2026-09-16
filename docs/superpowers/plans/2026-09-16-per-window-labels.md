# Per-Window Gesture Labels + Full Classical Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace whole-trial window labels with per-window labels derived from each trial's gesture ground-truth bounds (`groundTruthIndex` `[start, end]`), and add a full end-to-end classical-model test on a small subset of the real EMG-EPN612 data.

**Architecture:** `Dataset` gains per-trial `gesture_start`/`gesture_end` arrays; the EPN612 loader extracts them from `groundTruthIndex` (falling back to `startPointforGestureExecution`); the shared `window_dataset` stamps each window `0` (rest) or the trial's gesture label (1–5) based on whether the window overlaps `[start, end]`. The full test drives the whole classical pipeline on real data.

**Tech Stack:** Python 3.10, numpy, scikit-learn, pytest. (No new dependencies.)

## Global Constraints

- Package `hand_gesture_recognition` (import root `src/hand_gesture_recognition/`).
- Use the EXISTING conda env `hand_gesture_recognition`; DO NOT create a venv, DO NOT use `uv`, DO NOT `pip install`.
- Run tests with `python -m pytest` (NOT bare `pytest`); from the repo root. If output is hidden by the conda wrapper, append `2>&1 | more`.
- NEVER use `python -c "..."` (shell strips quotes).
- Commit messages single-token (e.g. `git commit -m feat-gesture-bounds`).
- Per-window label rule: window `[s, s+W)` of trial `i` → `0` if rest trial (`gesture_start[i] == -1`), or `s+W <= gesture_start[i]`, or `s >= gesture_end[i]`; else the trial's label (`1..5`).
- 6 classes: `0` = noGesture/rest, `1..5` = gestures.
- Overlapping windows stay configurable via `stride`/`overlap_percent` (default 0.2s window / 0.05s stride = 75% overlap).

---

### Task 1: Gesture bounds on `Dataset` + loader extraction + propagation

**Files:**
- Modify: `src/hand_gesture_recognition/data/datasets/base.py`
- Modify: `src/hand_gesture_recognition/data/fixtures.py`
- Modify: `src/hand_gesture_recognition/data/splits/base.py`
- Modify: `src/hand_gesture_recognition/data/loaders/epn612.py`
- Modify: `tests/unit/test_dataset_base.py`, `tests/unit/test_windowing.py`, `tests/unit/test_analytics.py`

**Interfaces:**
- Produces: `Dataset` with two new aligned fields `gesture_start: np.ndarray` (int, `-1` = rest) and `gesture_end: np.ndarray` (int, `-1` = rest). `_subset` and `make_synthetic_dataset` preserve them.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_epn612_loader.py` a test that the loader records gesture bounds. In `_write_user`, add a `groundTruthIndex` to gesture samples. Modify `_write_user`:

```python
def _write_user(root, subdir, user_id, n_samples=3, length=50):
    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "open"]
    samples = {}
    for i in range(n_samples):
        g = gestures[i % len(gestures)]
        sample = {
            "startPointforGestureExecution": 0,
            "gestureName": g,
            "emg": {
                f"ch{c}": [float(c + j % 5) for j in range(length)] for c in range(1, 9)
            },
        }
        if g != "noGesture":
            sample["groundTruthIndex"] = [10, 40]
        samples[f"idx_{i + 1}"] = sample
    ...
```

Add this test:

```python
def test_epn612_loader_extracts_gesture_bounds(tmp_path):
    _write_user(tmp_path, "trainingJSON", 1)
    ds = EPN612Loader().load(str(tmp_path))
    # idx_1 = noGesture (rest), idx_2/idx_3 = gestures with groundTruthIndex [10, 40]
    assert ds.gesture_start.tolist() == [-1, 10, 10]
    assert ds.gesture_end.tolist() == [-1, 40, 40]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest tests/unit/test_epn612_loader.py::test_epn612_loader_extracts_gesture_bounds -v`
Expected: FAIL — `AttributeError: 'Dataset' object has no attribute 'gesture_start'` (or `TypeError` on construction).

- [ ] **Step 3: Implement the data model + propagation**

`src/hand_gesture_recognition/data/datasets/base.py` — add fields to `Dataset`:

```python
@dataclass
class Dataset:
    """Raw, unwindowed dataset: one trial = one continuous signal.

    trials[i] has shape (time_steps, channels) and may vary in length across i.
    labels, subject_ids, trial_ids, gesture_start, gesture_end are aligned with
    trials. gesture_start/gesture_end are EMG sample indices delimiting the
    gesture segment within the trial; both are -1 for rest (no-gesture) trials.
    """

    trials: List[np.ndarray]
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    gesture_start: np.ndarray
    gesture_end: np.ndarray
    metadata: DatasetMetadata

    @property
    def n_trials(self) -> int:
        return len(self.trials)
```

`src/hand_gesture_recognition/data/fixtures.py` — in `make_synthetic_dataset`, after building the `trials` list, add to the returned `Dataset(...)`:

```python
        gesture_start=np.zeros(len(trials), dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
```

`src/hand_gesture_recognition/data/splits/base.py` — in `_subset`, add to the returned `Dataset(...)`:

```python
        gesture_start=dataset.gesture_start[idx],
        gesture_end=dataset.gesture_end[idx],
```

`src/hand_gesture_recognition/data/loaders/epn612.py` — add `gesture_starts: List[int] = []` and `gesture_ends: List[int] = []` before the file loop; inside the sample loop, after `gesture = sample.get("gestureName", "noGesture")` and the `if gesture not in label_map: continue` check, add bounds extraction:

```python
            if gesture == "noGesture":
                g_start, g_end = -1, -1
            else:
                gti = sample.get("groundTruthIndex")
                if gti is not None and len(gti) == 2:
                    g_start, g_end = int(gti[0]), int(gti[1])
                else:
                    g_start = int(sample.get("startPointforGestureExecution", 0))
                    g_end = x.shape[0]
```

Then in the `trials.append(...)` block, also append `gesture_starts.append(g_start)` and `gesture_ends.append(g_end)`, and in the returned `Dataset(...)` add:

```python
            gesture_start=np.asarray(gesture_starts, dtype=int),
            gesture_end=np.asarray(gesture_ends, dtype=int),
```

(Note: the loop currently computes `x` before the `gesture` check, so `x.shape[0]` is available for the fallback `g_end`.)

- [ ] **Step 4: Fix the other `Dataset(...)` constructions in tests**

Add the two fields to every direct `Dataset(...)` in the test files (full-trial bounds so existing assertions keep passing):

`tests/unit/test_dataset_base.py` (`test_dataset_n_trials_and_shapes`):

```python
        gesture_start=np.array([0, 0]),
        gesture_end=np.array([100, 100]),
```

`tests/unit/test_windowing.py` (`_ds`):

```python
        gesture_start=np.zeros(len(trials), dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
```

`tests/unit/test_analytics.py` (`_ds`):

```python
        gesture_start=np.array([0, 0]),
        gesture_end=np.array([100, 100]),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest -q`
Expected: all pass (the new loader test + full suite, ~39 tests).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m feat-gesture-bounds
```

---

### Task 2: Per-window labels in shared windowing

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/base.py`
- Test: `tests/unit/test_windowing.py`

**Interfaces:**
- Consumes: `Dataset.gesture_start` / `Dataset.gesture_end`.
- Produces: `window_dataset` returns `WindowedDataset` whose `y` is the per-window label (0 rest / 1–5 gesture).

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_windowing.py`:

```python
def test_per_window_labels_from_gesture_bounds():
    from hand_gesture_recognition.data.windowing.base import window_dataset
    from hand_gesture_recognition.configuration.schema import WindowingConfig

    ds = Dataset(
        trials=[np.zeros((50, 2), dtype=np.float32), np.zeros((50, 2), dtype=np.float32)],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([10, -1]),
        gesture_end=np.array([30, -1]),
        metadata=DatasetMetadata(
            name="x", version="0", sampling_rate=200.0,
            channels=["c1", "c2"], class_names=["noGesture", "fist"], content_hash="h",
        ),
    )
    wd = window_dataset(ds, WindowingConfig(window_size_samples=10, stride_samples=10))
    # trial 0 (gesture [10,30]) -> [0,1,1,0,0]; trial 1 (rest) -> [0,0,0,0,0]
    assert wd.y.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest tests/unit/test_windowing.py::test_per_window_labels_from_gesture_bounds -v`
Expected: FAIL — current `window_dataset` stamps every window with the trial label, so `y` is `[1,1,1,1,1,0,0,0,0,0]` not the expected.

- [ ] **Step 3: Implement per-window labeling**

In `src/hand_gesture_recognition/data/windowing/base.py`, change the loop to zip in `gesture_start`/`gesture_end` and compute the per-window label:

```python
    for trial, label, subj, tid, g_start, g_end in zip(
        dataset.trials,
        dataset.labels,
        dataset.subject_ids,
        dataset.trial_ids,
        dataset.gesture_start,
        dataset.gesture_end,
    ):
        length = trial.shape[0]
        if length < ws:
            if config.padding == "zero":
                pad = np.zeros((ws - length, trial.shape[1]), dtype=trial.dtype)
                trial = np.vstack([trial, pad])
                length = ws
            else:
                continue
        n_windows = (length - ws) // stride + 1
        for i in range(n_windows):
            s = i * stride
            e = s + ws
            # Per-window label from the gesture ground-truth bounds.
            if g_start < 0:            # rest trial
                w_label = 0
            elif e <= g_start:         # window fully before gesture
                w_label = 0
            elif s >= g_end:           # window fully after gesture
                w_label = 0
            else:                      # window overlaps the gesture segment
                w_label = int(label)
            xs.append(trial[s:e])
            ys.append(w_label)
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))
```

(Keep the rest of the function — the `WindowedDataset` return and empty-`xs` guard — unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest tests/unit/test_windowing.py -v`
Expected: PASS (3 existing + 1 new = 4).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m feat-per-window-labels
```

---

### Task 3: Full classical model test on real data

**Files:**
- Create: `tests/datasets/test_classical_full.py`
- Modify: `tests/datasets/test_epn612_real.py` (assert per-window labels against bounds)

**Interfaces:**
- Consumes: `EPN612Loader`, `split_dataset`, `assert_no_group_leakage`, `WindowingTransformer`, `TimeDomainFeatures`, `RandomForestEstimator`, `SklearnPredictionAdapter`, `evaluate`, `EvaluationConfig`, `SplitConfig`, `WindowingConfig`, `resolve_path`.

- [ ] **Step 1: Write the failing test**

Create `tests/datasets/test_classical_full.py`:

```python
"""Full end-to-end classical pipeline on a SMALL subset of the real EMG-EPN612 dataset."""
from __future__ import annotations

import os

import pytest

from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.classical.models.random_forest import RandomForestEstimator
from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer
from hand_gesture_recognition.configuration.schema import (
    EvaluationConfig,
    SplitConfig,
    WindowingConfig,
)
from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader
from hand_gesture_recognition.data.splits.base import assert_no_group_leakage, split_dataset
from hand_gesture_recognition.evaluation.base import evaluate
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.paths import resolve_path

DATA_PATH = resolve_path("resources/data/EPN612")
HAS_REAL_DATA = os.path.isdir(DATA_PATH)

pytestmark = pytest.mark.skipif(
    not HAS_REAL_DATA, reason="EMG-EPN612 dataset not present under resources/data/EPN612"
)

SUBJECTS = [f"train:user{i}" for i in range(1, 7)]  # 6 subjects


def test_full_classical_pipeline_real_data():
    ds = EPN612Loader(subjects=SUBJECTS).load(DATA_PATH)
    assert ds.n_trials > 0

    split = split_dataset(
        ds,
        SplitConfig(strategy="subject_wise", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=0),
    )
    assert_no_group_leakage(split)

    windower = WindowingTransformer(WindowingConfig(window_size_seconds=0.2, stride_seconds=0.05))
    train_wd = windower.fit_transform(split.train)
    test_wd = windower.transform(split.test)

    # Per-window labels: both rest (0) and gesture (1-5) present.
    assert 0 in train_wd.y
    assert set(train_wd.y.tolist()) - {0}

    feat = TimeDomainFeatures()
    X_train = feat.fit_transform(train_wd.X)
    X_test = feat.transform(test_wd.X)

    model = RandomForestEstimator(n_estimators=50, random_state=0)
    model.fit(X_train, train_wd.y)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    adapter = SklearnPredictionAdapter()
    pred = adapter.to_prediction(
        windowed=test_wd, y_pred=y_pred, y_score=y_score,
        run_id="classical-full-test", duration_s=0.0,
    )
    assert pred.backend == "classical"

    result = evaluate(
        pred,
        EvaluationConfig(metrics=["accuracy", "balanced_accuracy", "macro_f1"]),
        dataset_id=ds.metadata.name,
        split_info={"strategy": "subject_wise"},
    )
    assert result.metrics["accuracy"] > 1 / 6
    assert result.metrics["balanced_accuracy"] > 1 / 6
    assert result.confusion_matrix.shape == (6, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest tests/datasets/test_classical_full.py -v`
Expected: FAIL — `ModuleNotFoundError` (file doesn't exist yet) OR, if run after the file is created, the accuracy assertions may fail before Task 2's labeling lands. The first run is RED either way.

- [ ] **Step 3: Add a per-window-label assertion to the existing real-data test**

In `tests/datasets/test_epn612_real.py`, add a test verifying the window labels match the bounds for one gesture trial:

```python
def test_real_data_per_window_labels_match_bounds():
    from hand_gesture_recognition.data.windowing.base import window_dataset
    from hand_gesture_recognition.configuration.schema import WindowingConfig

    ds = EPN612Loader(subjects=["train:user1"]).load(DATA_PATH)
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))

    # For every window, label must be 0 when entirely outside [start, end], else
    # the trial's gesture label (1-5).
    for trial, g_start, g_end, subj, tid in zip(
        ds.trials, ds.gesture_start, ds.gesture_end, ds.subject_ids, ds.trial_ids
    ):
        for s in range(0, trial.shape[0] - 40, 10):
            e = s + 40
            if g_start < 0:
                expected = 0
            elif e <= g_start or s >= g_end:
                expected = 0
            else:
                expected = int(ds.labels[list(ds.trial_ids).index(tid)])
            # find this window in wd by (subject, trial, start)
            mask = (wd.metadata.subject_ids == subj) & (wd.metadata.trial_ids == tid) & (wd.metadata.window_start == s)
            assert wd.y[mask][0] == expected
```

- [ ] **Step 4: Run the full suite to verify**

Run: `cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition" && python -m pytest -q`
Expected: all pass (~41 tests).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m feat-full-classical-test
```

---

## Self-Review (plan author)

**Spec coverage:** gesture bounds on `Dataset` (Task 1), loader `groundTruthIndex` extraction with fallback (Task 1), per-window labeling (Task 2), synthetic fixture populated with full-trial bounds (Task 1, so unit tests keep passing), full classical test on a small real subset with no synthetic (Task 3), overlapping windows unchanged (Global Constraints). ✅

**Placeholder scan:** no TBD/TODO; every code step has concrete code.

**Type consistency:** `gesture_start`/`gesture_end` are `np.ndarray` (int) everywhere; `window_dataset` zips `dataset.gesture_start/gesture_end`; `_subset` and `make_synthetic_dataset` propagate them; the label rule matches the spec's rule verbatim.
