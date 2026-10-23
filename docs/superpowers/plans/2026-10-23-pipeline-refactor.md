# Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat-array data model and the god-class `ExperimentRunner` with a nested `Dataset → User → Trial → emg` model, strategy-based splitting/windowing/training, a `PreprocessingPipeline`, and a thin `Pipeline` + `PipelineBuilder` orchestrator — preserving current behavior.

**Architecture:** A nested object model is the "tall" representation for load/split/filter/transform; windowing is the single conversion point to the "wide" flat `WindowedDataset` ndarray that features/models/evaluation consume. Branching logic (`split.strategy`, `task.name`, `windowing.mode`, `backend`) moves into Strategy classes dispatched by a tiny registry. `ExperimentRunner` becomes a facade over a `Pipeline` (Template Method) assembled by a `PipelineBuilder`.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, scikit-learn (estimator convention), Hydra (`instantiate`/`_partial_`), PyTorch Lightning (neural backend), MLflow (tracking), pytest.

## Global Constraints

- **Behavior preservation:** windowing semantics, signal-processing order, split partitions (same seed → same partitions), and evaluation output must not change. Integration/smoke tests are the behavioral spec.
- **Hydra for leaf components:** models, optimizers, filters, feature extractors, datamodules, loaders are selected via `_target_` config — do NOT add registries for them.
- **Strategies only where branching exists today:** splitting, CV, windowing (absorbs `task`), and training/prediction (absorbs `backend`).
- **No preemptive ABCs/Protocols:** enforce contracts with type hints + tests; add an interface only when a second implementation demands it.
- **Immutability by convention** via `dataclasses.replace`; do not use `frozen=True` (existing code/tests mutate objects, e.g. `Dataset.metadata`).
- **Label/rest convention:** `DatasetMetadata.class_names[0]` is always the no-gesture class; rest trials have `start == end == -1` and `gesture_name == metadata.no_gesture_name`; integer labels are derived via `class_names.index(gesture_name)`.
- **Test runner:** `python -m pytest <path>` (do not wrap in `conda run`).

---

## Target File Structure

New files (created across phases):

```
src/hand_gesture_recognition/data/datasets/base.py      # rewritten: Dataset/User/Trial/DatasetMetadata
src/hand_gesture_recognition/data/windowing/base.py     # + window_emg primitive, Segment dataclass
src/hand_gesture_recognition/data/windowing/strategy.py # WindowingStrategy registry + 3 impls
src/hand_gesture_recognition/data/splits/strategy.py    # SplitStrategy + CVStrategy registries
src/hand_gesture_recognition/preprocessing/pipeline.py  # PreprocessingPipeline
src/hand_gesture_recognition/training/base.py           # TrainingStrategy base + registry
src/hand_gesture_recognition/training/classical.py      # ClassicalTrainingStrategy
src/hand_gesture_recognition/training/neural.py         # NeuralTrainingStrategy
src/hand_gesture_recognition/training/features.py       # FeatureExtractor (union application)
src/hand_gesture_recognition/runner/pipeline.py         # Pipeline (Template Method) + PipelineBuilder
src/hand_gesture_recognition/evaluation/grouped.py      # evaluate_grouped
```

Deleted files (across phases):

```
src/hand_gesture_recognition/data/windowing/two_level.py    # folded into base.py + strategy.py
src/hand_gesture_recognition/data/windowing/dispatch.py     # folded into strategy.py
src/hand_gesture_recognition/neural/datamodules/two_level.py# folded into emg.py (grouped)
```

Rewritten files: `data/loaders/epn612.py`, `data/fixtures.py`, `data/gesture_crop.py`,
`data/class_selection.py`, `data/splits/base.py`, `data/splits/folds.py`,
`classical/adapters/sklearn.py`, `neural/adapters/lightning.py`,
`neural/datamodules/emg.py`, `runner/run.py`, plus call sites in
`reporting/visualization.py`, `reporting/reporters.py`, `tracking/mlflow_tracker.py`,
`analytics/*`, and all affected tests.

---

## Phase 0 — Nested data model (foundation)

### Task 0.1: Define `DatasetMetadata`, `Trial`, `User`, `Dataset`

**Files:**
- Modify: `src/hand_gesture_recognition/data/datasets/base.py` (full rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces (used by every later task):
  - `DatasetMetadata(name, version, sampling_rate, channels, class_names, content_hash, provenance=dict, no_gesture_name="noGesture")`
  - `Trial(trial_id, user_id, gesture_name, start, end, emg)` with property `is_rest -> bool` (`start < 0`)
  - `User(user_id, trials, demographics=None)`
  - `Dataset(metadata, users)` with cached properties `trials`, `labels`, `subject_ids`, `trial_ids`, and `n_trials`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_dataset_base.py` (rewrite):

```python
import numpy as np

from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)


def _meta():
    return DatasetMetadata(
        name="x", version="0", sampling_rate=200.0,
        channels=["c1", "c2"], class_names=["noGesture", "fist", "open"],
        content_hash="h",
    )


def test_trial_is_rest():
    rest = Trial("t1", "u1", "noGesture", -1, -1, np.zeros((10, 2), np.float32))
    gesture = Trial("t2", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32))
    assert rest.is_rest is True
    assert gesture.is_rest is False


def test_dataset_flatten_and_derived_labels():
    u = User("u1", [
        Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32)),
        Trial("u1:t2", "u1", "noGesture", -1, -1, np.zeros((10, 2), np.float32)),
        Trial("u1:t3", "u1", "open", 0, 10, np.zeros((10, 2), np.float32)),
    ])
    ds = Dataset(_meta(), [u])
    assert ds.n_trials == 3
    assert [t.trial_id for t in ds.trials] == ["u1:t1", "u1:t2", "u1:t3"]
    # labels derived from class_names ordering: fist=1, noGesture=0, open=2
    assert ds.labels.tolist() == [1, 0, 2]
    assert ds.subject_ids.tolist() == ["u1", "u1", "u1"]
    assert ds.trial_ids.tolist() == ["u1:t1", "u1:t2", "u1:t3"]


def test_dataset_trials_cache_respects_replace():
    u = User("u1", [Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32))])
    ds = Dataset(_meta(), [u])
    first = ds.trials
    assert ds.trials is first  # cached
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_dataset_base.py -v`
Expected: FAIL (import error — new names don't exist).

- [ ] **Step 3: Write the implementation**

`src/hand_gesture_recognition/data/datasets/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DatasetMetadata:
    name: str
    version: str
    sampling_rate: float
    channels: List[str]
    class_names: List[str]
    content_hash: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    no_gesture_name: str = "noGesture"


@dataclass
class Trial:
    trial_id: str
    user_id: str
    gesture_name: str
    start: int
    end: int
    emg: np.ndarray  # (time_steps, channels)

    @property
    def is_rest(self) -> bool:
        return self.start < 0


@dataclass
class User:
    user_id: str
    trials: List[Trial]
    demographics: Optional[Dict[str, Any]] = None


@dataclass
class Dataset:
    """Nested dataset: users own trials; trials own the raw EMG signal.

    ``trials``/``labels``/``subject_ids``/``trial_ids`` are derived, cached
    flat views. Treat instances as immutable (transform via ``replace``).
    """

    metadata: DatasetMetadata
    users: List[User]

    @cached_property
    def trials(self) -> List[Trial]:
        return [t for u in self.users for t in u.trials]

    @cached_property
    def labels(self) -> np.ndarray:
        lut = {n: i for i, n in enumerate(self.metadata.class_names)}
        return np.asarray([lut[t.gesture_name] for t in self.trials], dtype=int)

    @cached_property
    def subject_ids(self) -> np.ndarray:
        return np.asarray([t.user_id for t in self.trials])

    @cached_property
    def trial_ids(self) -> np.ndarray:
        return np.asarray([t.trial_id for t in self.trials])

    @property
    def n_trials(self) -> int:
        return len(self.trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_dataset_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/datasets/base.py tests/unit/test_dataset_base.py
git commit -m "refactor: nested Dataset/User/Trial data model"
```

### Task 0.2: Add `from_trials` / `from_users` constructors and a `label_of` helper

**Files:**
- Modify: `src/hand_gesture_recognition/data/datasets/base.py`

**Interfaces:**
- Produces: `Dataset.from_trials(trials, metadata) -> Dataset` (groups trials by `user_id`, preserving order); `Dataset.from_users(users, metadata) -> Dataset`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_dataset_base.py`:

```python
def test_from_trials_groups_by_user():
    trials = [
        Trial("u1:t1", "u1", "fist", 0, 10, np.zeros((10, 2), np.float32)),
        Trial("u2:t1", "u2", "fist", 0, 10, np.zeros((10, 2), np.float32)),
        Trial("u1:t2", "u1", "open", 0, 10, np.zeros((10, 2), np.float32)),
    ]
    ds = Dataset.from_trials(trials, _meta())
    assert [u.user_id for u in ds.users] == ["u1", "u2"]
    assert len(ds.users[0].trials) == 2
    assert len(ds.users[1].trials) == 1
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_dataset_base.py::test_from_trials_groups_by_user -v`
Expected: FAIL (`from_trials` missing).

- [ ] **Step 3: Implement**

Add to `Dataset`:

```python
    @classmethod
    def from_trials(cls, trials: List[Trial], metadata: DatasetMetadata) -> "Dataset":
        by_user: Dict[str, List[Trial]] = {}
        order: List[str] = []
        for t in trials:
            if t.user_id not in by_user:
                by_user[t.user_id] = []
                order.append(t.user_id)
            by_user[t.user_id].append(t)
        return cls(metadata=metadata, users=[User(u, by_user[u]) for u in order])

    @classmethod
    def from_users(cls, users: List[User], metadata: DatasetMetadata) -> "Dataset":
        return cls(metadata=metadata, users=list(users))
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_dataset_base.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/datasets/base.py tests/unit/test_dataset_base.py
git commit -m "feat: Dataset.from_trials/from_users constructors"
```

### Task 0.3: Rewrite `make_synthetic_dataset`

**Files:**
- Modify: `src/hand_gesture_recognition/data/fixtures.py`

**Interfaces:**
- Consumes: `Dataset`, `User`, `Trial`, `DatasetMetadata` (Task 0.1).
- Produces: same `make_synthetic_dataset(...)` signature as today (defaults unchanged).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_dataset_base.py` add:

```python
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset


def test_synthetic_dataset_is_nested():
    ds = make_synthetic_dataset(seed=0)
    assert ds.n_trials == 4 * 6
    assert len(ds.users) == 4
    assert ds.trials[0].emg.shape == (100, 8)
    assert ds.trials[0].gesture_name == "class_0"  # seed 0, s=0,t=0 -> label 0
    assert ds.labels.tolist() == [(s + t) % 3 for s in range(4) for t in range(6)]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_dataset_base.py::test_synthetic_dataset_is_nested -v`
Expected: FAIL (fixture still builds flat arrays).

- [ ] **Step 3: Implement**

Rewrite `make_synthetic_dataset` body (keep signature):

```python
def make_synthetic_dataset(
    n_subjects=4, n_trials_per_subject=6, n_classes=3,
    length=100, n_channels=8, seed=0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    class_names = [f"class_{i}" for i in range(n_classes)]
    users: List[User] = []
    for s in range(n_subjects):
        user_trials = []
        for t in range(n_trials_per_subject):
            label = (s + t) % n_classes
            base = rng.normal(0, 1, size=(length, n_channels))
            signal = (base + label).astype(np.float32)
            user_trials.append(
                Trial(
                    trial_id=f"s{s}_t{t}",
                    user_id=f"s{s}",
                    gesture_name=class_names[label],
                    start=0,
                    end=length,
                    emg=signal,
                )
            )
        users.append(User(f"s{s}", user_trials))
    return Dataset(
        metadata=DatasetMetadata(
            name="synthetic", version="0", sampling_rate=200.0,
            channels=[f"ch{i}" for i in range(1, n_channels + 1)],
            class_names=class_names, content_hash="synthetic",
            provenance={"seed": seed},
        ),
        users=users,
    )
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_dataset_base.py tests/unit/test_windowing.py -v`
Expected: PASS only if downstream windowing is not yet rewritten — this will FAIL elsewhere; proceed to rewrite windowing in Phase 1. For this task's gate, run only the dataset tests:
`python -m pytest tests/unit/test_dataset_base.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/fixtures.py tests/unit/test_dataset_base.py
git commit -m "refactor: nested synthetic dataset fixture"
```

### Task 0.4: Rewrite `EPN612Loader` to build the nested model

**Files:**
- Modify: `src/hand_gesture_recognition/data/loaders/epn612.py`

**Interfaces:**
- Consumes: `Dataset`, `User`, `Trial`, `DatasetMetadata` (Task 0.1).
- Produces: `EPN612Loader.load(path) -> Dataset` with identical discovery/sampling semantics; demographics land on `User.demographics` (no more `subject_info` DataFrame).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_epn612_loader.py` add (uses a temp dir with one user file):

```python
def _write_user(path, cohort="training", user="user1"):
    import json, os
    d = os.path.join(path, cohort, user)
    os.makedirs(d, exist_ok=True)
    doc = {
        "generalInfo": {"samplingFrequencyInHertz": 200.0},
        "userInfo": {"age": 30},
        "trainingSamples": {
            "s1": {"emg": {"ch1": [1, 2, 3], "ch2": [1, 2, 3]},
                  "gestureName": "fist",
                  "groundTruthIndex": [0, 3]},
        },
    }
    with open(os.path.join(d, f"{user}.json"), "w") as f:
        json.dump(doc, f)
    return d


def test_loader_nested(tmp_path):
    from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader
    _write_user(tmp_path)
    ds = EPN612Loader().load(str(tmp_path))
    assert [u.user_id for u in ds.users] == ["train:user1"]
    assert ds.users[0].demographics["age"] == 30
    t = ds.trials[0]
    assert t.gesture_name == "fist"
    assert t.emg.shape == (3, 2)
    assert (t.start, t.end) == (0, 3)
    assert ds.labels.tolist() == [ds.metadata.class_names.index("fist")]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_epn612_loader.py::test_loader_nested -v`
Expected: FAIL (loader returns flat arrays / `subject_info`).

- [ ] **Step 3: Implement**

Rewrite the `load` loop to accumulate `User` objects keyed by `subject_id`, each holding `Trial` objects, and demographics on the `User`. Keep `_discover`, `_sample_users`, `GESTURE_LABELS`, `label_map` derivation, `channels` sorting, and the `noGesture`/`groundTruthIndex`/`startPointforGestureExecution` bounds logic **unchanged**. Replace the final `Dataset(...)` construction with:

```python
        users = []
        for sid in ordered_subject_ids:  # preserve first-seen order
            users.append(
                User(
                    user_id=sid,
                    trials=trials_by_user[sid],
                    demographics=subject_info_rows.get(sid),
                )
            )
        return Dataset(
            metadata=DatasetMetadata(
                name="epn612", version="2020", sampling_rate=sampling_rate,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory_cached(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
            users=users,
        )
```

(Accumulate `trials_by_user: Dict[str, List[Trial]]`, `ordered_subject_ids: List[str]`, and `subject_info_rows: Dict[str, Dict]` exactly as the current loop does, but appending `Trial(trial_id=f"{subject_id}:{section}:{key}", user_id=subject_id, gesture_name=gesture, start=g_start, end=g_end, emg=x)` instead of parallel lists.)

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_epn612_loader.py -v`
Expected: PASS (plus any loader-specific asserts already present; update those that referenced `subject_info`/flat arrays).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/loaders/epn612.py tests/unit/test_epn612_loader.py
git commit -m "refactor: EPN612Loader builds nested model with User demographics"
```

### Task 0.5: Rewrite `class_selection.py`

**Files:**
- Modify: `src/hand_gesture_recognition/data/class_selection.py`

**Interfaces:**
- Consumes: `Dataset`, `Trial` (Task 0.1).
- Produces (unchanged names, new semantics): `build_no_gesture_remap(dataset) -> (remap: Dict[int,int], class_names: List[str])`; `drop_no_gesture(dataset, remap, class_names) -> Dataset`; `remove_no_gesture(dataset) -> Dataset`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_class_selection.py` (rewrite):

```python
import numpy as np
from hand_gesture_recognition.data.class_selection import (
    build_no_gesture_remap, drop_no_gesture,
)
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata, Trial, User


def _ds():
    meta = DatasetMetadata(
        name="x", version="0", sampling_rate=200.0, channels=["c"],
        class_names=["noGesture", "fist", "open"], content_hash="h",
    )
    u = User("u1", [
        Trial("t1", "u1", "noGesture", -1, -1, np.zeros((5, 1), np.float32)),
        Trial("t2", "u1", "fist", 0, 5, np.ones((5, 1), np.float32)),
        Trial("t3", "u1", "open", 0, 5, np.full((5, 1), 2.0, np.float32)),
    ])
    return Dataset(meta, [u])


def test_remap_and_drop():
    remap, names = build_no_gesture_remap(_ds())
    assert names == ["fist", "open"]
    assert remap == {1: 0, 2: 1}
    out = drop_no_gesture(_ds(), remap, names)
    assert out.n_trials == 2
    assert out.labels.tolist() == [0, 1]
    assert out.metadata.class_names == ["fist", "open"]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_class_selection.py -v`
Expected: FAIL (still expects flat arrays).

- [ ] **Step 3: Implement**

```python
def build_no_gesture_remap(dataset: Dataset):
    kept = [t for t in dataset.trials if not t.is_rest]
    kept_names = []
    for t in kept:
        if t.gesture_name not in kept_names:
            kept_names.append(t.gesture_name)
    remap = {dataset.metadata.class_names.index(n): i for i, n in enumerate(kept_names)}
    return remap, kept_names


def drop_no_gesture(dataset, remap, new_class_names):
    from dataclasses import replace
    kept = [t for t in dataset.trials if not t.is_rest]
    new_meta = replace(dataset.metadata, class_names=new_class_names)
    return Dataset(metadata=new_meta, users=Dataset.from_trials(kept, new_meta).users)


def remove_no_gesture(dataset):
    remap, names = build_no_gesture_remap(dataset)
    return drop_no_gesture(dataset, remap, names)
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_class_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/class_selection.py tests/unit/test_class_selection.py
git commit -m "refactor: class selection over nested model"
```

### Task 0.6: Rewrite `gesture_crop.py`

**Files:**
- Modify: `src/hand_gesture_recognition/data/gesture_crop.py`

**Interfaces:**
- Consumes: `Dataset`, `Trial` (Task 0.1).
- Produces: `crop_to_gesture(dataset, config) -> Dataset` (same semantics: symmetric crop with shifted bounds; rest trials truncated to `no_gesture_max_length_samples`).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_gesture_crop.py` add:

```python
def test_crop_shifts_bounds():
    from hand_gesture_recognition.data.gesture_crop import crop_to_gesture
    from hand_gesture_recognition.configuration.schema import GestureCropConfig
    import numpy as np
    from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata, Trial, User
    emg = np.arange(100, dtype=np.float32)[:, None]
    ds = Dataset(
        DatasetMetadata("x", "0", 200.0, ["c"], ["noGesture", "fist"], "h"),
        [User("u1", [Trial("t1", "u1", "fist", 40, 60, emg)])],
    )
    out = crop_to_gesture(ds, GestureCropConfig(enabled=True, pad_samples=10))
    t = out.trials[0]
    assert t.emg.shape[0] == 40  # [30, 70)
    assert (t.start, t.end) == (10, 30)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_gesture_crop.py::test_crop_shifts_bounds -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Rewrite the per-trial loop to produce `Trial` objects (use `dataclasses.replace` on each `Trial`), keeping the median/max-length computation identical. Rebuild via `Dataset.from_trials(new_trials, replace(dataset.metadata, ...))` (metadata unchanged here).

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_gesture_crop.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/gesture_crop.py tests/unit/test_gesture_crop.py
git commit -m "refactor: gesture crop over nested model"
```

### Task 0.7: Adapt `splits/base.py` (`_subset`, `assert_no_group_leakage`) and `splits/folds.py`

**Files:**
- Modify: `src/hand_gesture_recognition/data/splits/base.py`, `src/hand_gesture_recognition/data/splits/folds.py`

**Interfaces:**
- Consumes: `Dataset`, `Dataset.from_trials/from_users`, `Trial.user_id` (Tasks 0.1–0.2).
- Produces (Phase 2 will refactor these into strategies; for now just make them work): `split_dataset(dataset, config) -> DataSplit`; `generate_folds(dataset, cv, split) -> List[DataSplit]`; `assert_no_group_leakage(split)`.

- [ ] **Step 1: Rewrite `_subset` to operate on trials**

Replace flat `_subset(dataset, idx)` with trial-index subsetting:

```python
def _subset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    chosen = [dataset.trials[int(i)] for i in idx]
    return Dataset.from_trials(chosen, dataset.metadata)
```

`split_dataset` keeps its strategy branches for now but uses the derived views:

```python
def split_dataset(dataset, config):
    config.validate()
    n = dataset.n_trials
    groups = dataset.subject_ids if config.split_on == "subject_ids" else dataset.trial_ids
    ...
```

(the `predefined` branch's `np.char.startswith(sids, "train:")` now reads `dataset.subject_ids.astype(str)` — unchanged since `subject_ids` is still an ndarray).

- [ ] **Step 2: Rewrite `_group_ids`/carve in `folds.py`** to use `dataset.subject_ids`/`dataset.trial_ids` (unchanged values); only `_subset` changes (already updated).

- [ ] **Step 3: Update `assert_no_group_leakage`**

```python
def assert_no_group_leakage(split, group_field="user_id"):
    def ids(d):
        return {getattr(t, group_field) for t in d.trials}
    tr, va, te = ids(split.train), ids(split.val), ids(split.test)
    if tr & va or tr & te or va & te:
        raise ValueError("group overlap between partitions")
```

- [ ] **Step 4: Run split tests**

Run: `python -m pytest tests/unit/test_splits.py tests/unit/test_cv.py -v`
Expected: PASS after updating any asserts that referenced `subject_info` (remove those).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/splits/base.py src/hand_gesture_recognition/data/splits/folds.py tests/unit/test_splits.py tests/unit/test_cv.py
git commit -m "refactor: splitting over nested model"
```

### Task 0.8: Adapt remaining call sites to make the suite green for Phase 0

**Files:**
- Modify: `src/hand_gesture_recognition/preprocessing/baseline.py`, `src/hand_gesture_recognition/data/windowing/base.py` (temporary shim), `src/hand_gesture_recognition/classical/adapters/sklearn.py`, `src/hand_gesture_recognition/neural/adapters/lightning.py`, `src/hand_gesture_recognition/runner/run.py`, and any `reporting`/`analytics`/`tracking` code that reads `dataset.labels/subject_ids/trial_ids/gesture_start/gesture_end`.

**Interfaces:**
- Consumes: everything from Phase 0.
- Produces: a tree that imports cleanly and passes the Phase-0 regression set (dataset, loader, class selection, gesture crop, splits, analytics, baseline unit tests).

- [ ] **Step 1: Mechanical rewrites** — replace these access patterns everywhere:

| Old | New |
|---|---|
| `dataset.trials` (was `List[np.ndarray]`) | `[t.emg for t in dataset.trials]` where a raw array list is required |
| `dataset.labels` | unchanged (derived ndarray) |
| `dataset.subject_ids` / `dataset.trial_ids` | unchanged (derived ndarray) |
| `dataset.gesture_start` / `dataset.gesture_end` | `np.asarray([t.start for t in dataset.trials])` / `np.asarray([t.end ...])` |
| `dataset.subject_info` | `{u.user_id: u.demographics for u in dataset.users}` |

`baseline.py`: iterate `for t in dataset.trials`, use `t.is_rest`/`t.emg`/`t.user_id`; rebuild with `replace(dataset, users=Dataset.from_trials(new_trials, dataset.metadata).users)`.

- [ ] **Step 2: Run the Phase-0 regression set**

Run: `python -m pytest tests/unit/test_dataset_base.py tests/unit/test_epn612_loader.py tests/unit/test_class_selection.py tests/unit/test_gesture_crop.py tests/unit/test_splits.py tests/unit/test_baseline.py tests/unit/test_analytics.py -v`
Expected: PASS. Windowing/training/evaluation tests will still fail — they are fixed in Phases 1–4. Do not merge Phase 0 until this set is green.

- [ ] **Step 3: Commit**

```bash
git add -A src/hand_gesture_recognition tests/unit
git commit -m "refactor: adapt baseline/analytics/adapters to nested model"
```

---

## Phase 1 — Windowing (compositional)

### Task 1.1: Extract `window_emg` primitive

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/base.py`

**Interfaces:**
- Consumes: `WindowingConfig.resolve()`.
- Produces (Phase 1+ core): 
  - `SlideResult = Tuple[np.ndarray, np.ndarray, WindowMetadata]` (X `(n, ws, ch)`, y `(n,)`, meta)
  - `window_emg(emg, label, config, subject_id, trial_id, prefix, rest_bounds=None, drop_rest=False) -> SlideResult`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_windowing.py` add:

```python
import numpy as np
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.windowing.base import window_emg


def _cfg(ws, stride):
    return WindowingConfig(window_size_samples=ws, stride_samples=stride, padding="none")


def test_window_emg_dumb_slide():
    emg = np.arange(10, dtype=np.float32)[:, None]
    X, y, meta = window_emg(emg, 1, _cfg(4, 2), "s1", "t1", "s1_t1")
    assert X.shape == (4, 4, 1)  # windows at 0,2,4,6
    assert y.tolist() == [1, 1, 1, 1]
    assert meta.window_start.tolist() == [0, 2, 4, 6]
    assert meta.subject_ids.tolist() == ["s1"] * 4


def test_window_emg_rest_detection():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10))
    # window at 0 is rest (ends at 4, not overlapping [4,10]); window at 2 overlaps
    assert y.tolist() == [0, 5, 5, 5]
    assert meta.window_start.tolist() == [0, 2, 4, 6]


def test_window_emg_drop_rest():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10), drop_rest=True)
    assert y.tolist() == [5, 5, 5]
    assert meta.window_start.tolist() == [2, 4, 6]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_windowing.py -v`
Expected: FAIL (`window_emg` missing).

- [ ] **Step 3: Implement**

Add to `base.py` (reusing `WindowMetadata`/`WindowedDataset`). `window_emg` resolves
window size/stride in samples (two-level big windows are also passed in samples):

```python
def _resolve_samples(config):
    if config.window_size_samples is None:
        raise ValueError("window_emg requires window_size_samples")
    stride = config.stride_samples or config.window_size_samples
    return config.window_size_samples, stride


def _empty_meta() -> WindowMetadata:
    return WindowMetadata(
        window_start=np.asarray([], dtype=int),
        window_end=np.asarray([], dtype=int),
        subject_ids=np.asarray([], dtype=object),
        trial_ids=np.asarray([], dtype=object),
        window_ids=np.asarray([], dtype=object),
    )


def _concat_meta(metas) -> WindowMetadata:
    if not metas:
        return _empty_meta()
    return WindowMetadata(
        window_start=np.concatenate([m.window_start for m in metas]),
        window_end=np.concatenate([m.window_end for m in metas]),
        subject_ids=np.concatenate([m.subject_ids for m in metas]),
        trial_ids=np.concatenate([m.trial_ids for m in metas]),
        window_ids=np.concatenate([m.window_ids for m in metas]),
    )


def window_emg(emg, label, config, subject_id, trial_id, prefix,
               rest_bounds=None, drop_rest=False):
    ws, stride = _resolve_samples(config)
    emg = np.asarray(emg, dtype=np.float32)
    length = emg.shape[0]
    if length < ws:
        if config.padding == "zero":
            pad = np.zeros((ws - length, emg.shape[1]), dtype=emg.dtype)
            emg = np.vstack([emg, pad])
            length = ws
        else:
            return np.zeros((0, ws, emg.shape[1]), np.float32), np.array([], int), _empty_meta()
    g_start, g_end = rest_bounds if rest_bounds is not None else (-1, -1)
    n = (length - ws) // stride + 1
    xs, ys, starts, ends = [], [], [], []
    for i in range(n):
        s = i * stride
        e = s + ws
        is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
        if drop_rest and is_rest:
            continue
        xs.append(emg[s:e])
        ys.append(0 if is_rest else int(label))
        starts.append(s)
        ends.append(e)
    wins = [f"{subject_id}_{trial_id}_{prefix}{i}" for i in range(len(xs))]
    meta = WindowMetadata(
        window_start=np.asarray(starts, int),
        window_end=np.asarray(ends, int),
        subject_ids=np.asarray([subject_id] * len(xs)),
        trial_ids=np.asarray([trial_id] * len(xs)),
        window_ids=np.asarray(wins),
    )
    X = np.stack(xs) if xs else np.zeros((0, ws, emg.shape[1]), np.float32)
    return X, np.asarray(ys, int), meta
```

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_windowing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/base.py tests/unit/test_windowing.py
git commit -m "feat: window_emg primitive (dumb slide + rest detection)"
```

### Task 1.2: Reimplement `window_dataset` on `window_emg`

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/base.py`

**Interfaces:**
- Produces: `window_dataset(dataset, config) -> WindowedDataset` with identical output to today (verified by existing `test_windowing.py`).

- [ ] **Step 1: Rewrite `window_dataset` to loop `window_emg` and concatenate:

```python
def window_dataset(dataset, config):
    drop_rest = getattr(dataset.metadata, "drop_rest_windows", False)
    Xs, ys, metas = [], [], []
    for t in dataset.trials:
        X, y, m = window_emg(t.emg, dataset.metadata.class_names.index(t.gesture_name),
                             config, t.user_id, t.trial_id, "window",
                             rest_bounds=None if t.is_rest else (t.start, t.end),
                             drop_rest=drop_rest)
        if len(y) == 0:
            continue
        Xs.append(X); ys.append(y); metas.append(m)
    if not Xs:
        ws, _ = _resolve_samples(config)
        n_ch = dataset.metadata and (len(dataset.metadata.channels) if dataset.metadata.channels else (dataset.trials[0].emg.shape[1] if dataset.trials else 0))
        return WindowedDataset(X=np.zeros((0, ws, n_ch), np.float32), y=np.asarray([], int), metadata=_empty_meta())
    return WindowedDataset(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0).astype(int),
        metadata=_concat_meta(metas),
    )
```

(Define `_concat_meta(metas)` in `base.py` alongside `_empty_meta` — concatenates each of the five `WindowMetadata` columns across the list.)

- [ ] **Step 2: Run existing windowing tests**

Run: `python -m pytest tests/unit/test_windowing.py tests/unit/test_gesture_windowing.py tests/unit/test_windowing_alignment.py -v`
Expected: PASS after reconciling any `drop_rest_windows` accesses to `getattr`.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/base.py
git commit -m "refactor: window_dataset via window_emg"
```

### Task 1.3: Reimplement `window_gestures` (classification) on `window_emg`

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/gesture.py`

- [ ] **Step 1: Rewrite to one window per non-rest trial:

```python
def window_gestures(dataset, config):
    ws, _ = _resolve_samples(config)
    xs, ys, metas = [], [], []
    for t in dataset.trials:
        if t.is_rest:
            continue
        seg = t.emg[t.start:t.end]
        if seg.shape[0] < ws:
            seg = np.vstack([seg, np.zeros((ws - seg.shape[0], seg.shape[1]), seg.dtype)])
        elif seg.shape[0] > ws:
            seg = seg[:ws]
        xs.append(seg)
        ys.append(dataset.metadata.class_names.index(t.gesture_name))
        metas.append(WindowMetadata(
            window_start=np.asarray([t.start], int),
            window_end=np.asarray([t.end], int),
            subject_ids=np.asarray([t.user_id]),
            trial_ids=np.asarray([t.trial_id]),
            window_ids=np.asarray([f"{t.user_id}_{t.trial_id}_gesture"]),
        ))
    n_ch = dataset.metadata and (len(dataset.metadata.channels) if dataset.metadata.channels else (xs[0].shape[1] if xs else 0))
    return WindowedDataset(
        X=np.stack(xs) if xs else np.zeros((0, ws, n_ch), np.float32),
        y=np.asarray(ys, dtype=int),
        metadata=_concat_meta(metas),
    )
```

(Add `_concat_meta(metas)` in `base.py` — a helper that concatenates the five `WindowMetadata` columns of a list into one; reuse it in Task 1.2's `window_dataset` too.)

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_gesture_windowing.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/gesture.py
git commit -m "refactor: window_gestures via segments"
```

### Task 1.4: `WindowingStrategy` registry + two-level composition; delete `two_level.py`/`dispatch.py`

**Files:**
- Create: `src/hand_gesture_recognition/data/windowing/strategy.py`
- Modify: `src/hand_gesture_recognition/data/windowing/base.py` (add `big_window_id` to `WindowMetadata` as an optional field defaulting to `None`)
- Delete: `src/hand_gesture_recognition/data/windowing/two_level.py`, `src/hand_gesture_recognition/data/windowing/dispatch.py`

**Interfaces:**
- Consumes: `window_emg`, `window_dataset`, `window_gestures` (Tasks 1.1–1.3).
- Produces:
  - `@register_windowing(name)` decorator + `get_windowing_strategy(mode_or_task) -> WindowingStrategy`
  - `WindowingStrategy.window(split_or_dataset, config, signal_steps) -> WindowedPartitions | GroupedWindowedPartitions`
  - `GroupedWindowedDataset(big: WindowedDataset, small_windows: List[np.ndarray], big_window_id: np.ndarray, y: np.ndarray, metadata)` + `GroupedWindowedPartitions(train, val, test)`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_two_level_windowing.py` (rewrite to new API):

```python
import numpy as np
from hand_gesture_recognition.data.windowing.strategy import (
    GroupedWindowedPartitions, get_windowing_strategy,
)
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata, Trial, User
from hand_gesture_recognition.configuration.schema import WindowingConfig


def _ds():
    meta = DatasetMetadata("x", "0", 200.0, ["c1", "c2"], ["noGesture", "fist"], "h")
    u = User("s1", [
        Trial("t1", "s1", "fist", 0, 100, np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1)),
        Trial("t2", "s1", "noGesture", -1, -1, np.zeros((100, 2), np.float32)),
    ])
    return Dataset(meta, [u])


def test_two_level_compositional():
    cfg = WindowingConfig(mode="two_level", window_size_samples=20, stride_samples=10,
                          big_window_size_samples=60, big_stride_samples=30)
    s = get_windowing_strategy("two_level")
    g = s.window(_ds(), cfg, signal_steps=[])  # Phase 1: returns GroupedWindowedDataset
    # big windows: t1 -> 2, t2 -> 2 => 4
    assert g.big.y.tolist() == [1, 1, 0, 0]
    assert g.small_windows[0].shape == (5, 20, 2)
    assert g.big.X.shape == (4, 60, 2)
```

(Note: in Phase 1 the strategy `window` takes a `Dataset` and returns `WindowedDataset` or `GroupedWindowedDataset` directly; the `DataSplit`/`WindowedPartitions` wrappers arrive in Phase 4.)

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_two_level_windowing.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Implement**

`strategy.py`:

```python
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset, WindowedPartitions, window_dataset,
)
from hand_gesture_recognition.data.windowing.gesture import window_gestures

_REGISTRY: Dict[str, type] = {}


def register_windowing(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_windowing_strategy(name):
    return _REGISTRY[name]()


@dataclass
class GroupedWindowedDataset:
    big: WindowedDataset               # one sample per big window (classical view)
    small_windows: List[np.ndarray]    # per big window: (n_small, small_ws, ch)
    big_window_id: np.ndarray          # (n_big,) links small groups to big windows
    y: np.ndarray
    metadata: object


@dataclass
class GroupedWindowedPartitions:
    train: GroupedWindowedDataset
    val: GroupedWindowedDataset
    test: GroupedWindowedDataset


def _big_config(config, sampling_rate):
    from dataclasses import replace
    big_ws, big_stride = config.resolve_big(sampling_rate)
    return replace(config, window_size_samples=big_ws, stride_samples=big_stride)


def _build_windowed(xs, ys, metas, config):
    if not xs:
        return WindowedDataset(
            X=np.zeros((0, 0, 0), np.float32), y=np.asarray([], int),
            metadata=_empty_meta(),
        )
    from hand_gesture_recognition.data.windowing.base import WindowMetadata
    X = np.stack(xs)
    return WindowedDataset(
        X=X,
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.concatenate([m.window_start for m in metas]),
            window_end=np.concatenate([m.window_end for m in metas]),
            subject_ids=np.concatenate([m.subject_ids for m in metas]),
            trial_ids=np.concatenate([m.trial_ids for m in metas]),
            window_ids=np.concatenate([m.window_ids for m in metas]),
        ),
    )


class RecognitionWindowing:
    def window(self, dataset, config, signal_steps=None):
        return window_dataset(dataset, config)


class ClassificationWindowing:
    def window(self, dataset, config, signal_steps=None):
        return window_gestures(dataset, config)


class TwoLevelWindowing:
    def window(self, dataset, config, signal_steps=None):
        signal_steps = signal_steps or []
        big_cfg = _big_config(config, dataset.metadata.sampling_rate)
        big_X, big_y, big_metas = [], [], []
        smalls, big_ids = [], []
        for t in dataset.trials:
            X, y, m = window_emg(
                t.emg, dataset.metadata.class_names.index(t.gesture_name), big_cfg,
                t.user_id, t.trial_id, "big",
                rest_bounds=None if t.is_rest else (t.start, t.end),
                drop_rest=getattr(dataset.metadata, "drop_rest_windows", False),
            )
            for j in range(len(y)):
                big = X[j]
                for step in signal_steps:
                    big = step.transform(big)
                sX, _sy, _sm = window_emg(
                    big, int(y[j]), config, t.user_id, t.trial_id, f"big{j}_",
                    rest_bounds=None, drop_rest=False,
                )
                smalls.append(sX)
                big_ids.append(f"{t.trial_id}_big{j}")
                big_X.append(big)
                big_y.append(int(y[j]))
                # per-big-window metadata: take the j-th row of m
                big_metas.append(_row_meta(m, j))
        big_wd = _build_windowed(big_X, big_y, big_metas, config)
        return GroupedWindowedDataset(
            big=big_wd, small_windows=smalls, big_window_id=np.asarray(big_ids),
            y=np.asarray(big_y, dtype=int), metadata=big_wd.metadata,
        )


def _row_meta(m, j):
    from hand_gesture_recognition.data.windowing.base import WindowMetadata
    return WindowMetadata(
        window_start=m.window_start[j:j + 1],
        window_end=m.window_end[j:j + 1],
        subject_ids=m.subject_ids[j:j + 1],
        trial_ids=m.trial_ids[j:j + 1],
        window_ids=m.window_ids[j:j + 1],
    )


register_windowing("recognition")(RecognitionWindowing)
register_windowing("classification")(ClassificationWindowing)
register_windowing("two_level")(TwoLevelWindowing)
```

Two-level small pass reuses `window_emg` with `rest_bounds=None` per big window (label = big window's label). `_big_config(config)` builds a `WindowingConfig` whose `window_size_samples=stride` come from `resolve_big` (already on `WindowingConfig`).

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_two_level_windowing.py -v`
Expected: PASS.

- [ ] **Step 5: Delete old modules and commit**

```bash
git rm src/hand_gesture_recognition/data/windowing/two_level.py src/hand_gesture_recognition/data/windowing/dispatch.py
git add src/hand_gesture_recognition/data/windowing/
git commit -m "refactor: compositional two-level windowing via strategy"
```

---

## Phase 2 — Split & CV strategies

### Task 2.1: `SplitStrategy` registry

**Files:**
- Create: `src/hand_gesture_recognition/data/splits/strategy.py`
- Modify: `src/hand_gesture_recognition/data/splits/base.py` (re-export `split_dataset` via registry)

**Interfaces:**
- Consumes: `Dataset`, `Dataset.from_trials`, `SplitConfig`.
- Produces: `register_split(name)` / `get_split_strategy(name)`; `SplitStrategy.split(dataset, config) -> DataSplit`.

- [ ] **Step 1: Move each `if strategy == "..."` branch of `split_dataset` into its own class (`RandomSplit`, `StratifiedSplit`, `GroupWiseSplit`, `SessionWiseSplit`, `TrialWiseSplit`, `PredefinedSplit`), all implementing `split(self, dataset, config) -> DataSplit`. `split_dataset` becomes `return get_split_strategy(config.strategy).split(dataset, config)`.

- [ ] **Step 2: Run split regression**

Run: `python -m pytest tests/unit/test_splits.py -v`
Expected: PASS (same partitions for same seed — the group counts and trial indices are unchanged because the logic is moved verbatim).

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/data/splits/
git commit -m "refactor: split strategy registry"
```

### Task 2.2: `CVStrategy` registry

**Files:**
- Modify: `src/hand_gesture_recognition/data/splits/folds.py`, `strategy.py`

- [ ] **Step 1: Move `kfold` and `loso` generation into `KFoldCVStrategy` and `LosoCVStrategy` (each `generate(dataset, cv, split) -> List[DataSplit]`), reusing `_carve_val_group`/`_carve_val_trial` verbatim. `generate_folds` becomes `return get_cv_strategy(cv.method).generate(dataset, cv, split)`.

- [ ] **Step 2: Run CV regression**

Run: `python -m pytest tests/unit/test_cv.py tests/integration/test_cv.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/data/splits/
git commit -m "refactor: CV strategy registry"
```

---

## Phase 3 — Preprocessing pipeline

### Task 3.1: `PreprocessingPipeline`

**Files:**
- Create: `src/hand_gesture_recognition/preprocessing/pipeline.py`

**Interfaces:**
- Consumes: `BaselineRemoval`, signal transformers (`signal.py`), `crop_to_gesture`, `DataSplit`, `Dataset`.
- Produces:
  - `PreprocessingPipeline(pre_steps: List, post_steps: List)`
  - `.fit(train_split) -> PreprocessingPipeline` (fits stateful post-steps on train partition)
  - `.transform(split: DataSplit) -> DataSplit`
  - `.transform_dataset(dataset) -> Dataset` (pre-steps only)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_preprocessing.py` add:

```python
import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata, Trial, User
from hand_gesture_recognition.data.splits.base import DataSplit
from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval
from hand_gesture_recognition.preprocessing.pipeline import PreprocessingPipeline
from hand_gesture_recognition.preprocessing.signal import RectifyTransformer


def _ds(values):
    meta = DatasetMetadata("x", "0", 200.0, ["c"], ["noGesture", "fist"], "h")
    rest = User("u1", [Trial("t0", "u1", "noGesture", -1, -1, np.full((4, 1), 5.0, np.float32))])
    gest = User("u1", [Trial("t1", "u1", "fist", 0, 4, np.full((4, 1), 10.0, np.float32))])
    return Dataset(meta, [rest, gest])


def test_pipeline_fits_baseline_on_train_then_rectifies():
    train = _ds(10.0)
    val = _ds(10.0)
    split = DataSplit(train=train, val=val, test=val)
    p = PreprocessingPipeline(pre_steps=[], post_steps=[BaselineRemoval(mode="global"), RectifyTransformer()])
    p.fit(split)
    out = p.transform(split)
    # rest level 5 subtracted from all trials, then rectified to >= 0
    assert np.all(out.train.trials[1].emg >= 0.0)
    assert np.all(out.val.trials[0].emg >= 0.0)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_preprocessing.py -v`
Expected: FAIL (`PreprocessingPipeline` not defined).

- [ ] **Step 3: Implement** `PreprocessingPipeline` with explicit `fit`/`transform` semantics; `transform` maps `transform_dataset` over `train/val/test` after applying the fitted post-steps (baseline fit on train, transform applied to all three).

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_preprocessing.py tests/unit/test_baseline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/preprocessing/pipeline.py tests/unit/test_preprocessing.py
git commit -m "feat: PreprocessingPipeline with fit/transform"
```

### Task 3.2: Wire preprocessing into the runner (pre-split crop + post-split baseline/signal)

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`

- [ ] **Step 1: Replace `_gesture_crop`/`_signal_processing`/baseline branches in `_run_partition` with a single `PreprocessingPipeline` built from config (`pre_steps=[crop]` when recognition, `post_steps=[baseline] + signal_steps`), `fit` on train, `transform` the split. Preserve the two-level special case (per-big-window filtering) by passing `signal_steps` to the windowing strategy instead (see Phase 4).

- [ ] **Step 2: Run smoke tests**

Run: `python -m pytest tests/integration/test_smoke.py tests/integration/test_task_classification.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py
git commit -m "refactor: runner uses PreprocessingPipeline"
```

---

## Phase 4 — Training strategies

### Task 4.1: Shared `FeatureExtractor` application

**Files:**
- Create: `src/hand_gesture_recognition/training/features.py`

**Interfaces:**
- Consumes: `FeatureExtractionConfig.resolve_features()`, Hydra `instantiate`.
- Produces: `FeatureExtractor(feature_specs)` with `.fit_transform(X) -> np.ndarray` and `.transform(X) -> np.ndarray` (FeatureUnion-style column concatenation), and `.transform_windows(list_of_X) -> np.ndarray` for per-small-window two-level application.

- [ ] **Step 1: Extract the concatenation loop currently in `_train_and_predict_classical` into `FeatureExtractor`. Add a unit test that two extractors (`TimeDomainFeatures` + a stub) concatenate column-wise.

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_frequency_domain.py -v`
Expected: PASS (plus the new feature test).

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/training/features.py tests/unit
git commit -m "feat: shared FeatureExtractor application"
```

### Task 4.2: `TrainingStrategy` base + registry

**Files:**
- Create: `src/hand_gesture_recognition/training/base.py`

**Interfaces:**
- Produces: `register_training(name)` / `get_training_strategy(name)`; `TrainingStrategy.train_predict(windowed, class_names, cfg) -> PartitionPredictions`.

- [ ] **Step 1: Implement the registry (same pattern as `register_windowing`). The base class is a plain class with the single method signature above (no ABC — only two concrete implementations exist).

- [ ] **Step 2: Commit**

```bash
git add src/hand_gesture_recognition/training/base.py
git commit -m "feat: TrainingStrategy registry"
```

### Task 4.3: `ClassicalTrainingStrategy`

**Files:**
- Create: `src/hand_gesture_recognition/training/classical.py`

- [ ] **Step 1: Move the body of `_train_and_predict_classical` into `ClassicalTrainingStrategy.train_predict`, replacing the inline preprocessing/feature/model/predict code with `FeatureExtractor` + the sklearn `SklearnPredictionAdapter`. Two-level classical = flatten the grouped result's `big` view and reuse the same path (replaces `_train_and_predict_classical_two_level`).

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_two_level_classical.py tests/integration/test_partition_evaluation.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/training/classical.py
git commit -m "refactor: ClassicalTrainingStrategy"
```

### Task 4.4: `NeuralTrainingStrategy` + single grouped datamodule

**Files:**
- Create: `src/hand_gesture_recognition/training/neural.py`
- Modify: `src/hand_gesture_recognition/neural/datamodules/emg.py`
- Delete: `src/hand_gesture_recognition/neural/datamodules/two_level.py`

- [ ] **Step 1: Extend `EMGDataModule` (or add `GroupedEMGDataset`) to accept a grouped result: when given `GroupedWindowedPartitions`, build sequences of small-window features per big window (feature extraction + padding to `max_small_per_big`), reusing `FeatureExtractor`. This absorbs `TwoLevelEMGDataset`.

- [ ] **Step 2: Move the body of `_train_and_predict_neural` and `_train_and_predict_neural_two_level` into `NeuralTrainingStrategy.train_predict`, which picks the flat vs grouped datamodule based on the windowed input type (no `if mode` in the runner).

- [ ] **Step 3: Run**

Run: `python -m pytest tests/integration/test_two_level_neural.py tests/integration/test_cross_backend_parity.py tests/unit/test_neural.py -v`
Expected: PASS (delete/replace `tests/unit/test_two_level_datamodule.py`).

- [ ] **Step 4: Commit**

```bash
git rm src/hand_gesture_recognition/neural/datamodules/two_level.py tests/unit/test_two_level_datamodule.py
git add src/hand_gesture_recognition/training/neural.py src/hand_gesture_recognition/neural/datamodules/emg.py
git commit -m "refactor: NeuralTrainingStrategy + single grouped datamodule"
```

---

## Phase 5 — Pipeline + Builder

### Task 5.1: `Pipeline` (Template Method)

**Files:**
- Create: `src/hand_gesture_recognition/runner/pipeline.py`

**Interfaces:**
- Produces: `Pipeline` with ordered phases `load → analytics → preprocess → split/CV → window → train/predict → evaluate → track/report`; hook methods are overridable; strategies are constructor-injected.

- [ ] **Step 1: Implement `Pipeline.run() -> EvaluationResult` as a sequence of phase calls, moving the orchestration logic out of `ExperimentRunner.run()` verbatim. Each phase is a small method (no task/backend conditionals — those now live in strategies).

- [ ] **Step 2: Commit**

```bash
git add src/hand_gesture_recognition/runner/pipeline.py
git commit -m "feat: Pipeline template method"
```

### Task 5.2: `PipelineBuilder`

**Files:**
- Modify: `src/hand_gesture_recognition/runner/pipeline.py`

- [ ] **Step 1: Add a fluent `PipelineBuilder` that resolves strategies from `ExperimentConfig` (`get_split_strategy`, `get_cv_strategy`, `get_windowing_strategy`, `get_training_strategy`) and builds a `Pipeline`. Provide `PipelineBuilder(cfg).build() -> Pipeline`.

- [ ] **Step 2: Commit**

```bash
git add src/hand_gesture_recognition/runner/pipeline.py
git commit -m "feat: PipelineBuilder"
```

### Task 5.3: Reduce `ExperimentRunner` to a facade

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`

- [ ] **Step 1: `ExperimentRunner.run()` becomes: start tracker → build pipeline → `pipeline.run()` → finish tracker. Move analytics/visualization/reporting calls into the pipeline phases. Delete all `_train_and_predict_*`, `_window`, `_signal_processing`, `_gesture_crop` methods.

- [ ] **Step 2: Run the full smoke set**

Run: `python -m pytest tests/integration/test_smoke.py tests/integration/test_two_level_classical.py tests/integration/test_two_level_neural.py tests/integration/test_cross_backend_parity.py tests/integration/test_task_classification.py tests/integration/test_partition_evaluation.py tests/integration/test_cv.py tests/integration/test_replay.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py
git commit -m "refactor: ExperimentRunner reduced to facade"
```

---

## Phase 6 — Grouped evaluation + hardening

### Task 6.1: Enrich `Prediction` with lineage

**Files:**
- Modify: `src/hand_gesture_recognition/prediction/base.py`, `src/hand_gesture_recognition/classical/adapters/sklearn.py`, `src/hand_gesture_recognition/neural/adapters/lightning.py`

- [ ] **Step 1: Add `subject_ids` and `trial_ids` to `Prediction`. `SklearnPredictionAdapter` copies them from `windowed.metadata`; `LightningPredictionAdapter` collects `subject_id`/`trial_id` from `predict_step` dicts (already present — see `neural/models/base.py` `predict_step`). Update `tests/unit/test_prediction.py`.

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_prediction.py tests/integration/test_partition_evaluation.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/prediction/base.py src/hand_gesture_recognition/classical/adapters/sklearn.py src/hand_gesture_recognition/neural/adapters/lightning.py tests/unit/test_prediction.py
git commit -m "feat: Prediction carries subject/trial lineage"
```

### Task 6.2: `evaluate_grouped`

**Files:**
- Create: `src/hand_gesture_recognition/evaluation/grouped.py`

- [ ] **Step 1: Write the failing test

`tests/unit/test_evaluation.py` add:

```python
from hand_gesture_recognition.evaluation.grouped import evaluate_grouped
from hand_gesture_recognition.prediction.base import Prediction
from hand_gesture_recognition.configuration.schema import EvaluationConfig
import numpy as np


def test_evaluate_grouped_by_subject():
    p = Prediction(
        run_id="r", sample_ids=np.array(["a0", "a1", "b0"]),
        subject_ids=np.array(["A", "A", "B"]),
        trial_ids=np.array(["a", "a", "b"]),
        y_true=np.array([0, 0, 1]), y_pred=np.array([0, 1, 1]),
        y_score=None, prediction_duration_s=0.0, backend="classical",
    )
    out = evaluate_grouped(p, by="subject_id", config=EvaluationConfig(),
                           dataset_id="d", class_names=["c0", "c1"])
    assert set(out) == {"A", "B"}
    assert out["A"].metrics["accuracy"] == 0.5
    assert out["B"].metrics["accuracy"] == 1.0
```

- [ ] **Step 2: Run to verify FAIL**

Run: `python -m pytest tests/unit/test_evaluation.py::test_evaluate_grouped_by_subject -v`
Expected: FAIL.

- [ ] **Step 3: Implement** `evaluate_grouped(prediction, by, config, dataset_id, class_names)` — split the prediction's aligned arrays by the chosen key (`"subject_id"`, `"trial_id"`, or `"y_true"`), skip empty groups (log a warning), and call `evaluate()` per group.

- [ ] **Step 4: Run to verify PASS**

Run: `python -m pytest tests/unit/test_evaluation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/evaluation/grouped.py tests/unit/test_evaluation.py
git commit -m "feat: evaluate_grouped (per-subject/per-gesture)"
```

### Task 6.3: Add `evaluation.breakdown_by` config + reporting, and final cleanup

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`, `src/hand_gesture_recognition/configuration/loader.py`, `src/hand_gesture_recognition/runner/pipeline.py`, `src/hand_gesture_recognition/reporting/reporters.py`

- [ ] **Step 1: Add `breakdown_by: List[str] = field(default_factory=list)` to `EvaluationConfig`; coerce in `loader._coerce` (`EvaluationConfig(**(d.get("evaluation") or {}))` already passes it through). In the pipeline's evaluate phase, when `breakdown_by` is non-empty, run `evaluate_grouped` per key and log per-group metrics to the tracker (metric name `f"{metric}_{group}"`); extend `render_report` to emit a per-group table.

- [ ] **Step 2: Add a schema test** in `tests/unit/test_config_schema.py` asserting `breakdown_by=["subject"]` parses and validates.

- [ ] **Step 3: Full suite**

Run: `python -m pytest -q`
Expected: PASS (excluding any long-running `tests/datasets/` real-data tests, which may be marked/skipped).

- [ ] **Step 4: Remove dead aliases (`classical/pipelines/windowing.py` if unused), add docstrings, and commit**

```bash
git add -A src tests
git commit -m "feat: grouped evaluation + cleanup"
```

---

## Self-Review notes

- **Spec coverage:** Phase 0 = data model; Phase 1 = compositional windowing (kills `TwoLevelWindowedDataset`); Phase 2 = split/CV strategies; Phase 3 = preprocessing pipeline; Phase 4 = training strategy (kills `TwoLevelEMGDataset`); Phase 5 = Pipeline + Builder (kills god class); Phase 6 = grouped evaluation + `breakdown_by`.
- **Type consistency:** `Dataset.trials/labels/subject_ids/trial_ids` are defined once (Task 0.1) and consumed identically everywhere after. `window_emg` signature is the single windowing primitive; `GroupedWindowedDataset(big, small_windows, big_window_id, y, metadata)` is the two-level contract used by Phases 4–6. `Prediction.subject_ids/trial_ids` are defined in Task 6.1 and consumed by Task 6.2.
- **Ordering preserved:** gesture-crop (pre-split) → baseline → signal (post-split) per spec §5.3; two-level per-big-window filtering preserved by passing `signal_steps` into the windowing strategy.
