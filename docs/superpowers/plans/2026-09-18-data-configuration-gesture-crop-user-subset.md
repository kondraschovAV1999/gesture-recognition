# Data Configuration: Gesture-Context Cropping & Per-Cohort User Subsetting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two data-configuration features: (1) crop each gesture trial to the gesture region ± a symmetric context margin (trimming `noGesture` trials to a comparable length), and (2) load a configurable, seeded-random number of users per cohort instead of all users.

**Architecture:** A `GestureCropConfig` (new config group `gesture_crop`, default-on) drives a pure `crop_to_gesture(dataset, config) -> Dataset` transform applied in the runner between load and signal-processing. `EPN612Loader` gains `max_train_users`/`max_test_users`/`user_seed` and samples users during discovery so only the chosen users are parsed.

**Tech Stack:** Python 3.10, numpy, dataclasses, Hydra/OmegaConf.

## Global Constraints

- Package name `hand_gesture_recognition`; source under `src/hand_gesture_recognition/`.
- Windows shell: run tests with `python -m pytest` (never bare `pytest`); commit with single-token `git commit -m <token>`; never use `python -c "..."` (quote-stripping) — write a `.py` file instead.
- MLflow backend is `sqlite:///mlflow.db`; tests must override `tracking.tracking_uri` to a `tmp_path`.
- Synthetic dataset has `gesture_start=0, gesture_end=length` for every trial, so `crop_to_gesture` is a **no-op** on synthetic data — existing smoke tests must stay green with cropping enabled.
- `crop_to_gesture` must NOT change `labels`, `subject_ids`, `trial_ids`, `metadata`, or `subject_info` — only `trials` and `gesture_start`/`gesture_end`.
- Shipped defaults: `gesture_crop: default` (enabled, `pad_samples: 100`); `dataset.loader.max_train_users: 50`, `max_test_users: 50`, `user_seed: 42`.

---

### Task 1: `GestureCropConfig` schema + config group wiring

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/configuration/loader.py`
- Create: `configs/gesture_crop/none.yaml`, `configs/gesture_crop/default.yaml`
- Modify: `configs/config.yaml`
- Test: `tests/unit/test_config_loader.py`

**Interfaces:**
- Consumes: existing `ExperimentConfig`, `_coerce`, Hydra config layout.
- Produces: `GestureCropConfig(enabled=False, pad_samples=100, no_gesture_max_length_samples=None)` and `ExperimentConfig.gesture_crop` — used by Tasks 2 and 3.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_loader.py`:

```python
def test_gesture_crop_config_default_enabled():
    cfg = load_config(mode="compose")
    assert cfg.gesture_crop.enabled is True
    assert cfg.gesture_crop.pad_samples == 100
    assert cfg.gesture_crop.no_gesture_max_length_samples is None


def test_gesture_crop_config_can_be_disabled():
    cfg = load_config(mode="compose", overrides=["gesture_crop=none"])
    assert cfg.gesture_crop.enabled is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_config_loader.py -q`
Expected: FAIL — `load_config` errors: `Could not find gesture_crop` (unknown config group) or `ExperimentConfig` has no `gesture_crop` attribute.

- [ ] **Step 3: Implement**

**(a)** In `src/hand_gesture_recognition/configuration/schema.py`, add the dataclass after `WindowingConfig` (line ~64):

```python
@dataclass
class GestureCropConfig:
    enabled: bool = False
    pad_samples: int = 100  # symmetric context kept before AND after the gesture
    no_gesture_max_length_samples: Optional[int] = None  # None = auto-derive
```

Add the field to `ExperimentConfig` (after the `signal_processing` field, ~line 194):

```python
    gesture_crop: GestureCropConfig = field(default_factory=GestureCropConfig)
```

**(b)** In `src/hand_gesture_recognition/configuration/loader.py`, add `GestureCropConfig` to the `from hand_gesture_recognition.configuration.schema import (...)` list, and in `_coerce` add this argument to the `ExperimentConfig(...)` call (next to `signal_processing=...`):

```python
        gesture_crop=GestureCropConfig(**(d.get("gesture_crop") or {})),
```

**(c)** Create `configs/gesture_crop/none.yaml`:

```yaml
# @package gesture_crop
enabled: false
```

Create `configs/gesture_crop/default.yaml`:

```yaml
# @package gesture_crop
enabled: true
pad_samples: 100
```

**(d)** In `configs/config.yaml`, add the default group to the `defaults` list (right after `- signal_processing: emg`):

```yaml
  - gesture_crop: default
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config_loader.py -q`
Expected: PASS (2 new tests + existing).

Then run the full suite to catch config-serialization regressions (especially the replay roundtrip, which now serializes `gesture_crop`):
Run: `python -m pytest -q`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py configs/gesture_crop/none.yaml configs/gesture_crop/default.yaml configs/config.yaml tests/unit/test_config_loader.py
git commit -m feat-gesture-crop-config
```

---

### Task 2: `crop_to_gesture` transform

**Files:**
- Create: `src/hand_gesture_recognition/data/gesture_crop.py`
- Test: `tests/unit/test_gesture_crop.py` (new file)

**Interfaces:**
- Consumes: `GestureCropConfig` (Task 1), `Dataset` (`trials`, `gesture_start`, `gesture_end`, `n_trials`).
- Produces: `crop_to_gesture(dataset: Dataset, config: GestureCropConfig) -> Dataset`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_gesture_crop.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.gesture_crop import crop_to_gesture


def _dataset():
    # gesture trial: length 500, gesture [200, 300]; noGesture trial: length 500
    return Dataset(
        trials=[
            np.arange(500, dtype=np.float32).reshape(500, 1),
            np.arange(500, dtype=np.float32).reshape(500, 1),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([200, -1]),
        gesture_end=np.array([300, -1]),
        metadata=DatasetMetadata(
            name="test",
            version="0",
            sampling_rate=200.0,
            channels=["ch1"],
            class_names=["noGesture", "fist"],
            content_hash="x",
        ),
    )


def test_crop_gesture_trial_shifts_bounds():
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    ds = crop_to_gesture(_dataset(), cfg)
    # gesture [200,300] + 100 pad -> crop [100,400] -> length 300
    assert ds.trials[0].shape[0] == 300
    assert ds.gesture_start[0] == 100  # 200 - 100
    assert ds.gesture_end[0] == 200    # 300 - 100
    # noGesture trial truncated to auto = 2*100 + median(100) = 300
    assert ds.trials[1].shape[0] == 300
    assert ds.gesture_start[1] == -1


def test_crop_clamps_at_trial_edges():
    # gesture near start: [10, 60], length 200, pad 100
    ds = _dataset()
    ds.gesture_start = np.array([10, -1])
    ds.gesture_end = np.array([60, -1])
    ds.trials[0] = np.arange(200, dtype=np.float32).reshape(200, 1)
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    # crop_start = max(0, 10-100)=0; crop_end = min(200, 60+100)=160
    assert out.trials[0].shape[0] == 160
    assert out.gesture_start[0] == 10  # 10 - 0
    assert out.gesture_end[0] == 60    # 60 - 0


def test_crop_explicit_no_gesture_length():
    cfg = GestureCropConfig(enabled=True, pad_samples=100, no_gesture_max_length_samples=250)
    ds = crop_to_gesture(_dataset(), cfg)
    assert ds.trials[1].shape[0] == 250  # explicit value wins over auto


def test_crop_disabled_returns_same_object():
    ds = _dataset()
    cfg = GestureCropConfig(enabled=False, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    assert out is ds


def test_crop_preserves_labels_and_ids():
    ds = _dataset()
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    assert np.array_equal(out.labels, ds.labels)
    assert np.array_equal(out.trial_ids, ds.trial_ids)
    assert np.array_equal(out.subject_ids, ds.subject_ids)
    assert out.metadata is ds.metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_gesture_crop.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.gesture_crop'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/hand_gesture_recognition/data/gesture_crop.py`:

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.data.datasets.base import Dataset


def crop_to_gesture(dataset: Dataset, config: GestureCropConfig) -> Dataset:
    """Crop each trial to the gesture region plus a symmetric context margin.

    Gesture trials (gesture_start >= 0) are cropped to
    [gesture_start - pad, gesture_end + pad] (clamped to the trial bounds) and
    their gesture bounds are shifted into the new coordinates. noGesture trials
    (gesture_start < 0) are truncated to no_gesture_max_length_samples (auto =
    2*pad + median gesture length). Trials already shorter than the target are
    left unchanged.
    """
    if not config.enabled:
        return dataset

    pad = config.pad_samples
    gesture_lens = [
        int(g_end) - int(g_start)
        for g_start, g_end in zip(dataset.gesture_start, dataset.gesture_end)
        if int(g_start) >= 0
    ]
    median_len = int(np.median(gesture_lens)) if gesture_lens else None
    max_len = config.no_gesture_max_length_samples
    if max_len is None and median_len is not None:
        max_len = 2 * pad + median_len

    new_trials = []
    new_starts = []
    new_ends = []
    for trial, g_start, g_end in zip(
        dataset.trials, dataset.gesture_start, dataset.gesture_end
    ):
        g_start = int(g_start)
        g_end = int(g_end)
        length = trial.shape[0]
        if g_start < 0:
            # noGesture trial: truncate to max_len when set and longer
            if max_len is not None and length > max_len:
                trial = trial[:max_len]
            new_trials.append(trial)
            new_starts.append(g_start)  # -1
            new_ends.append(g_end)      # -1
        else:
            crop_start = max(0, g_start - pad)
            crop_end = min(length, g_end + pad)
            new_trials.append(trial[crop_start:crop_end])
            new_starts.append(g_start - crop_start)
            new_ends.append(g_end - crop_start)

    return replace(
        dataset,
        trials=new_trials,
        gesture_start=np.asarray(new_starts, dtype=int),
        gesture_end=np.asarray(new_ends, dtype=int),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_gesture_crop.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/gesture_crop.py tests/unit/test_gesture_crop.py
git commit -m feat-gesture-crop-transform
```

---

### Task 3: Runner `_gesture_crop` stage

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/unit/test_gesture_crop.py` (append)

**Interfaces:**
- Consumes: `crop_to_gesture` (Task 2), `cfg.gesture_crop` (Task 1).
- Produces: `ExperimentRunner._gesture_crop(dataset) -> Dataset` and the `gesture_crop` stage in `run()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_gesture_crop.py` (add `import mlflow` is NOT needed; add the imports below):

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_runner_gesture_crop_respects_flag(tmp_path):
    cfg = load_config(mode="compose", overrides=["dataset.name=synthetic"])
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    runner = ExperimentRunner(cfg)
    ds = _dataset()

    cfg.gesture_crop.enabled = False
    assert runner._gesture_crop(ds) is ds

    cfg.gesture_crop.enabled = True
    cropped = runner._gesture_crop(ds)
    assert cropped.trials[0].shape[0] == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_gesture_crop.py::test_runner_gesture_crop_respects_flag -q`
Expected: FAIL with `AttributeError: 'ExperimentRunner' object has no attribute '_gesture_crop'`.

- [ ] **Step 3: Implement**

**(a)** Add the `_gesture_crop` method to `ExperimentRunner` (place it right after `_signal_processing`, ~line 119 in `src/hand_gesture_recognition/runner/run.py`):

```python
    def _gesture_crop(self, dataset):
        if not self.cfg.gesture_crop.enabled:
            return dataset
        from hand_gesture_recognition.data.gesture_crop import crop_to_gesture

        return crop_to_gesture(dataset, self.cfg.gesture_crop)
```

**(b)** In `run()`, insert a `gesture_crop` stage between `analytics` and `signal_processing`, and thread the cropped dataset through. Change the current:

```python
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
            with stage(self.logger, "signal_processing"):
                dataset = self._signal_processing(raw_dataset)
```

to:

```python
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
            with stage(self.logger, "gesture_crop"):
                cropped = self._gesture_crop(raw_dataset)
            with stage(self.logger, "signal_processing"):
                dataset = self._signal_processing(cropped)
```

**(c)** Change the `_visualize` call in the `track_and_report` stage from
`self._visualize(raw_dataset, dataset, split, predictions)` to
`self._visualize(cropped, dataset, split, predictions)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_gesture_crop.py -q`
Expected: PASS (6 tests).

Then run the full suite (the smoke tests now go through the crop stage — no-op on synthetic):
Run: `python -m pytest -q`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/unit/test_gesture_crop.py
git commit -m feat-gesture-crop-runner
```

---

### Task 4: `EPN612Loader` per-cohort user sampling + shipped defaults

**Files:**
- Modify: `src/hand_gesture_recognition/data/loaders/epn612.py`
- Modify: `configs/dataset/epn612.yaml`
- Test: `tests/unit/test_epn612_loader.py` (append)

**Interfaces:**
- Consumes: existing `_discover`, `load`, `_write_user` test fixture.
- Produces: `EPN612Loader(subjects=None, max_train_users=None, max_test_users=None, user_seed=42)`; `load()` samples users per cohort before parsing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_epn612_loader.py`:

```python
def test_epn612_loader_max_users_sampling(tmp_path):
    for uid in range(1, 6):
        _write_user(tmp_path, "trainingJSON", uid)
    for uid in range(1, 5):
        _write_user(tmp_path, "testingJSON", uid)
    ds = EPN612Loader(max_train_users=2, max_test_users=2, user_seed=42).load(str(tmp_path))
    sids = set(ds.subject_ids.tolist())
    assert len({s for s in sids if s.startswith("train:")}) == 2
    assert len({s for s in sids if s.startswith("test:")}) == 2


def test_epn612_loader_max_users_reproducible(tmp_path):
    for uid in range(1, 8):
        _write_user(tmp_path, "trainingJSON", uid)
    a = EPN612Loader(max_train_users=3, user_seed=7).load(str(tmp_path))
    b = EPN612Loader(max_train_users=3, user_seed=7).load(str(tmp_path))
    assert set(a.subject_ids.tolist()) == set(b.subject_ids.tolist())


def test_epn612_loader_max_users_clamps(tmp_path):
    for uid in range(1, 4):
        _write_user(tmp_path, "trainingJSON", uid)
    ds = EPN612Loader(max_train_users=10, user_seed=42).load(str(tmp_path))
    assert len({s for s in ds.subject_ids.tolist() if s.startswith("train:")}) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_epn612_loader.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'max_train_users'`.

- [ ] **Step 3: Implement**

**(a)** In `src/hand_gesture_recognition/data/loaders/epn612.py`, replace the `__init__` (lines 26–28) with:

```python
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,
        max_train_users: Optional[int] = None,
        max_test_users: Optional[int] = None,
        user_seed: int = 42,
    ):
        # subjects are cohort-qualified ids, e.g. "train:user1" / "test:user3"
        self.subjects = set(subjects) if subjects is not None else None
        self.max_train_users = max_train_users
        self.max_test_users = max_test_users
        self.user_seed = user_seed
```

**(b)** In `load()`, change the line `files = self._discover(path)` to:

```python
        files = self._discover(path)
        if self.max_train_users is not None or self.max_test_users is not None:
            self._sample_users(files, path)
```

**(c)** Add the `_sample_users` method (before `_discover`):

```python
    def _sample_users(self, files: List[str], path: str) -> None:
        """Restrict `self.subjects` to a seeded-random sample per cohort."""
        rng = np.random.default_rng(self.user_seed)
        train_dirs: set = set()
        test_dirs: set = set()
        for f in files:
            rel = os.path.relpath(f, path)
            parts = rel.split(os.sep)
            cohort_dir = parts[0] if parts else ""
            subject_dir = parts[1] if len(parts) > 1 else ""
            if "training" in cohort_dir.lower():
                train_dirs.add(subject_dir)
            else:
                test_dirs.add(subject_dir)
        train_dirs = sorted(train_dirs)
        test_dirs = sorted(test_dirs)

        def sample(dirs, n):
            if n is None:
                return set(dirs)
            k = min(n, len(dirs))
            return set(
                rng.choice(np.asarray(dirs, dtype=object), size=k, replace=False).tolist()
            )

        allowed = {f"train:{u}" for u in sample(train_dirs, self.max_train_users)}
        allowed |= {f"test:{u}" for u in sample(test_dirs, self.max_test_users)}
        if self.subjects is not None:
            allowed &= self.subjects
        self.subjects = allowed
```

Note: `np` and `os` and `List` are already imported in this module; the existing per-file subject filter in `load()` then does the actual filtering via `self.subjects`.

**(d)** In `configs/dataset/epn612.yaml`, update the `loader` block:

```yaml
loader:
  _target_: hand_gesture_recognition.data.loaders.epn612.EPN612Loader
  subjects: null  # e.g. override with: dataset.loader.subjects=[train:user1,train:user2]
  max_train_users: 50
  max_test_users: 50
  user_seed: 42
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_epn612_loader.py -q`
Expected: PASS (3 new tests + existing).

Then run the full suite:
Run: `python -m pytest -q`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/loaders/epn612.py configs/dataset/epn612.yaml tests/unit/test_epn612_loader.py
git commit -m feat-user-subset-loader
```

---

## Self-Review Notes (for the implementer)

- Spec coverage: gesture cropping config (Task 1), `crop_to_gesture` (Task 2), runner stage (Task 3), loader user sampling + shipped defaults (Task 4). All spec sections covered.
- The crop transform's auto `noGesture` trim = `2*pad + median(gesture length)`; explicit `no_gesture_max_length_samples` overrides it (Task 2).
- Existing tests stay green because cropping is a no-op on the synthetic dataset (full-trial gesture) and loader sampling only affects real-data runs.
