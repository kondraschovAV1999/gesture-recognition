# User-Specific Baseline (`per_user`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `baseline.mode: global | per_user` so the pipeline can subtract a per-subject rest baseline (estimated from each subject's own rest trials) instead of the global average.

**Architecture:** `BaselineRemoval` gains a `mode` parameter: `global` keeps today's single per-channel mean; `per_user` builds a `{subject_id → per-channel-mean}` map in `fit(train)` and subtracts each trial's own subject's baseline in `transform`. The runner passes `cfg.baseline.mode` through. Config validation rejects `per_user` under a subject-level split (fail-loudly), and `per_user` raises on any subject without a rest trial in train / without a baseline.

**Tech Stack:** Python 3.10, Hydra, NumPy, scikit-learn, pytest.

## Global Constraints

- Default `mode: global` preserves current behavior; all existing tests stay green.
- `mode` restricted to `("global", "per_user")`.
- `per_user` requires a **trial-level** split (`random`/`stratified`/`trial_wise`); config validation raises otherwise.
- `per_user` **fails loudly** (raises) when a subject has no rest trials in train (fit) or no baseline (transform) — no silent fallback.
- Baseline is estimated from **rest trials only** (`gesture_start < 0`), per subject.
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: `BaselineRemoval(mode)` + unit tests

**Files:**
- Modify: `src/hand_gesture_recognition/preprocessing/baseline.py`
- Create: `tests/unit/test_baseline.py`

**Interfaces:**
- Produces: `BaselineRemoval(mode: str = "global")`; `fit(train: Dataset) -> BaselineRemoval`; `transform(dataset: Dataset) -> Dataset` (unchanged signatures, new `mode`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_baseline.py`:

```python
from dataclasses import replace

import numpy as np
import pytest

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval


def _dataset():
    # s1: gesture trial (1.0) + rest trial (5.0); s2: rest trial (9.0)
    return Dataset(
        trials=[
            np.full((40, 2), 1.0, dtype=np.float32),
            np.full((30, 2), 5.0, dtype=np.float32),
            np.full((20, 2), 9.0, dtype=np.float32),
        ],
        labels=np.array([1, 0, 0]),
        subject_ids=np.array(["s1", "s1", "s2"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        gesture_start=np.array([10, -1, -1]),
        gesture_end=np.array([30, -1, -1]),
        metadata=DatasetMetadata(
            name="x", version="0", sampling_rate=200.0,
            channels=["c1", "c2"], class_names=["rest", "fist"], content_hash="h",
        ),
    )


def test_global_mode_unchanged():
    rem = BaselineRemoval(mode="global").fit(_dataset())
    # global = mean of both rest trials (5 + 9) / 2 = 7 per channel
    assert np.allclose(rem.baseline_, np.array([7.0, 7.0]))
    out = rem.transform(_dataset())
    assert np.allclose(out.trials[0], -6.0)  # 1 - 7
    assert np.allclose(out.trials[2], 2.0)   # 9 - 7


def test_per_user_builds_subject_baselines():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    assert set(rem.baseline_.keys()) == {"s1", "s2"}
    assert np.allclose(rem.baseline_["s1"], 5.0)
    assert np.allclose(rem.baseline_["s2"], 9.0)


def test_per_user_subtracts_own_subject_baseline():
    ds = _dataset()
    rem = BaselineRemoval(mode="per_user").fit(ds)
    out = rem.transform(ds)
    assert np.allclose(out.trials[0], -4.0)  # s1 gesture: 1 - 5
    assert np.allclose(out.trials[1], 0.0)   # s1 rest: 5 - 5
    assert np.allclose(out.trials[2], 0.0)   # s2 rest: 9 - 9


def test_per_user_raises_when_subject_missing_rest():
    ds = _dataset()
    ds.labels = np.array([1, 0, 1])
    ds.gesture_start = np.array([10, -1, 0])
    ds.gesture_end = np.array([30, -1, 20])
    with pytest.raises(ValueError, match="no rest trials"):
        BaselineRemoval(mode="per_user").fit(ds)


def test_per_user_raises_on_unseen_subject():
    rem = BaselineRemoval(mode="per_user").fit(_dataset())
    ds2 = replace(
        _dataset(),
        trials=_dataset().trials + [np.full((10, 2), 3.0, dtype=np.float32)],
        labels=np.append(_dataset().labels, 0),
        subject_ids=np.append(_dataset().subject_ids, "s3"),
        trial_ids=np.append(_dataset().trial_ids, "t4"),
        gesture_start=np.append(_dataset().gesture_start, -1),
        gesture_end=np.append(_dataset().gesture_end, -1),
    )
    with pytest.raises(ValueError, match="no baseline for subject"):
        rem.transform(ds2)


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="must be fit"):
        BaselineRemoval(mode="per_user").transform(_dataset())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_baseline.py -v`
Expected: FAIL — `BaselineRemoval` has no `mode` parameter (`TypeError` on the first test).

- [ ] **Step 3: Implement `BaselineRemoval(mode)`**

Replace `src/hand_gesture_recognition/preprocessing/baseline.py` with:

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator

from hand_gesture_recognition.data.datasets.base import Dataset


class BaselineRemoval(BaseEstimator):
    """Subtract a per-channel mean rest level.

    ``mode="global"`` estimates a single per-channel mean from the *train
    partition's* noGesture trials (``gesture_start < 0``) and subtracts it from
    every trial. ``mode="per_user"`` estimates one per-channel mean **per
    subject** from that subject's own rest trials and subtracts each trial's own
    subject's baseline (raising if a subject has no rest trials or no baseline).
    """

    def __init__(self, mode: str = "global"):
        self.mode = mode
        self.baseline_ = None

    def fit(self, train: Dataset) -> "BaselineRemoval":
        if self.mode == "global":
            rest = [t for t, gs in zip(train.trials, train.gesture_start) if int(gs) < 0]
            if not rest:
                raise ValueError(
                    "baseline removal requires noGesture (rest) trials in the train partition"
                )
            self.baseline_ = np.mean(np.concatenate(rest, axis=0), axis=0).astype(
                np.float32
            )
        elif self.mode == "per_user":
            self.baseline_ = {}
            for subj in np.unique(train.subject_ids.astype(str)):
                rest = [
                    t
                    for t, gs, s in zip(
                        train.trials, train.gesture_start, train.subject_ids
                    )
                    if str(s) == subj and int(gs) < 0
                ]
                if not rest:
                    raise ValueError(
                        f"subject {subj} has no rest trials in the train partition; "
                        "per_user baseline requires rest trials for every subject"
                    )
                self.baseline_[subj] = np.mean(
                    np.concatenate(rest, axis=0), axis=0
                ).astype(np.float32)
        else:
            raise ValueError(f"unknown baseline mode: {self.mode!r}")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if self.baseline_ is None:
            raise ValueError("BaselineRemoval must be fit before transform")
        if self.mode == "global":
            trials = [t.astype(np.float32) - self.baseline_ for t in dataset.trials]
        else:
            trials = []
            for t, s in zip(dataset.trials, dataset.subject_ids):
                b = self.baseline_.get(str(s))
                if b is None:
                    raise ValueError(
                        f"no baseline for subject {s}; per_user baseline requires "
                        "the subject to appear in train"
                    )
                trials.append(t.astype(np.float32) - b)
        return replace(dataset, trials=trials)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_baseline.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/preprocessing/baseline.py tests/unit/test_baseline.py
git commit -m task1-per-user-baseline
```

---

### Task 2: Config — `mode` field, validation, `per_user` preset

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `configs/baseline/default.yaml`
- Create: `configs/baseline/per_user.yaml`
- Test: `tests/unit/test_config_schema.py` (append), `tests/unit/test_config_loader.py` (append)

**Interfaces:**
- Produces: `BaselineConfig.mode: str = "global"`.
- Produces: validation errors for bad `mode` and for `per_user` + subject-level split.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_schema.py`:

```python
def test_baseline_mode_defaults_global():
    from hand_gesture_recognition.configuration.schema import BaselineConfig
    assert BaselineConfig().mode == "global"


def test_baseline_rejects_bad_mode():
    cfg = _base_experiment()
    cfg.baseline.mode = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_per_user_requires_trial_level_split():
    cfg = _base_experiment()
    cfg.baseline.enabled = True
    cfg.baseline.mode = "per_user"
    cfg.split.strategy = "subject_wise"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)
```

Append to `tests/unit/test_config_loader.py`:

```python
def test_compose_loads_per_user_baseline():
    cfg = load_config(mode="compose", overrides=["baseline=per_user"])
    assert cfg.baseline.mode == "per_user"
    assert cfg.baseline.enabled is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: FAIL (`BaselineConfig` has no `mode`; no `baseline/per_user.yaml` preset).

- [ ] **Step 3: Add the `mode` field**

In `schema.py`, change `BaselineConfig`:

```python
@dataclass
class BaselineConfig:
    enabled: bool = False
    mode: str = "global"  # "global" | "per_user"
```

- [ ] **Step 4: Add validation**

In `validate_experiment_config` (after the `task` block, before the `seed` check), add:

```python
    if cfg.baseline.mode not in ("global", "per_user"):
        raise ConfigValidationError(
            f"baseline.mode={cfg.baseline.mode!r} not in ('global', 'per_user')"
        )
    if (
        cfg.baseline.enabled
        and cfg.baseline.mode == "per_user"
        and cfg.split.strategy in ("subject_wise", "session_wise", "predefined")
    ):
        raise ConfigValidationError(
            "baseline.mode=per_user requires a trial-level split "
            "(random/stratified/trial_wise)"
        )
```

- [ ] **Step 5: Add the config files**

In `configs/baseline/default.yaml`, add `mode: global` after `enabled: true`.

Create `configs/baseline/per_user.yaml`:

```yaml
# @package baseline
enabled: true
mode: per_user
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS (all tests, including the 4 new ones)

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py configs/baseline/default.yaml configs/baseline/per_user.yaml tests/unit/test_config_schema.py tests/unit/test_config_loader.py
git commit -m task2-per-user-baseline-config
```

---

### Task 3: Runner wiring + docs

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Modify: `docs/experiments/configuration.md`
- Modify: `docs/experiments/running.md`

**Interfaces:**
- Consumes: `BaselineConfig.mode` (Task 2), `BaselineRemoval(mode)` (Task 1).

- [ ] **Step 1: Pass `mode` at both runner call sites**

In `src/hand_gesture_recognition/runner/run.py`, replace `BaselineRemoval().fit(...)` with `BaselineRemoval(mode=self.cfg.baseline.mode).fit(...)` at **both** locations:

1. In `run()` (the visualization-only baseline): `viz_baseline = BaselineRemoval().fit(cropped)` → `viz_baseline = BaselineRemoval(mode=self.cfg.baseline.mode).fit(cropped)`.

2. In `_run_partition` (the training baseline): `baseline_remover = BaselineRemoval().fit(split.train)` → `baseline_remover = BaselineRemoval(mode=self.cfg.baseline.mode).fit(split.train)`.

- [ ] **Step 2: Update `configuration.md`**

In the `### baseline` section of `docs/experiments/configuration.md`, add the `mode` row and a sentence:

```markdown
| `mode` | `global` \| `per_user` | `global` |

`mode=per_user` estimates one baseline per subject from that subject's own rest
trials (subtracting each trial's own subject's baseline) — requires a
trial-level split (`random`/`stratified`/`trial_wise`). Preset: `baseline=per_user`.
```

- [ ] **Step 3: Update `running.md`**

In `docs/experiments/running.md`, add a row to the "Common overrides" table (after the `dataset.include_no_gesture` row):

```markdown
| `baseline=per_user` | per-subject rest baseline (trial-level splits only) |
```

- [ ] **Step 4: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py docs/experiments/configuration.md docs/experiments/running.md
git commit -m task3-per-user-baseline-runner-docs
```

---

## Self-Review

- **Spec coverage:** §1 config (`mode` field, `per_user.yaml`) → Task 2; §2 `BaselineRemoval(mode)` → Task 1; §3 runner passes `mode` → Task 3; §4 validation → Task 2; §5 tests/docs → Tasks 1–3.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test commands include expected outcomes.
- **Type consistency:** `BaselineRemoval(mode)` is defined in Task 1 and consumed in Task 3 with the same parameter name; `BaselineConfig.mode` is defined in Task 2 and read in Task 3 as `self.cfg.baseline.mode`; the `per_user` preset name matches the `baseline=per_user` override used in Task 2's loader test.
