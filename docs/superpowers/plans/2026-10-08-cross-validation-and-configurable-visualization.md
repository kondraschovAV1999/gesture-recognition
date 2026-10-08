# Cross-Validation, Split Docs, and Configurable Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable cross-validation (k-fold / LOSO), document trial-level vs subject-wise splitting, and make visualization configurable (consistent scale, per-gesture channel selection driving stage-progression and feature-over-windows plots).

**Architecture:** New `cv` and `visualization` config groups (typed dataclasses in `configuration/schema.py`, coerced in `configuration/loader.py`). A `generate_folds` function produces `List[DataSplit]`; the runner's post-split core is extracted into `_run_partition` and looped per fold. Visualization gets a `resolve_selection` source-of-truth, a `compute_stages` snapshot helper, and three plot functions honoring the selection and a shared y-scale.

**Tech Stack:** Python 3.10, dataclasses + Hydra/OmegaConf config, scikit-learn (KFold/GroupKFold/LeaveOneGroupOut), matplotlib, mlflow, pytest.

## Global Constraints

- Python 3.10 (`pyproject.toml` `requires-python = ">=3.10"`); existing deps only (scikit-learn>=1.4, numpy, pandas, hydra-core, omegaconf, mlflow, matplotlib, torch, lightning).
- Follow existing conventions: empty `__init__.py` files, import via full module paths, typed dataclass config in `configuration/schema.py`, validation via `validate_experiment_config`.
- Config group files use `# @package <group>` header; defaults live in `configs/config.yaml`.
- **Baseline/synthetic caveat (robustness rule):** the committed config default is `baseline: none` and the full suite is green. However, `make_synthetic_dataset` has no rest trials, so if baseline is ever enabled (e.g. `baseline=default`), `BaselineRemoval.fit` raises. **Every new integration test MUST therefore override `baseline=none`** so it stays green regardless of the ambient default. Run only the specific test file for each task (the full suite is green but slow due to neural smoke tests).
- Run commands from the repo root: `cd "E:\Research\Dissertation\codes\hand-gesture-recognition"`.
- On this Windows shell, `git commit -m` with spaces breaks under the conda wrapper — use hyphenated commit messages (e.g. `-m feat-add-cv-config`).

---

## Phase 1 — Configuration schema & config files

### Task 1: Add `cv` + `visualization` config dataclasses, rename `reporting.visualize`

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/configuration/loader.py`
- Test: `tests/unit/test_config_schema.py`

**Interfaces:**
- Consumes: existing `ConfigValidationError`, `SplitConfig`, `ExperimentConfig`.
- Produces: `CrossValidationConfig`, `VisualizationConfig`, `VisualizationSelectionConfig`, `FeatureOverWindowsConfig` (all with `.validate()` where applicable); `ExperimentConfig.cv` and `ExperimentConfig.visualization`; `ReportingConfig` without `visualize`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config_schema.py`:

```python
def test_cv_config_defaults_disabled():
    from hand_gesture_recognition.configuration.schema import CrossValidationConfig

    cv = CrossValidationConfig()
    assert cv.enabled is False
    assert cv.method == "kfold"


def test_cv_config_validation_rejects_bad_method():
    from hand_gesture_recognition.configuration.schema import (
        CrossValidationConfig,
    )

    with pytest.raises(ConfigValidationError):
        CrossValidationConfig(method="bogus").validate()


def test_visualization_config_defaults():
    from hand_gesture_recognition.configuration.schema import (
        FeatureOverWindowsConfig,
        VisualizationConfig,
    )

    v = VisualizationConfig()
    assert v.enabled is True
    assert v.consistent_scale is True
    assert v.feature_over_windows.enabled is True
    assert v.feature_over_windows.features == []


def test_experiment_config_has_cv_and_visualization():
    cfg = _base_experiment()
    assert cfg.cv.enabled is False
    assert cfg.visualization.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config_schema.py -q`
Expected: FAIL with `ImportError: cannot import name 'CrossValidationConfig'`.

- [ ] **Step 3: Write minimal implementation**

In `schema.py`, add these dataclasses right after `SplitConfig` (before `_CLASSICAL_FEATURE_TYPES`):

```python
@dataclass
class CrossValidationConfig:
    enabled: bool = False
    method: str = "kfold"          # kfold | loso
    n_splits: int = 5              # kfold only
    group_folds: bool = True       # kfold only: subject-wise vs trial-level
    include_val: bool = True
    val_ratio: float = 0.2         # fraction of the training fold carved into val
    seed: int = 42
    report_per_fold: bool = True

    def validate(self) -> None:
        if self.method not in ("kfold", "loso"):
            raise ConfigValidationError(f"unknown cv method: {self.method!r}")
        if self.method == "kfold" and self.n_splits < 2:
            raise ConfigValidationError(
                f"cv.n_splits must be >= 2, got {self.n_splits}"
            )
        if not (0.0 < self.val_ratio < 1.0):
            raise ConfigValidationError(
                f"cv.val_ratio must be in (0,1), got {self.val_ratio}"
            )


@dataclass
class VisualizationSelectionConfig:
    gestures: List[str] = field(default_factory=list)   # [] = all classes
    channels: List[int] = field(default_factory=list)   # [] = all channels (1-based)
    by_gesture: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class FeatureOverWindowsConfig:
    enabled: bool = True
    features: List[str] = field(default_factory=list)   # names; [] = all
    max_windows: int = 0                                # 0 = all


@dataclass
class VisualizationConfig:
    enabled: bool = True
    consistent_scale: bool = True
    selection: VisualizationSelectionConfig = field(
        default_factory=VisualizationSelectionConfig
    )
    stage_progression: bool = True
    feature_over_windows: FeatureOverWindowsConfig = field(
        default_factory=FeatureOverWindowsConfig
    )
```

Modify `ReportingConfig` to remove the `visualize` field:

```python
@dataclass
class ReportingConfig:
    formats: List[str] = field(default_factory=lambda: ["json", "html"])
```

Add the two fields to `ExperimentConfig` (after `reporting`):

```python
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
```

In `validate_experiment_config`, add `cfg.cv.validate()` after `cfg.classical.validate()`.

Also, in `configs/experiment/default.yaml`, remove the `visualize: true` line (part of the same rename, so config compose stays consistent):

```yaml
reporting:
  formats: [json, html]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_config_schema.py -q`
Expected: PASS (note: the two new dataclasses also satisfy existing schema tests).

- [ ] **Step 5: Wire into the config loader**

Modify `configuration/loader.py`:
- Add to the schema import block: `CrossValidationConfig`, `FeatureOverWindowsConfig`, `VisualizationConfig`, `VisualizationSelectionConfig`.
- In `_coerce`, after the `reporting=...` line, build the nested visualization dataclass explicitly (OmegaConf won't auto-coerce nested dataclasses):

```python
        vis_raw = d.get("visualization", {})
        sel_raw = vis_raw.get("selection", {})
        fow_raw = vis_raw.get("feature_over_windows", {})
        visualization=VisualizationConfig(
            enabled=vis_raw.get("enabled", True),
            consistent_scale=vis_raw.get("consistent_scale", True),
            selection=VisualizationSelectionConfig(**sel_raw),
            stage_progression=vis_raw.get("stage_progression", True),
            feature_over_windows=FeatureOverWindowsConfig(**fow_raw),
        ),
        cv=CrossValidationConfig(**d.get("cv", {})),
```

Run: `python -m pytest tests/unit/test_config_schema.py -q` (still PASS).

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py tests/unit/test_config_schema.py configs/experiment/default.yaml
git commit -m feat-add-cv-and-visualization-config-schema
```

### Task 2: Add `cv` and `visualization` config group files

**Files:**
- Create: `configs/cv/none.yaml`, `configs/cv/kfold.yaml`, `configs/cv/loso.yaml`
- Create: `configs/visualization/default.yaml`
- Modify: `configs/config.yaml`
- Modify: `configs/experiment/default.yaml`

**Interfaces:**
- Produces: Hydra groups `cv` (`none`/`kfold`/`loso`) and `visualization` (`default`), selected via `cv=kfold` / `visualization=default`.

- [ ] **Step 1: Create the cv config files**

`configs/cv/none.yaml`:
```yaml
# @package cv
enabled: false
```

`configs/cv/kfold.yaml`:
```yaml
# @package cv
enabled: true
method: kfold
n_splits: 5
group_folds: true
include_val: true
val_ratio: 0.2
seed: 42
report_per_fold: true
```

`configs/cv/loso.yaml`:
```yaml
# @package cv
enabled: true
method: loso
include_val: true
val_ratio: 0.2
seed: 42
report_per_fold: true
```

- [ ] **Step 2: Create the visualization config file**

`configs/visualization/default.yaml`:
```yaml
# @package visualization
enabled: true
consistent_scale: true
selection:
  gestures: []
  channels: []
  by_gesture: {}
stage_progression: true
feature_over_windows:
  enabled: true
  features: []
  max_windows: 0
```

- [ ] **Step 3: Register the groups**

In `configs/config.yaml`, add to the `defaults:` list (order matters — put after `experiment`):
```yaml
  - cv: none
  - visualization: default
```

- [ ] **Step 4: Verify config composes**

Run: `python -m pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py::test_compose_mode_returns_validated_config tests/unit/test_config_loader.py::test_replay_mode_roundtrips -q`
Expected: PASS for all three (the loader tests compose/replay a full config and now exercise the new groups).

- [ ] **Step 5: Commit**

```bash
git add configs/cv configs/visualization configs/config.yaml configs/experiment/default.yaml
git commit -m feat-add-cv-and-visualization-config-groups
```

---

## Phase 2 — Cross-validation

### Task 3: `trial_wise` split alias

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/data/splits/base.py`
- Test: `tests/unit/test_splits.py`

**Interfaces:**
- Produces: `SplitConfig(strategy="trial_wise")` behaves as trial-level random split.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_splits.py`:

```python
def test_trial_wise_split_is_trial_level():
    ds = make_synthetic_dataset(n_subjects=4, n_trials_per_subject=5, seed=0)
    cfg = SplitConfig(
        strategy="trial_wise",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    split = split_dataset(ds, cfg)
    assert (
        split.train.n_trials + split.val.n_trials + split.test.n_trials
        == ds.n_trials
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_splits.py::test_trial_wise_split_is_trial_level -q`
Expected: FAIL with `ConfigValidationError: unknown split strategy: trial_wise`.

- [ ] **Step 3: Implement**

In `schema.py`, add `"trial_wise"` to the allowlist tuple in `SplitConfig.validate`.

In `data/splits/base.py`, change the `elif config.strategy == "random":` line to:
```python
    elif config.strategy in ("random", "trial_wise"):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_splits.py -q`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/data/splits/base.py tests/unit/test_splits.py
git commit -m feat-add-trial-wise-split-alias
```

### Task 4: Fold generation — `data/splits/folds.py`

**Files:**
- Create: `src/hand_gesture_recognition/data/splits/folds.py`
- Test: `tests/unit/test_cv.py`

**Interfaces:**
- Consumes: `DataSplit`, `_subset`, `assert_no_group_leakage` (from `data.splits.base`); `CrossValidationConfig`, `SplitConfig`; `Dataset`.
- Produces: `generate_folds(dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig) -> List[DataSplit]`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cv.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import assert_no_group_leakage
from hand_gesture_recognition.data.splits.folds import generate_folds


def test_kfold_group_folds_no_leakage():
    ds = make_synthetic_dataset(n_subjects=6, n_trials_per_subject=4, seed=0)
    cv = CrossValidationConfig(
        enabled=True, method="kfold", n_splits=3, group_folds=True, seed=1
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=1))
    assert len(folds) == 3
    for f in folds:
        assert_no_group_leakage(f)
        assert f.test.n_trials > 0
        assert f.train.n_trials > 0


def test_kfold_trial_folds_cover_all_trials_and_empty_val():
    ds = make_synthetic_dataset(n_subjects=3, n_trials_per_subject=6, seed=0)
    cv = CrossValidationConfig(
        enabled=True,
        method="kfold",
        n_splits=2,
        group_folds=False,
        include_val=False,
        seed=0,
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="random", seed=0))
    assert len(folds) == 2
    assert folds[0].val.n_trials == 0
    assert sum(f.test.n_trials for f in folds) == ds.n_trials


def test_loso_one_fold_per_subject():
    ds = make_synthetic_dataset(n_subjects=5, n_trials_per_subject=2, seed=0)
    cv = CrossValidationConfig(enabled=True, method="loso", include_val=False, seed=0)
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=0))
    assert len(folds) == 5
    for f in folds:
        assert len(set(f.test.subject_ids.tolist())) == 1
        assert_no_group_leakage(f)


def test_kfold_group_val_carved_from_train():
    ds = make_synthetic_dataset(n_subjects=8, n_trials_per_subject=3, seed=0)
    cv = CrossValidationConfig(
        enabled=True, method="kfold", n_splits=4, group_folds=True, seed=2
    )
    folds = generate_folds(ds, cv, SplitConfig(strategy="subject_wise", seed=2))
    for f in folds:
        assert f.val.n_trials > 0
        assert_no_group_leakage(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_cv.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.splits.folds'`.

- [ ] **Step 3: Write the implementation**

Create `src/hand_gesture_recognition/data/splits/folds.py`:

```python
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.splits.base import (
    DataSplit,
    _subset,
    assert_no_group_leakage,
)


def _group_ids(dataset: Dataset, split: SplitConfig) -> np.ndarray:
    return (
        dataset.subject_ids
        if split.split_on == "subject_ids"
        else dataset.trial_ids
    )


def _carve_val_group(
    train_idx: np.ndarray, groups: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a subject-wise val subset out of train_idx (group-level)."""
    rng = np.random.default_rng(seed)
    train_groups = np.unique(groups[train_idx])
    rng.shuffle(train_groups)
    n_val = max(1, int(round(len(train_groups) * val_ratio)))
    val_groups = set(train_groups[:n_val].tolist())
    val_idx = np.asarray([i for i in train_idx if groups[i] in val_groups], dtype=int)
    new_train = np.asarray(
        [i for i in train_idx if groups[i] not in val_groups], dtype=int
    )
    return new_train, val_idx


def _carve_val_trial(
    train_idx: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a trial-wise val subset out of train_idx."""
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def _empty(dataset: Dataset) -> Dataset:
    return _subset(dataset, np.asarray([], dtype=int))


def generate_folds(
    dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig
) -> List[DataSplit]:
    """Produce the list of DataSplits for a cross-validation run."""
    cv.validate()
    n = dataset.n_trials
    indices = np.arange(n)
    folds: List[DataSplit] = []

    def carve(train_idx: np.ndarray):
        if not cv.include_val:
            return train_idx, np.asarray([], dtype=int)
        if cv.method == "loso" or cv.group_folds:
            return _carve_val_group(train_idx, _group_ids(dataset, split), cv.val_ratio, cv.seed)
        return _carve_val_trial(train_idx, cv.val_ratio, cv.seed)

    if cv.method == "loso":
        groups = _group_ids(dataset, split)
        for g in np.unique(groups):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
        return folds

    # kfold
    if cv.group_folds:
        groups = _group_ids(dataset, split)
        splitter = GroupKFold(n_splits=cv.n_splits)
        for train_idx, test_idx in splitter.split(indices, groups=groups):
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
    else:
        splitter = KFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.seed)
        for train_idx, test_idx in splitter.split(indices):
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
    return folds
```

Note: `_subset` with an empty index array returns a `Dataset` with empty lists and a subject_info filtered to no subjects — safe downstream because the runner skips empty-val evaluation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_cv.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/splits/folds.py tests/unit/test_cv.py
git commit -m feat-add-cross-validation-fold-generation
```

### Task 5: Fold aggregation — `evaluation/aggregate.py`

**Files:**
- Create: `src/hand_gesture_recognition/evaluation/aggregate.py`
- Test: `tests/unit/test_cv.py` (append)

**Interfaces:**
- Consumes: `EvaluationResult` (from `evaluation.base`).
- Produces: `aggregate_folds(results: List[EvaluationResult], run_id: str, dataset_id: str) -> EvaluationResult` — metrics contain `{m}` (mean) and `{m}_std`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cv.py`:

```python
from hand_gesture_recognition.evaluation.aggregate import aggregate_folds
from hand_gesture_recognition.evaluation.base import EvaluationResult


def test_aggregate_folds_mean_std():
    results = [
        EvaluationResult(run_id="r", dataset_id="d", metrics={"accuracy": 0.9, "macro_f1": 0.8}),
        EvaluationResult(run_id="r", dataset_id="d", metrics={"accuracy": 0.7, "macro_f1": 0.6}),
    ]
    agg = aggregate_folds(results, "r", "d")
    assert agg.metrics["accuracy"] == 0.8
    assert abs(agg.metrics["accuracy_std"] - 0.1) < 1e-9
    assert agg.metrics["macro_f1"] == 0.7
    assert agg.run_id == "r"
    assert agg.dataset_id == "d"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_cv.py::test_aggregate_folds_mean_std -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.evaluation.aggregate'`.

- [ ] **Step 3: Implement**

Create `src/hand_gesture_recognition/evaluation/aggregate.py`:

```python
from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.evaluation.base import EvaluationResult


def aggregate_folds(
    results: List[EvaluationResult], run_id: str, dataset_id: str
) -> EvaluationResult:
    """Mean and std of each metric across folds, as a single EvaluationResult."""
    if not results:
        return EvaluationResult(
            run_id=run_id, dataset_id=dataset_id, split={"cv": {"n_folds": 0}}
        )
    metrics: dict = {}
    for m in results[0].metrics:
        vals = [r.metrics[m] for r in results]
        metrics[m] = float(np.mean(vals))
        metrics[f"{m}_std"] = float(np.std(vals))
    return EvaluationResult(
        run_id=run_id,
        dataset_id=dataset_id,
        split={"cv": {"n_folds": len(results)}},
        metrics=metrics,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_cv.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/evaluation/aggregate.py tests/unit/test_cv.py
git commit -m feat-add-fold-aggregation
```

### Task 6: Runner refactor — `_run_partition` + CV loop + CV reporting

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/integration/test_cv.py` (new)

**Interfaces:**
- Consumes: `generate_folds`, `aggregate_folds`, `build_no_gesture_remap`, `drop_no_gesture`, `BaselineRemoval`, existing `_window`, `_signal_processing`, `_train_and_predict_*`, `evaluate`.
- Produces: `ExperimentRunner._run_partition(self, split, class_names, remap, dataset_name) -> Tuple[PartitionPredictions, Dict[str, Optional[EvaluationResult]]]`; `_report_cv(...)`; `_write_cv_summary(...)`; `run()` returns an aggregate `EvaluationResult` when `cv.enabled`.

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_cv.py`:

```python
import mlflow

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_cv_kfold_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "cv=kfold",
            "cv.n_splits=3",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()

    assert "accuracy" in result.metrics
    assert "accuracy_std" in result.metrics
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "report.json").exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=cfg.tracking.tracking_uri)
    artifacts = [a.path for a in client.list_artifacts(runner.tracker.run_id)]
    assert any(a.startswith("confusion_matrix_fold0_test") for a in artifacts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_cv.py -q`
Expected: FAIL — `AttributeError: 'ExperimentConfig' object has no attribute 'cv'` (config not yet wired into the runner).

- [ ] **Step 3: Refactor `run()`**

Replace the body of `ExperimentRunner.run()` (from the `try:` block down to the `return result` line) with:

```python
        try:
            with stage(self.logger, "load_dataset"):
                raw_dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
            with stage(self.logger, "gesture_crop"):
                cropped = self._gesture_crop(raw_dataset)

            # Global class set: no-gesture remap computed once from the FULL
            # dataset so every fold/partition shares the same mapping.
            class_names = cropped.metadata.class_names
            remap = None
            if not cfg.dataset.include_no_gesture:
                from hand_gesture_recognition.data.class_selection import (
                    build_no_gesture_remap,
                )

                remap, class_names = build_no_gesture_remap(cropped)
            dataset_name = cropped.metadata.name

            # Visualization-only baseline (fit on the FULL cropped dataset, a
            # diagnostic; training baselines are fit per partition in
            # _run_partition). Fitted only when visualization needs it.
            viz_baseline = None
            if cfg.baseline.enabled and cfg.visualization.enabled:
                from hand_gesture_recognition.preprocessing.baseline import (
                    BaselineRemoval,
                )

                viz_baseline = BaselineRemoval().fit(cropped)

            if cfg.cv.enabled:
                from hand_gesture_recognition.data.splits.folds import generate_folds

                with stage(self.logger, "split"):
                    folds = generate_folds(cropped, cfg.cv, cfg.split)
                fold_test_results = []
                for i, fold in enumerate(folds):
                    with stage(self.logger, f"fold_{i}"):
                        _, results = self._run_partition(
                            fold, class_names, remap, dataset_name
                        )
                    if results["test"] is not None:
                        fold_test_results.append(results["test"])
                from hand_gesture_recognition.evaluation.aggregate import aggregate_folds

                result = aggregate_folds(fold_test_results, cfg.run_id, dataset_name)
                self.logger.info(f"CV metrics (mean): {result.metrics}")
                split, predictions = None, None
            else:
                with stage(self.logger, "split"):
                    split = split_dataset(cropped, cfg.split)
                    if cfg.split.strategy in (
                        "subject_wise",
                        "session_wise",
                        "predefined",
                    ):
                        assert_no_group_leakage(split)
                predictions, results = self._run_partition(
                    split, class_names, remap, dataset_name
                )
                result = results["test"]

            # processed dataset for visualization (baseline first, then signal).
            processed_dataset = (
                viz_baseline.transform(cropped) if viz_baseline is not None else cropped
            )
            processed_dataset = self._apply_signal_processing(processed_dataset)

            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(cropped)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                if cfg.cv.enabled:
                    self._report_cv(result, fold_test_results, class_names)
                    run_dir = os.path.join(
                        self.output_root, "evaluation", self.cfg.run_id
                    )
                    self._write_cv_summary(fold_test_results, run_dir)
                else:
                    for part, r in results.items():
                        if r is None:
                            continue
                        self.tracker.log_results(
                            r, partition=(None if part == "test" else part)
                        )
                        self.tracker.log_confusion_matrix(r, class_names, partition=part)
                self._visualize(
                    cropped,
                    processed_dataset,
                    split,
                    predictions,
                    class_names,
                    viz_baseline,
                    is_cv=cfg.cv.enabled,
                )
                self._persist_artifacts(cropped, result)
            self.tracker.finish("FINISHED")
            return result
        except Exception:
            self.tracker.finish("FAILED")
            raise
```

Add the new methods right after `run()` (before `_load_dataset`):

```python
    def _run_partition(self, split, class_names, remap, dataset_name):
        """Run baseline -> class selection -> signal -> window -> train -> evaluate
        on a single DataSplit. Returns (predictions, {part: result-or-None})."""
        baseline_remover = None
        if self.cfg.baseline.enabled:
            from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval

            baseline_remover = BaselineRemoval().fit(split.train)
            split = DataSplit(
                train=baseline_remover.transform(split.train),
                val=baseline_remover.transform(split.val),
                test=baseline_remover.transform(split.test),
            )
        if not self.cfg.dataset.include_no_gesture:
            from hand_gesture_recognition.data.class_selection import drop_no_gesture

            split = DataSplit(
                train=drop_no_gesture(split.train, remap, class_names),
                val=drop_no_gesture(split.val, remap, class_names),
                test=drop_no_gesture(split.test, remap, class_names),
            )
        split = self._signal_processing(split)
        with stage(self.logger, "windowing"):
            windowed = self._window(split)
        with stage(self.logger, "train"):
            if self.cfg.backend == "classical":
                predictions = self._train_and_predict_classical(windowed)
            else:
                predictions = self._train_and_predict_neural(
                    windowed, n_classes=len(class_names)
                )
        results = {}
        for part in ("train", "val", "test"):
            pred = getattr(predictions, part)
            if pred is None or np.asarray(pred.y_true).size == 0:
                results[part] = None
                continue
            results[part] = evaluate(
                pred,
                self.cfg.evaluation,
                dataset_id=dataset_name,
                split_info={"strategy": self.cfg.split.strategy},
                class_names=class_names,
            )
        return predictions, results

    def _report_cv(self, result, fold_results, class_names):
        self.tracker.log_metrics(result.metrics)
        if self.cfg.cv.report_per_fold:
            for i, r in enumerate(fold_results):
                self.tracker.log_results(r, partition=f"fold{i}")
                self.tracker.log_confusion_matrix(
                    r, class_names, partition=f"fold{i}_test"
                )

    def _write_cv_summary(self, fold_results, run_dir):
        import json

        os.makedirs(run_dir, exist_ok=True)
        summary = {
            "n_folds": len(fold_results),
            "folds": [
                {"fold": i, "metrics": r.metrics}
                for i, r in enumerate(fold_results)
            ],
        }
        path = os.path.join(run_dir, "cv_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.tracker.log_artifact(path)
```

Remove the now-unused `split`/`baseline_remover`/`class_selection`/`signal_processing`/`windowing`/`train`/`evaluate` stage blocks that were previously inline in `run()` (they are all now in `_run_partition`). Keep `_apply_signal_steps`, `_apply_signal_processing`, `_signal_processing`, `_window`, `_train_and_predict_*` unchanged.

- [ ] **Step 4: Update `_visualize` signature**

Change the `_visualize` method signature to accept `viz_baseline` and `is_cv` (implementation updated in Task 12; for now just make it accept and ignore them so the call site compiles):

```python
    def _visualize(
        self,
        raw_dataset,
        processed_dataset,
        split,
        predictions,
        class_names,
        viz_baseline=None,
        is_cv=False,
    ) -> None:
        if not self.cfg.reporting.visualize:  # TEMPORARY: replaced in Task 12
            return
```

Wait — `reporting.visualize` no longer exists after Task 1. The runner's `_visualize` currently reads `self.cfg.reporting.visualize`. In this task, replace that guard temporarily with `self.cfg.visualization.enabled` (the real guard from Task 12). The full `_visualize` body is rewritten in Task 12, but the runner must at least run in this task. Use the temporary guard:

```python
    def _visualize(
        self,
        raw_dataset,
        processed_dataset,
        split,
        predictions,
        class_names,
        viz_baseline=None,
        is_cv=False,
    ) -> None:
        if not self.cfg.visualization.enabled:
            return
        # TEMPORARY: existing body, minus the reporting.visualize guard, will be
        # fully replaced in Task 12. Keep the current plot calls as-is for now.
        from hand_gesture_recognition.reporting.visualization import (
            plot_fft_spectrum,
            plot_predictions_on_signal,
            plot_raw_signal_by_class,
            plot_signal_data_flow,
            plot_signal_processing_comparison,
        )

        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        for p in plot_raw_signal_by_class(raw_dataset, run_dir, None):
            self.tracker.log_artifact(p)
        if not is_cv:
            for part, ds, pred in (
                ("val", split.val, predictions.val),
                ("test", split.test, predictions.test),
            ):
                p = os.path.join(run_dir, f"predictions_on_signal_{part}.png")
                plot_predictions_on_signal(
                    ds, pred, self.cfg.windowing, class_names, path=p
                )
                self.tracker.log_artifact(p)
        if self.cfg.signal_processing:
            p = os.path.join(run_dir, "signal_processing_comparison.png")
            plot_signal_processing_comparison(raw_dataset, processed_dataset, path=p)
            self.tracker.log_artifact(p)
        if not is_cv:
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
        feature_specs = self.cfg.classical.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        if any("frequency_domain" in str(f.get("_target_", "")) for f in feature_specs):
            p = os.path.join(run_dir, "fft_spectrum.png")
            plot_fft_spectrum(
                processed_dataset,
                self.cfg.windowing,
                sampling_rate=processed_dataset.metadata.sampling_rate,
                path=p,
            )
            self.tracker.log_artifact(p)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_cv.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_cv.py
git commit -m feat-add-cv-loop-and-partition-refactor
```

---

## Phase 3 — Documentation

### Task 7: Split-strategy documentation + cross-links

**Files:**
- Create: `docs/experiments/splitting.md`
- Modify: `docs/architecture/overview.md`
- Modify: `README.md`

**Interfaces:** none (prose only).

- [ ] **Step 1: Write the splitting doc**

Create `docs/experiments/splitting.md`:

```markdown
# Splitting strategies

The framework splits a `Dataset` into train / val / test **before** windowing,
at one of two granularities: **trial-level** or **subject-wise** (group-level).

## Trial-level splits (same subject may appear in train and test)

| Strategy | Behavior |
|---|---|
| `random` | Shuffles *trials* and assigns them to train/val/test by ratio. |
| `stratified` | Like `random`, but preserves class proportions (scikit-learn `train_test_split` with `stratify=labels`). |
| `trial_wise` | Explicit alias of `random`; named for discoverability. |

With a trial-level split, different trials of the *same subject* can land in
train and test. This maximizes use of data and can improve within-subject
accuracy, but it risks **subject leakage** (the model sees that subject's
signal statistics during training), so reported accuracy may be optimistic for
unseen users.

## Subject-wise splits (a subject appears in exactly one partition)

| Strategy | Behavior |
|---|---|
| `subject_wise` | Shuffles *subjects* and assigns whole subjects to one partition. |
| `session_wise` | Same grouping mechanism keyed by `split_on` (e.g. session id). |
| `predefined` | Uses the dataset's own cohort markers (e.g. EPN612 `train:`/`test:` prefixes); test cohort held out entirely. |

Subject-wise splits guarantee no subject appears in more than one partition,
which is the honest setting for cross-user generalization. The runner asserts
`assert_no_group_leakage` for these strategies.

## Choosing a strategy

- **Cross-user generalization / realistic deployment** → `subject_wise` (or
  `predefined` for EPN612).
- **Within-user diagnostics / maximum data efficiency** → `random` /
  `stratified` / `trial_wise`.
- **Cross-validation** → set `cv.method=kfold` with `cv.group_folds=true`
  (subject-wise folds) or `cv.group_folds=false` (trial-level folds), or
  `cv.method=loso` (leave-one-subject-out).
```

- [ ] **Step 2: Cross-link**

In `docs/architecture/overview.md`, add a link row to the docs table after the "EMG-EPN612 dataset" row:

```markdown
| [Splitting strategies](docs/experiments/splitting.md) | Trial-level vs subject-wise splits, leakage, and choosing a strategy |
```

In `README.md`, add the same row to its documentation table.

- [ ] **Step 3: Commit**

```bash
git add docs/experiments/splitting.md docs/architecture/overview.md README.md
git commit -m docs-splitting-strategies
```

---

## Phase 4 — Visualization

### Task 8: Feature-name constants on extractors

**Files:**
- Modify: `src/hand_gesture_recognition/features/time_domain.py`
- Modify: `src/hand_gesture_recognition/features/frequency_domain.py`
- Test: `tests/unit/test_visualization.py` (append)

**Interfaces:**
- Produces: `TimeDomainFeatures.FEATURE_NAMES`, `FrequencyDomainFeatures.FEATURE_NAMES` (list of str).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_visualization.py`:

```python
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.features.frequency_domain import FrequencyDomainFeatures


def test_feature_name_constants():
    assert TimeDomainFeatures.FEATURE_NAMES == [
        "rms", "std", "min", "max", "mean", "max_over_mean", "min_over_max",
    ]
    assert FrequencyDomainFeatures.FEATURE_NAMES == [
        "total_power", "peak_frequency", "mean_frequency", "median_frequency",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py::test_feature_name_constants -q`
Expected: FAIL with `AttributeError: type object 'TimeDomainFeatures' has no attribute 'FEATURE_NAMES'`.

- [ ] **Step 3: Implement**

In `time_domain.py`, add a class attribute to `TimeDomainFeatures`:

```python
class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """..."""
    FEATURE_NAMES = ["rms", "std", "min", "max", "mean", "max_over_mean", "min_over_max"]
```

In `frequency_domain.py`, add to `FrequencyDomainFeatures`:

```python
class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    """..."""
    FEATURE_NAMES = ["total_power", "peak_frequency", "mean_frequency", "median_frequency"]
```

(Place the attribute right under the class docstring.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/features/time_domain.py src/hand_gesture_recognition/features/frequency_domain.py tests/unit/test_visualization.py
git commit -m feat-add-feature-name-constants
```

### Task 9: `resolve_selection`, consistent-scale helper, and plot A

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Test: `tests/unit/test_visualization.py` (append)

**Interfaces:**
- Produces:
  - `resolve_selection(selection: VisualizationSelectionConfig, class_names: List[str], n_channels: int) -> Dict[str, List[int]]` (gesture name → 0-based channel indices).
  - `plot_raw_signal(dataset, example_tid=None, path=None, channels=None, consistent_scale=False)`.
  - `plot_raw_signal_by_class(dataset, run_dir, class_names=None, selection=None, consistent_scale=False) -> List[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_visualization.py`:

```python
import matplotlib.pyplot as plt

from hand_gesture_recognition.configuration.schema import VisualizationSelectionConfig
from hand_gesture_recognition.reporting.visualization import (
    plot_raw_signal,
    plot_raw_signal_by_class,
    resolve_selection,
)


def test_resolve_selection_defaults_to_all():
    sel = VisualizationSelectionConfig()
    assert resolve_selection(sel, ["a", "b"], 3) == {"a": [0, 1, 2], "b": [0, 1, 2]}


def test_resolve_selection_channels_and_by_gesture():
    sel = VisualizationSelectionConfig(
        gestures=["a", "b"], channels=[1, 3], by_gesture={"b": [2]}
    )
    assert resolve_selection(sel, ["a", "b", "c"], 4) == {"a": [0, 2], "b": [1]}


def test_plot_raw_signal_by_class_respects_selection(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=3, n_classes=2, length=100, n_channels=4, seed=0
    )
    sel = VisualizationSelectionConfig(gestures=["class_0"], channels=[1, 2])
    selection = resolve_selection(sel, ds.metadata.class_names, 4)
    paths = plot_raw_signal_by_class(
        ds, str(tmp_path), class_names=ds.metadata.class_names,
        selection=selection, consistent_scale=True,
    )
    assert len(paths) == 1
    assert os.path.exists(str(tmp_path / "raw_signal_class_0.png"))


def test_consistent_scale_shares_ylim():
    ds = make_synthetic_dataset(
        n_subjects=1, n_trials_per_subject=1, n_classes=1, length=100, n_channels=3, seed=0
    )
    ds.trials[0][:, 1] *= 100.0
    ds.trials[0][:, 2] *= 0.01
    fig = plot_raw_signal(ds, channels=[0, 1, 2], consistent_scale=True)
    ylims = [tuple(ax.get_ylim()) for ax in fig.axes]
    assert ylims[0] == ylims[1] == ylims[2]
    plt.close(fig)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_selection'`.

- [ ] **Step 3: Implement**

At the top of `visualization.py`, add `Dict`, `Tuple` to the typing import and `replace` from dataclasses; add these imports:

```python
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from hand_gesture_recognition.configuration.schema import VisualizationSelectionConfig
```

Add the helper functions after the existing `_shade_gesture` function:

```python
def _shared_ylim(axes, traces, pad=0.05):
    """Set a single shared y-range across all axes from the given traces."""
    lo, hi = np.inf, -np.inf
    for arr in traces:
        if arr is None or arr.size == 0:
            continue
        lo = min(lo, float(np.nanmin(arr)))
        hi = max(hi, float(np.nanmax(arr)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    span = hi - lo
    for ax in axes:
        ax.set_ylim(lo - pad * span, hi + pad * span)


def resolve_selection(
    selection: VisualizationSelectionConfig, class_names: List[str], n_channels: int
) -> Dict[str, List[int]]:
    """Gesture name -> 0-based channel indices. Empty lists mean 'all'."""
    gestures = selection.gestures or list(class_names)
    all_channels = list(range(n_channels))
    out: Dict[str, List[int]] = {}
    for g in gestures:
        ch = selection.by_gesture.get(g)
        if ch is None:
            ch = selection.channels
        idx = [int(c) - 1 for c in ch] if ch else all_channels
        out[g] = sorted(set(idx))
    return out
```

Replace the existing `plot_raw_signal` function with a version that accepts `channels` and `consistent_scale`:

```python
def plot_raw_signal(
    dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    channels: Optional[List[int]] = None,
    consistent_scale: bool = False,
):
    """One trial's raw signal for the given channels (0-based), with the
    ground-truth gesture segment shaded. When ``consistent_scale`` is true, all
    channel subplots share one y-range."""
    if dataset.n_trials == 0:
        return None
    tid = example_tid or str(dataset.trial_ids[0])
    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    fs = dataset.metadata.sampling_rate
    n_ch = trial.shape[1]
    if channels is None:
        channels = list(range(n_ch))

    fig, axes = plt.subplots(
        len(channels), 1, figsize=(12, 1.9 * len(channels)), squeeze=False
    )
    axes = axes[:, 0]
    t = np.arange(trial.shape[0]) / fs
    plotted = []
    for row, c in enumerate(channels):
        axes[row].plot(t, trial[:, c], lw=0.6)
        axes[row].set_ylabel(f"ch{c + 1}")
        plotted.append(trial[:, c])
    _shade_gesture(axes, int(dataset.gesture_start[i]), int(dataset.gesture_end[i]), fs)
    if consistent_scale:
        _shared_ylim(axes, plotted)

    axes[0].set_title(f"raw signal — {tid}")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
```

Replace `plot_raw_signal_by_class` with a version honoring `selection` and `consistent_scale` (preserving the default behavior exactly when `selection is None`):

```python
def plot_raw_signal_by_class(
    dataset: Dataset,
    run_dir: str,
    class_names: Optional[List[str]] = None,
    selection: Optional[Dict[str, List[int]]] = None,
    consistent_scale: bool = False,
) -> List[str]:
    """Write one raw-signal figure per class. When ``selection`` is provided,
    only its gestures (with their channels) are plotted; otherwise every class
    with all channels (legacy behavior)."""
    paths: List[str] = []
    if dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else dataset.metadata.class_names
    if selection is not None:
        present = set(int(l) for l in np.unique(dataset.labels))
        for g_name, channels in selection.items():
            try:
                g_idx = names.index(g_name)
            except ValueError:
                continue
            if g_idx not in present:
                continue
            idx = np.where(dataset.labels == g_idx)[0]
            tid = str(dataset.trial_ids[idx[0]])
            path = os.path.join(run_dir, f"raw_signal_{g_name}.png")
            plot_raw_signal(
                dataset, example_tid=tid, path=path,
                channels=channels, consistent_scale=consistent_scale,
            )
            paths.append(path)
        return paths
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
Expected: PASS (8 tests) — existing 3 + 5 new.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py tests/unit/test_visualization.py
git commit -m feat-add-selection-and-consistent-scale
```

### Task 10: Stage progression — `compute_stages`, `stage_label`, `plot_stage_progression`

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Test: `tests/unit/test_visualization.py` (append)

**Interfaces:**
- Consumes: `BaselineRemoval` (optional), signal-processing transformers (via `hydra.utils.instantiate`).
- Produces:
  - `stage_label(step_cfg: Dict) -> str`
  - `compute_stages(dataset, signal_processing_steps, baseline_remover) -> List[Tuple[str, Dataset]]`
  - `plot_stage_progression(stages, selection, class_names, consistent_scale, run_dir) -> List[str]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_visualization.py`:

```python
from hand_gesture_recognition.reporting.visualization import (
    compute_stages,
    plot_stage_progression,
    stage_label,
)


def test_stage_label_and_compute_stages_order():
    ds = make_synthetic_dataset(
        n_subjects=1, n_trials_per_subject=1, n_classes=1, length=100, n_channels=2, seed=0
    )
    steps = [
        {"_target_": "hand_gesture_recognition.preprocessing.signal.RectifyTransformer"}
    ]
    stages = compute_stages(ds, steps, None)
    labels = [s[0] for s in stages]
    assert labels == ["raw", "rectified"]
    assert stage_label(steps[0]) == "rectified"


def test_plot_stage_progression_one_figure_per_stage_gesture(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=3, n_classes=2, length=100, n_channels=4, seed=0
    )
    steps = [
        {"_target_": "hand_gesture_recognition.preprocessing.signal.RectifyTransformer"}
    ]
    stages = compute_stages(ds, steps, None)
    selection = {"class_0": [0, 1]}
    paths = plot_stage_progression(
        stages, selection, ds.metadata.class_names, True, str(tmp_path)
    )
    assert len(paths) == 2  # (raw, class_0) and (rectified, class_0)
    assert os.path.exists(str(tmp_path / "stage_progression_raw_class_0.png"))
    assert os.path.exists(str(tmp_path / "stage_progression_rectified_class_0.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_stages'`.

- [ ] **Step 3: Implement**

Add `import re` and `from hydra.utils import instantiate` at the top of `visualization.py`. Add the helpers and plot function (place after `plot_raw_signal_by_class`):

```python
def _slug(label: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", label).strip("_") or "stage"


def stage_label(step_cfg: Dict) -> str:
    target = str(step_cfg.get("_target_", ""))
    cls = target.rsplit(".", 1)[-1]
    if cls == "ButterworthFilter":
        t = step_cfg.get("type")
        if t == "bandpass":
            return f"bandpass_{step_cfg.get('lowcut')}_{step_cfg.get('highcut')}"
        if t == "notch":
            return f"notch_{step_cfg.get('frequency')}"
        if t == "lowpass":
            return f"lowpass_{step_cfg.get('highcut')}"
        if t == "highpass":
            return f"highpass_{step_cfg.get('lowcut')}"
        return str(t or "butterworth")
    if cls == "RectifyTransformer":
        return "rectified"
    if cls == "EnvelopeExtractor":
        return "envelope"
    return cls or "step"


def compute_stages(
    dataset: Dataset, signal_processing_steps: List[Dict], baseline_remover=None
) -> List[Tuple[str, Dataset]]:
    """Snapshot the dataset at each stage, in training-path order:
    raw -> baseline_removed (if any) -> each signal-processing step."""
    stages: List[Tuple[str, Dataset]] = [("raw", dataset)]
    cur = dataset
    if baseline_remover is not None:
        cur = baseline_remover.transform(cur)
        stages.append(("baseline_removed", cur))
    for step_cfg in signal_processing_steps:
        step = instantiate(step_cfg)
        cur = replace(cur, trials=[step.transform(t) for t in cur.trials])
        stages.append((stage_label(step_cfg), cur))
    return stages


def _plot_stage_channels(
    ds: Dataset, tid: str, channels: List[int], label: str,
    gesture: str, consistent_scale: bool, path: str,
):
    i = _trial_index(ds, tid)
    trial = ds.trials[i]
    fs = ds.metadata.sampling_rate
    fig, axes = plt.subplots(
        len(channels), 1, figsize=(12, 1.9 * len(channels)), squeeze=False
    )
    axes = axes[:, 0]
    t = np.arange(trial.shape[0]) / fs
    plotted = []
    for row, c in enumerate(channels):
        axes[row].plot(t, trial[:, c], lw=0.6)
        axes[row].set_ylabel(f"ch{c + 1}")
        plotted.append(trial[:, c])
    if consistent_scale:
        _shared_ylim(axes, plotted)
    axes[0].set_title(f"{gesture} — {label}")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_stage_progression(
    stages: List[Tuple[str, Dataset]],
    selection: Dict[str, List[int]],
    class_names: List[str],
    consistent_scale: bool,
    run_dir: str,
) -> List[str]:
    """One figure per (stage, gesture), showing the gesture's selected channels."""
    paths: List[str] = []
    if not stages:
        return paths
    raw = stages[0][1]
    if raw.n_trials == 0:
        return paths
    names = class_names if class_names is not None else raw.metadata.class_names
    present = set(int(l) for l in np.unique(raw.labels))
    for label, ds in stages:
        for g_name, channels in selection.items():
            try:
                g_idx = names.index(g_name)
            except ValueError:
                continue
            if g_idx not in present:
                continue
            idx = np.where(raw.labels == g_idx)[0]
            tid = str(raw.trial_ids[idx[0]])
            path = os.path.join(
                run_dir, f"stage_progression_{_slug(label)}_{g_name}.png"
            )
            _plot_stage_channels(ds, tid, channels, label, g_name, consistent_scale, path)
            paths.append(path)
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py tests/unit/test_visualization.py
git commit -m feat-add-stage-progression-plot
```

### Task 11: Feature-over-windows plot

**Files:**
- Modify: `src/hand_gesture_recognition/reporting/visualization.py`
- Test: `tests/unit/test_visualization.py` (append)

**Interfaces:**
- Consumes: `window_dataset`, `TimeDomainFeatures`, `FrequencyDomainFeatures`.
- Produces: `plot_feature_over_windows(processed_dataset, windowing, selection, features, max_windows, class_names, run_dir) -> List[str]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_visualization.py`:

```python
from hand_gesture_recognition.reporting.visualization import plot_feature_over_windows
from hand_gesture_recognition.configuration.schema import WindowingConfig


def test_plot_feature_over_windows(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2, n_trials_per_subject=3, n_classes=2, length=100, n_channels=4, seed=0
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)
    selection = {"class_0": [0, 1]}
    paths = plot_feature_over_windows(
        ds, windowing, selection, ["rms"], 0, ds.metadata.class_names, str(tmp_path)
    )
    assert len(paths) == 1
    assert os.path.exists(str(tmp_path / "feature_over_windows_class_0.png"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_visualization.py::test_plot_feature_over_windows -q`
Expected: FAIL — `ImportError: cannot import name 'plot_feature_over_windows'`.

- [ ] **Step 3: Implement**

Add imports to `visualization.py`:

```python
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures
from hand_gesture_recognition.features.frequency_domain import FrequencyDomainFeatures
```

Add the helpers and plot function (place after `plot_stage_progression`):

```python
def _feature_matrix(X: np.ndarray, sampling_rate: float):
    """(n_windows, 11*n_channels) matrix + flattened per-column names."""
    td = TimeDomainFeatures().transform(X)
    fd = FrequencyDomainFeatures(sampling_rate=sampling_rate).transform(X)
    n_ch = X.shape[-1]
    td_names = [
        f"{n}_ch{c + 1}" for n in TimeDomainFeatures.FEATURE_NAMES for c in range(n_ch)
    ]
    fd_names = [
        f"{n}_ch{c + 1}" for n in FrequencyDomainFeatures.FEATURE_NAMES for c in range(n_ch)
    ]
    return np.concatenate([td, fd], axis=1), td_names + fd_names


def plot_feature_over_windows(
    processed_dataset: Dataset,
    windowing,
    selection: Dict[str, List[int]],
    features: List[str],
    max_windows: int,
    class_names: List[str],
    run_dir: str,
) -> List[str]:
    """One figure per selected gesture: rows = features, cols = channels,
    value vs window index."""
    paths: List[str] = []
    if processed_dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else processed_dataset.metadata.class_names
    present = set(int(l) for l in np.unique(processed_dataset.labels))
    wd = window_dataset(processed_dataset, windowing)
    if wd.X.shape[0] == 0:
        return paths
    fs = processed_dataset.metadata.sampling_rate
    X_feat, feat_names = _feature_matrix(wd.X, fs)

    for g_name, channels in selection.items():
        try:
            g_idx = names.index(g_name)
        except ValueError:
            continue
        if g_idx not in present:
            continue
        idx = np.where(processed_dataset.labels == g_idx)[0]
        tid = str(processed_dataset.trial_ids[idx[0]])
        mask = wd.metadata.trial_ids == tid
        if not mask.any():
            continue
        path = os.path.join(run_dir, f"feature_over_windows_{g_name}.png")
        _plot_features(X_feat[mask], feat_names, channels, features, max_windows, g_name, path)
        paths.append(path)
    return paths


def _plot_features(X_feat, feat_names, channels, features, max_windows, g_name, path):
    wanted = features or list(
        dict.fromkeys(n.rsplit("_ch", 1)[0] for n in feat_names)
    )
    if max_windows and X_feat.shape[0] > max_windows:
        X_feat = X_feat[:max_windows]
    fig, axes = plt.subplots(
        len(wanted), len(channels), figsize=(2.5 * len(channels), 1.8 * len(wanted)),
        squeeze=False,
    )
    w = np.arange(X_feat.shape[0])
    for r, feat in enumerate(wanted):
        row_traces = []
        for c_idx, ch in enumerate(channels):
            col = f"{feat}_ch{ch + 1}"
            ax = axes[r, c_idx]
            if col in feat_names:
                j = feat_names.index(col)
                ax.plot(w, X_feat[:, j], lw=0.8)
                row_traces.append(X_feat[:, j])
            else:
                ax.set_visible(False)
            ax.set_xlabel("window")
            if c_idx == 0:
                ax.set_ylabel(feat)
        if row_traces:
            _shared_ylim(axes[r, :], row_traces)
    axes[0, 0].set_title(f"features over windows — {g_name}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_visualization.py -q`
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/reporting/visualization.py tests/unit/test_visualization.py
git commit -m feat-add-feature-over-windows-plot
```

### Task 12: Wire visualization config into the runner (final `_visualize`)

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/integration/test_visualization_config.py` (new)

**Interfaces:**
- Consumes: `resolve_selection`, `compute_stages`, `plot_stage_progression`, `plot_feature_over_windows`, `plot_raw_signal_by_class` (all with selection/scale), `BaselineRemoval`.
- Produces: final `_visualize(...)` that honors `cfg.visualization` and skips prediction overlays in CV mode.

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_visualization_config.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_visualization_config_produces_selected_plots(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "reporting.formats=[json]",
            "visualization.selection.gestures=[class_0]",
            "visualization.selection.channels=[1,2]",
            "visualization.stage_progression=true",
            "visualization.feature_over_windows.features=[rms]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "raw_signal_class_0.png").exists()
    assert (out / "stage_progression_raw_class_0.png").exists()
    assert (out / "feature_over_windows_class_0.png").exists()
    # selection limited to class_0, so class_1 artifacts are absent
    assert not (out / "raw_signal_class_1.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_visualization_config.py -q`
Expected: FAIL — `raw_signal_class_0.png` still written via the temporary (non-selection) path, or `stage_progression_raw_class_0.png` missing.

- [ ] **Step 3: Replace `_visualize` with the final version**

Replace the entire `_visualize` method in `run.py` (from the `def _visualize` line through the end of the method) with:

```python
    def _visualize(
        self,
        raw_dataset,
        processed_dataset,
        split,
        predictions,
        class_names,
        viz_baseline=None,
        is_cv=False,
    ) -> None:
        vis = self.cfg.visualization
        if not vis.enabled:
            return
        from hand_gesture_recognition.reporting.visualization import (
            compute_stages,
            plot_fft_spectrum,
            plot_feature_over_windows,
            plot_predictions_on_signal,
            plot_raw_signal_by_class,
            plot_signal_data_flow,
            plot_signal_processing_comparison,
            plot_stage_progression,
            resolve_selection,
        )

        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)

        n_channels = (
            len(raw_dataset.metadata.channels)
            if raw_dataset.metadata.channels
            else (raw_dataset.trials[0].shape[1] if raw_dataset.trials else 0)
        )
        selection = resolve_selection(vis.selection, class_names, n_channels)

        # A: raw signal per selected gesture/channel
        for p in plot_raw_signal_by_class(
            raw_dataset,
            run_dir,
            class_names=class_names,
            selection=selection,
            consistent_scale=vis.consistent_scale,
        ):
            self.tracker.log_artifact(p)

        # B: stage progression (one figure per (stage, gesture))
        if vis.stage_progression:
            steps = [instantiate(s) for s in self.cfg.signal_processing]
            stages = compute_stages(raw_dataset, steps, viz_baseline)
            for p in plot_stage_progression(
                stages, selection, class_names, vis.consistent_scale, run_dir
            ):
                self.tracker.log_artifact(p)

        # C: feature-over-windows (one figure per gesture)
        if vis.feature_over_windows.enabled:
            for p in plot_feature_over_windows(
                processed_dataset,
                self.cfg.windowing,
                selection,
                vis.feature_over_windows.features,
                vis.feature_over_windows.max_windows,
                class_names,
                run_dir,
            ):
                self.tracker.log_artifact(p)

        # Prediction overlay + data flow: single-split only.
        if not is_cv:
            for part, ds, pred in (
                ("val", split.val, predictions.val),
                ("test", split.test, predictions.test),
            ):
                p = os.path.join(run_dir, f"predictions_on_signal_{part}.png")
                plot_predictions_on_signal(
                    ds, pred, self.cfg.windowing, class_names, path=p
                )
                self.tracker.log_artifact(p)

        if self.cfg.signal_processing:
            p = os.path.join(run_dir, "signal_processing_comparison.png")
            plot_signal_processing_comparison(raw_dataset, processed_dataset, path=p)
            self.tracker.log_artifact(p)

        if not is_cv:
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

        feature_specs = self.cfg.classical.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        if any("frequency_domain" in str(f.get("_target_", "")) for f in feature_specs):
            p = os.path.join(run_dir, "fft_spectrum.png")
            plot_fft_spectrum(
                processed_dataset,
                self.cfg.windowing,
                sampling_rate=processed_dataset.metadata.sampling_rate,
                path=p,
            )
            self.tracker.log_artifact(p)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_visualization_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_visualization_config.py
git commit -m feat-wire-visualization-config-into-runner
```

---

## Final verification

- [ ] Run the full **new + unaffected** unit/integration suites relevant to this work:

```bash
python -m pytest tests/unit/test_cv.py tests/unit/test_config_schema.py tests/unit/test_splits.py tests/unit/test_visualization.py tests/integration/test_cv.py tests/integration/test_visualization_config.py -q
```

Expected: PASS. (The pre-existing failures in `test_config_loader.py` and the synthetic smoke/partition tests remain, out of scope.)

## Out-of-scope / follow-ups

- Fix pre-existing `gesture_crop` stale test and the baseline-on-synthetic breakage (recommended: add rest trials to `make_synthetic_dataset`, or default `baseline: none`, or override `baseline=none` in the synthetic smoke/partition tests + README quick-start).
- `consistent_scale` for `plot_predictions_on_signal` (kept all-channels, no shared scale in this plan to limit churn; it uses the legacy auto-scale).
