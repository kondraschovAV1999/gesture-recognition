# Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 "Foundation" vertical slice of the ML research framework from `ml_research_framework_spec_v3.md`: a configurable, reproducible end-to-end classical-ML experiment on the EMG-EPN612 dataset — from Hydra configuration and replay, through dataset loading, splitting, shared windowing, analytics, an sklearn pipeline, a backend-normalized `Prediction`, evaluation, MLflow tracking (with dataset content hash), and JSON/HTML reporting.

**Architecture:** A thin Experiment Runner orchestrates independently-testable stages, all wired together via Hydra `instantiate`/`_target_`. The runner never touches sklearn/Lightning output directly: a single shared windowing module produces a `WindowedDataset`, and a `Prediction` adapter normalizes sklearn output into a backend-independent `Prediction` consumed by a framework-level evaluator. A single Configuration Loading Module is the only code that calls Hydra's composition API (compose + replay + validate).

**Tech Stack:** Python 3.10, Hydra (`hydra-core`) + OmegaConf, scikit-learn, NumPy, pandas, MLflow, Matplotlib, pytest. (PyTorch/Lightning are deliberately **out of scope** for Phase 1 — neural is Phase 2.)

## Global Constraints

- Python **3.10** (conda env `hand-gesture-recognition` is already created and active).
- Package manager: **pip** with `pyproject.toml` (setuptools). No poetry/uv.
- Import root: **`src/hand_gesture_recognition/`** as a single top-level package (see "Key decisions" below for why this deviates from the spec's flat `src/data`, `src/logging`, … tree).
- Windowing must have **exactly one** implementation (`hand_gesture_recognition/data/windowing/`), imported by nothing that depends on sklearn or Lightning.
- **Split-before-window** ordering is a hard rule; subject/session leakage is tested, not just assumed.
- The **Configuration Loading Module** (`hand_gesture_recognition/configuration/loader.py`) is the *only* module allowed to call `hydra.compose` / `OmegaConf.load` for config resolution. Every other component receives a validated `ExperimentConfig` dataclass.
- Evaluation accepts **only** `Prediction` objects — never raw sklearn/Lightning output.
- Every `_target_`-instantiated pluggable component is constructed via `hydra.utils.instantiate` in the runner, never hardcoded dispatch.
- No hidden defaults that affect research results: every default lives in the structured config schema and is recorded in the resolved config.
- `output/` is git-ignored; the raw EPN612 dataset under `resources/data/EPN612/` is git-ignored too (content-hash, not bytes-in-git, is the reproducibility anchor).
- All generated artifacts are grouped under `output/<kind>/<run_id>/`.
- **Windows shell caveat:** the execution shell wraps commands via `conda run cmd.exe /c`, which strips double quotes and splits on spaces. Therefore: (a) do NOT use inline `python -c "..."` with quoted code — write a temporary script file or use `pytest`/`python -m`; (b) do NOT use `git commit -m "multi word message"` — use a single-token hyphenated message (`git commit -m feat-config-schema`) or a message file (`git commit -F <file>`); (c) do NOT put filesystem paths in Hydra overrides — set them on the typed config object in code instead.

## Key Decisions (deviations from spec, documented)

1. **Single top-level package `hand_gesture_recognition`.** The spec's tree puts generic modules (`data/`, `logging/`, `tracing/`, `runner/`) directly under `src/`. Importing `logging` as a top-level module would shadow the Python stdlib `logging`, and `data`/`runner`/`tracking` are collision-prone generic names. We place everything under `src/hand_gesture_recognition/` (→ `hand_gesture_recognition.data`, `hand_gesture_recognition.configuration`, `hand_gesture_recognition.logging`, …), preserving every submodule name and responsibility from the spec while making the codebase a proper installable package. This is a strong technical reason to amend ADR-style item "project structure" from Section 4.
2. **Dataclasses + ConfigStore (not Pydantic)** for structured configs. Phase 1's validation needs are narrow (windowing sample/time consistency, split ratios, enum-like fields); a small hand-written validator on dataclasses keeps deps minimal and stays idiomatic to Hydra. Pydantic can be introduced later if schema complexity grows.
3. **Pluggable component configs are `DictConfig`s carrying `_target_`.** Preprocessing steps, feature extractors, and the model are represented as nested config nodes with a `_target_` key. The strongly-typed dataclass layer covers the fields that need real validation (dataset, split, windowing, seed, backend, tracking/reporting). This is the standard Hydra `instantiate` idiom and keeps the schema focused.
4. **First dataset loader = EPN612** (real data), plus a tiny synthetic fixture builder for tests (so correctness tests never depend on 612 JSON files being present).
5. **Phase 1 model = Random Forest** as the single classical estimator (satisfies "one classical model" while still exercising the full pipeline). One preprocessing step (standard scaler) and one feature extractor (handcrafted time-domain features) are included as concrete `_target_` examples.

## File Structure

```
hand-gesture-recognition/            # git repo root (git init here)
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── ml_research_framework_spec_v3.md   # spec moved here
│   └── superpowers/plans/2026-09-13-phase1-foundation.md
├── src/hand_gesture_recognition/
│   ├── __init__.py
│   ├── configuration/
│   │   ├── __init__.py
│   │   ├── schema.py          # dataclasses + ConfigStore + cross-field validation
│   │   ├── errors.py          # ConfigValidationError, ConfigReplayError
│   │   └── loader.py          # compose / replay / validate / -> ExperimentConfig
│   ├── data/
│   │   ├── __init__.py
│   │   ├── hashing.py         # content hash over source files
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   └── base.py        # Dataset, DatasetMetadata
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # DatasetLoader interface
│   │   │   └── epn612.py      # EMG-EPN612 JSON loader
│   │   ├── fixtures.py        # synthetic dataset builder (tests + smoke)
│   │   ├── splits/
│   │   │   ├── __init__.py
│   │   │   └── base.py        # split strategies + leakage check
│   │   └── windowing/
│   │       ├── __init__.py
│   │       └── base.py        # WindowedDataset + window_dataset()
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── stats.py           # dataset/signal statistics
│   │   └── qc.py              # quality-control checks (warning/error/info)
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── base.py            # Prediction dataclass + PredictionAdapter ABC
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   └── sklearn.py     # SklearnPredictionAdapter
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   └── windowing.py   # WindowingTransformer (sklearn wrapper)
│   │   └── models/
│   │       ├── __init__.py
│   │       └── random_forest.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── time_domain.py     # handcrafted feature transformer
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── base.py            # EvaluationResult + evaluate()
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── mlflow_tracker.py  # MLflow init/log/artifact/close
│   ├── logging_setup/
│   │   ├── __init__.py
│   │   └── config.py          # central logging setup (avoids stdlib clash)
│   ├── tracing/
│   │   ├── __init__.py
│   │   └── stage.py           # stage timing/trace context manager
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── reporters.py       # JsonReporter, HtmlReporter
│   └── runner/
│       ├── __init__.py
│       └── run.py             # ExperimentRunner
├── configs/
│   ├── config.yaml
│   ├── dataset/epn612.yaml
│   ├── split/subject_wise.yaml
│   ├── windowing/default.yaml
│   ├── classical/random_forest.yaml
│   ├── evaluation/basic.yaml
│   ├── analytics/basic.yaml
│   └── experiment/default.yaml
├── resources/
│   └── data/EPN612/            # symlink/copy of the uploaded dataset
├── output/                     # git-ignored
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

### Task 0: Repository scaffold, packaging, and dependency install

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/hand_gesture_recognition/__init__.py`
- Create: `tests/__init__.py`, `tests/unit/__init__.py`
- Move: `ml_research_framework_spec_v3.md` → `docs/ml_research_framework_spec_v3.md`
- Modify: nothing else (new repo)

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `hand_gesture_recognition` (version `0.1.0`); installed deps; git repo.

- [ ] **Step 1: `git init` and move the spec**

```bash
cd /d "E:\Research\Dissertation\codes\hand-gesture-recognition"
git init
mkdir docs
move ml_research_framework_spec_v3.md docs\ml_research_framework_spec_v3.md
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hand-gesture-recognition"
version = "0.1.0"
description = "Reusable research/benchmarking framework for time-series and tabular ML experiments"
requires-python = ">=3.10"
dependencies = [
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "scikit-learn>=1.4",
    "numpy>=1.24",
    "pandas>=2.0",
    "mlflow>=2.10",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
.pytest_cache/

# Generated artifacts
output/

# Datasets (content-hash anchored, not committed)
resources/data/EPN612/

# MLflow local store
mlruns/
```

- [ ] **Step 4: Write package `__init__.py` files**

Create `src/hand_gesture_recognition/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`, `tests/fixtures/__init__.py` (all empty files).

- [ ] **Step 5: Install the package (editable) and dev deps**

```bash
pip install -e .
pip install pytest
```

Expected: installs `hand-gesture-recognition 0.1.0`, hydra-core, scikit-learn, numpy, pandas, mlflow, matplotlib, pytest.

- [ ] **Step 6: Verify the package imports and commit**

```bash
pip show hand-gesture-recognition pytest
git add -A
git commit -m "chore: scaffold package, pyproject, and gitignore"
```

Expected output: `pip show` lists `hand-gesture-recognition` version `0.1.0` and `pytest`.

---

### Task 1: Structured configuration schema (dataclasses + validation)

**Files:**
- Create: `src/hand_gesture_recognition/configuration/__init__.py`
- Create: `src/hand_gesture_recognition/configuration/errors.py`
- Create: `src/hand_gesture_recognition/configuration/schema.py`
- Test: `tests/unit/test_config_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ConfigValidationError` (exception).
  - `WindowingConfig.resolve(sampling_rate: float) -> ResolvedWindowing` where `ResolvedWindowing = (window_size_samples:int, stride_samples:int)`.
  - `SplitConfig`, `DatasetConfig`, `EvaluationConfig`, `TrackingConfig`, `ReportingConfig`, `ClassicalConfig`, `ExperimentConfig` dataclasses.
  - `validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig` (raises `ConfigValidationError` on bad input).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config_schema.py`:

```python
import pytest
from hand_gesture_recognition.configuration.schema import (
    ExperimentConfig,
    WindowingConfig,
    validate_experiment_config,
)
from hand_gesture_recognition.configuration.errors import ConfigValidationError


def _base_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        run_id="test",
        backend="classical",
        seed=42,
        deterministic=False,
        windowing=WindowingConfig(window_size_samples=40, stride_samples=10),
    )


def test_window_size_samples_and_seconds_consistent():
    cfg = WindowingConfig(window_size_samples=40, window_size_seconds=0.2, stride_samples=10)
    resolved = cfg.resolve(sampling_rate=200)
    assert resolved == (40, 10)


def test_window_size_inconsistency_raises():
    cfg = WindowingConfig(window_size_samples=30, window_size_seconds=0.2)
    with pytest.raises(ConfigValidationError):
        cfg.resolve(sampling_rate=200)


def test_stride_seconds_derived_from_samples():
    cfg = WindowingConfig(window_size_samples=40, stride_seconds=0.05)
    resolved = cfg.resolve(sampling_rate=200)
    assert resolved == (40, 10)


def test_experiment_validation_passes_on_defaults():
    cfg = _base_experiment()
    out = validate_experiment_config(cfg)
    assert out.backend == "classical"
    assert out.schema_version == "1.0"


def test_experiment_validation_rejects_bad_backend():
    cfg = _base_experiment()
    cfg.backend = "neural"  # Phase 1 does not implement neural
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hand_gesture_recognition.configuration'`.

- [ ] **Step 3: Write `errors.py`**

```python
class ConfigError(Exception):
    """Base class for configuration errors."""


class ConfigValidationError(ConfigError):
    """Raised when a resolved config violates the schema or cross-field rules."""


class ConfigReplayError(ConfigError):
    """Raised when replaying a persisted config fails (missing/incompatible)."""
```

- [ ] **Step 4: Write `schema.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hand_gesture_recognition.configuration.errors import ConfigValidationError

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("classical",)  # neural arrives in Phase 2


@dataclass
class WindowingConfig:
    window_size_samples: Optional[int] = None
    window_size_seconds: Optional[float] = None
    stride_samples: Optional[int] = None
    stride_seconds: Optional[float] = None
    overlap_percent: Optional[float] = None
    label_granularity: str = "per_window"  # per_window | sequence
    padding: str = "none"  # none | zero

    def resolve(self, sampling_rate: float) -> Tuple[int, int]:
        """Return (window_size_samples, stride_samples), deriving/validating.

        Fails loudly if two representations of the same quantity disagree.
        """
        if sampling_rate <= 0:
            raise ConfigValidationError(f"sampling_rate must be > 0, got {sampling_rate}")

        ws = self.window_size_samples
        if self.window_size_seconds is not None:
            derived = round(self.window_size_seconds * sampling_rate)
            if ws is not None and ws != derived:
                raise ConfigValidationError(
                    f"window_size_samples={ws} != window_size_seconds="
                    f"{self.window_size_seconds}s * {sampling_rate}Hz = {derived}"
                )
            ws = derived
        if ws is None:
            raise ConfigValidationError(
                "window size must be given as window_size_samples or window_size_seconds"
            )
        if ws < 1:
            raise ConfigValidationError(f"window_size_samples must be >= 1, got {ws}")

        stride = self.stride_samples
        if self.stride_seconds is not None:
            derived = round(self.stride_seconds * sampling_rate)
            if stride is not None and stride != derived:
                raise ConfigValidationError(
                    f"stride_samples={stride} != stride_seconds="
                    f"{self.stride_seconds}s * {sampling_rate}Hz = {derived}"
                )
            stride = derived
        if stride is None:
            if self.overlap_percent is not None:
                stride = round(ws * (1.0 - self.overlap_percent / 100.0))
            else:
                stride = ws  # non-overlapping default
        if stride < 1:
            raise ConfigValidationError(f"stride must be >= 1, got {stride}")
        return ws, stride


@dataclass
class DatasetConfig:
    name: str = "epn612"
    path: str = "resources/data/EPN612"
    sampling_rate: float = 200.0
    loader: Dict[str, Any] = field(
        default_factory=lambda: {"_target_": "hand_gesture_recognition.data.loaders.epn612.EPN612Loader"}
    )


@dataclass
class SplitConfig:
    strategy: str = "subject_wise"  # random | stratified | subject_wise | session_wise | predefined
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    split_on: str = "subject_ids"  # which Dataset field carries group identity

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.0 < self.train_ratio < 1.0):
            raise ConfigValidationError(f"train_ratio must be in (0,1), got {self.train_ratio}")
        if abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(f"split ratios must sum to 1.0, got {total}")
        if self.strategy not in (
            "random", "stratified", "subject_wise", "session_wise", "predefined",
        ):
            raise ConfigValidationError(f"unknown split strategy: {self.strategy}")


@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator",
            "n_estimators": 100,
            "random_state": 42,
        }
    )


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1"]
    )
    average: str = "macro"


@dataclass
class AnalyticsConfig:
    enabled: bool = True
    run_qc: bool = True


@dataclass
class TrackingConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "hand_gesture_recognition"


@dataclass
class ReportingConfig:
    formats: List[str] = field(default_factory=lambda: ["json", "html"])


@dataclass
class ExperimentConfig:
    run_id: Optional[str] = None
    backend: str = "classical"
    seed: int = 42
    deterministic: bool = False
    schema_version: str = SCHEMA_VERSION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.backend not in SUPPORTED_BACKENDS:
        raise ConfigValidationError(
            f"backend={cfg.backend!r} not supported in Phase 1 (expected one of {SUPPORTED_BACKENDS})"
        )
    if cfg.seed is None:
        raise ConfigValidationError("seed must be set for reproducibility")
    cfg.split.validate()
    # Validate windowing consistency against the configured sampling rate.
    cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)
    if cfg.schema_version != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"schema_version={cfg.schema_version!r} != {SCHEMA_VERSION!r}"
        )
    return cfg
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(config): structured config schema with cross-field validation"
```

---

### Task 2: Dataset representation and content hashing

**Files:**
- Create: `src/hand_gesture_recognition/data/__init__.py`
- Create: `src/hand_gesture_recognition/data/hashing.py`
- Create: `src/hand_gesture_recognition/data/datasets/__init__.py`
- Create: `src/hand_gesture_recognition/data/datasets/base.py`
- Test: `tests/unit/test_dataset_base.py`

**Interfaces:**
- Consumes: nothing (pure data structures + hashing).
- Produces:
  - `DatasetMetadata` dataclass: `name:str, version:str, sampling_rate:float, channels:List[str], class_names:List[str], content_hash:str, provenance:Dict[str,Any]`.
  - `Dataset` dataclass: `trials:List[np.ndarray]` (each `(time_steps, channels)`), `labels:np.ndarray`, `subject_ids:np.ndarray`, `trial_ids:np.ndarray`, `metadata:DatasetMetadata`. Property `n_trials`.
  - `hash_files(paths: Iterable[str]) -> str` (SHA-256 hex) and `hash_directory(root: str) -> str` (deterministic over all files, sorted by relative path).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_dataset_base.py`:

```python
import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.hashing import hash_files, hash_directory


def _meta() -> DatasetMetadata:
    return DatasetMetadata(
        name="test", version="1", sampling_rate=200.0,
        channels=["ch1", "ch2"], class_names=["a", "b"], content_hash="h",
        provenance={},
    )


def test_dataset_n_trials_and_shapes():
    d = Dataset(
        trials=[np.zeros((100, 2)), np.ones((100, 2))],
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        metadata=_meta(),
    )
    assert d.n_trials == 2
    assert d.trials[0].shape == (100, 2)
    assert d.labels.tolist() == [0, 1]


def test_hash_files_is_deterministic(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = hash_files([str(f)])
    h2 = hash_files([str(f)])
    assert h1 == h2
    assert len(h1) == 64


def test_hash_directory_ignores_filenames_but_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("1")
    (tmp_path / "a.txt").write_text("2")
    h = hash_directory(str(tmp_path))
    assert isinstance(h, str) and len(h) == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_dataset_base.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `hashing.py`**

```python
from __future__ import annotations

import hashlib
import os
from typing import Iterable, List


def hash_files(paths: Iterable[str]) -> str:
    """SHA-256 over the concatenation of (relative path, bytes) pairs."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8", errors="replace"))
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def hash_directory(root: str) -> str:
    """Deterministic hash of every file under root, sorted by relative path."""
    rel_files: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            rel_files.append(os.path.join(dirpath, name))
    return hash_files(rel_files)
```

- [ ] **Step 4: Write `datasets/base.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

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


@dataclass
class Dataset:
    """Raw, unwindowed dataset: one trial = one continuous signal.

    trials[i] has shape (time_steps, channels) and may vary in length across i.
    labels, subject_ids, trial_ids are aligned with trials.
    """

    trials: List[np.ndarray]
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    metadata: DatasetMetadata

    @property
    def n_trials(self) -> int:
        return len(self.trials)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_dataset_base.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(data): Dataset/DatasetMetadata types and content hashing"
```

---

### Task 3: EPN612 loader and synthetic fixture

**Files:**
- Create: `src/hand_gesture_recognition/data/loaders/__init__.py`
- Create: `src/hand_gesture_recognition/data/loaders/base.py`
- Create: `src/hand_gesture_recognition/data/loaders/epn612.py`
- Create: `src/hand_gesture_recognition/data/fixtures.py`
- Test: `tests/unit/test_epn612_loader.py`

**Interfaces:**
- Consumes: `Dataset`, `DatasetMetadata`, `hash_directory`.
- Produces:
  - `DatasetLoader` ABC with `load(self, path: str) -> Dataset`.
  - `EPN612Loader(subjects: Optional[Iterable[int]] = None)` → `load(path) -> Dataset` parsing `trainingJSON/user*/user*.json` + `testingJSON/user*/user*.json`.
  - `make_synthetic_dataset(n_subjects, n_trials_per_subject, n_classes, length, n_channels, seed) -> Dataset` (test/smoke fixture).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_epn612_loader.py`:

```python
import json

import numpy as np

from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader


def _write_user(root, subdir, user_id, n_samples=3, length=50):
    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "open"]
    samples = {}
    for i in range(n_samples):
        samples[f"idx_{i+1}"] = {
            "startPointforGestureExecution": 0,
            "gestureName": gestures[i % len(gestures)],
            "emg": {f"ch{c}": [float(c + j % 5) for j in range(length)] for c in range(1, 9)},
        }
    doc = {
        "generalInfo": {
            "samplingFrequencyInHertz": 200,
            "myoPredictionLabel": {
                "noGesture": 0, "fist": 1, "waveIn": 2,
                "waveOut": 3, "open": 4, "pinch": 5,
            },
        },
        "trainingSamples": samples,
        "testingSamples": {},
    }
    (d / f"user{user_id}.json").write_text(json.dumps(doc))


def test_epn612_loader_builds_dataset(tmp_path):
    for sub, uid in [("trainingJSON", 1), ("trainingJSON", 2), ("testingJSON", 3)]:
        _write_user(tmp_path, sub, uid)
    loader = EPN612Loader()
    ds = loader.load(str(tmp_path))

    assert ds.n_trials == 9  # 3 users * 3 training samples each
    assert ds.metadata.sampling_rate == 200.0
    assert ds.metadata.channels == [f"ch{i}" for i in range(1, 9)]
    assert set(ds.metadata.class_names) == {"noGesture", "fist", "waveIn", "waveOut", "open", "pinch"}
    assert ds.trials[0].shape == (50, 8)
    # subject ids disambiguated by cohort folder, one per trial
    assert set(ds.subject_ids.tolist()) == {"train:1", "train:2", "test:3"}
    assert ds.metadata.content_hash


def test_epn612_loader_subject_filter(tmp_path):
    for sub, uid in [("trainingJSON", 1), ("trainingJSON", 2)]:
        _write_user(tmp_path, sub, uid)
    ds = EPN612Loader(subjects=[1]).load(str(tmp_path))
    assert set(ds.subject_ids.tolist()) == {"train:1"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_epn612_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `loaders/base.py` and `fixtures.py`**

`src/hand_gesture_recognition/data/loaders/base.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod

from hand_gesture_recognition.data.datasets.base import Dataset


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> Dataset:
        """Load a dataset from `path` into the framework Dataset representation."""
        raise NotImplementedError
```

`src/hand_gesture_recognition/data/fixtures.py`:

```python
from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def make_synthetic_dataset(
    n_subjects: int = 4,
    n_trials_per_subject: int = 6,
    n_classes: int = 3,
    length: int = 100,
    n_channels: int = 8,
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []
    trial_ids: List[str] = []
    for s in range(n_subjects):
        for t in range(n_trials_per_subject):
            label = (s + t) % n_classes
            base = rng.normal(0, 1, size=(length, n_channels))
            signal = base + label  # class-separable offset
            trials.append(signal.astype(np.float32))
            labels.append(label)
            subject_ids.append(f"s{s}")
            trial_ids.append(f"s{s}_t{t}")
    return Dataset(
        trials=trials,
        labels=np.asarray(labels, dtype=int),
        subject_ids=np.asarray(subject_ids),
        trial_ids=np.asarray(trial_ids),
        metadata=DatasetMetadata(
            name="synthetic",
            version="0",
            sampling_rate=200.0,
            channels=[f"ch{i}" for i in range(1, n_channels + 1)],
            class_names=[f"class_{i}" for i in range(n_classes)],
            content_hash="synthetic",
            provenance={"seed": seed},
        ),
    )
```

- [ ] **Step 4: Write `loaders/epn612.py`**

```python
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.hashing import hash_directory
from hand_gesture_recognition.data.loaders.base import DatasetLoader

GESTURE_LABELS = {
    "noGesture": 0, "fist": 1, "waveIn": 2, "waveOut": 3, "open": 4, "pinch": 5,
}


class EPN612Loader(DatasetLoader):
    def __init__(self, subjects: Optional[Iterable[int]] = None):
        self.subjects = set(subjects) if subjects is not None else None

    def load(self, path: str) -> Dataset:
        files = self._discover(path)
        trials: List[np.ndarray] = []
        labels: List[int] = []
        subject_ids: List[str] = []
        trial_ids: List[str] = []
        sampling_rate: Optional[float] = None
        class_names = list(GESTURE_LABELS.keys())

        for f in files:
            doc = json.load(open(f, encoding="utf-8"))
            user_id = doc["userInfo"]["name"]
            sampling_rate = float(doc["generalInfo"]["samplingFrequencyInHertz"])
            # The dataset ships two cohorts (trainingJSON / testingJSON) that
            # reuse the same local user numbering; disambiguate subject ids by
            # cohort folder so subject-wise splits cannot collide.
            rel = os.path.relpath(f, path)
            group = "train" if "training" in rel.lower() else "test"
            subject_id = f"{group}:{user_id}"
            # Phase 1 loads trainingSamples only; the framework's own split
            # strategy (subject_wise / random) creates train/val/test.
            for key, sample in doc.get("trainingSamples", {}).items():
                emg = sample.get("emg")
                if not emg:
                    continue
                channels = sorted(emg.keys(), key=lambda c: int(c.replace("ch", "")))
                x = np.stack([np.asarray(emg[c], dtype=np.float32) for c in channels], axis=1)
                gesture = sample.get("gestureName", "noGesture")
                if gesture not in GESTURE_LABELS:
                    continue
                trials.append(x)
                labels.append(GESTURE_LABELS[gesture])
                subject_ids.append(subject_id)
                trial_ids.append(f"{subject_id}:{key}")

        if not trials:
            raise FileNotFoundError(f"No EPN612 user files found under {path}")

        return Dataset(
            trials=trials,
            labels=np.asarray(labels, dtype=int),
            subject_ids=np.asarray(subject_ids),
            trial_ids=np.asarray(trial_ids),
            metadata=DatasetMetadata(
                name="epn612",
                version="2020",
                sampling_rate=sampling_rate or 200.0,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
        )

    def _discover(self, path: str) -> List[str]:
        pattern = os.path.join(path, "**", "user*.json")
        files = sorted(glob.glob(pattern, recursive=True))
        if self.subjects is not None:
            files = [f for f in files if self._subject_num(f) in self.subjects]
        return files

    @staticmethod
    def _subject_num(f: str) -> int:
        base = os.path.basename(f)  # e.g. user12.json
        return int("".join(ch for ch in base if ch.isdigit()))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_epn612_loader.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(data): EPN612 JSON loader and synthetic fixture"
```

---

### Task 4: Split strategies and leakage check

**Files:**
- Create: `src/hand_gesture_recognition/data/splits/__init__.py`
- Create: `src/hand_gesture_recognition/data/splits/base.py`
- Test: `tests/unit/test_splits.py`

**Interfaces:**
- Consumes: `Dataset`, `SplitConfig`.
- Produces:
  - `DataSplit` dataclass with fields `train: Dataset`, `val: Dataset`, `test: Dataset`.
  - `split_dataset(dataset: Dataset, config: SplitConfig) -> DataSplit`.
  - `assert_no_group_leakage(split: DataSplit, group_field: str = "subject_ids") -> None` (raises `ValueError` on overlap).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_splits.py`:

```python
import pytest

from hand_gesture_recognition.configuration.schema import SplitConfig
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import assert_no_group_leakage, split_dataset


def test_subject_wise_split_has_no_leakage():
    ds = make_synthetic_dataset(n_subjects=6, n_trials_per_subject=4, seed=0)
    cfg = SplitConfig(strategy="subject_wise", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=1)
    split = split_dataset(ds, cfg)
    assert_no_group_leakage(split)  # raises on any subject overlap
    # every trial lands in exactly one partition
    total = split.train.n_trials + split.val.n_trials + split.test.n_trials
    assert total == ds.n_trials


def test_random_split_preserves_count():
    ds = make_synthetic_dataset(n_subjects=4, n_trials_per_subject=5, seed=0)
    cfg = SplitConfig(strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=7)
    split = split_dataset(ds, cfg)
    assert split.train.n_trials + split.val.n_trials + split.test.n_trials == ds.n_trials


def test_leakage_detection_raises():
    ds = make_synthetic_dataset(n_subjects=2, n_trials_per_subject=3, seed=0)
    cfg = SplitConfig(strategy="random", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=0)
    split = split_dataset(ds, cfg)
    # force overlap
    split.val = split.train
    with pytest.raises(ValueError):
        assert_no_group_leakage(split)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `splits/base.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from hand_gesture_recognition.configuration.errors import ConfigValidationError
from hand_gesture_recognition.configuration.schema import SplitConfig
from hand_gesture_recognition.data.datasets.base import Dataset


@dataclass
class DataSplit:
    train: Dataset
    val: Dataset
    test: Dataset


def _subset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    return Dataset(
        trials=[dataset.trials[i] for i in idx],
        labels=dataset.labels[idx],
        subject_ids=dataset.subject_ids[idx],
        trial_ids=dataset.trial_ids[idx],
        metadata=dataset.metadata,
    )


def split_dataset(dataset: Dataset, config: SplitConfig) -> DataSplit:
    config.validate()
    n = dataset.n_trials
    rng = np.random.default_rng(config.seed)

    if config.strategy in ("subject_wise", "session_wise"):
        groups = dataset.subject_ids if config.split_on == "subject_ids" else dataset.trial_ids
        unique = np.unique(groups)
        rng.shuffle(unique)
        n_train = max(1, int(round(len(unique) * config.train_ratio)))
        n_val = max(1, int(round(len(unique) * config.val_ratio)))
        train_g, val_g, test_g = unique[:n_train], unique[n_train:n_train + n_val], unique[n_train + n_val:]
        if len(test_g) == 0:
            test_g = unique[n_train + n_val:]
        train_idx = np.where(np.isin(groups, train_g))[0]
        val_idx = np.where(np.isin(groups, val_g))[0]
        test_idx = np.where(np.isin(groups, test_g))[0]
    elif config.strategy == "random":
        perm = rng.permutation(n)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        train_idx, val_idx, test_idx = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    elif config.strategy == "stratified":
        from sklearn.model_selection import train_test_split
        train_idx, rest_idx = train_test_split(
            np.arange(n), train_size=config.train_ratio,
            stratify=dataset.labels, random_state=config.seed,
        )
        val_ratio = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=1 - val_ratio,
            stratify=dataset.labels[rest_idx], random_state=config.seed,
        )
    elif config.strategy == "predefined":
        # EPN612 ships two cohorts (trainingJSON / testingJSON); "predefined"
        # treats train-cohort subjects as train and test-cohort subjects as test.
        sids = dataset.subject_ids.astype(str)
        train_idx = np.where(np.char.startswith(sids, "train:"))[0]
        test_idx = np.where(np.char.startswith(sids, "test:"))[0]
        val_idx = np.array([], dtype=int)
    else:
        raise ConfigValidationError(f"unknown strategy {config.strategy}")

    return DataSplit(
        train=_subset(dataset, train_idx),
        val=_subset(dataset, val_idx),
        test=_subset(dataset, test_idx),
    )


def assert_no_group_leakage(split: DataSplit, group_field: str = "subject_ids") -> None:
    def ids(d: Dataset) -> set:
        return set(getattr(d, group_field).tolist())

    tr, va, te = ids(split.train), ids(split.val), ids(split.test)
    if tr & va:
        raise ValueError(f"{group_field} overlap between train and val: {tr & va}")
    if tr & te:
        raise ValueError(f"{group_field} overlap between train and test: {tr & te}")
    if va & te:
        raise ValueError(f"{group_field} overlap between val and test: {va & te}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_splits.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(data): split strategies with subject/session leakage check"
```

---

### Task 5: Shared windowing (backend-agnostic)

**Files:**
- Create: `src/hand_gesture_recognition/data/windowing/__init__.py`
- Create: `src/hand_gesture_recognition/data/windowing/base.py`
- Test: `tests/unit/test_windowing.py`

**Interfaces:**
- Consumes: `Dataset`, `WindowingConfig` (and its `.resolve()`).
- Produces:
  - `WindowMetadata` dataclass: `window_start:np.ndarray, window_end:np.ndarray, subject_ids:np.ndarray, trial_ids:np.ndarray`.
  - `WindowedDataset` dataclass: `X:np.ndarray` `(n_windows, window_size, channels)`, `y:np.ndarray` `(n_windows,)`, `metadata:WindowMetadata`.
  - `window_dataset(dataset: Dataset, config: WindowingConfig) -> WindowedDataset` (no sklearn/Lightning imports).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_windowing.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.base import window_dataset


def _ds(lengths=(100, 60), labels=(0, 1), subjects=("s1", "s2")):
    trials = [np.arange(l, dtype=np.float32)[:, None].repeat(2, axis=1) for l in lengths]
    return Dataset(
        trials=trials,
        labels=np.array(labels),
        subject_ids=np.array(subjects),
        trial_ids=np.array(["t1", "t2"]),
        metadata=DatasetMetadata(
            name="x", version="0", sampling_rate=200.0,
            channels=["c1", "c2"], class_names=["a", "b"], content_hash="h",
        ),
    )


def test_window_counts_and_shapes():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    # each 100-sample trial -> ceil((100-40)/10)+1 = 7 windows
    assert wd.X.shape == (14, 40, 2)
    assert wd.y.shape == (14,)
    assert len(wd.metadata.window_start) == 14


def test_window_labels_and_subject_metadata_carry_through():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    assert wd.y[:7].tolist() == [0] * 7
    assert wd.y[7:].tolist() == [1] * 7
    assert set(wd.metadata.subject_ids.tolist()) == {"s1", "s2"}


def test_window_seconds_derived():
    ds = _ds(lengths=(100, 100), labels=(0, 1))
    cfg = WindowingConfig(window_size_seconds=0.2, stride_seconds=0.05)  # 40 / 10 @200Hz
    wd = window_dataset(ds, cfg)
    assert wd.X.shape[1] == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_windowing.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `windowing/base.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset


@dataclass
class WindowMetadata:
    window_start: np.ndarray  # (n_windows,)
    window_end: np.ndarray    # (n_windows,)
    subject_ids: np.ndarray   # (n_windows,)
    trial_ids: np.ndarray     # (n_windows,)


@dataclass
class WindowedDataset:
    X: np.ndarray            # (n_windows, window_size, channels)
    y: np.ndarray            # (n_windows,)
    metadata: WindowMetadata


def window_dataset(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    ws, stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    subjects: List[str] = []
    trials: List[str] = []

    for trial, label, subj, tid in zip(
        dataset.trials, dataset.labels, dataset.subject_ids, dataset.trial_ids
    ):
        length = trial.shape[0]
        if length < ws:
            if config.padding == "zero":
                pad = np.zeros((ws - length, trial.shape[1]), dtype=trial.dtype)
                trial = np.vstack([trial, pad])
                length = ws
            else:
                continue  # drop trials shorter than a window (config.padding == "none")
        n_windows = (length - ws) // stride + 1
        for i in range(n_windows):
            s = i * stride
            e = s + ws
            xs.append(trial[s:e])
            ys.append(int(label))
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))

    return WindowedDataset(
        X=np.stack(xs, axis=0) if xs else np.zeros((0, ws, dataset.trials[0].shape[1]), dtype=np.float32),
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_windowing.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(data): shared backend-agnostic windowing"
```

---

### Task 6: Configuration Loading Module (compose + replay + validate)

**Files:**
- Create: `src/hand_gesture_recognition/configuration/loader.py`
- Test: `tests/unit/test_config_loader.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `validate_experiment_config`, `register_configs`, `ConfigValidationError`, `ConfigReplayError`.
- Produces:
  - `load_config(mode: str = "compose", overrides: List[str] = [], resolved_path: Optional[str] = None) -> ExperimentConfig`.
  - `save_resolved_config(cfg: ExperimentConfig, path: str) -> None` (persists `resolved.yaml`).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config_loader.py`:

```python
import pytest

from hand_gesture_recognition.configuration.errors import ConfigValidationError, ConfigReplayError
from hand_gesture_recognition.configuration.loader import load_config, save_resolved_config


def test_compose_mode_returns_validated_config():
    cfg = load_config(mode="compose", overrides=["seed=42", "backend=classical"])
    assert cfg.backend == "classical"
    assert cfg.seed == 42
    assert cfg.schema_version == "1.0"


def test_compose_mode_rejects_invalid_override():
    with pytest.raises(ConfigValidationError):
        load_config(mode="compose", overrides=["backend=neural"])


def test_compose_mode_rejects_inconsistent_windowing():
    with pytest.raises(ConfigValidationError):
        load_config(mode="compose", overrides=[
            "windowing.window_size_samples=30",
            "windowing.window_size_seconds=0.2",
        ])


def test_replay_mode_roundtrips(tmp_path):
    cfg = load_config(mode="compose", overrides=["seed=7"])
    p = tmp_path / "resolved.yaml"
    save_resolved_config(cfg, str(p))
    replayed = load_config(mode="replay", resolved_path=str(p))
    assert replayed.seed == 7
    assert replayed.backend == "classical"
    assert replayed == cfg


def test_replay_mode_missing_file_raises(tmp_path):
    with pytest.raises(ConfigReplayError):
        load_config(mode="replay", resolved_path=str(tmp_path / "nope.yaml"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `loader.py`**

```python
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hand_gesture_recognition.configuration.errors import ConfigReplayError, ConfigValidationError
from hand_gesture_recognition.configuration.schema import (
    AnalyticsConfig,
    ClassicalConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ReportingConfig,
    SCHEMA_VERSION,
    SplitConfig,
    TrackingConfig,
    WindowingConfig,
    validate_experiment_config,
)

_CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs")
)


def _coerce(d: Dict[str, Any]) -> ExperimentConfig:
    """Build a fully-typed ExperimentConfig from a plain resolved dict.

    Nested dataclasses are constructed explicitly so the rest of the framework
    always receives typed sub-configs (DatasetConfig, SplitConfig, ...) rather
    than plain dicts.
    """
    return ExperimentConfig(
        run_id=d.get("run_id"),
        backend=d.get("backend", "classical"),
        seed=d.get("seed", 42),
        deterministic=d.get("deterministic", False),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
        dataset=DatasetConfig(**d.get("dataset", {})),
        split=SplitConfig(**d.get("split", {})),
        windowing=WindowingConfig(**d.get("windowing", {})),
        classical=ClassicalConfig(**d.get("classical", {})),
        evaluation=EvaluationConfig(**d.get("evaluation", {})),
        analytics=AnalyticsConfig(**d.get("analytics", {})),
        tracking=TrackingConfig(**d.get("tracking", {})),
        reporting=ReportingConfig(**d.get("reporting", {})),
    )


def _compose(overrides: List[str]) -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=_CONFIGS_DIR):
        raw = compose(config_name="config", overrides=overrides)
    d = OmegaConf.to_container(raw, resolve=True)
    return validate_experiment_config(_coerce(d))


def _replay(resolved_path: str) -> ExperimentConfig:
    if not os.path.exists(resolved_path):
        raise ConfigReplayError(f"resolved config not found: {resolved_path}")
    raw = OmegaConf.load(resolved_path)
    d = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(d, dict):
        raise ConfigReplayError(f"resolved config {resolved_path} is not a mapping")
    return validate_experiment_config(_coerce(d))


def load_config(
    mode: str = "compose",
    overrides: Optional[List[str]] = None,
    resolved_path: Optional[str] = None,
) -> ExperimentConfig:
    """Single entry point for configuration resolution (compose or replay)."""
    overrides = overrides or []
    if mode == "compose":
        return _compose(overrides)
    if mode == "replay":
        if resolved_path is None:
            raise ConfigReplayError("replay mode requires resolved_path")
        return _replay(resolved_path)
    raise ConfigValidationError(f"unknown config load mode: {mode!r}")


def save_resolved_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), path)
```

- [ ] **Step 4: Write the minimal Hydra configs required by `compose`**

`configs/config.yaml`:

```yaml
defaults:
  - dataset: epn612
  - split: subject_wise
  - windowing: default
  - classical: random_forest
  - evaluation: basic
  - analytics: basic
  - experiment: default
  - _self_

hydra:
  run:
    dir: output/test_runs/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: output/test_runs
  job:
    chdir: false
```

Each config group below uses a `# @package` directive so Hydra nests its keys
under the matching top-level field (`dataset`, `split`, `windowing`, ...), while
the `experiment` group contributes top-level fields (`backend`, `seed`,
`tracking`, `reporting`).

`configs/dataset/epn612.yaml`:

```yaml
# @package dataset
name: epn612
path: resources/data/EPN612
sampling_rate: 200
loader:
  _target_: hand_gesture_recognition.data.loaders.epn612.EPN612Loader
```

`configs/split/subject_wise.yaml`:

```yaml
# @package split
strategy: subject_wise
train_ratio: 0.6
val_ratio: 0.2
test_ratio: 0.2
seed: 42
split_on: subject_ids
```

`configs/windowing/default.yaml` (all fields listed explicitly so every key is
overridable from the CLI, including the `null` sample-based slots):

```yaml
# @package windowing
window_size_samples: null
window_size_seconds: 0.2
stride_samples: null
stride_seconds: 0.05
overlap_percent: null
label_granularity: per_window
padding: none
```

`configs/classical/random_forest.yaml`:

```yaml
# @package classical
features:
  - _target_: hand_gesture_recognition.features.time_domain.TimeDomainFeatures
preprocessing:
  - _target_: sklearn.preprocessing.StandardScaler
model:
  _target_: hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator
  n_estimators: 100
  random_state: 42
```

`configs/evaluation/basic.yaml`:

```yaml
# @package evaluation
metrics: [accuracy, balanced_accuracy, macro_f1]
average: macro
```

`configs/analytics/basic.yaml`:

```yaml
# @package analytics
enabled: true
run_qc: true
```

`configs/experiment/default.yaml`:

```yaml
# @package _global_
backend: classical
seed: 42
deterministic: false
tracking:
  tracking_uri: mlruns
  experiment_name: hand_gesture_recognition
reporting:
  formats: [json, html]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_loader.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(config): compose/replay Configuration Loading Module"
```

---

### Task 7: Analytics (statistics, QC checks)

**Files:**
- Create: `src/hand_gesture_recognition/analytics/__init__.py`
- Create: `src/hand_gesture_recognition/analytics/stats.py`
- Create: `src/hand_gesture_recognition/analytics/qc.py`
- Test: `tests/unit/test_analytics.py`

**Interfaces:**
- Consumes: `Dataset`.
- Produces:
  - `dataset_summary(dataset: Dataset) -> dict` (counts, class distribution, channel stats as JSON-serializable dict).
  - `run_quality_checks(dataset: Dataset) -> List[dict]` where each entry is `{"level": "warning"|"error"|"info", "check": str, "message": str}`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_analytics.py`:

```python
import numpy as np

from hand_gesture_recognition.analytics.qc import run_quality_checks
from hand_gesture_recognition.analytics.stats import dataset_summary
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def _ds():
    meta = DatasetMetadata(
        name="x", version="0", sampling_rate=200.0,
        channels=["c1", "c2"], class_names=["a", "b"], content_hash="h",
    )
    return Dataset(
        trials=[np.ones((100, 2)), np.full((100, 2), np.nan)],
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s2"]),
        trial_ids=np.array(["t1", "t2"]),
        metadata=meta,
    )


def test_dataset_summary_counts():
    s = dataset_summary(_ds())
    assert s["n_trials"] == 2
    assert s["n_subjects"] == 2
    assert s["n_channels"] == 2
    assert s["class_distribution"] == {"a": 1, "b": 1}


def test_qc_detects_nan():
    checks = run_quality_checks(_ds())
    msgs = " ".join(c["message"] for c in checks)
    assert any(c["check"] == "nan_inf" for c in checks)
    assert "NaN" in msgs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_analytics.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `stats.py` and `qc.py`**

`src/hand_gesture_recognition/analytics/stats.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def dataset_summary(dataset: Dataset) -> Dict[str, Any]:
    label_names = dataset.metadata.class_names
    class_counts = Counter(int(l) for l in dataset.labels)
    all_signals = np.vstack(dataset.trials)
    per_channel = {
        ch: {
            "mean": float(all_signals[:, i].mean()) if np.isfinite(all_signals[:, i].mean()) else None,
            "std": float(np.nanstd(all_signals[:, i])),
            "min": float(np.nanmin(all_signals[:, i])),
            "max": float(np.nanmax(all_signals[:, i])),
            "rms": float(np.sqrt(np.nanmean(all_signals[:, i] ** 2))),
        }
        for i, ch in enumerate(dataset.metadata.channels)
    }
    return {
        "name": dataset.metadata.name,
        "content_hash": dataset.metadata.content_hash,
        "sampling_rate": dataset.metadata.sampling_rate,
        "n_trials": dataset.n_trials,
        "n_subjects": int(np.unique(dataset.subject_ids).size),
        "n_channels": len(dataset.metadata.channels),
        "class_distribution": {label_names[int(k)]: v for k, v in sorted(class_counts.items())},
        "per_channel": per_channel,
    }
```

`src/hand_gesture_recognition/analytics/qc.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def run_quality_checks(dataset: Dataset) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for i, trial in enumerate(dataset.trials):
        if np.isnan(trial).any() or np.isinf(trial).any():
            checks.append({
                "level": "error", "check": "nan_inf",
                "message": f"trial {dataset.trial_ids[i]} contains NaN/Inf values",
            })
    counts = np.bincount(dataset.labels.astype(int))
    if counts.size and counts.min() == 0:
        checks.append({
            "level": "warning", "check": "empty_class",
            "message": "one or more classes have zero trials",
        })
    if not checks:
        checks.append({"level": "info", "check": "ok", "message": "no issues detected"})
    return checks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_analytics.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(analytics): dataset statistics and quality-control checks"
```

---

### Task 8: Prediction object and classical adapter

**Files:**
- Create: `src/hand_gesture_recognition/prediction/__init__.py`
- Create: `src/hand_gesture_recognition/prediction/base.py`
- Create: `src/hand_gesture_recognition/classical/__init__.py`
- Create: `src/hand_gesture_recognition/classical/adapters/__init__.py`
- Create: `src/hand_gesture_recognition/classical/adapters/sklearn.py`
- Test: `tests/unit/test_prediction.py`

**Interfaces:**
- Consumes: `WindowedDataset` (for window ids).
- Produces:
  - `Prediction` dataclass: `run_id:str, sample_ids:np.ndarray, y_true:np.ndarray, y_pred:np.ndarray, y_score:Optional[np.ndarray], prediction_duration_s:float, backend:str`.
  - `PredictionAdapter` ABC with `to_prediction(...) -> Prediction`.
  - `SklearnPredictionAdapter` whose `to_prediction(windowed, y_pred, y_score, run_id, duration_s)` builds a `Prediction` with `backend="classical"` and `sample_ids = windowed.metadata.trial_ids`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_prediction.py`:

```python
import numpy as np

from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.data.windowing.base import WindowMetadata, WindowedDataset


def _windowed(n=10):
    return WindowedDataset(
        X=np.zeros((n, 40, 8), dtype=np.float32),
        y=np.arange(n) % 3,
        metadata=WindowMetadata(
            window_start=np.arange(n),
            window_end=np.arange(n) + 40,
            subject_ids=np.array([f"s{i % 2}" for i in range(n)]),
            trial_ids=np.array([f"t{i}" for i in range(n)]),
        ),
    )


def test_sklearn_adapter_builds_prediction():
    wd = _windowed()
    adapter = SklearnPredictionAdapter()
    pred = adapter.to_prediction(
        windowed=wd,
        y_pred=np.array([0] * 10),
        y_score=np.ones((10, 3)) / 3,
        run_id="r1",
        duration_s=0.1,
    )
    assert pred.backend == "classical"
    assert pred.run_id == "r1"
    assert pred.sample_ids.tolist() == [f"t{i}" for i in range(10)]
    assert pred.y_true.tolist() == (np.arange(10) % 3).tolist()
    assert pred.y_score.shape == (10, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_prediction.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `prediction/base.py`**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Prediction:
    run_id: str
    sample_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: Optional[np.ndarray]
    prediction_duration_s: float
    backend: str  # "classical" | "neural"


class PredictionAdapter(ABC):
    @abstractmethod
    def to_prediction(self, **kwargs) -> Prediction:
        """Normalize backend-native output into a framework Prediction."""
        raise NotImplementedError
```

- [ ] **Step 4: Write `classical/adapters/sklearn.py`**

```python
from __future__ import annotations

from typing import Optional

import numpy as np

from hand_gesture_recognition.data.windowing.base import WindowedDataset
from hand_gesture_recognition.prediction.base import Prediction, PredictionAdapter


class SklearnPredictionAdapter(PredictionAdapter):
    def to_prediction(
        self,
        windowed: WindowedDataset,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray],
        run_id: str,
        duration_s: float,
    ) -> Prediction:
        return Prediction(
            run_id=run_id,
            sample_ids=windowed.metadata.trial_ids,
            y_true=windowed.y,
            y_pred=np.asarray(y_pred),
            y_score=y_score,
            prediction_duration_s=duration_s,
            backend="classical",
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_prediction.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(prediction): Prediction dataclass and sklearn adapter"
```

---

### Task 9: Evaluation layer

**Files:**
- Create: `src/hand_gesture_recognition/evaluation/__init__.py`
- Create: `src/hand_gesture_recognition/evaluation/base.py`
- Test: `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: `Prediction`, `EvaluationConfig`.
- Produces:
  - `EvaluationResult` dataclass: `run_id:str, dataset_id:str, split:dict, metrics:Dict[str,float], per_class:Dict[str,Dict[str,float]], confusion_matrix:np.ndarray`.
  - `evaluate(prediction: Prediction, config: EvaluationConfig, dataset_id: str, split_info: dict) -> EvaluationResult`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_evaluation.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import EvaluationConfig
from hand_gesture_recognition.evaluation.base import evaluate
from hand_gesture_recognition.prediction.base import Prediction


def _pred():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    return Prediction(
        run_id="r1", sample_ids=np.arange(6), y_true=y_true, y_pred=y_pred,
        y_score=np.eye(3)[y_pred], prediction_duration_s=0.0, backend="classical",
    )


def test_evaluate_computes_accuracy_and_f1():
    cfg = EvaluationConfig(metrics=["accuracy", "macro_f1"])
    res = evaluate(_pred(), cfg, dataset_id="d1", split_info={"strategy": "random"})
    assert res.run_id == "r1"
    assert res.metrics["accuracy"] == 5 / 6
    assert 0.0 <= res.metrics["macro_f1"] <= 1.0
    assert res.confusion_matrix.shape == (3, 3)
    assert set(res.per_class.keys()) == {"0", "1", "2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_evaluation.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `evaluation/base.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from hand_gesture_recognition.configuration.schema import EvaluationConfig
from hand_gesture_recognition.prediction.base import Prediction

_METRIC_FNS = {
    "accuracy": lambda yt, yp, **k: accuracy_score(yt, yp),
    "balanced_accuracy": lambda yt, yp, **k: balanced_accuracy_score(yt, yp),
    "macro_f1": lambda yt, yp, **k: f1_score(yt, yp, average="macro"),
    "weighted_f1": lambda yt, yp, **k: f1_score(yt, yp, average="weighted"),
}


@dataclass
class EvaluationResult:
    run_id: str
    dataset_id: str
    split: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def evaluate(
    prediction: Prediction,
    config: EvaluationConfig,
    dataset_id: str,
    split_info: Dict[str, Any],
) -> EvaluationResult:
    y_true = np.asarray(prediction.y_true)
    y_pred = np.asarray(prediction.y_pred)
    metrics = {m: _METRIC_FNS[m](y_true, y_pred) for m in config.metrics if m in _METRIC_FNS}
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {
        str(c): {"f1": f1_score(y_true, y_pred, labels=[c], average="macro")}
        for c in classes
    }
    return EvaluationResult(
        run_id=prediction.run_id,
        dataset_id=dataset_id,
        split=split_info,
        metrics=metrics,
        per_class=per_class,
        confusion_matrix=cm,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_evaluation.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(evaluation): backend-independent EvaluationResult and metrics"
```

---

### Task 10: Classical pipeline (windowing transformer, features, model)

**Files:**
- Create: `src/hand_gesture_recognition/classical/pipelines/__init__.py`
- Create: `src/hand_gesture_recognition/classical/pipelines/windowing.py`
- Create: `src/hand_gesture_recognition/classical/models/__init__.py`
- Create: `src/hand_gesture_recognition/classical/models/random_forest.py`
- Create: `src/hand_gesture_recognition/features/__init__.py`
- Create: `src/hand_gesture_recognition/features/time_domain.py`
- Test: `tests/unit/test_classical_pipeline.py`

**Interfaces:**
- Consumes: `Dataset`, `WindowingConfig`, `window_dataset`.
- Produces:
  - `WindowingTransformer(window_config: WindowingConfig)`: sklearn transformer; `fit_transform(dataset: Dataset) -> WindowedDataset` (returns the shared windowing output, not a flat array — the pipeline flattens downstream); `transform(dataset) -> WindowedDataset`.
  - `TimeDomainFeatures`: sklearn transformer mapping `(n_windows, time, ch)` → `(n_windows, n_ch * n_features)` (RMS, MAV, WL).
  - `RandomForestEstimator(**kwargs)`: thin `RandomForestClassifier` subclass.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_classical_pipeline.py`:

```python
import numpy as np
from sklearn.pipeline import Pipeline

from hand_gesture_recognition.classical.models.random_forest import RandomForestEstimator
from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer
from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.features.time_domain import TimeDomainFeatures


def test_windowing_transformer_calls_shared_windowing():
    ds = make_synthetic_dataset(n_subjects=2, n_trials_per_subject=2, length=100, seed=0)
    t = WindowingTransformer(WindowingConfig(window_size_samples=40, stride_samples=10))
    wd = t.fit_transform(ds)
    assert wd.X.ndim == 3  # (windows, time, channels) -> shared representation
    assert wd.X.shape[1] == 40


def test_full_classical_pipeline_fits_and_predicts():
    ds = make_synthetic_dataset(n_subjects=4, n_trials_per_subject=6, length=100, seed=0)
    win = WindowingTransformer(WindowingConfig(window_size_samples=40, stride_samples=20))
    feat = TimeDomainFeatures()
    model = RandomForestEstimator(n_estimators=20, random_state=0)
    # manual wiring (the Runner wires these via hydra.utils.instantiate)
    wd = win.fit_transform(ds)
    X = feat.fit_transform(wd.X)
    model.fit(X, wd.y)
    preds = model.predict(X)
    assert preds.shape == wd.y.shape
    assert set(preds.tolist()).issubset(set(ds.metadata.class_names) or set(range(3)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_classical_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the transformers and model**

`src/hand_gesture_recognition/classical/pipelines/windowing.py`:

```python
from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import WindowedDataset, window_dataset


class WindowingTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper around the shared windowing implementation."""

    def __init__(self, window_config: WindowingConfig):
        self.window_config = window_config

    def fit(self, X, y=None):
        return self

    def transform(self, X: Dataset) -> WindowedDataset:
        return window_dataset(X, self.window_config)

    def fit_transform(self, X: Dataset, y=None, **fit_params) -> WindowedDataset:
        return self.transform(X)
```

`src/hand_gesture_recognition/features/time_domain.py`:

```python
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Handcrafted time-domain features: RMS, mean absolute value, waveform length."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_windows, time, channels)
        X = np.asarray(X, dtype=np.float32)
        rms = np.sqrt(np.mean(X ** 2, axis=1))
        mav = np.mean(np.abs(X), axis=1)
        wl = np.sum(np.abs(np.diff(X, axis=1)), axis=1)
        return np.concatenate([rms, mav, wl], axis=1)
```

`src/hand_gesture_recognition/classical/models/random_forest.py`:

```python
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class RandomForestEstimator(RandomForestClassifier):
    """Random Forest classifier (thin wrapper for _target_ instantiation)."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_classical_pipeline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(classical): windowing transformer, time-domain features, RF model"
```

---

### Task 11: Tracking (MLflow), logging, and tracing

**Files:**
- Create: `src/hand_gesture_recognition/tracking/__init__.py`
- Create: `src/hand_gesture_recognition/tracking/mlflow_tracker.py`
- Create: `src/hand_gesture_recognition/logging_setup/__init__.py`
- Create: `src/hand_gesture_recognition/logging_setup/config.py`
- Create: `src/hand_gesture_recognition/tracing/__init__.py`
- Create: `src/hand_gesture_recognition/tracing/stage.py`
- Test: `tests/unit/test_tracking.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `EvaluationResult`, `Dataset`.
- Produces:
  - `MlflowTracker(config: TrackingConfig)`: `start(run_id)`, `log_params(cfg)`, `log_dataset(dataset)`, `log_results(result)`, `log_artifact(path)`, `finish(status)`. Uses `mlflow.set_tracking_uri`, `start_run`, `log_param/log_metric/log_artifact`.
  - `get_logger(name: str)` returning a configured logger writing to `output/logs/<run_id>/application.log`.
  - `stage(name)` context manager emitting start/end timing to the logger.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_tracking.py`:

```python
import mlflow
import pytest

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.tracking.mlflow_tracker import MlflowTracker


def test_tracker_logs_params_and_metrics(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-1")
    t.log_params({"seed": 42, "window_size": 40})
    t.log_metrics({"accuracy": 0.9})
    t.finish("FINISHED")
    # assert the run was recorded
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    run = client.get_run(t.run_id)
    assert run.data.params["seed"] == "42"
    assert run.data.metrics["accuracy"] == 0.9
    assert run.info.status == "FINISHED"


def test_tracker_marks_failed(tmp_path):
    cfg = TrackingConfig(tracking_uri=str(tmp_path / "mlruns"), experiment_name="t")
    t = MlflowTracker(cfg)
    t.start("run-2")
    t.finish("FAILED")
    client = mlflow.tracking.MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    assert client.get_run(t.run_id).info.status == "FAILED"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `mlflow_tracker.py`**

```python
from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow

from hand_gesture_recognition.configuration.schema import TrackingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.evaluation.base import EvaluationResult


class MlflowTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.run_id: Optional[str] = None
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def start(self, run_id: str) -> None:
        mlflow.start_run(run_name=run_id)
        self.run_id = mlflow.active_run().info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_dataset(self, dataset: Dataset) -> None:
        mlflow.log_params({
            "dataset_name": dataset.metadata.name,
            "dataset_version": dataset.metadata.version,
            "dataset_content_hash": dataset.metadata.content_hash,
            "sampling_rate": dataset.metadata.sampling_rate,
        })

    def log_results(self, result: EvaluationResult) -> None:
        mlflow.log_metrics(result.metrics)

    def log_artifact(self, local_path: str) -> None:
        mlflow.log_artifact(local_path)

    def finish(self, status: str) -> None:
        mlflow.end_run(status=status)
```

- [ ] **Step 4: Write `logging_setup/config.py` and `tracing/stage.py`**

`src/hand_gesture_recognition/logging_setup/config.py`:

```python
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, run_id: Optional[str] = None, log_dir: str = "output/logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if run_id:
        d = os.path.join(log_dir, run_id)
        os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(os.path.join(d, "application.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
```

`src/hand_gesture_recognition/tracing/stage.py`:

```python
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def stage(logger: logging.Logger, name: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info(f"[stage] {name} started")
    try:
        yield
    except Exception:
        logger.exception(f"[stage] {name} FAILED")
        raise
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[stage] {name} finished in {elapsed:.3f}s")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_tracking.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(tracking): MLflow tracker, centralized logging, stage tracing"
```

---

### Task 12: Reporting (JSON + HTML)

**Files:**
- Create: `src/hand_gesture_recognition/reporting/__init__.py`
- Create: `src/hand_gesture_recognition/reporting/reporters.py`
- Test: `tests/unit/test_reporting.py`

**Interfaces:**
- Consumes: `EvaluationResult`, `ExperimentConfig`, `Dataset`.
- Produces:
  - `render_report(result: EvaluationResult, config: ExperimentConfig, dataset: Dataset) -> dict` (serializable report).
  - `JsonReporter.write(report, path) -> None`.
  - `HtmlReporter.write(report, path) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_reporting.py`:

```python
import json

import numpy as np

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.evaluation.base import EvaluationResult
from hand_gesture_recognition.reporting.reporters import JsonReporter, render_report


def test_render_report_serializable():
    res = EvaluationResult(
        run_id="r1", dataset_id="d1", metrics={"accuracy": 0.9},
        confusion_matrix=np.array([[5, 0], [0, 5]]),
    )
    report = render_report(res, ExperimentConfig(run_id="r1"), dataset=None)
    s = json.dumps(report)
    assert "accuracy" in s
    assert report["run_id"] == "r1"


def test_json_reporter_writes_file(tmp_path):
    res = EvaluationResult(run_id="r1", dataset_id="d1", metrics={"accuracy": 0.9})
    report = render_report(res, ExperimentConfig(run_id="r1"), dataset=None)
    p = tmp_path / "report.json"
    JsonReporter().write(report, str(p))
    assert json.loads(p.read_text())["run_id"] == "r1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `reporters.py`**

```python
from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.evaluation.base import EvaluationResult


def render_report(
    result: EvaluationResult,
    config: ExperimentConfig,
    dataset: Optional[Dataset],
) -> Dict[str, Any]:
    return {
        "run_id": result.run_id,
        "dataset_id": result.dataset_id,
        "backend": config.backend,
        "seed": config.seed,
        "dataset": {
            "name": dataset.metadata.name,
            "content_hash": dataset.metadata.content_hash,
        } if dataset else {},
        "split": result.split,
        "metrics": result.metrics,
        "confusion_matrix": result.confusion_matrix.tolist(),
    }


class JsonReporter:
    def write(self, report: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


class HtmlReporter:
    def write(self, report: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        metrics = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
            for k, v in report["metrics"].items()
        )
        doc = (
            "<html><head><title>Experiment Report</title></head><body>"
            f"<h1>Run {html.escape(report['run_id'])}</h1>"
            f"<table border='1'><tr><th>Metric</th><th>Value</th></tr>{metrics}</table>"
            "</body></html>"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_reporting.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(reporting): JSON and HTML reporters"
```

---

### Task 13: Experiment Runner + end-to-end smoke test

**Files:**
- Create: `src/hand_gesture_recognition/runner/__init__.py`
- Create: `src/hand_gesture_recognition/runner/run.py`
- Create: `src/hand_gesture_recognition/__main__.py`
- Test: `tests/integration/test_smoke.py`

**Interfaces:**
- Consumes: `load_config`, `save_resolved_config`, `get_logger`, `stage`, `MlflowTracker`, `EPN612Loader`/`make_synthetic_dataset`, `split_dataset`, `assert_no_group_leakage`, `window_dataset`, `SklearnPredictionAdapter`, `evaluate`, `render_report`, reporters, `hydra.utils.instantiate`.
- Produces:
  - `ExperimentRunner(cfg: ExperimentConfig) -> None.run()` executing the full classical pipeline and writing all artifacts under `output/`.
  - `run_experiment(mode, overrides, resolved_path)` CLI entry point.

- [ ] **Step 1: Write the failing smoke test**

`tests/integration/test_smoke.py`:

```python
import os

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_classical_smoke(tmp_path, monkeypatch):
    cfg = load_config(mode="compose", overrides=[
        "dataset.name=synthetic",
        "split.strategy=random",
        "reporting.formats=[json]",
    ])
    # filesystem paths are set directly on the typed config (avoids Hydra
    # override parsing of Windows backslash/colon path separators)
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output"
    assert (out / "evaluation" / cfg.run_id / "resolved.yaml").exists()
    assert (out / "evaluation" / cfg.run_id / "report.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/integration/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `runner/run.py`**

```python
from __future__ import annotations

import os
import time
import uuid

from hydra.utils import instantiate

from hand_gesture_recognition.configuration.loader import save_resolved_config
from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import assert_no_group_leakage, split_dataset
from hand_gesture_recognition.evaluation.base import evaluate
from hand_gesture_recognition.logging_setup.config import get_logger
from hand_gesture_recognition.reporting.reporters import HtmlReporter, JsonReporter, render_report
from hand_gesture_recognition.tracking.mlflow_tracker import MlflowTracker
from hand_gesture_recognition.tracing.stage import stage

OUTPUT_ROOT = os.environ.get("HAND_GESTURE_RECOGNITION_OUTPUT", "output")


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cfg.run_id = cfg.run_id or f"{cfg.backend}-{uuid.uuid4().hex[:8]}"
        self.logger = get_logger("hand_gesture_recognition.runner", run_id=self.cfg.run_id)
        self.tracker = MlflowTracker(cfg.tracking)

    def run(self) -> None:
        cfg = self.cfg
        self.tracker.start(cfg.run_id)
        try:
            with stage(self.logger, "load_dataset"):
                dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(dataset)
            with stage(self.logger, "split"):
                split = split_dataset(dataset, cfg.split)
                assert_no_group_leakage(split)
            with stage(self.logger, "train"):
                prediction = self._train_and_predict(split.train, split.test)
            with stage(self.logger, "evaluate"):
                result = evaluate(
                    prediction, cfg.evaluation,
                    dataset_id=dataset.metadata.name,
                    split_info={"strategy": cfg.split.strategy},
                )
            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(dataset)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                self.tracker.log_results(result)
                self._persist_artifacts(dataset, result)
            self.tracker.finish("FINISHED")
        except Exception:
            self.tracker.finish("FAILED")
            raise

    def _load_dataset(self):
        if self.cfg.dataset.name == "synthetic":
            return make_synthetic_dataset(seed=self.cfg.seed)
        loader = instantiate(self.cfg.dataset.loader)
        return loader.load(self.cfg.dataset.path)

    def _analytics(self, dataset) -> None:
        if not self.cfg.analytics.enabled:
            return
        from hand_gesture_recognition.analytics.stats import dataset_summary
        import json
        d = os.path.join(OUTPUT_ROOT, "analytics", dataset.metadata.name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "dataset_summary.json"), "w") as f:
            json.dump(dataset_summary(dataset), f, indent=2)

    def _train_and_predict(self, train, test):
        from hand_gesture_recognition.classical.pipelines.windowing import WindowingTransformer

        # shared windowing via the spec-mandated sklearn wrapper (Section 6.2)
        windower = WindowingTransformer(self.cfg.windowing)
        train_wd = windower.fit_transform(train)
        test_wd = windower.transform(test)

        # handcrafted features (3D -> 2D), then preprocessing (scaler, fit on
        # train only) — both constructed via hydra.utils.instantiate (ADR #18).
        X_train, X_test = train_wd.X, test_wd.X
        for step_cfg in self.cfg.classical.features:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_test = step.transform(X_test)
        for step_cfg in self.cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_test = step.transform(X_test)

        model = instantiate(self.cfg.classical.model)
        t0 = time.perf_counter()
        model.fit(X_train, train_wd.y)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        duration = time.perf_counter() - t0

        adapter = SklearnPredictionAdapter()
        return adapter.to_prediction(
            windowed=test_wd, y_pred=y_pred, y_score=y_score,
            run_id=self.cfg.run_id, duration_s=duration,
        )

    def _persist_artifacts(self, dataset, result) -> None:
        run_dir = os.path.join(OUTPUT_ROOT, "evaluation", self.cfg.run_id)
        cfg_dir = os.path.join(run_dir, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        resolved_path = os.path.join(cfg_dir, "resolved.yaml")
        save_resolved_config(self.cfg, resolved_path)
        self.tracker.log_artifact(resolved_path)

        report = render_report(result, self.cfg, dataset)
        if "json" in self.cfg.reporting.formats:
            JsonReporter().write(report, os.path.join(run_dir, "report.json"))
        if "html" in self.cfg.reporting.formats:
            HtmlReporter().write(report, os.path.join(run_dir, "report.html"))
```

`src/hand_gesture_recognition/__main__.py`:

```python
from __future__ import annotations

import argparse

from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="compose", choices=["compose", "replay"])
    p.add_argument("--resolved", default=None, help="path to resolved.yaml for replay")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(mode=args.mode, overrides=args.overrides, resolved_path=args.resolved)
    ExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `pytest tests/integration/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full test suite and a real CLI invocation**

```bash
pytest
python -m hand_gesture_recognition --mode compose "split.strategy=subject_wise"
```

Expected: full suite green; CLI run creates `output/evaluation/<run_id>/{config/resolved.yaml, report.json, report.html}` and an `mlruns/` entry with the dataset content hash.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(runner): Experiment Runner and end-to-end smoke test"
```

---

## Self-Review (completed by plan author)

**Spec coverage (Phase 1 scope, spec §24):**
- Config Loading Module compose + replay + structured validation → Tasks 1, 6. ✅
- Hydra structured configuration → Tasks 1, 6. ✅
- One dataset loader (EPN612) → Task 3. ✅
- Common `Dataset` representation → Task 2. ✅
- Train/val/test split with leakage check → Task 4. ✅
- Shared windowing → Task 5. ✅
- Analytics (stats + QC) → Task 7. ✅
- sklearn classical pipeline → Task 10. ✅
- One classical model → Task 10. ✅
- `Prediction` + classical adapter → Task 8. ✅
- Basic evaluation → Task 9. ✅
- MLflow + dataset content hash → Tasks 2, 11, 13. ✅
- JSON/HTML reporting → Task 12. ✅

**Testing-strategy properties (spec §27) applicable to Phase 1:** windowing math round-trip (Task 5 + Task 1 resolve tests), split leakage (Task 4), config validation of inconsistent sample/time windowing (Task 1), config replay parity (Task 6), end-to-end smoke test (Task 13). ✅ Cross-backend parity and HPO-isolation tests are intentionally deferred to Phase 2/3.

**Placeholder scan:** no TBD/TODO/"similar to Task N"; every code step has concrete code.

**Type consistency:** `WindowingConfig.resolve(sampling_rate)->(int,int)` is used consistently in Tasks 5 and 1; `window_dataset(Dataset, WindowingConfig)->WindowedDataset` used in Tasks 5, 10, 13; `Prediction` fields match between Tasks 8 and 9; `EvaluationResult.metrics` dict used in Tasks 9, 11, 12, 13; `MlflowTracker.start(run_id)/log_*/finish(status)` consistent between Tasks 11 and 13.

**Note on `configs/`** — Task 6 writes the `configs/` tree consumed by `hydra.compose`. The runner (Task 13) builds the classical pipeline using the spec-mandated `WindowingTransformer` (shared windowing, Section 6.2) followed by `hydra.utils.instantiate` of the configured features, preprocessing steps, and model (ADR #18), then normalizes output through the `SklearnPredictionAdapter`.
