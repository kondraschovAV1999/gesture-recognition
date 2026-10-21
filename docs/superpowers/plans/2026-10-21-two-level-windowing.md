# Two-Level Windowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `windowing.mode: two_level` so recognition can segment a trial into **big windows**, signal-process each big window, slide **small windows** inside it, and classify at the **big-window** level (classical: features on the big window; neural: per-small-window features fed to an LSTM).

**Architecture:** A new `window_dataset_two_level` builder emits `TwoLevelWindowedDataset` (stacked `big_windows` + per-big-window `small_windows` + big-window labels + `WindowMetadata`). A thin `window_big_windows` flatten reuses it for the classical path and the prediction-overlay visualizations. The runner branches on `windowing.mode`: it skips per-trial signal processing for `two_level` (filtering happens per big window inside the builder), and routes classical to the existing `_train_and_predict_classical` (on flattened big windows) and neural to a new `TwoLevelEMGDataModule` + the existing `LSTMModule` with `input_size = n_features`.

**Tech Stack:** Python 3.10, Hydra, NumPy, scikit-learn, PyTorch Lightning, pytest.

## Global Constraints

- Default `mode: single` preserves current behavior; all existing tests stay green.
- `mode` restricted to `("single", "two_level")`; `two_level` requires a resolvable big window and `big_size >= small_size`.
- Signal processing is applied **per big window** in `two_level` mode (never per trial, never per small window).
- Rest is identified **structurally** by `gesture_start < 0`; big-window labels use the same overlap rule as `window_dataset` (`rest` if no overlap with `[gesture_start, gesture_end]`, else the gesture class).
- Both `TwoLevelWindowedDataset` and the flattened big-window `WindowedDataset` reuse the existing `WindowMetadata` type; big-window ids are `f"{subj}_{tid}_big{i}"`.
- Neural reuses `LSTMModule` (no new model class) with `n_channels = n_features`; the existing `ClassifierModule`/`LightningPredictionAdapter` are unchanged.
- No changes to the `classification` task, `window_gestures`, or single-level recognition.
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: Config — `mode`, big-window fields, `resolve_big`, validation, preset

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `configs/windowing/default.yaml`
- Create: `configs/windowing/two_level.yaml`
- Test: `tests/unit/test_config_schema.py` (append), `tests/unit/test_config_loader.py` (append)

**Interfaces:**
- Produces: `WindowingConfig.mode: str = "single"`; big fields `big_window_size_samples`, `big_window_size_seconds`, `big_stride_samples`, `big_stride_seconds`, `big_overlap_percent`.
- Produces: `WindowingConfig.resolve_big(sampling_rate: float) -> Tuple[int, int]`.
- Produces: validation errors for bad `mode` and for `big_size < small_size`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_schema.py`:

```python
def test_windowing_mode_defaults_single():
    from hand_gesture_recognition.configuration.schema import WindowingConfig
    assert WindowingConfig().mode == "single"


def test_windowing_rejects_bad_mode():
    cfg = _base_experiment()
    cfg.windowing.mode = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_two_level_requires_big_at_least_small():
    cfg = _base_experiment()
    cfg.windowing.mode = "two_level"
    cfg.windowing.window_size_samples = 40
    cfg.windowing.stride_samples = 10
    cfg.windowing.big_window_size_samples = 20  # < small
    cfg.windowing.big_stride_samples = 10
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_resolve_big_derives_and_validates():
    from hand_gesture_recognition.configuration.schema import WindowingConfig
    cfg = WindowingConfig(
        big_window_size_seconds=0.5, big_stride_seconds=0.1
    )  # 100 / 20 @ 200 Hz
    assert cfg.resolve_big(sampling_rate=200.0) == (100, 20)
```

Append to `tests/unit/test_config_loader.py`:

```python
def test_compose_loads_two_level_windowing():
    cfg = load_config(mode="compose", overrides=["windowing=two_level"])
    assert cfg.windowing.mode == "two_level"
    assert cfg.windowing.big_window_size_samples is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: FAIL (`WindowingConfig` has no `mode`/`big_*`; no `resolve_big`; no `windowing/two_level.yaml` preset).

- [ ] **Step 3: Add fields and `resolve_big` to the schema**

In `src/hand_gesture_recognition/configuration/schema.py`, change `WindowingConfig` to add `mode` and the `big_*` fields (keep the existing `resolve` method body unchanged), and add `resolve_big`:

```python
@dataclass
class WindowingConfig:
    mode: str = "single"  # "single" | "two_level"
    window_size_samples: Optional[int] = None
    window_size_seconds: Optional[float] = None
    stride_samples: Optional[int] = None
    stride_seconds: Optional[float] = None
    overlap_percent: Optional[float] = None
    label_granularity: str = "per_window"  # per_window | sequence
    padding: str = "none"  # none | zero
    # --- big window (used only when mode == "two_level") ---
    big_window_size_samples: Optional[int] = None
    big_window_size_seconds: Optional[float] = None
    big_stride_samples: Optional[int] = None
    big_stride_seconds: Optional[float] = None
    big_overlap_percent: Optional[float] = None

    def resolve(self, sampling_rate: float) -> Tuple[int, int]:
        # (existing body — leave unchanged)
        ...

    def resolve_big(self, sampling_rate: float) -> Tuple[int, int]:
        """Return (big_window_size_samples, big_stride_samples), mirroring resolve()."""
        if sampling_rate <= 0:
            raise ConfigValidationError(
                f"sampling_rate must be > 0, got {sampling_rate}"
            )
        ws = self.big_window_size_samples
        if self.big_window_size_seconds is not None:
            derived = round(self.big_window_size_seconds * sampling_rate)
            if ws is not None and ws != derived:
                raise ConfigValidationError(
                    f"big_window_size_samples={ws} != big_window_size_seconds="
                    f"{self.big_window_size_seconds}s * {sampling_rate}Hz = {derived}"
                )
            ws = derived
        if ws is None:
            raise ConfigValidationError(
                "big window size must be given as big_window_size_samples "
                "or big_window_size_seconds"
            )
        if ws < 1:
            raise ConfigValidationError(
                f"big_window_size_samples must be >= 1, got {ws}"
            )
        stride = self.big_stride_samples
        if self.big_stride_seconds is not None:
            derived = round(self.big_stride_seconds * sampling_rate)
            if stride is not None and stride != derived:
                raise ConfigValidationError(
                    f"big_stride_samples={stride} != big_stride_seconds="
                    f"{self.big_stride_seconds}s * {sampling_rate}Hz = {derived}"
                )
            stride = derived
        if stride is None:
            if self.big_overlap_percent is not None:
                stride = round(ws * (1.0 - self.big_overlap_percent / 100.0))
            else:
                stride = ws
        if stride < 1:
            raise ConfigValidationError(f"big stride must be >= 1, got {stride}")
        return ws, stride
```

- [ ] **Step 4: Add validation**

In `validate_experiment_config`, replace the line
`cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)` with:

```python
    if cfg.windowing.mode not in ("single", "two_level"):
        raise ConfigValidationError(
            f"windowing.mode={cfg.windowing.mode!r} not in ('single', 'two_level')"
        )
    small_size, _ = cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)
    if cfg.windowing.mode == "two_level":
        big_size, _ = cfg.windowing.resolve_big(sampling_rate=cfg.dataset.sampling_rate)
        if big_size < small_size:
            raise ConfigValidationError(
                f"two_level requires big window size ({big_size}) >= "
                f"small window size ({small_size})"
            )
```

- [ ] **Step 5: Add the config files**

In `configs/windowing/default.yaml`, add `mode: single` as the first line (after the `# @package windowing` comment).

Create `configs/windowing/two_level.yaml`:

```yaml
# @package windowing
mode: two_level
window_size_samples: 20      # small window
stride_samples: 10
big_window_size_samples: 60  # big window
big_stride_samples: 30
padding: none
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS (all tests, including the 5 new ones)

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py configs/windowing/default.yaml configs/windowing/two_level.yaml tests/unit/test_config_schema.py tests/unit/test_config_loader.py
git commit -m task1-two-level-config
```

---

### Task 2: Data structures + `window_dataset_two_level` + `window_big_windows`

**Files:**
- Create: `src/hand_gesture_recognition/data/windowing/two_level.py`
- Test: `tests/unit/test_two_level_windowing.py`

**Interfaces:**
- Produces: `TwoLevelWindowedDataset` (`big_windows: np.ndarray`, `small_windows: List[np.ndarray]`, `y: np.ndarray`, `metadata: WindowMetadata`).
- Produces: `TwoLevelWindowedPartitions` (`train`/`val`/`test`).
- Produces: `window_dataset_two_level(dataset, config, signal_steps) -> TwoLevelWindowedDataset` (where `signal_steps` are already-instantiated transform objects with `.transform(segment)`).
- Produces: `window_big_windows(dataset, config, signal_steps) -> WindowedDataset`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_two_level_windowing.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.two_level import (
    window_big_windows,
    window_dataset_two_level,
)


def _dataset():
    # trial 0: gesture [0,100] label 1; trial 1: rest label 0
    return Dataset(
        trials=[
            np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1),
            np.zeros((100, 2), dtype=np.float32),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([0, -1]),
        gesture_end=np.array([100, -1]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist"],
            content_hash="h",
        ),
    )


def _config():
    return WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )


def test_big_and_small_window_counts():
    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[])
    # trial 0 (len 100): big windows [0,60], [30,90] -> 2; trial 1 (rest): 2 -> 4 total
    assert len(twd.big_windows) == 4
    assert twd.y.tolist() == [1, 1, 0, 0]
    # each 60-sample big window -> (60-20)//10 + 1 = 5 small windows
    assert twd.small_windows[0].shape == (5, 20, 2)
    assert twd.big_windows.shape == (4, 60, 2)


def test_big_window_labels_and_metadata():
    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[])
    assert twd.metadata.window_start.tolist() == [0, 30, 0, 30]
    assert twd.metadata.window_end.tolist() == [60, 90, 60, 90]
    assert twd.metadata.window_ids.tolist() == [
        "s1_t1_big0", "s1_t1_big1", "s1_t2_big0", "s1_t2_big1"
    ]
    assert twd.metadata.trial_ids.tolist() == ["t1", "t1", "t2", "t2"]


def test_signal_steps_applied_per_big_window():
    class Scale:
        def transform(self, x):
            return x * 2.0

    twd = window_dataset_two_level(_dataset(), _config(), signal_steps=[Scale()])
    assert np.allclose(twd.big_windows[0], _dataset().trials[0][0:60] * 2.0)


def test_drop_rest_big_windows():
    ds = _dataset()
    ds.metadata.drop_rest_windows = True
    twd = window_dataset_two_level(ds, _config(), signal_steps=[])
    assert twd.y.tolist() == [1, 1]
    assert len(twd.big_windows) == 2


def test_window_big_windows_flat():
    wd = window_big_windows(_dataset(), _config(), signal_steps=[])
    assert wd.X.shape == (4, 60, 2)
    assert wd.y.tolist() == [1, 1, 0, 0]
    assert wd.metadata.window_ids[0] == "s1_t1_big0"


def test_empty_input_returns_empty():
    ds = _dataset()
    ds.gesture_start = np.array([-1, -1])
    ds.gesture_end = np.array([-1, -1])
    twd = window_dataset_two_level(ds, _config(), signal_steps=[])
    assert twd.big_windows.shape[0] == 0
    assert twd.y.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_two_level_windowing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.data.windowing.two_level'`

- [ ] **Step 3: Implement `two_level.py`**

Create `src/hand_gesture_recognition/data/windowing/two_level.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowMetadata,
)


@dataclass
class TwoLevelWindowedDataset:
    big_windows: np.ndarray            # (n_big, big_size, channels) — signal-processed
    small_windows: List[np.ndarray]    # per big window: (n_small_i, small_size, channels)
    y: np.ndarray                      # (n_big,) big-window labels
    metadata: WindowMetadata           # big-window level


@dataclass
class TwoLevelWindowedPartitions:
    train: TwoLevelWindowedDataset
    val: TwoLevelWindowedDataset
    test: TwoLevelWindowedDataset


def window_dataset_two_level(
    dataset: Dataset,
    config: WindowingConfig,
    signal_steps: List[Any],
) -> TwoLevelWindowedDataset:
    """Slide big windows, filter each, slide small windows inside, label big.

    ``signal_steps`` are already-instantiated transform objects (each with a
    ``transform(segment)`` method), applied per big window.
    """
    big_size, big_stride = config.resolve_big(sampling_rate=dataset.metadata.sampling_rate)
    small_size, small_stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    drop_rest = dataset.metadata.drop_rest_windows
    n_channels = (
        len(dataset.metadata.channels)
        if dataset.metadata.channels
        else (dataset.trials[0].shape[1] if dataset.trials else 0)
    )

    big_windows: List[np.ndarray] = []
    small_windows: List[np.ndarray] = []
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
        length = trial.shape[0]
        if length < big_size:
            if config.padding == "zero":
                pad = np.zeros((big_size - length, trial.shape[1]), dtype=trial.dtype)
                trial = np.vstack([trial, pad])
                length = big_size
            else:
                continue  # drop trials shorter than a big window
        n_big = (length - big_size) // big_stride + 1
        for i in range(n_big):
            s = i * big_stride
            e = s + big_size
            # A big window is "rest" if it does not overlap [g_start, g_end]
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
            if drop_rest and is_rest:
                continue
            big_seg = trial[s:e]
            for step in signal_steps:
                big_seg = step.transform(big_seg)
            blen = big_seg.shape[0]
            smalls = []
            if blen >= small_size:
                n_small = (blen - small_size) // small_stride + 1
                for j in range(n_small):
                    ss = j * small_stride
                    smalls.append(big_seg[ss:ss + small_size])
            if not smalls:
                continue  # defensive: no small window fits
            big_windows.append(big_seg)
            small_windows.append(np.stack(smalls, axis=0))
            ys.append(0 if is_rest else int(label))
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))
            wins.append(f"{subj}_{tid}_big{i}")

    return TwoLevelWindowedDataset(
        big_windows=np.stack(big_windows, axis=0)
        if big_windows
        else np.zeros((0, big_size, n_channels), dtype=np.float32),
        small_windows=small_windows,
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
            window_ids=np.asarray(wins),
        ),
    )


def window_big_windows(
    dataset: Dataset,
    config: WindowingConfig,
    signal_steps: List[Any],
) -> WindowedDataset:
    """Flat big-window view (for the classical path and prediction overlays)."""
    twd = window_dataset_two_level(dataset, config, signal_steps)
    return WindowedDataset(X=twd.big_windows, y=twd.y, metadata=twd.metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_two_level_windowing.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/two_level.py tests/unit/test_two_level_windowing.py
git commit -m task2-two-level-windowing
```

---

### Task 3: Runner — two-level dispatch + classical path

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Create: `tests/integration/test_two_level_classical.py`

**Interfaces:**
- Consumes: `window_dataset_two_level` (Task 2).
- Produces: `_window` returns `TwoLevelWindowedPartitions` for `two_level`; `_run_partition` skips per-trial signal processing; `_train_and_predict_classical_two_level` flattens big windows.

- [ ] **Step 1: Write the failing integration test**

Create `tests/integration/test_two_level_classical.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_classical_two_level_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=classical",
            "windowing=two_level",
            "windowing.big_window_size_samples=60",
            "windowing.big_stride_samples=30",
            "windowing.window_size_samples=20",
            "windowing.stride_samples=10",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_two_level_classical.py -v`
Expected: FAIL — `_window` still uses single-level windowing, so `windowing=two_level` has no effect.

- [ ] **Step 3: Dispatch `_window` on mode**

In `src/hand_gesture_recognition/runner/run.py`, replace the `_window` method with:

```python
    def _window(self, split):
        from hand_gesture_recognition.data.windowing.base import WindowedPartitions
        from hand_gesture_recognition.data.windowing.dispatch import (
            build_windowed_dataset,
        )

        if self.cfg.windowing.mode == "two_level":
            from hand_gesture_recognition.data.windowing.two_level import (
                TwoLevelWindowedPartitions,
                window_dataset_two_level,
            )

            steps = (
                [instantiate(s) for s in self.cfg.signal_processing]
                if self.cfg.signal_processing
                else []
            )
            windowed = TwoLevelWindowedPartitions(
                train=window_dataset_two_level(split.train, self.cfg.windowing, steps),
                val=window_dataset_two_level(split.val, self.cfg.windowing, steps),
                test=window_dataset_two_level(split.test, self.cfg.windowing, steps),
            )
            for part in ("train", "val", "test"):
                if getattr(windowed, part).y.shape[0] == 0:
                    self.logger.warning(
                        f"partition '{part}' has 0 big windows after two-level windowing"
                    )
            return windowed

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

- [ ] **Step 4: Skip per-trial signal processing for two-level**

In `_run_partition`, replace `split = self._signal_processing(split)` with:

```python
        if self.cfg.windowing.mode != "two_level":
            # two-level filters per big window inside the builder instead
            split = self._signal_processing(split)
```

- [ ] **Step 5: Route the classical branch and add the flatten helper**

In `_run_partition`, replace the `with stage(self.logger, "train"):` block with:

```python
        with stage(self.logger, "train"):
            if self.cfg.backend == "classical":
                if self.cfg.windowing.mode == "two_level":
                    predictions = self._train_and_predict_classical_two_level(windowed)
                else:
                    predictions = self._train_and_predict_classical(windowed)
            else:
                predictions = self._train_and_predict_neural(
                    windowed, n_classes=len(class_names)
                )
```

Then add this method (directly after `_train_and_predict_classical`):

```python
    def _train_and_predict_classical_two_level(self, windowed) -> PartitionPredictions:
        from hand_gesture_recognition.data.windowing.base import (
            WindowedDataset,
            WindowedPartitions,
        )

        def flatten(twd) -> WindowedDataset:
            return WindowedDataset(X=twd.big_windows, y=twd.y, metadata=twd.metadata)

        flat = WindowedPartitions(
            train=flatten(windowed.train),
            val=flatten(windowed.val),
            test=flatten(windowed.test),
        )
        return self._train_and_predict_classical(flat)
```

- [ ] **Step 6: Run the integration test**

Run: `pytest tests/integration/test_two_level_classical.py -v`
Expected: PASS (classical two-level completes; 3×3 confusion matrix)

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_two_level_classical.py
git commit -m task3-two-level-classical
```

### Task 4: Neural two-level — `TwoLevelEMGDataModule` + LSTM-over-features

**Files:**
- Create: `src/hand_gesture_recognition/neural/datamodules/two_level.py`
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/unit/test_two_level_datamodule.py`, `tests/integration/test_two_level_neural.py`

**Interfaces:**
- Consumes: `TwoLevelWindowedPartitions` (Task 2), `classical.resolve_features` (existing), `LSTMModule` (existing).
- Produces: `TwoLevelEMGDataModule`; runner's `_compute_n_features` and `_train_and_predict_neural_two_level`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_two_level_datamodule.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.two_level import (
    TwoLevelWindowedPartitions,
    window_dataset_two_level,
)
from hand_gesture_recognition.neural.datamodules.two_level import TwoLevelEMGDataModule


def _partitions():
    ds = Dataset(
        trials=[np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1)],
        labels=np.array([1]),
        subject_ids=np.array(["s1"]),
        trial_ids=np.array(["t1"]),
        gesture_start=np.array([0]),
        gesture_end=np.array([100]),
        metadata=DatasetMetadata(
            name="x", version="0", sampling_rate=200.0,
            channels=["c1", "c2"], class_names=["rest", "fist"], content_hash="h",
        ),
    )
    cfg = WindowingConfig(
        mode="two_level", window_size_samples=20, stride_samples=10,
        big_window_size_samples=60, big_stride_samples=30,
    )
    twd = window_dataset_two_level(ds, cfg, signal_steps=[])
    return TwoLevelWindowedPartitions(train=twd, val=twd, test=twd)


def test_two_level_datamodule_shapes():
    partitions = _partitions()
    feature_specs = [
        {"_target_": "hand_gesture_recognition.features.time_domain.TimeDomainFeatures"}
    ]
    n_features = 7 * 2  # 7 time-domain features x 2 channels
    dm = TwoLevelEMGDataModule(
        windowed=partitions, feature_specs=feature_specs,
        n_features=n_features, max_small_per_big=6, batch_size=2,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # 2 big windows, seq len 6, 14 features
    assert batch["x"].shape == (2, 6, n_features)
    assert batch["y_true"].shape == (2,)
    assert batch["window_id"][0] == "s1_t1_big0"
    assert batch["trial_id"][0] == "t1"
```

Create `tests/integration/test_two_level_neural.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_neural_two_level_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "split.strategy=random",
            "backend=neural",
            "windowing=two_level",
            "windowing.big_window_size_samples=60",
            "windowing.big_stride_samples=30",
            "windowing.window_size_samples=20",
            "windowing.stride_samples=10",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
    assert result.confusion_matrix.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_two_level_datamodule.py tests/integration/test_two_level_neural.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hand_gesture_recognition.neural.datamodules.two_level'`

- [ ] **Step 3: Implement `TwoLevelEMGDataModule`**

Create `src/hand_gesture_recognition/neural/datamodules/two_level.py`:

```python
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.data.windowing.two_level import (
    TwoLevelWindowedDataset,
    TwoLevelWindowedPartitions,
)


class TwoLevelEMGDataset(Dataset):
    """One item per big window: a fixed-length sequence of small-window features."""

    def __init__(
        self,
        twd: TwoLevelWindowedDataset,
        feature_specs: List[Any],
        n_features: int,
        max_small_per_big: int,
    ):
        self.y_true = twd.y
        self.window_ids = twd.metadata.window_ids
        self.trial_ids = twd.metadata.trial_ids
        self.subject_ids = twd.metadata.subject_ids
        self.sequences = []
        for smalls in twd.small_windows:
            parts = []
            for spec in feature_specs:
                step = instantiate(spec)
                parts.append(step.transform(smalls))
            feats = np.concatenate(parts, axis=1).astype(np.float32)  # (n_small, n_features)
            if feats.shape[0] > max_small_per_big:
                feats = feats[:max_small_per_big]
            elif feats.shape[0] < max_small_per_big:
                pad = np.zeros(
                    (max_small_per_big - feats.shape[0], n_features), dtype=np.float32
                )
                feats = np.vstack([feats, pad])
            self.sequences.append(feats)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "x": torch.as_tensor(self.sequences[idx], dtype=torch.float32),
            "y_true": torch.as_tensor(self.y_true[idx], dtype=torch.long),
            "window_id": self.window_ids[idx],
            "trial_id": self.trial_ids[idx],
            "subject_id": self.subject_ids[idx],
        }


class TwoLevelEMGDataModule(LightningDataModule):
    def __init__(
        self,
        windowed: TwoLevelWindowedPartitions,
        feature_specs: List[Any],
        n_features: int,
        max_small_per_big: int,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.windowed = windowed
        self.feature_specs = feature_specs
        self.n_features = n_features
        self.max_small_per_big = max_small_per_big
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = TwoLevelEMGDataset(
            self.windowed.train, self.feature_specs, self.n_features, self.max_small_per_big
        )
        self._val = TwoLevelEMGDataset(
            self.windowed.val, self.feature_specs, self.n_features, self.max_small_per_big
        )
        self._test = TwoLevelEMGDataset(
            self.windowed.test, self.feature_specs, self.n_features, self.max_small_per_big
        )

    def _loader(self, ds, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
```

- [ ] **Step 4: Run the unit test**

Run: `pytest tests/unit/test_two_level_datamodule.py -v`
Expected: PASS

- [ ] **Step 5: Wire the neural two-level runner path**

In `src/hand_gesture_recognition/runner/run.py`, first update the `else` branch of the `with stage(self.logger, "train"):` block (added in Task 3) to route two-level neural:

```python
            else:
                if self.cfg.windowing.mode == "two_level":
                    predictions = self._train_and_predict_neural_two_level(
                        windowed, n_classes=len(class_names)
                    )
                else:
                    predictions = self._train_and_predict_neural(
                        windowed, n_classes=len(class_names)
                    )
```

Then add these two methods directly after `_train_and_predict_classical_two_level`:

```python
    def _compute_n_features(self, feature_specs, windowed) -> int:
        for part in ("train", "val", "test"):
            for smalls in getattr(windowed, part).small_windows:
                if smalls.shape[0] > 0:
                    parts = []
                    for spec in feature_specs:
                        step = instantiate(spec)
                        parts.append(step.transform(smalls[:1]))
                    return int(np.concatenate(parts, axis=1).shape[1])
        return 0

    def _train_and_predict_neural_two_level(
        self, windowed, n_classes: int
    ) -> PartitionPredictions:
        import logging

        import lightning.pytorch as pl

        from hand_gesture_recognition.neural.adapters.lightning import (
            LightningPredictionAdapter,
        )
        from hand_gesture_recognition.neural.datamodules.two_level import (
            TwoLevelEMGDataModule,
        )

        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
        pl.seed_everything(self.cfg.seed, workers=True)

        feature_specs = self.cfg.classical.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        n_features = self._compute_n_features(feature_specs, windowed)
        max_small_per_big = max(
            (
                len(smalls)
                for part in ("train", "val", "test")
                for smalls in getattr(windowed, part).small_windows
            ),
            default=1,
        )
        dm = TwoLevelEMGDataModule(
            windowed=windowed,
            feature_specs=feature_specs,
            n_features=n_features,
            max_small_per_big=max_small_per_big,
            batch_size=self.cfg.neural.datamodule.get("batch_size", 32),
            num_workers=self.cfg.neural.datamodule.get("num_workers", 0),
        )
        # Reuse the configured neural model (LSTM by default) with the feature
        # dimension as its input_size.
        model_partial = instantiate(
            self.cfg.neural.model,
            n_channels=n_features,
            n_classes=n_classes,
            _partial_=True,
        )
        model = model_partial(optimizer=self.cfg.neural.optimizer)
        callbacks = None
        if self.cfg.neural.early_stopping:
            callbacks = [instantiate(self.cfg.neural.early_stopping)]
        trainer_partial = instantiate(self.cfg.neural.trainer, _partial_=True)
        trainer = trainer_partial(callbacks=callbacks)

        trainer.fit(model, datamodule=dm)

        t0 = time.perf_counter()
        y_train_pred_raw = trainer.predict(model, dataloaders=dm.train_dataloader())
        train_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_val_pred_raw = trainer.predict(model, dataloaders=dm.val_dataloader())
        val_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_test_pred_raw = trainer.predict(model, dataloaders=dm.test_dataloader())
        test_duration = time.perf_counter() - t0

        adapter = LightningPredictionAdapter()
        return PartitionPredictions(
            train=adapter.to_prediction(
                raw_predictions=y_train_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=train_duration,
            ),
            val=adapter.to_prediction(
                raw_predictions=y_val_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=val_duration,
            ),
            test=adapter.to_prediction(
                raw_predictions=y_test_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=test_duration,
            ),
        )
```

- [ ] **Step 6: Run the integration test**

Run: `pytest tests/integration/test_two_level_neural.py -v`
Expected: PASS (neural two-level completes; 3×3 confusion matrix)

- [ ] **Step 7: Commit**

```bash
git add src/hand_gesture_recognition/neural/datamodules/two_level.py src/hand_gesture_recognition/runner/run.py tests/unit/test_two_level_datamodule.py tests/integration/test_two_level_neural.py
git commit -m task4-two-level-neural
```

<!-- PART3 -->

---

### Task 5: Two-level-aware prediction overlays

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/dispatch.py`
- Test: `tests/unit/test_two_level_windowing.py` (append one test)

**Interfaces:**
- Consumes: `window_big_windows` (Task 2).
- Produces: `build_windowed_dataset` now returns the flat big-window view when `config.mode == "two_level"`, so the prediction-overlay visualizations (which already call it) align to big-window bounds.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_two_level_windowing.py`:

```python
def test_build_windowed_dataset_two_level():
    from hand_gesture_recognition.data.windowing.dispatch import build_windowed_dataset
    wd = build_windowed_dataset(_dataset(), _config(), "recognition")
    assert wd.X.shape == (4, 60, 2)
    assert wd.metadata.window_ids[0] == "s1_t1_big0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_two_level_windowing.py::test_build_windowed_dataset_two_level -v`
Expected: FAIL (the dispatcher ignores `mode` and calls `window_dataset`, yielding small-window samples).

- [ ] **Step 3: Make the dispatcher two-level-aware**

In `src/hand_gesture_recognition/data/windowing/dispatch.py`, add the two-level branch at the top of `build_windowed_dataset`:

```python
def build_windowed_dataset(dataset, config, task):
    if config.mode == "two_level":
        from hand_gesture_recognition.data.windowing.two_level import window_big_windows
        return window_big_windows(dataset, config, [])
    if task == "classification":
        return window_gestures(dataset, config)
    return window_dataset(dataset, config)
```

- [ ] **Step 4: Run the test file**

Run: `pytest tests/unit/test_two_level_windowing.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/dispatch.py tests/unit/test_two_level_windowing.py
git commit -m task5-two-level-visualization
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/experiments/configuration.md`
- Modify: `docs/experiments/running.md`
- Modify: `docs/architecture/overview.md`

- [ ] **Step 1: Document `mode` in `configuration.md`**

In the `### windowing` section of `docs/experiments/configuration.md`, add the `mode` and `big_*` rows to the field table and a short paragraph:

```markdown
| `mode` | `single` \| `two_level` | `single` |
| `big_window_size_samples` / `big_window_size_seconds` | one or the other (two_level only) | `null` |
| `big_stride_samples` / `big_stride_seconds` / `big_overlap_percent` | one of the three (two_level only) | `null` |

`windowing=two_level` enables two-level windowing: the trial is segmented into
**big windows** (filtered independently), then **small windows** are slid inside
each big window, and the model classifies at the **big-window** level (classical
extracts features on the big window; neural extracts features per small window
and feeds the sequence to an LSTM). Preset: `windowing=two_level`.
```

- [ ] **Step 2: Add the override + example to `running.md`**

In `docs/experiments/running.md`, add a row to the "Common overrides" table (after the `windowing.window_size_seconds` row):

```markdown
| `windowing=two_level` | two-level windowing: big → small windows, classify at the big-window level |
```

and add an example to the Examples block:

```bash
# Two-level windowing: big windows filtered, small windows -> LSTM, big-window labels
python -m hand_gesture_recognition --mode compose windowing=two_level dataset.name=synthetic backend=neural
```

- [ ] **Step 3: Update the architecture overview**

In `docs/architecture/overview.md`, append to step 5 ("Windowing") of the data-flow list:

```markdown
   When `windowing.mode=two_level`, the shared windowing instead segments each
   trial into big windows (signal-processed per big window), slides small
   windows inside each, and labels at the big-window level; classical extracts
   features on the big window, neural extracts features per small window and
   feeds the sequence to an LSTM.
```

and add a bullet to "Key design decisions":

```markdown
7. **Two-level windowing** — `windowing.mode=two_level` adds a coarse big-window
   level (per-big-window filtering + big-window classification) over the fine
   small-window level, sharing the same `WindowMetadata`/`evaluate` machinery.
```

- [ ] **Step 4: Commit**

```bash
git add docs/experiments/configuration.md docs/experiments/running.md docs/architecture/overview.md
git commit -m task6-two-level-docs
```

---

## Self-Review

- **Spec coverage:** §1 config (`mode`, `big_*`, `resolve_big`, validation, preset) → Task 1; §2 data structures → Task 2; §3 builder `window_dataset_two_level` → Task 2; §4 per-big-window signal processing → Task 2 (inside builder) + Task 3 (runner skips per-trial); §5 classical path → Task 3 (flatten + reuse); §6 neural path → Task 4 (`TwoLevelEMGDataModule` + `LSTMModule(input_size=n_features)`); §7 runner wiring → Tasks 3–4; §8 evaluation/ground truth → unchanged `evaluate` on big-window labels (invariant across Tasks 2–4) + Task 2 unit tests for overlap labels; §9 visualization → Task 5.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test commands include expected outcomes.
- **Type consistency:** `window_dataset_two_level(dataset, config, signal_steps)` and `window_big_windows(dataset, config, signal_steps)` are defined in Task 2 and consumed in Tasks 3 and 5 with the same signatures. `TwoLevelWindowedDataset`/`TwoLevelWindowedPartitions` are defined in Task 2 and consumed in Tasks 3–4 with the same field names (`big_windows`, `small_windows`, `y`, `metadata`). `resolve_big(sampling_rate)` is defined in Task 1 and consumed in Task 2. `mode` is defined in Task 1 and read in Tasks 3–5. The `_train_and_predict_classical_two_level`/`_train_and_predict_neural_two_level`/`_compute_n_features` methods are defined in Tasks 3–4 and called in the `_run_partition` branch they add.

