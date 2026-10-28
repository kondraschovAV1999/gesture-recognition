# Shape Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a per-stage shape summary (console + `shape_summary.json` artifact) using symbolic dimension names, for a representative (first non-rest) trial, in both windowing modes.

**Architecture:** A new pure function `compute_shape_summary(dataset, windowing, feature_specs, task_name)` in `utils/observability/shapes.py` windows one representative trial via the existing windowing strategies and reports structural shapes with symbolic keys. The pipeline calls it once per run (independent of `visualization.enabled`), logs it, and persists `shape_summary.json`.

**Tech Stack:** Python 3.10, numpy, hydra (`instantiate`), pytest.

## Global Constraints

- Symbolic dimension keys exactly: `channels`, `datapoints_per_channel`, `big_windows`, `big_window_size`, `small_windows`, `small_window_size`, `windows`, `window_size`, `features_per_small_window`, `features_per_window`.
- Two-level stages: `input`, `big_windows`, `small_windows`, `features`. Single-level stages: `input`, `windows`, `features`.
- Representative trial = first non-rest trial (`next(t for t in dataset.trials if not t.is_rest)`).
- `features_per_small_window`/`features_per_window` = features per channel per window, derived from the resolved feature specs (11 time, 4 freq, 15 combined; a subset yields the subset count).
- Reuse existing windowing strategies; do NOT reimplement windowing. Single-level strategy name = `task_name` (recognition → `window_dataset`, classification → `window_gestures`), two-level = `two_level`.
- Do NOT commit `configs/config.yaml`, `configs/feature_extraction/default.yaml`, or `configs/neural/lstm.yaml`.

---

### Task 1: `compute_shape_summary` function + unit tests

**Files:**
- Create: `src/hand_gesture_recognition/utils/observability/shapes.py`
- Test: `tests/unit/test_shapes.py` (new)

**Interfaces:**
- Consumes: `Dataset`, `WindowingConfig` from `hand_gesture_recognition.configuration.schema`; `get_windowing_strategy` from `hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy`; `TIME_DOMAIN_FEATURE_NAMES`, `FREQUENCY_DOMAIN_FEATURE_NAMES` from `hand_gesture_recognition.configuration.schema.feature_extraction`.
- Produces: `compute_shape_summary(dataset, windowing, feature_specs, task_name="recognition") -> Dict[str, Any]`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_shapes.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
    WindowingConfig,
)
from hand_gesture_recognition.utils.observability.shapes import compute_shape_summary

_TIME_SPEC = [
    {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
    }
]


def _ds(n_channels=8, length=994):
    meta = DatasetMetadata("x", "0", 200.0, [f"ch{i}" for i in range(n_channels)],
                           ["noGesture", "fist"], "h")
    rest = Trial("t_rest", "s1", "noGesture", -1, -1, np.zeros((length, n_channels), np.float32))
    fist = Trial("t_fist", "s1", "fist", 0, length, np.ones((length, n_channels), np.float32))
    return Dataset(meta, [User("s1", [rest, fist])])


def test_two_level_shape_summary():
    ds = _ds()
    windowing = WindowingConfig(
        mode="two_level",
        window_size_samples=25, stride_samples=10,
        big_window_size_samples=250, big_stride_samples=50,
    )
    s = compute_shape_summary(ds, windowing, _TIME_SPEC)
    assert s["mode"] == "two_level"
    assert s["input"] == {"channels": 8, "datapoints_per_channel": 994}
    assert s["big_windows"]["channels"] == 8
    assert s["big_windows"]["big_window_size"] == 250
    assert s["big_windows"]["big_windows"] == (994 - 250) // 50 + 1
    assert s["small_windows"]["small_window_size"] == 25
    assert s["small_windows"]["small_windows"] == (250 - 25) // 10 + 1
    assert s["small_windows"]["big_windows"] == s["big_windows"]["big_windows"]
    assert s["features"] == {
        "channels": 8,
        "big_windows": s["big_windows"]["big_windows"],
        "small_windows": s["small_windows"]["small_windows"],
        "features_per_small_window": 11,
    }


def test_single_level_shape_summary():
    ds = _ds()
    windowing = WindowingConfig(window_size_samples=25, stride_samples=10)
    s = compute_shape_summary(ds, windowing, _TIME_SPEC, task_name="recognition")
    assert s["mode"] == "single"
    assert s["input"] == {"channels": 8, "datapoints_per_channel": 994}
    assert s["windows"]["window_size"] == 25
    assert s["windows"]["windows"] == (994 - 25) // 10 + 1
    assert s["features"]["features_per_window"] == 11


def test_subset_features_per_channel():
    ds = _ds(n_channels=2, length=100)
    windowing = WindowingConfig(window_size_samples=25, stride_samples=10)
    subset_spec = [
        {
            "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures",
            "features": ["rms", "mav"],
        }
    ]
    s = compute_shape_summary(ds, windowing, subset_spec)
    assert s["features"]["features_per_window"] == 2


def test_no_non_rest_trial_returns_empty():
    meta = DatasetMetadata("x", "0", 200.0, ["ch1"], ["noGesture", "fist"], "h")
    ds = Dataset(meta, [User("s1", [Trial("t", "s1", "noGesture", -1, -1, np.zeros((50, 1), np.float32))])])
    windowing = WindowingConfig(window_size_samples=10, stride_samples=5)
    assert compute_shape_summary(ds, windowing, _TIME_SPEC) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_shapes.py -q`
Expected: FAIL — `ModuleNotFoundError: hand_gesture_recognition.utils.observability.shapes`.

- [ ] **Step 3: Write the module**

Create `src/hand_gesture_recognition/utils/observability/shapes.py`:

```python
from __future__ import annotations

from typing import Any, Dict, List

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
    TIME_DOMAIN_FEATURE_NAMES,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def _features_per_channel(feature_specs: List[Dict[str, Any]]) -> int:
    """Number of handcrafted features per channel from the resolved specs."""
    total = 0
    for spec in feature_specs:
        target = spec.get("_target_", "")
        if "TimeDomainFeatures" in target:
            names = spec.get("features") or TIME_DOMAIN_FEATURE_NAMES
        elif "FrequencyDomainFeatures" in target:
            names = spec.get("features") or FREQUENCY_DOMAIN_FEATURE_NAMES
        else:
            continue
        total += len(names)
    return total


def compute_shape_summary(
    dataset,
    windowing,
    feature_specs: List[Dict[str, Any]],
    task_name: str = "recognition",
) -> Dict[str, Any]:
    """Report the shape at each pipeline stage for a representative trial.

    Picks the first non-rest trial and reports symbolic shapes (channels-first):

    - two_level: input -> big_windows -> small_windows -> features
    - single:    input -> windows -> features

    Returns ``{}`` when there is no non-rest trial.
    """
    trial = next((t for t in dataset.trials if not t.is_rest), None)
    if trial is None:
        return {}

    channels = trial.emg.shape[1]
    datapoints = trial.emg.shape[0]
    summary: Dict[str, Any] = {
        "mode": windowing.mode,
        "input": {"channels": channels, "datapoints_per_channel": datapoints},
    }

    single = dataset.replace_trials([trial])
    features_per_channel = _features_per_channel(feature_specs)

    if windowing.mode == "two_level":
        grouped = get_windowing_strategy("two_level").window(single, windowing, [])
        big_windows = grouped.big.X.shape[0]
        big_window_size = grouped.big.X.shape[1]
        small_windows = grouped.small_windows[0].shape[0] if grouped.small_windows else 0
        small_window_size = (
            grouped.small_windows[0].shape[1] if grouped.small_windows else 0
        )
        summary["big_windows"] = {
            "channels": channels,
            "big_windows": big_windows,
            "big_window_size": big_window_size,
        }
        summary["small_windows"] = {
            "channels": channels,
            "big_windows": big_windows,
            "small_windows": small_windows,
            "small_window_size": small_window_size,
        }
        summary["features"] = {
            "channels": channels,
            "big_windows": big_windows,
            "small_windows": small_windows,
            "features_per_small_window": features_per_channel,
        }
    else:
        strategy = get_windowing_strategy(task_name)
        wd = strategy.window(single, windowing, [])
        windows = wd.X.shape[0]
        window_size = wd.X.shape[1]
        summary["windows"] = {
            "channels": channels,
            "windows": windows,
            "window_size": window_size,
        }
        summary["features"] = {
            "channels": channels,
            "windows": windows,
            "features_per_window": features_per_channel,
        }
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_shapes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/utils/observability/shapes.py tests/unit/test_shapes.py
git commit -m "feat-shape-summary-computation"
```

---

### Task 2: Wire into pipeline + integration test + docs

**Files:**
- Modify: `src/hand_gesture_recognition/runner/pipeline.py` (`_track_and_report` + new `_log_shape_summary`)
- Test: `tests/integration/test_shape_summary.py` (new)
- Modify: `docs/experiments/configuration.md` (one line)

**Interfaces:**
- Consumes: `compute_shape_summary` from `hand_gesture_recognition.utils.observability.shapes` (Task 1).
- Produces: `shape_summary.json` in `output/evaluation/<run_id>/`, logged to MLflow.

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/test_shape_summary.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_run_writes_shape_summary(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "backend=classical",
            "windowing=default",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    runner.run()
    out = tmp_path / "output" / "evaluation" / cfg.run_id
    assert (out / "shape_summary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_shape_summary.py -q`
Expected: FAIL — no `shape_summary.json` written.

- [ ] **Step 3: Wire into the pipeline**

In `src/hand_gesture_recognition/runner/pipeline.py`, add a call to a new `_log_shape_summary` method inside `_track_and_report`, right before the `self._visualize(...)` call (after the split/report block, before visualize):

```python
        self._log_shape_summary(processed_dataset)
        self._visualize(
            cropped,
            processed_dataset,
            split,
            predictions,
            class_names,
            viz_baseline,
            is_cv=self.cfg.cv.enabled,
        )
```

And add the method (place it just above `_persist_artifacts`):

```python
    def _log_shape_summary(self, processed_dataset) -> None:
        """Log a per-stage shape summary (console + shape_summary.json + MLflow)."""
        from hand_gesture_recognition.utils.observability.shapes import (
            compute_shape_summary,
        )

        feature_specs = self.cfg.feature_extraction.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        summary = compute_shape_summary(
            processed_dataset,
            self.cfg.windowing,
            feature_specs,
            task_name=self.cfg.task.name,
        )
        if not summary:
            return
        self.logger.info(f"shape_summary: {json.dumps(summary)}")
        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "shape_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.tracker.log_artifact(path)
```

`json` and `os` are already imported at the top of `pipeline.py`.

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `python -m pytest tests/integration/test_shape_summary.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs**

In `docs/experiments/configuration.md`, under the observability/reporting area, add one line noting the new artifact. Append near the `reporting.formats` row (around line 260, after the reporting table):

```markdown
Each run also writes a `shape_summary.json` (input → windows → features shapes
with symbolic dimension names) into `output/evaluation/<run_id>/` and logs it
to MLflow.
```

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/runner/pipeline.py tests/integration/test_shape_summary.py docs/experiments/configuration.md
git commit -m "feat-wire-shape-summary-into-pipeline"
```

---

## Self-Review

- **Spec coverage:** two-level + single-level shapes → Task 1 (compute) + Task 2 (integration); symbolic names → Task 1 keys; representative trial → Task 1; console + JSON + MLflow → Task 2; docs → Task 2 Step 5.
- **Placeholder scan:** no TBD; all code/test steps fully specified.
- **Type consistency:** `compute_shape_summary(dataset, windowing, feature_specs, task_name)` signature identical in Task 1 (definition) and Task 2 (call site). Symbolic keys identical in spec, Task 1, and Task 2.
