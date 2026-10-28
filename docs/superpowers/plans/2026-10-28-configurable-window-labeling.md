# Configurable Window Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract window ground-truth labeling into a `ThresholdLabeling` strategy with a configurable threshold (default 0.5 majority), replacing the hardcoded any-overlap rule.

**Architecture:** A new `ThresholdLabeling` class (with `is_rest(window_start, window_end, g_start, g_end) -> bool`) lives in `windowing/labeling.py`; `WindowingConfig` gains `label_threshold: float = 0.5`; `window_emg` builds the labeling from the config and calls `is_rest` in place of the inline overlap check. The label assignment (`0 if is_rest else label`) and `drop_rest` logic are unchanged.

**Tech Stack:** Python 3.10, numpy, pytest.

## Global Constraints

- `is_rest` returns a **boolean** (not a class index) — the rest label is index 0 and may collide with a real gesture label when `noGesture` is dropped.
- Rule (strict `>`): `overlap = max(0, min(end, g_end) - max(start, g_start))`; rest iff `overlap / (end - start) <= threshold`; rest trial (`g_start < 0`) is always rest.
- Default `label_threshold = 0.5`; `0.0` reproduces the old any-overlap behavior.
- Small windows (`rest_bounds=None`) are unaffected (they inherit the big window's label).
- Do NOT commit `configs/config.yaml`, `configs/feature_extraction/default.yaml`, or `configs/neural/lstm.yaml`.

---

### Task 1: `ThresholdLabeling` + config field + unit tests

**Files:**
- Create: `src/hand_gesture_recognition/pipeline/feature_extraction/windowing/labeling.py`
- Modify: `src/hand_gesture_recognition/configuration/schema/windowing.py` (add `label_threshold` field)
- Test: `tests/unit/test_labeling.py` (new)

**Interfaces:**
- Produces: `ThresholdLabeling(threshold: float = 0.5)` with `is_rest(window_start, window_end, g_start, g_end) -> bool`.
- Produces: `WindowingConfig.label_threshold: float = 0.5`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_labeling.py`:

```python
from hand_gesture_recognition.pipeline.feature_extraction.windowing.labeling import (
    ThresholdLabeling,
)


def test_majority_threshold_strict():
    l = ThresholdLabeling(0.5)
    assert l.is_rest(0, 4, 0, 10) is False    # 100% inside -> gesture
    assert l.is_rest(3, 7, 4, 10) is False    # 75% inside -> gesture
    assert l.is_rest(2, 6, 4, 10) is True     # 50% inside == threshold -> rest (strict)
    assert l.is_rest(0, 4, 8, 10) is True     # 0% inside -> rest


def test_threshold_zero_reproduces_overlap():
    l = ThresholdLabeling(0.0)
    assert l.is_rest(2, 6, 4, 10) is False    # any overlap -> gesture
    assert l.is_rest(0, 4, 4, 10) is True     # ends exactly at g_start -> no overlap -> rest


def test_rest_trial_always_rest():
    assert ThresholdLabeling(0.5).is_rest(0, 4, -1, -1) is True
    assert ThresholdLabeling(0.0).is_rest(0, 4, -1, -1) is True
```

Add to `tests/unit/test_config_schema.py` (near the other `WindowingConfig` assertions) a defaults check:

```python
def test_windowing_label_threshold_default():
    from hand_gesture_recognition.configuration.schema import WindowingConfig
    assert WindowingConfig().label_threshold == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_labeling.py tests/unit/test_config_schema.py::test_windowing_label_threshold_default -q`
Expected: FAIL (module/attribute missing).

- [ ] **Step 3: Write the module + config field**

Create `src/hand_gesture_recognition/pipeline/feature_extraction/windowing/labeling.py`:

```python
from __future__ import annotations


class ThresholdLabeling:
    """Threshold-based window ground-truth labeling.

    A window is ``rest`` (labeled noGesture) unless the fraction of its samples
    that fall inside the gesture interval ``[g_start, g_end]`` strictly exceeds
    ``threshold``. With ``threshold=0.0`` this reproduces the old any-overlap
    behavior; the default ``0.5`` is majority overlap.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def is_rest(self, window_start: int, window_end: int, g_start: int, g_end: int) -> bool:
        """True if the window should be labeled noGesture."""
        if g_start < 0:  # rest trial has no gesture bounds
            return True
        overlap = max(0, min(window_end, g_end) - max(window_start, g_start))
        return overlap / (window_end - window_start) <= self.threshold
```

In `src/hand_gesture_recognition/configuration/schema/windowing.py`, add the field to `WindowingConfig` (right after `padding`, before the big-window block):

```python
    label_threshold: float = 0.5  # fraction of samples that must be in the gesture
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_labeling.py tests/unit/test_config_schema.py::test_windowing_label_threshold_default -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/feature_extraction/windowing/labeling.py src/hand_gesture_recognition/configuration/schema/windowing.py tests/unit/test_labeling.py tests/unit/test_config_schema.py
git commit -m "feat-threshold-window-labeling-strategy"
```

---

### Task 2: Wire into `window_emg` + update affected tests

**Files:**
- Modify: `src/hand_gesture_recognition/pipeline/feature_extraction/windowing/base.py`
- Test: `tests/unit/test_windowing.py`
- Test: `tests/datasets/test_epn612_real.py`
- Test: `tests/unit/test_windowing.py` (add configurable-threshold test)

**Interfaces:**
- Consumes: `ThresholdLabeling` from `hand_gesture_recognition.pipeline.feature_extraction.windowing.labeling` (Task 1).

- [ ] **Step 1: Wire `window_emg`**

In `src/hand_gesture_recognition/pipeline/feature_extraction/windowing/base.py`, add the import at the top:

```python
from hand_gesture_recognition.pipeline.feature_extraction.windowing.labeling import (
    ThresholdLabeling,
)
```

Then in `window_emg`, after the `rest_bounds` unpacking block (the lines that set `g_start = g_end = None` and `if rest_bounds is not None: g_start, g_end = rest_bounds`), add:

```python
    labeling = (
        ThresholdLabeling(config.label_threshold) if rest_bounds is not None else None
    )
```

and replace the loop body's label computation:

```python
        if rest_bounds is None:
            is_rest = False
        else:
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
```

with:

```python
        if rest_bounds is None:
            is_rest = False
        else:
            is_rest = labeling.is_rest(s, e, g_start, g_end)
```

(`xs.append`, `ys.append(0 if is_rest else int(label))`, `drop_rest` handling, `starts`, `ends`, `wins` stay exactly as-is.)

- [ ] **Step 2: Update affected tests in `tests/unit/test_windowing.py`**

`test_window_emg_rest_detection` (lines 28-35) — the window at index 1 `[2,6]` is now 50% inside → rest:

```python
def test_window_emg_rest_detection():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10)
    )
    # window at 0 is rest (no overlap); window at 2 is 50% inside -> rest (majority);
    # windows at 4 and 6 are fully inside -> gesture.
    assert y.tolist() == [0, 0, 5, 5]
    assert meta.window_start.tolist() == [0, 2, 4, 6]
```

`test_window_emg_drop_rest` (lines 47-54):

```python
def test_window_emg_drop_rest():
    emg = np.ones((10, 1), np.float32)
    X, y, meta = window_emg(
        emg, 5, _cfg(4, 2), "s1", "t1", "s1_t1", rest_bounds=(4, 10), drop_rest=True
    )
    assert y.tolist() == [5, 5]
    assert meta.window_start.tolist() == [4, 6]
    assert X.shape == (2, 4, 1)
```

`test_window_dataset_drops_rest_windows_when_no_gesture_dropped` (lines 114-162) — both parts:

```python
def test_window_dataset_drops_rest_windows_when_no_gesture_dropped():
    emg_len = 140
    start, end = 20, 80
    trial = Trial(
        "t1", "s1", "fist", start, end,
        np.arange(emg_len, dtype=np.float32)[:, None],
    )

    dropped_meta = DatasetMetadata(
        name="x", version="0", sampling_rate=200.0, channels=["c1"],
        class_names=["fist"], content_hash="h", no_gesture_name="noGesture",
    )
    dropped = Dataset.from_trials([trial], dropped_meta)
    wd = window_dataset(dropped, _cfg(40, 20))
    # windows [0,40] and [60,100] are only 50% inside [20,80] -> rest -> dropped
    assert wd.y.tolist() == [0, 0]
    assert wd.metadata.window_start.tolist() == [20, 40]
    assert (wd.metadata.window_start >= end).sum() == 0
    assert (wd.metadata.window_end <= start).sum() == 0

    kept_meta = DatasetMetadata(
        name="x", version="0", sampling_rate=200.0, channels=["c1"],
        class_names=["noGesture", "fist"], content_hash="h",
    )
    kept = Dataset.from_trials([trial], kept_meta)
    wd_kept = window_dataset(kept, _cfg(40, 20))
    assert wd_kept.y.tolist() == [0, 1, 1, 0, 0, 0]
    assert wd_kept.metadata.window_start.tolist() == [0, 20, 40, 60, 80, 100]
```

Add a new test (at the end of `tests/unit/test_windowing.py`) verifying `label_threshold` flows through `window_dataset`:

```python
def test_window_dataset_respects_label_threshold():
    trial = Trial("t1", "s1", "fist", 4, 10, np.ones((10, 1), np.float32))
    meta = DatasetMetadata(
        name="x", version="0", sampling_rate=200.0, channels=["c1"],
        class_names=["noGesture", "fist"], content_hash="h",
    )
    ds = Dataset.from_trials([trial], meta)
    majority = window_dataset(
        ds, WindowingConfig(window_size_samples=4, stride_samples=2, label_threshold=0.5)
    )
    overlap = window_dataset(
        ds, WindowingConfig(window_size_samples=4, stride_samples=2, label_threshold=0.0)
    )
    assert majority.y.tolist() == [0, 0, 1, 1]  # window [2,6] is 50% -> rest
    assert overlap.y.tolist() == [0, 1, 1, 1]   # window [2,6] overlaps -> gesture
```

- [ ] **Step 3: Update `tests/datasets/test_epn612_real.py`**

In `test_real_data_per_window_labels_match_bounds` (lines 76-86), replace the `expected` computation with the threshold rule:

```python
        if g_start < 0:  # rest trial
            expected = 0
        else:
            overlap = max(0, min(e, g_end) - max(s, g_start))
            expected = label if overlap / (e - s) > 0.5 else 0
        assert wd.y[i] == expected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_labeling.py tests/unit/test_windowing.py tests/unit/test_windowing_alignment.py tests/unit/test_two_level_windowing.py tests/unit/test_gesture_windowing.py -q`
Expected: PASS.

Also run the real-data test if the dataset is present:
`python -m pytest tests/datasets/test_epn612_real.py::test_real_data_per_window_labels_match_bounds -q`
Expected: PASS (if `resources/data/EPN612` is available; otherwise note it skipped/absent).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/feature_extraction/windowing/base.py tests/unit/test_windowing.py tests/datasets/test_epn612_real.py
git commit -m "feat-wire-threshold-labeling-into-windowing"
```

---

### Task 3: Documentation

**Files:**
- Modify: `docs/experiments/configuration.md` (windowing section)

- [ ] **Step 1: Document `label_threshold`**

In `docs/experiments/configuration.md`, in the `windowing` section (near the table describing `mode`, `window_size_samples`, etc.), add a row/note:

```markdown
| `label_threshold` | `float` | Fraction of a window's samples that must fall inside the gesture interval for the window to keep the gesture label (majority = `0.5`, default; `0.0` = any overlap). |
```

Place it with the other `windowing` fields so it reads naturally.

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/configuration.md
git commit -m "docs-window-label-threshold"
```

---

## Self-Review

- **Spec coverage:** ThresholdLabeling → Task 1; config field → Task 1; wiring → Task 2; behavior-change test updates → Task 2; docs → Task 3.
- **Placeholder scan:** no TBD; all code/test content specified.
- **Type consistency:** `ThresholdLabeling.is_rest(window_start, window_end, g_start, g_end)` signature identical in Task 1 (def) and Task 2 (call). `label_threshold` field name consistent across spec, Task 1 (schema), Task 2 (usage), Task 3 (docs).
