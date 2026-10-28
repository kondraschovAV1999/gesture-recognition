# Configurable window ground-truth labeling

Date: 2026-10-28
Status: approved (pending implementation)

## Context

Window ground-truth labeling is hardcoded in `window_emg`
(`pipeline/feature_extraction/windowing/base.py`): a window gets the trial's
gesture label if it overlaps the gesture interval `[start, end]` at all,
otherwise `noGesture`. This "any-overlap" rule labels boundary windows
optimistically and is not tunable.

## Goals

1. Extract window labeling into a separate strategy (`ThresholdLabeling`).
2. Make the threshold configurable (default 50% majority).
3. Default becomes majority overlap; `threshold=0` reproduces the old
   any-overlap behavior.

## Non-goals

- Only one strategy for now (`ThresholdLabeling`); no strategy registry.
- No change to classification windowing (`window_gestures`), which takes the
  whole `[start, end]` segment.

## Design

### `ThresholdLabeling`

New module `pipeline/feature_extraction/windowing/labeling.py`:

```python
class ThresholdLabeling:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def is_rest(self, window_start, window_end, g_start, g_end) -> bool:
        """True if the window should be labeled noGesture."""
        if g_start < 0:  # rest trial (no gesture bounds)
            return True
        overlap = max(0, min(window_end, g_end) - max(window_start, g_start))
        return overlap / (window_end - window_start) <= self.threshold
```

Returns a boolean (not a class index) because the "rest" label is always
index 0 and may collide with a real gesture label when `noGesture` is dropped
(e.g. `class_names=["fist"]`); the boolean keeps rest vs. gesture distinct
regardless of index remapping.

### Config

`WindowingConfig` gains one field:

```python
label_threshold: float = 0.5
```

Semantics (strict `>`): `0.0` = any overlap (old behavior), `0.5` = majority
(default); higher thresholds require a larger fraction of the window to fall
inside the gesture.

### Wiring

`window_emg(...)` builds `ThresholdLabeling(config.label_threshold)` when
`rest_bounds` is provided and replaces the inline `is_rest` expression with a
call to `labeling.is_rest(s, e, g_start, g_end)`. The label assignment
(`0 if is_rest else int(label)`) and `drop_rest` logic are unchanged. The
change flows through `window_dataset` (recognition) and `TwoLevelWindowing`
(big windows; small windows already use `rest_bounds=None` and inherit the big
window's label).

## Behavior change

The default label for boundary windows changes. For example, a window that is
exactly 50% inside the gesture interval becomes `noGesture` under the new
default (strict `>`), whereas it was `gesture` before. Tests that assert the
old any-overlap labels are updated accordingly.

## Testing

- Unit tests for `ThresholdLabeling`: majority, strict boundary (`== threshold`
  → rest), `threshold=0` reproduces any-overlap, `threshold=1` full-inside,
  rest trial.
- Update `test_windowing.py` (`test_window_emg_rest_detection`,
  `test_window_emg_drop_rest`,
  `test_window_dataset_drops_rest_windows_when_no_gesture_dropped`) and
  `tests/datasets/test_epn612_real.py::test_real_data_per_window_labels_match_bounds`
  to the new majority rule.
- Add a test that a configurable `label_threshold` flows through
  `window_dataset`.

## Documentation

- Note `windowing.label_threshold` in `docs/experiments/configuration.md`.
