# Data Configuration: Gesture-Context Cropping and Per-Cohort User Subsetting

**Date:** 2026-09-18
**Status:** Approved (pending implementation plan)

## Context

Two data-configuration needs for scaling EPN612 experiments:

1. **Class imbalance.** In `window_dataset`, a window is labeled `noGesture`
   unless it overlaps the ground-truth gesture segment `[gesture_start,
   gesture_end]`. Because the gesture is a short part of each trial, most
   windows become `noGesture` — heavy imbalance. We want to keep only the
   region around each gesture (with a configurable context margin) so the
   `noGesture` class doesn't dominate.

2. **Number of users.** EPN612 has ~306 users per cohort (`trainingJSON` and
   `testingJSON`). We want to run experiments on a configurable number of
   randomly selected users per cohort (e.g. 50 train + 50 test) instead of all.

## Goals

- Crop each gesture trial to `[gesture_start − pad, gesture_end + pad]`, keeping
  the ±`pad` context labeled `noGesture` and the gesture core labeled as the
  gesture class.
- Trim dedicated `noGesture` trials to a length comparable to a cropped gesture
  trial.
- Select a configurable, seeded-random number of users per cohort before loading
  (so only the chosen users are parsed).
- Keep the change deterministic and configurable via a config group (schema
  default `enabled: false`; shipped `config.yaml` opts into
  `gesture_crop: default`), with `50/50` users as the shipped default.

## Non-goals

- No change to the split logic, leakage checks, windowing, or evaluation.
- No relabeling of the padding into the gesture class (padding stays
  `noGesture`).
- No window-level class-distribution analytics (cropping affects window labels,
  not trial-level labels; analytics stays trial-level).
- No subject sampling for the synthetic dataset (it has no cohort folders).

## Design

### Feature 1 — Gesture-context cropping

**Config** (`src/hand_gesture_recognition/configuration/schema.py`):

```python
@dataclass
class GestureCropConfig:
    enabled: bool = False
    pad_samples: int = 100                 # symmetric context before AND after
    no_gesture_max_length_samples: Optional[int] = None  # None = auto-derive
```

- `ExperimentConfig.gesture_crop: GestureCropConfig = field(default_factory=GestureCropConfig)`.
- `_coerce` adds `gesture_crop=GestureCropConfig(**d.get("gesture_crop", {}))`.
- New Hydra config group `gesture_crop`:
  - `configs/gesture_crop/none.yaml` → `enabled: false`.
  - `configs/gesture_crop/default.yaml` → `enabled: true`, `pad_samples: 100`.
- `configs/config.yaml` `defaults` gains `- gesture_crop: default`.

**Transform** (`src/hand_gesture_recognition/data/gesture_crop.py`):

```python
def crop_to_gesture(dataset: Dataset, config: GestureCropConfig) -> Dataset
```

Per trial:
- **Gesture trial** (`gesture_start >= 0`): crop to
  `[max(0, g_start − pad), min(length, g_end + pad)]`. Shift bounds to the new
  coordinates: `g_start' = g_start − crop_start`, `g_end' = g_end − crop_start`.
- **noGesture trial** (`gesture_start < 0`): if `no_gesture_max_length_samples`
  is set, truncate to that many samples; if `None`, truncate to
  `2×pad + median(gesture length)` where the median is over gesture trials. If
  there are no gesture trials, leave noGesture trials unchanged. Short trials
  are left as-is (no zero-padding).
- Returns `replace(dataset, trials=..., gesture_start=..., gesture_end=...)`;
  `labels`, `subject_ids`, `trial_ids`, `metadata`, `subject_info` are unchanged.

**Runner** (`src/hand_gesture_recognition/runner/run.py`): a deterministic stage
between load and signal processing:

```
load → gesture_crop → signal_processing → split → windowing → …
```

```python
with stage(self.logger, "gesture_crop"):
    cropped = self._gesture_crop(raw_dataset)
with stage(self.logger, "signal_processing"):
    dataset = self._signal_processing(cropped)
```

where `_gesture_crop` returns the dataset unchanged when
`cfg.gesture_crop.enabled` is false. `_visualize` receives the **cropped** raw
(`cropped`) so `raw_signal_*.png` and the before/after comparison show the data
the model actually sees, with aligned bounds. Analytics runs on the uncropped
`raw_dataset` (trial-level, unaffected).

### Feature 2 — Per-cohort user subsetting

**Loader** (`src/hand_gesture_recognition/data/loaders/epn612.py`):

```python
def __init__(self, subjects=None, max_train_users=None, max_test_users=None, user_seed=42):
```

- `max_train_users` / `max_test_users`: max number of random users to load from
  the training / testing cohort. `None` = all users in that cohort.
- `user_seed`: seed for the random selection.
- In `load()`, after `_discover` and before the existing parse loop: derive the
  unique physical user dirs per cohort, sample
  `min(max_train_users, available)` train users and
  `min(max_test_users, available)` test users with
  `np.random.default_rng(user_seed)`, build the allowed qualified-id set
  (`train:userX`, `test:userX`), intersect with the explicit `subjects` filter
  if one is set, and filter the file list to allowed ids. Only the chosen users
  are parsed.
- Sampling is deterministic for a fixed `user_seed`.

**Config** (`configs/dataset/epn612.yaml`):

```yaml
loader:
  _target_: hand_gesture_recognition.data.loaders.epn612.EPN612Loader
  subjects: null
  max_train_users: 50
  max_test_users: 50
  user_seed: 42
```

## Edge cases

- Gesture bounds at/near trial edges: clamped to `[0, length]`.
- `max_users` > available users in a cohort: all available users are used (no
  error).
- No gesture trials in a dataset: noGesture auto-trim is skipped.
- Cropping is a no-op for the synthetic dataset (its gesture spans the whole
  trial, so `[0−pad, length+pad]` clamps to the full trial).

## Testing

- Unit: `crop_to_gesture` — bounds shift after crop, clamping at edges, noGesture
  trimming (explicit and auto), no-op when disabled, short trials unchanged.
- Unit: loader user sampling — `max_train_users`/`max_test_users` yield the
  expected subject-id counts; `user_seed` reproducibility; `max_users` larger
  than available clamps.
- Integration: one run with `gesture_crop=default` + `max_*_users` set, asserting
  the pipeline completes and artifacts are produced.
- Existing tests stay green (synthetic unaffected; loader sampling only changes
  real-data runs).

## Out of scope

- Asymmetric before/after padding (single symmetric `pad_samples` for now).
- Window-level class-distribution analytics.
- Non-random (e.g. first-N or stratified) user selection.
