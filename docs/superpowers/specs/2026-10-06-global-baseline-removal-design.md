# Global Baseline Removal (Leakage-Free Preprocessing)

**Date:** 2026-10-06
**Status:** Proposed

## Context

The framework's preprocessing must obey the data-leakage golden rule: *any
parameter estimated from data must be fit on training data only, frozen, then
applied to validation and test.*

Current state (correct but incomplete):

- `signal_processing` (`ButterworthFilter` bandpass/notch, `RectifyTransformer`)
  is **deterministic** — `fit()` returns `self`, `transform()` uses fixed
  config — so running it pre-split is safe.
- `classical.preprocessing` (`PerChannelStandardScaler`) is **learned** and
  already fit on `X_train` only (post-windowing, post-split). Correct.
- There is **no baseline removal** today: no subtraction of the resting
  (no-gesture) EMG level.

This feature adds baseline removal: subtract the resting per-channel level
($\mu_c = \operatorname{mean}(x_c^{NG})$, estimated from the training no-gesture
data only) as the **first** preprocessing step, before the existing filter
chain. Because the estimate is data-dependent, it must happen **after the
split**.

The no-gesture class toggle (`include_no_gesture`) interacts with this:
`noGesture` trials are *calibration* data for the baseline (always usable for
estimation) and *prediction* data only when `include_no_gesture=true`. Baseline
estimation must therefore see `noGesture` trials even when
`include_no_gesture=false`, and class selection (dropping `noGesture`) must
happen **after** baseline estimation.

## Goals

- Add a `baseline` config group (`enabled: bool = false`).
- `BaselineRemoval` transformer: fit on the **train partition's `noGesture`
  trials** (`gesture_start < 0`), computing a **global per-channel mean**;
  frozen; `transform` subtracts it from every trial of train/val/test.
- Baseline runs **first** — on the raw signal, before `signal_processing`.
- `noGesture` trials remain available for baseline estimation **regardless of
  `include_no_gesture`**; they are dropped from the model only afterwards, when
  `include_no_gesture=false`.
- Keep every learned/estimated quantity fit-on-train-only.

## Non-goals

- No per-subject baseline (deferred — subject-wise/predefined splits make a
  train-only subject-specific baseline impossible; needs a separate calibration
  protocol).
- No median/robust/other statistics — v1 uses the per-channel **mean**.
- No pre-gesture-segment source — v1 estimates from dedicated `noGesture`
  trials only.
- No config knobs beyond `enabled` (YAGNI).

## Design

### 1. Config

`src/hand_gesture_recognition/configuration/schema.py`:

```python
@dataclass
class BaselineConfig:
    enabled: bool = False
```

- `ExperimentConfig.baseline: BaselineConfig = field(default_factory=BaselineConfig)`.
- `_coerce` adds `baseline=BaselineConfig(**(d.get("baseline") or {}))`.
- New Hydra config group `baseline`:
  - `configs/baseline/none.yaml` → `enabled: false`.
  - `configs/baseline/default.yaml` → `enabled: true`.
- `configs/config.yaml` `defaults` gains `- baseline: none`.

No validation couples `baseline.enabled` to `include_no_gesture` — they are
orthogonal (see §4).

### 2. Transformer

New module `src/hand_gesture_recognition/preprocessing/baseline.py`:

```python
class BaselineRemoval(BaseEstimator):
    def fit(self, train: Dataset) -> "BaselineRemoval":
        # noGesture trials: gesture_start < 0
        rest = [t for t, gs in zip(train.trials, train.gesture_start) if gs < 0]
        # global per-channel mean over all rest samples
        self.baseline_ = np.mean(np.concatenate(rest, axis=0), axis=0)  # (channels,)

    def transform(self, dataset: Dataset) -> Dataset:
        trials = [t - self.baseline_ for t in dataset.trials]
        return replace(dataset, trials=trials)
```

- `fit` sees **train only**; `transform` applies the frozen `baseline_` to any
  partition.
- `baseline_` is a 1-D array of length `n_channels`, broadcast over the time
  axis.

### 3. Runner restructure

New pipeline order:

```
load → analytics (raw) → gesture_crop → split
     → baseline removal (fit train noGesture → subtract train/val/test)
     → class selection (if include_no_gesture=false)
     → signal_processing (bandpass → notch → rectify)
     → windowing → train → evaluate
```

Concrete changes in `src/hand_gesture_recognition/runner/run.py`:

1. **Split moves earlier** — `split_dataset(cropped, cfg.split)` runs on the
   cropped, pre-signal-processing dataset. Safe: split keys on
   `subject_ids`/`trial_ids`, independent of signal values.
2. **`remove_no_gesture` moves post-split** (currently it runs right after
   load). It now runs **after** baseline estimation and **after** split, applied
   per-partition.
3. **`signal_processing` moves post-split** — after baseline + class selection,
   applied per-partition (deterministic, so output is identical to today).
4. **New `baseline` stage** — between split and class selection.

### 4. `remove_no_gesture` refactor (global remapping)

The existing `remove_no_gesture` remaps labels via `np.unique(labels[keep])`
*per dataset*. Once applied per-partition, a partition missing a gesture class
would remap inconsistently across partitions. So split it into a global
(computed once) and a per-partition (apply) step in
`src/hand_gesture_recognition/data/class_selection.py`:

```python
def build_no_gesture_remap(dataset: Dataset) -> tuple[dict[int, int], list[str]]:
    # global: computed once from the full (pre-split) dataset
    keep = dataset.gesture_start >= 0
    kept_labels = np.unique(dataset.labels[keep])
    remap = {int(old): new for new, old in enumerate(kept_labels)}
    new_class_names = [dataset.metadata.class_names[int(o)] for o in kept_labels]
    return remap, new_class_names


def drop_no_gesture(dataset: Dataset, remap: dict, new_class_names: list[str]) -> Dataset:
    # per-partition: drop rest trials, remap labels via the GLOBAL map
    keep = dataset.gesture_start >= 0
    new_labels = np.asarray([remap[int(l)] for l in dataset.labels[keep]], dtype=int)
    ...  # filter trials/subject_ids/trial_ids/gesture_start/gesture_end/subject_info
    return replace(dataset, ..., metadata=replace(
        dataset.metadata, class_names=new_class_names, drop_rest_windows=True))
```

The runner computes `remap`/`new_class_names` once (pre-split) when
`include_no_gesture=false`, then applies `drop_no_gesture` to each partition
after baseline.

`n_classes` and evaluation `class_names` are taken from the **post-selection**
`class_names` (5 entries when `include_no_gesture=false`), not the raw metadata.

### 5. Orthogonality with `include_no_gesture`

| `include_no_gesture` | `baseline.enabled` | Behavior |
|---|---|---|
| `true`  | `false` | 6-class, no baseline (unchanged) |
| `true`  | `true`  | 6-class + baseline from train noGesture |
| `false` | `false` | 5-class, no baseline |
| `false` | `true`  | 5-class; baseline estimated from noGesture, then noGesture dropped |

No `ConfigValidationError` between the two flags.

## Data flow

```
raw Dataset (6 classes, noGesture present)
 ├─ analytics (raw QC)
 ├─ gesture_crop
 ├─ split ────────────────────────────────► train / val / test (noGesture present)
 ├─ baseline.fit(train) → baseline_ (channels,)   [train noGesture only]
 ├─ baseline.transform(train/val/test)            [subtract baseline_]
 ├─ drop_no_gesture(train/val/test)               [if include_no_gesture=false]
 ├─ signal_processing(train/val/test)             [bandpass → notch → rectify]
 └─ windowing → train → evaluate
```

## Edge cases

- **No `noGesture` trials in train** → `fit` raises a clear error (not a silent
  no-op), since the baseline is undefined without rest data.
- **`include_no_gesture=false` + `baseline.enabled=true`** → works: baseline
  from noGesture, then noGesture dropped (the core correction).
- **Global remap consistency** → guaranteed by computing the remap once
  pre-split; partitions can never remap differently.
- **Empty partition after `drop_no_gesture`** → the existing `_window`
  empty-partition warning fires (unchanged).
- **Visualization** → `signal_processing` moving post-split means the
  "processed dataset" for plots is reconstructed by applying
  `signal_processing` to a display copy of the cropped dataset (deterministic,
  identical), or taken from the processed partitions. Handled in the plan.

## Testing (deferred per user instruction; listed for when tests resume)

- Unit `BaselineRemoval`: `fit` uses only train noGesture trials (assert
  val/test noGesture are ignored); `baseline_` equals the per-channel mean of
  train rest samples; `transform` subtracts and preserves non-signal fields.
- Unit `drop_no_gesture`: global remap applied identically across partitions
  (a partition missing a class still maps correctly).
- Leakage regression: baseline value unchanged whether val/test data is
  perturbed.
- Integration: `baseline=default` + `include_no_gesture=false` completes; CM is
  5×5; `baseline=default` + `include_no_gesture=true` completes (6×6 CM).
- Existing suite stays green (default `baseline: none`, `include_no_gesture:
  true`).

## Out of scope

- Per-subject baseline / subject-specific calibration.
- Robust statistics (median), per-gesture baseline, or pre-gesture-segment
  source.
- Any change to `PerChannelStandardScaler` or the feature extractors.
