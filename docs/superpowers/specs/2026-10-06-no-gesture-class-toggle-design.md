# No-Gesture Class Toggle (Option A — true 5-class reduction)

**Date:** 2026-10-06
**Status:** Proposed

## Context

The EPN612 pipeline is currently a **6-class** problem:
`noGesture=0, fist=1, waveIn=2, waveOut=3, open=4, pinch=5`.

`window_dataset` labels every window that does **not** overlap a gesture's
`[gesture_start, gesture_end]` segment as `0` (rest), and dedicated `noGesture`
trials (`gesture_start < 0`) are also label `0`. Because gestures are short
relative to the trial, most windows are "no gesture".

Two prior, partial knobs exist:

- `evaluation.exclude_classes` — drops *true* labels from the aggregate metric
  math only. The confusion matrix is still built with
  `labels=np.arange(len(class_names))` (always 6×6), the per-class table still
  includes predicted `noGesture`, and every plot still renders `noGesture`.
  It does **not** actually remove the class.
- `gesture_crop` — crops trials to the gesture region + context, but keeps
  `noGesture` as a class (padding windows and rest trials remain label 0).

**Decision (Option A):** add a single configuration parameter that turns the
pipeline into a **5-class** problem by *removing* the no-gesture class from the
dataset before training, windowing, and evaluation — not merely hiding it at
scoring time. When the class is off, the model trains, predicts, and is scored
**only on gesture-active windows**.

## Goals

- Add `dataset.include_no_gesture: bool = true`. When `false`:
  - Drop dedicated `noGesture` trials entirely.
  - Drop rest windows (windows fully outside `[gesture_start, gesture_end]`)
    from every partition, so the windowed dataset contains **only** gesture
    windows.
  - Remap remaining gesture labels to contiguous `0..4`
    (`fist→0, waveIn→1, waveOut→2, open→3, pinch→4`).
  - `metadata.class_names` becomes the 5 gesture names (no `noGesture`).
  - The confusion matrix, per-class metrics, and all plots are computed
    **only on the selected (5-class) data** — never on the raw 6-class set.
- Keep `true` the default so existing 6-class behavior is unchanged.
- Keep everything deterministic, config-driven, and backend-agnostic.

## Non-goals

- No online/two-stage activity detection (that was Option B — rejected).
- No change to the loader's parsing of `noGesture` trials (the loader stays
  honest and still loads them; removal happens as a post-load transform).
- No change to the split logic, leakage checks, or `evaluate()` internals.
- No general "arbitrary class selection" API — this is specific to the
  no-gesture/rest class (the only structurally-identifiable rest class).

## Design

### 1. Config

`src/hand_gesture_recognition/configuration/schema.py`:

```python
@dataclass
class DatasetConfig:
    ...
    include_no_gesture: bool = True
```

`configs/dataset/epn612.yaml`:

```yaml
name: epn612
path: resources/data/EPN612
sampling_rate: 200
include_no_gesture: true   # false => 5-class (rest removed)
loader:
  ...
```

No new config group — a plain field is enough, toggled via
`dataset.include_no_gesture=false` on the CLI or in the YAML.

### 2. Dataset metadata signal

`src/hand_gesture_recognition/data/datasets/base.py`:

```python
@dataclass
class DatasetMetadata:
    ...
    class_names: List[str]
    ...
    drop_rest_windows: bool = False   # windowing drops (not labels) rest windows
```

`drop_rest_windows` is an **internal** signal set by the transform (never set by
users directly). Windowing reads it so that the decision "drop vs label-0" lives
in exactly one place (`window_dataset`), which every caller — including the
visualization functions that re-window for alignment — already invokes with the
`Dataset`. This is what keeps predictions and plots aligned with zero signature
churn.

### 3. Transform — `remove_no_gesture`

New module `src/hand_gesture_recognition/data/class_selection.py`:

```python
def remove_no_gesture(dataset: Dataset) -> Dataset:
    ...
```

Logic (identifies rest **structurally**, not by name, so it is robust to a
custom `myoPredictionLabel` map):

1. `keep = dataset.gesture_start >= 0` (rest trials are exactly `gesture_start < 0`).
2. `old_kept = np.unique(dataset.labels[keep])` — the labels that remain.
3. Remap `old_kept → 0..N-1` in ascending order.
4. `new_class_names = [dataset.metadata.class_names[o] for o in old_kept]`.
5. Return `replace(dataset, trials=<filtered>, labels=<remapped>,
   subject_ids=<filtered>, trial_ids=<filtered>, gesture_start=<filtered>,
   gesture_end=<filtered>, metadata=replace(dataset.metadata,
   class_names=new_class_names, drop_rest_windows=True))`.
6. `subject_info` is filtered to subjects that still have trials (mirrors
   `splits.base._subset`).

For the standard EPN612 map this yields `fist, waveIn, waveOut, open, pinch`
(labels `0..4`). For the synthetic dataset there are no rest trials
(`gesture_start == 0` everywhere), so the transform is a no-op aside from
setting `drop_rest_windows=True` (which also has no effect because every window
overlaps the full-length "gesture").

### 4. Windowing — drop rest windows

`src/hand_gesture_recognition/data/windowing/base.py`:

```python
drop_rest = dataset.metadata.drop_rest_windows
...
for i in range(n_windows):
    s = i * stride; e = s + ws
    is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
    if drop_rest and is_rest:
        continue            # skip rest windows entirely
    # ... existing label assignment unchanged for drop_rest=False ...
```

- `drop_rest=False` (default): behavior identical to today.
- `drop_rest=True`: rest windows are **not emitted**; only gesture-overlapping
  windows (label `1..5` → already remapped to `0..4`) remain.

After the transform there are no `g_start < 0` trials, so the `is_rest` check
effectively reduces to "fully outside `[g_start, g_end]`".

### 5. Runner ordering

`src/hand_gesture_recognition/runner/run.py`:

```
load → remove_no_gesture (if !include_no_gesture) → analytics → gesture_crop
     → signal_processing → split → windowing → train → evaluate
```

```python
with stage(self.logger, "load_dataset"):
    raw_dataset = self._load_dataset()
if not cfg.dataset.include_no_gesture:
    from hand_gesture_recognition.data.class_selection import remove_no_gesture
    raw_dataset = remove_no_gesture(raw_dataset)
```

- Analytics, QC, `class_distribution.png`, and every plot run on the 5-class
  dataset, so `noGesture` never appears in any output.
- `n_classes = len(dataset.metadata.class_names)` = 5 (neural), and sklearn's
  `predict_proba` returns 5 columns, both automatically.
- `gesture_crop` still runs but its `noGesture` branch is never hit; its padding
  only shifts the window grid near the gesture boundary (see Interaction note).

### 6. Evaluation & confusion matrix — selected data only

No change to `evaluate()` is required, but the design explicitly guarantees the
invariant the user asked for: **`prediction.y_true` / `y_pred` passed to
`evaluate()` (and therefore the confusion matrix and every score) contain only
the selected 5-class, rest-dropped windows.** This holds because windowing
(§4) never emits rest windows, so `windowed.train/val/test.y` and the adapters'
outputs are already reduced.

- `confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))` becomes
  5×5 (no `noGesture` row or column).
- `per_class` is built from `unique(y_true ∪ y_pred) ⊆ {0..4}` → only the 5
  gestures appear.
- `render_report`, `mlflow_tracker.log_confusion_matrix`, and all plot
  functions receive the 5-entry `class_names`.

### 7. Config validation — guard the stale `exclude_classes` footgun

`validate_experiment_config` gains:

```python
if not cfg.dataset.include_no_gesture and 0 in cfg.evaluation.exclude_classes:
    raise ConfigValidationError(
        "evaluation.exclude_classes=[0] is ambiguous in 5-class mode "
        "(0 is now 'fist'); remove it or use the 6-class mode."
    )
```

(Fail loudly, consistent with the framework's schema philosophy.)

### 8. Minor visualization fix

`reporting/visualization.py` uses `if int(dataset.labels[i]) != 0` to "pick the
first non-noGesture trial". Under 5-class mode `0` is `fist`, so this heuristic
is now wrong (it skips `fist` as the example). Since the transform already
removed all rest trials, simplify to picking `dataset.trial_ids[0]`.

## Data flow (5-class mode)

```
EPN612Loader (6 classes, raw)
  └─ remove_no_gesture ──> 5 classes, no rest trials, drop_rest_windows=True
       ├─ analytics (5 classes)
       ├─ gesture_crop (no-op on rest branch)
       ├─ signal_processing
       ├─ split (subject-wise, unchanged)
       └─ window_dataset (drops rest windows) ──> train/val/test (gesture windows only, y ∈ 0..4)
            ├─ train (n_classes=5)
            ├─ predict (y_pred ∈ 0..4)
            └─ evaluate (5×5 CM, per-class = 5 gestures, selected data only)
```

## Interaction with `gesture_crop`

In 5-class mode `crop_to_gesture`'s rest-trial branch never fires. Its `±pad`
context windows fall outside `[g_start, g_end]` and are dropped by windowing,
so the padding contributes nothing but a grid-alignment shift near the gesture
boundary. **Recommendation:** run 5-class experiments with `gesture_crop: none`
so windowing anchors on the natural trial grid; document this in the config
comment. (No hard validation — both are valid, just slightly different window
boundaries.)

## Edge cases

- **No rest trials** (synthetic): transform is a no-op; `drop_rest_windows` set
  but inert.
- **A partition becomes empty** after dropping (e.g., a subject with only
  `noGesture` trials in a subject-wise split): `window_dataset` already returns
  an empty `WindowedDataset`; training on an empty partition would fail loudly
  downstream. Log a clear warning in `_window` when any partition has 0 windows.
- **Trial shorter than a window**: existing `padding: none` drop behavior
  unchanged (the trial contributes no windows).
- **Window straddling the gesture boundary**: kept if it overlaps the segment
  (same rule the labeler uses today); fully-outside windows are dropped.
- **Custom `myoPredictionLabel`**: rest identified by `gesture_start < 0`, so
  label remapping and `class_names` reduction are correct regardless of the
  map's integer coding.

## Testing

- **Unit `class_selection.py`**: rest trials dropped; labels remapped
  contiguous `0..4`; `class_names` = 5 gestures; `drop_rest_windows=True` set;
  `subject_info` filtered; identity/no-op when no rest trials; idempotent.
- **Unit `window_dataset`** (extend `test_windowing.py`): with
  `drop_rest_windows=True`, only gesture-overlapping windows are emitted and
  `y` contains no rest label; with `False`, behavior is byte-identical to today.
- **Unit `evaluate`**: on a 5-class prediction, `confusion_matrix.shape == (5,5)`
  and `set(per_class) == {fist, waveIn, waveOut, open, pinch}` (no noGesture).
- **Unit config**: `include_no_gesture=False` + `exclude_classes=[0]` raises
  `ConfigValidationError`; default `True` keeps current behavior.
- **Integration**: one 6-class run (unchanged) and one 5-class run
  (`dataset.include_no_gesture=false`, `gesture_crop=none`) both complete;
  the 5-class run's report/CM/plots contain no `noGesture` and the CM is 5×5.
- **Existing tests stay green** (synthetic and default 6-class paths unchanged).

## Out of scope

- Online activity detection / two-stage gating.
- General multi-class exclusion beyond the rest class.
- Per-class window-distribution analytics.
- Remapping that preserves original integer indices (we remap to contiguous
  `0..4` by design — required for `cross_entropy`/`n_classes`).
