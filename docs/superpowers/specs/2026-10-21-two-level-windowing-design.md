# Two-Level Windowing (recognition mode) — Big → Small windows

**Date:** 2026-10-21
**Status:** Proposed

## Context

Recognition currently uses a **single-level** sliding window: `window_dataset`
slides one fixed-size window with a stride over each continuous trial, labels
each window (`0` = rest, `1..N` = gesture), and both backends consume that flat
`WindowedDataset`.

The user needs a **two-level** windowing scheme for recognition, where a coarse
**big window** is segmented first, and within each big window a fine **small
window** is segmented. This serves two purposes:

- **Filtering context** — deterministic signal processing (bandpass, notch,
  rectify) is applied **per big window** rather than per whole trial.
- **Coarse classification unit** — the model predicts a label for the **big
  window** (not the small window):
  - **Classical** extracts handcrafted features on the **big window** directly.
  - **Neural** extracts features on **each small window**, stacks them into a
    sequence of `n_small` feature vectors, and feeds that sequence to an
    **LSTM → FC → softmax → argmax** to produce the big-window label.

This is a **recognition-mode** feature only; the `classification` task is
unchanged (one sample per gesture). The default (`mode=single`) is byte-for-byte
identical to today's behavior.

## Goals

- Add `windowing.mode: single | two_level` (default `single`); both levels have
  their own fixed size, stride, and overlap (all configurable).
- Add a `window_dataset_two_level` builder emitting a `TwoLevelWindowedDataset`
  (big windows + small windows, aligned).
- Apply **signal processing per big window** in `two_level` mode.
- Classical: features on the **big window** → classifier → big-window label
  (small windows unused).
- Neural: features per **small window** → `n_small × n_features` sequence →
  `LSTMModule` (with `input_size = n_features`) → big-window label.
- Evaluate at the **big-window level**, reusing the existing `Prediction` /
  `evaluate` machinery unchanged.
- Keep everything config-driven, deterministic, backend-agnostic, and
  dataset-agnostic (rest identified structurally by `gesture_start < 0`).

## Non-goals

- No changes to single-level (`mode=single`) behavior.
- No changes to the `classification` task or `window_gestures`.
- No new neural model class — reuse `LSTMModule`.
- No online/streaming inference; the pipeline remains offline over recorded
  trials.
- No small-window-level evaluation output (big-window only).

## Design

### 1. Config — `WindowingConfig` extension

`src/hand_gesture_recognition/configuration/schema.py`:

```python
@dataclass
class WindowingConfig:
    mode: str = "single"  # "single" | "two_level"
    # --- small window (existing fields) ---
    window_size_samples: Optional[int] = None
    window_size_seconds: Optional[float] = None
    stride_samples: Optional[int] = None
    stride_seconds: Optional[float] = None
    overlap_percent: Optional[float] = None
    label_granularity: str = "per_window"
    padding: str = "none"
    # --- big window (used only when mode == "two_level") ---
    big_window_size_samples: Optional[int] = None
    big_window_size_seconds: Optional[float] = None
    big_stride_samples: Optional[int] = None
    big_stride_seconds: Optional[float] = None
    big_overlap_percent: Optional[float] = None
```

- `resolve_big(sampling_rate) -> Tuple[int, int]` mirrors `resolve()` exactly
  for the big-window `size`/`stride`/`overlap` trio (with the same fail-loudly
  consistency checks).
- Validation (in `validate_experiment_config` or `WindowingConfig.validate`):
  - `mode in ("single", "two_level")`.
  - When `mode == "two_level"`, big-window size must be resolvable **and**
    `big_size >= small_size` (otherwise no small window fits).

`configs/windowing/default.yaml` gains a commented `mode: single` (and the
`big_*` fields are left `null`/unset for single mode). A new
`configs/windowing/two_level.yaml` preset carries `mode: two_level` plus big and
small sizes/stride/overlap.

### 2. Data structures

New module `src/hand_gesture_recognition/data/windowing/two_level.py`:

```python
@dataclass
class TwoLevelWindowedDataset:
    big_windows: List[np.ndarray]    # signal-processed big windows, each (big_size, channels)
    small_windows: List[np.ndarray]  # per big window: (n_small_i, small_size, channels)
    y: np.ndarray                    # (n_big,) big-window labels
    metadata: WindowMetadata         # big-window level: window_start/end, ids, trial/subject

@dataclass
class TwoLevelWindowedPartitions:
    train: TwoLevelWindowedDataset
    val: TwoLevelWindowedDataset
    test: TwoLevelWindowedDataset
```

`metadata` reuses the existing `WindowMetadata`; `window_start`/`window_end`
hold **big-window** bounds, and `window_ids` are `f"{subj}_{tid}_big{i}"`
(unique, traceable). This reuses the existing `WindowMetadata` type so the
`Prediction`/evaluation/visualization machinery sees the same field names.

### 3. Window builder — `window_dataset_two_level`

```python
def window_dataset_two_level(
    dataset: Dataset,
    config: WindowingConfig,
    signal_steps: List[Any],
) -> TwoLevelWindowedDataset:
    ...
```

Per trial (mirroring `window_dataset`'s structure):

1. Resolve `(big_size, big_stride)` and `(small_size, small_stride)`.
2. Slide **big windows** over the trial (padding behavior from `config.padding`).
3. For each big window segment `trial[s:e]`:
   - Apply the **signal-processing steps** to that segment (per-big-window
     filtering; §4).
   - Slide **small windows** within the filtered big window → `(n_small_i,
     small_size, channels)`.
   - Compute the **big-window label** (§8): `rest` if the big window does not
     overlap `[gesture_start, gesture_end]`, else the gesture class. If
     `drop_rest_windows` (no-gesture excluded), skip rest big windows entirely
     — mirroring `window_dataset`'s `is_rest` / `drop_rest` logic at the big scale.
4. Append big window, its small windows, its label, and its metadata.

Return `TwoLevelWindowedDataset` with an empty guard mirroring `window_dataset`.

### 4. Per-big-window signal processing

In `two_level` mode, the deterministic signal-processing steps are applied to
**each big window's segment** (inside `window_dataset_two_level`), not to the
whole trial. The runner therefore **skips** its usual per-trial
`_signal_processing` when `mode == "two_level"` and instead passes
`cfg.signal_processing` steps into `window_dataset_two_level`. This is the
structural reason the builder takes `signal_steps` as an argument.

### 5. Classical path

Big windows **are** the samples. The runner converts `TwoLevelWindowedDataset`
into a flat `WindowedDataset` (`X = stack(big_windows)`, `y`, `metadata`) and
feeds it to the **existing** `_train_and_predict_classical` unchanged —
features are extracted on the big windows via `classical.resolve_features`, and
the sklearn model classifies them. No new classical code beyond the small
`big → WindowedDataset` flatten.

### 6. Neural path

New `src/hand_gesture_recognition/neural/datamodules/two_level.py`:
`TwoLevelEMGDataModule` wraps `TwoLevelWindowedPartitions` and yields, per big
window, a sample `x` of shape `(n_small, n_features)` — the handcrafted feature
vectors of its small windows (extracted with `classical.resolve_features`, e.g.
time-domain → `7 * n_channels` per small window), padded/truncated to a fixed
sequence length `max_small_per_big` so batches are rectangular.

The model is the **existing `LSTMModule`** instantiated with
`n_channels = n_features` (its `input_size`), so `forward(x)` with
`x: (batch, n_small, n_features)` works unchanged; output is `FC → softmax →
argmax` → big-window label. The existing `ClassifierModule` plumbing
(`predict_step` → `{logits, y_true, window_id, trial_id, subject_id}`) and
`LightningPredictionAdapter` are unchanged — the two-level datamodule just
emits `window_id` as the **big-window** id and `y_true` as the big-window label.

The runner's neural branch computes `n_features` by running the configured
feature extractor on a single small window (or analytically) and passes it to
the model as `n_channels`.

### 7. Runner wiring

`src/hand_gesture_recognition/runner/run.py`:

- `_window(split)` returns `WindowedPartitions` when `mode == "single"` (via
  `build_windowed_dataset`, as today) and `TwoLevelWindowedPartitions` when
  `mode == "two_level"` (via `window_dataset_two_level`).
- `_run_partition` skips per-trial `_signal_processing` for `two_level` (it is
  done per big window inside the builder) and branches the train/predict step:
  - classical → flatten big windows → existing classical path;
  - neural → `TwoLevelEMGDataModule` + `LSTMModule(input_size=n_features)`.
- Evaluation is unchanged: `evaluate` runs on big-window predictions exactly as
  on any other samples.

### 8. Evaluation & ground truth

Big-window ground truth uses the **same overlap rule** as `window_dataset`: a
big window is `rest` if it does not overlap `[gesture_start, gesture_end]` (or
the trial is itself a rest trial), else it is the trial's gesture class. When
`dataset.include_no_gesture=false` (or the `classification` remap is forced),
rest big-windows are dropped (`drop_rest_windows`), exactly as single-level.
Evaluation (aggregate + per-class + confusion matrix) is computed on big-window
predictions through the existing `evaluate`.

### 9. Visualization

The prediction-overlay plots (`plot_predictions_on_signal`,
`plot_signal_data_flow`) already re-window via `build_windowed_dataset`; for
`two_level` the runner passes the two-level builder so overlays align to
big-window bounds. (Plot A/B/C and FFT remain single-level diagnostics and are
unchanged; noted, not reworked.)

## Data flow — `two_level`

```
trial (raw, continuous)
  └─ window_dataset_two_level(dataset, config, signal_steps)
       ├─ slide BIG windows (big_size / big_stride / big_overlap)
       ├─ signal-process each big window (per-big-window filtering)
       ├─ slide SMALL windows within each big window (small_size / small_stride / overlap)
       └─ big-window label = rest if no overlap [g_start, g_end], else gesture class
            └─ TwoLevelWindowedDataset (big_windows, small_windows, y, metadata)
  └─ backend branch
       classical: features on big_windows → sklearn classifier → big-window label
       neural:    features per small window → (n_small, n_features) → LSTMModule → big-window label
  └─ evaluate at big-window level (unchanged)
```

## Interaction with `gesture_crop` / `baseline` / no-gesture

- `gesture_crop`: still runs for `two_level` (recognition-mode pre-step),
  exactly as for single-level; it crops the trial before windowing.
- `baseline`: unchanged; baseline removal still runs per partition before
  windowing (on trials), then the two-level builder signal-processes per big
  window.
- no-gesture class: `include_no_gesture=false` (or classification's forced
  remap) drops rest big-windows via the same `drop_rest_windows` signal.

## Edge cases

- **Big window smaller than a small window** → validation error (config).
- **Trial shorter than a big window** → `padding: none` drops it; `padding:
  zero` pads it (mirroring `window_dataset`).
- **A big window yields zero small windows** (e.g. filtered length < small_size):
  skip that big window.
- **Rest trial** (`gesture_start < 0`): every big window is labeled rest; all
  dropped when no-gesture is excluded.
- **Empty partition** → empty `TwoLevelWindowedDataset`; the existing 0-window
  warning fires.
- **Variable `n_small` across big windows** (padding/stride edge effects):
  neural batches pad/truncate to `max_small_per_big` for rectangular tensors.

## Testing

- **Unit `window_dataset_two_level`** (new `tests/unit/test_two_level_windowing.py`):
  big/small window counts and shapes; per-big-window signal processing applied
  (assert a known filter changes the big window's samples); big-window label
  `rest` vs gesture from overlap; `drop_rest_windows` skips rest big windows;
  metadata (`window_start`/`window_end`/`window_ids`/`trial_ids`/`subject_ids`)
  correct; empty input; padding/truncation edge cases.
- **Unit config**: `mode` validation; `two_level` requires resolvable big size
  and `big_size >= small_size`; `resolve_big` consistency checks.
- **Unit neural datamodule** (`TwoLevelEMGDataModule`): batch `x` shape
  `(batch, n_small, n_features)`, `y_true`/`window_id`/`trial_id`/`subject_id`
  correct, padding to `max_small_per_big`.
- **Integration**: synthetic `mode=two_level` end-to-end for **both** backends
  (classical features-on-big-window; neural feature-sequence LSTM) completes
  and reports big-window-level metrics.
- **Existing tests stay green** (default `mode=single` unchanged).

## Out of scope

- Small-window-level predictions / voting (rejected earlier — big-window only).
- Changes to the `classification` task, `window_gestures`, or single-level
  recognition.
- New neural architectures beyond reusing `LSTMModule`.
- Streaming/online deployment.
