# Configuring the Project

Everything about a run — dataset, split, cross-validation, windowing, signal
processing, model, metrics, tracking, and visualization — is configured through
**Hydra** hierarchical configs under `configs/`. You rarely edit code; you edit
config or pass overrides on the command line.

## How configuration works

- `configs/config.yaml` holds the `defaults:` list — which preset each config
  *group* uses by default.
- Each group is a directory `configs/<group>/` full of preset `.yaml` files.
- **Switch a group's preset** from the CLI with `<group>=<preset>` (e.g.
  `neural=lstm`, `split=strategy_choice`).
- **Override any single field** with `<group>.<field>=<value>` (e.g.
  `windowing.window_size_seconds=0.2`), nested with dots (e.g.
  `neural.trainer.max_epochs=10`).
- Lists use Hydra syntax: `visualization.selection.gestures=[fist,open]`.

The current defaults (plus the inlined top-level fields):

```yaml
defaults:
  - dataset: epn612
  - windowing: default
  - signal_processing: emg
  - gesture_crop: default
  - baseline: none
  - classical: random_forest
  - feature_extraction: default
  - neural: lstm
  - evaluation: basic
  - analytics: basic
  - cv: none
  - visualization: default
  - _self_

# experiment (inlined from configs/experiment/default.yaml)
backend: neural
seed: 42
deterministic: true
tracking:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: hand_gesture_recognition
reporting:
  formats: [json, html]

# task (inlined from configs/task/*.yaml)
task:
  name: recognition

# split (inlined from configs/split/subject_wise.yaml)
split:
  strategy: group_wise
  split_on: subject_ids
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  seed: 42
```

The `experiment`, `task`, and `split` groups are **inlined directly in
`configs/config.yaml`** — there are no separate preset files for them. Set
their fields on the CLI with `task.name=...`, `split.strategy=...`, and
`split.split_on=...` (e.g. `task.name=classification`, `split.strategy=group_wise`).

## Config groups reference

### `dataset`

| Field | Values | Default |
|---|---|---|
| `name` | `epn612` \| `synthetic` | `epn612` |
| `path` | path to the raw data | `resources/data/EPN612` |
| `sampling_rate` | Hz | `200` |
| `include_no_gesture` | `true` (6-class) \| `false` (5-class, rest dropped) | `true` |
| `loader.subjects` | `null` or `[train:user1,...]` | `null` |
| `loader.max_train_users` / `max_test_users` | int or `null` | `25` / `25` |
| `loader.user_seed` | int | `42` |

`dataset.name=synthetic` loads a fast in-memory fixture (no files on disk).

### `task`

| Field | Values | Default |
|---|---|---|
| `name` | `classification` \| `recognition` | `recognition` |

Selects the problem formulation:

- **`recognition`** — the continuous-stream, sliding-window pipeline
  (`window_dataset`): windows slide over each trial, labeled `0` (rest) or the
  gesture class. This is the default and matches historical behavior.
- **`classification`** — the dataset is treated as pre-cut clips with one
  gesture per trial: `window_gestures` slices each trial's gesture segment
  (`emg[start:end]`) into exactly one sample per trial, pad/truncates it to
  `window_size`, and drops rest trials. In this mode the no-gesture class is
  always excluded (so `dataset.include_no_gesture` is ignored), and
  `gesture_crop` is not applied.

```bash
# classification: one sample per gesture
python -m hand_gesture_recognition --mode compose task.name=classification dataset.name=synthetic
```

### `split`

| Field | Values | Default |
|---|---|---|
| `strategy` | `random` \| `stratified` \| `trial_wise` \| `group_wise` \| `session_wise` \| `predefined` | `group_wise` |
| `train_ratio` / `val_ratio` / `test_ratio` | sum to `1.0` | `0.6` / `0.2` / `0.2` |
| `seed` | int | `42` |
| `split_on` | `subject_ids` \| `trial_ids` | `subject_ids` |

Trial-level strategies (`random`, `stratified`, `trial_wise`) let the same
subject appear in train and test; subject-wise strategies
(`group_wise`, `session_wise`, `predefined`) keep each subject in exactly one
partition. See [Splitting strategies](splitting.md).

### `cv` (cross-validation)

| Field | Values | Default |
|---|---|---|
| `enabled` | bool | `false` |
| `method` | `kfold` \| `loso` | `kfold` |
| `n_splits` | int (kfold only) | `5` |
| `group_folds` | `true` (subject-wise folds) \| `false` (trial-level folds) | `true` |
| `include_val` | bool (carve a val subset out of each train fold) | `true` |
| `val_ratio` | fraction of the training fold used as val | `0.2` |
| `seed` | int | `42` |
| `report_per_fold` | bool (log per-fold metrics + confusion matrices) | `true` |

When `enabled=true`, the single train/val/test split is replaced by N folds
(`cv=kfold` or `cv=loso`), and the reported metrics are the **mean ± std**
across folds. Presets: `cv=none` (off), `cv=kfold`, `cv=loso`.

```bash
# 5-fold subject-wise cross-validation
python -m hand_gesture_recognition --mode compose dataset.name=synthetic baseline=none cv=kfold

# Leave-one-subject-out
python -m hand_gesture_recognition --mode compose cv=loso

# Trial-level (non-subject-wise) folds
python -m hand_gesture_recognition --mode compose cv=kfold cv.group_folds=false
```

### `windowing`

| Field | Values | Default |
|---|---|---|
| `window_size_samples` / `window_size_seconds` | one or the other | `24` / `null` |
| `stride_samples` / `stride_seconds` / `overlap_percent` | one of the three | `16` / `null` / `null` |
| `label_granularity` | `per_window` \| `sequence` | `per_window` |
| `padding` | `none` \| `zero` | `none` |
| `mode` | `single` \| `two_level` | `single` |
| `big_window_size_samples` / `big_window_size_seconds` | one or the other (two_level only) | `null` |
| `big_stride_samples` / `big_stride_seconds` / `big_overlap_percent` | one of the three (two_level only) | `null` |
| `label_threshold` | fraction of a window's samples that must fall inside the gesture interval to keep the gesture label (`0.0` = any overlap) | `0.5` (majority) |

`windowing=two_level` enables two-level windowing: the trial is segmented into
**big windows** (filtered independently), then **small windows** are slid inside
each big window, and the model classifies at the **big-window** level (classical
extracts features on the big window; neural extracts features per small window
and feeds the sequence to an LSTM). Preset: `windowing=two_level`.

### `signal_processing` (deterministic, before split)

A **list** of `_target_` transforms, applied in order. `signal_processing=emg`
defaults to bandpass (10–95 Hz) → notch (60 Hz) → rectify; `signal_processing=none`
applies nothing.

| Component | Params |
|---|---|
| `ButterworthFilter` | `type` = `bandpass`/`bandstop`/`lowpass`/`highpass`/`notch`, `lowcut`/`highcut`/`frequency`, `order`, `fs` |
| `RectifyTransformer` | — (full-wave rectification) |
| `EnvelopeExtractor` | `window_size` |

### `gesture_crop`

| Field | Values | Default |
|---|---|---|
| `enabled` | bool | `true` |
| `pad_samples` | context kept before/after the gesture | `25` |

`gesture_crop=none` disables cropping. When using `dataset.include_no_gesture=false`
(5-class), pair it with `gesture_crop=none`.

### `baseline`

| Field | Values | Default |
|---|---|---|
| `enabled` | bool | `false` (`baseline: none`) |
| `mode` | `global` \| `per_user` | `global` |

When enabled, a global per-channel rest baseline is estimated from the train
partition's `noGesture` trials and subtracted before signal processing.
`baseline=default` enables it; `baseline=none` disables it. (Requires rest
trials, so it only works on real data with a `noGesture` class — not synthetic.)

`mode=per_user` estimates one baseline per subject from that subject's own rest
trials (subtracting each trial's own subject's baseline) — requires a
trial-level split (`random`/`stratified`/`trial_wise`). Preset: `baseline=per_user`.

### `feature_extraction`

| Field | Type | Meaning |
|---|---|---|
| `feature_type` | `time_domain` \| `frequency_domain` \| `combined` | Which extractor(s) to run |
| `features` | list of **names** (strings) | Which features to compute; `[]` = all |

`feature_type` picks the handcrafted feature representation:

| `feature_type` | extractors |
|---|---|
| `time_domain` | `TimeDomainFeatures` (11 per-channel features) |
| `frequency_domain` | `FrequencyDomainFeatures` (4 per-channel features) |
| `combined` | both, concatenated column-wise |

`features` is a list of feature names (not channel numbers):

```yaml
feature_extraction:
  feature_type: time_domain
  features: [rms, mav, wl]     # compute only these three
```

Feature names:

- Time-domain: `rms`, `std`, `min`, `max`, `mean`, `max_over_mean`,
  `min_over_max`, `mav`, `iemg`, `variance`, `wl`
- Frequency-domain: `total_power`, `peak_frequency`, `mean_frequency`,
  `median_frequency`

### `classical`

| Field | Values | Default |
|---|---|---|
| `preprocessing` | list of learned transformers (fit on train) | `[PerChannelStandardScaler]` |
| `model` | `_target_` + kwargs | `RandomForestEstimator` (`n_estimators: 100`) |

### `neural`

| Field | Values | Default (`neural=lstm`) |
|---|---|---|
| `datamodule` | `EMGDataModule` + `batch_size`, `num_workers`, `shuffle` | `batch_size: 32` |
| `model` | `DNNModule` \| `LSTMModule` \| `CNNModule` \| `CNNLSTMModule` | `LSTMModule` |
| `optimizer` | `_target_` (`torch.optim.Adam`/`SGD`/`AdamW`/`RMSprop`) + `lr` | `Adam`, `lr: 0.001` |
| `early_stopping` | `EarlyStopping` or `null` | `val_loss`, patience 5 |
| `trainer` | `lightning.pytorch.Trainer` kwargs | `max_epochs: 100`, `accelerator: auto` |

Select the architecture with `neural=dnn` / `lstm` / `cnn` / `cnn_lstm`.

### `evaluation`

| Field | Values | Default |
|---|---|---|
| `metrics` | list from `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1` | `[accuracy, balanced_accuracy, macro_f1]` |
| `average` | `macro` \| `weighted` | `macro` |
| `exclude_classes` | list of label indices to drop from scoring | `[]` |

### `analytics`

| Field | Values | Default |
|---|---|---|
| `enabled` | bool | `true` |
| `run_qc` | bool (write quality checks + class-distribution plot) | `true` |

### `tracking` and `reporting` (inlined in `config.yaml`)

| Field | Default |
|---|---|
| `tracking.tracking_uri` | `sqlite:///mlflow.db` |
| `tracking.experiment_name` | `hand_gesture_recognition` |
| `reporting.formats` | `[json, html]` |

Each run also writes a `shape_summary.json` (input → windows → features shapes
with symbolic dimension names) into `output/evaluation/<run_id>/` and logs it
to MLflow.

## Visualization

The `visualization` group controls all generated plots. The single
`selection` block is the **source of truth** that drives three plot families:
(A) raw-signal-per-class, (B) stage progression, and (C) feature-over-windows.

```yaml
# configs/visualization/default.yaml
enabled: true                 # master toggle for all plots
consistent_scale: true        # share one y-range across channels within a figure
selection:
  gestures: []                # which gestures to plot (names); [] = all
  channels: []                # which channels to show (1-based numbers); [] = all
  by_gesture: {}              # per-gesture channel override: {gesture_name: [channels]}
stage_progression: true       # plot B: signal at each preprocessing stage
feature_over_windows:
  enabled: true               # plot C
  features: []                # which features to plot (names); [] = all
  max_windows: 0              # cap the number of windows; 0 = all
```

### `selection` — choosing gestures and channels

| Field | Type | Meaning |
|---|---|---|
| `gestures` | list of **names** (strings) | Which gestures to plot. `[]` = all classes. |
| `channels` | list of **numbers** (1-based) | Which channels to show. `[]` = all channels. |
| `by_gesture` | dict `{name: [channels]}` | Overrides `channels` for one specific gesture. |

**Gesture names** must match `metadata.class_names` exactly (case-sensitive).
For the EPN612 dataset these are:

```
noGesture, fist, waveIn, waveOut, open, pinch
```

For synthetic data they are `class_0`, `class_1`, `class_2`, ….

**Channels** are 1-based numbers. EPN612 has 8 channels, so `[1, 2, 3]` means
the first three; `[]` means all 8.

**`by_gesture`** is a per-gesture override of the global `channels` list. The
resolution rule is:

> For gesture `g`: if `g` is listed in `by_gesture`, use those channels;
> otherwise use the global `channels`.

#### Examples

**Same channels for both gestures:**
```yaml
selection:
  gestures: [fist, open]
  channels: [1, 2, 3]
```
→ plots `fist` and `open`, each showing channels 1–3.

**Per-gesture channels via `by_gesture`:**
```yaml
selection:
  gestures: [fist, open]
  channels: [1, 2]        # global default
  by_gesture:
    open: [3, 4]          # override just for "open"
```
→ `fist` shows channels 1–2; `open` shows channels 3–4.

**All gestures, special-case one:**
```yaml
selection:
  gestures: []            # every class
  channels: [1, 2]
  by_gesture:
    pinch: [5, 6, 7]
```
→ every gesture shows channels 1–2, except `pinch` shows channels 5–7.

The same selection is applied to all three plot families — if you select
`open` with channel `1`, then stage progression (B) and feature-over-windows
(C) are rendered only for `open`, channel `1`.

### `feature_over_windows.features` (plot C)

Takes **feature names** (not channel numbers — channels come from `selection`):

- Time-domain: `rms`, `std`, `min`, `max`, `mean`, `max_over_mean`,
  `min_over_max`, `mav`, `iemg`, `variance`, `wl`
- Frequency-domain: `total_power`, `peak_frequency`, `mean_frequency`,
  `median_frequency`

```yaml
feature_over_windows:
  features: [rms, mean_frequency]   # plot only these two features per window
```

## Examples

```bash
# Default: neural LSTM on EPN612, subject-wise split
python -m hand_gesture_recognition --mode compose

# Classical Random Forest on synthetic, trial-level split
python -m hand_gesture_recognition --mode compose \
  dataset.name=synthetic backend=classical split.strategy=random baseline=none

# 5-fold subject-wise CV, classical, on synthetic
python -m hand_gesture_recognition --mode compose \
  dataset.name=synthetic backend=classical baseline=none cv=kfold

# Visualization: only fist/open, channel 1, with per-gesture override
python -m hand_gesture_recognition --mode compose \
  dataset.name=synthetic baseline=none split.strategy=random \
  visualization.selection.gestures=[fist,open] \
  visualization.selection.channels=[1] \
  visualization.selection.by_gesture.open=[3,4]
```

## Reproducibility

Every run persists its fully-resolved config to
`output/evaluation/<run_id>/config/resolved.yaml`. Reproduce it exactly with:

```bash
python -m hand_gesture_recognition --mode replay \
  --resolved output/evaluation/<run_id>/config/resolved.yaml
```
