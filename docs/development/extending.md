# Switching & Extending Components

Everything pluggable is selected by a `_target_` string in a Hydra config and
instantiated with `hydra.utils.instantiate`. To **switch** a component you edit
the config; to **add** a component you write the class and add a config entry.
No runner code changes.

The mechanism:

```yaml
# configs/<group>/<name>.yaml
_target_: full.import.path.ToClass   # import path of the class
arg1: value                          # constructor kwargs, passed as-is
```

`hydra.utils.instantiate(cfg_node)` imports the class and calls
`Class(arg1=value, ...)`.

## Config layout

```
configs/
├── config.yaml              # the defaults list (which config each group uses)
├── dataset/                 # dataset loaders
├── split/                   # split strategies
├── windowing/               # window size/stride
├── signal_processing/       # deterministic signal transforms (before split)
├── classical/               # classical preprocessing/features/model
├── neural/                  # neural datamodule/optimizer/model/trainer
├── evaluation/              # metrics
├── analytics/               # analytics flags
└── experiment/              # backend, seed, tracking, reporting
```

`config.yaml` picks the default for each group:

```yaml
defaults:
  - dataset: epn612
  - split: subject_wise
  - windowing: default
  - classical: random_forest
  - neural: dnn
  - evaluation: basic
  - analytics: basic
  - experiment: default
  - _self_
```

Switch any group from the CLI with `group=choice`, e.g. `neural=lstm`.

## Switching a dataset loader

`configs/dataset/epn612.yaml`:

```yaml
# @package dataset
name: epn612
path: resources/data/EPN612
sampling_rate: 200
loader:
  _target_: hand_gesture_recognition.data.loaders.epn612.EPN612Loader
  subjects: null      # override with dataset.loader.subjects=[train:user1,...]
```

**To add a new loader** (e.g. a CSV loader), implement the interface:

```python
# src/hand_gesture_recognition/data/loaders/csv_loader.py
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.loaders.base import DatasetLoader

class CsvTimeSeriesLoader(DatasetLoader):
    def load(self, path: str) -> Dataset:
        ...  # build a Dataset (trials, labels, subject_ids, trial_ids,
             # gesture_start/end, metadata, optional subject_info)
        return dataset
```

Then add `configs/dataset/csv.yaml` and select it with `dataset=csv`.

## Switching the split strategy

`configs/split/subject_wise.yaml` sets `strategy`. Built-in strategies:
`random`, `stratified`, `subject_wise`, `session_wise`, `predefined`.

```bash
python -m hand_gesture_recognition --mode compose split.strategy=predefined
```

**To add a new strategy**, extend `split_dataset` in
`data/splits/base.py` (add a `elif config.strategy == "..."` branch that
computes `train_idx`/`val_idx`/`test_idx`) and register the name in
`SplitConfig.validate()`.

## Signal processing (deterministic, before split)

`configs/signal_processing/` holds presets (e.g. `emg.yaml` = bandpass + notch
+ rectify). Select with `signal_processing=emg` (default `none`). Components
live in `src/preprocessing/signal.py` (`RectifyTransformer`,
`ButterworthFilter`, `EnvelopeExtractor`); learned preprocessing lives in
`src/preprocessing/learned.py` (`PerChannelStandardScaler`). To add a signal
transform, subclass `sklearn.base.BaseEstimator, TransformerMixin` with
`transform(X) -> X` (operating on `(time, channels)` or `(windows, time,
channels)` along `axis=-2`) and reference it in the `signal_processing` list.

## Switching a classical model

`configs/classical/random_forest.yaml`:

```yaml
# @package classical
feature_type: time_domain   # time_domain | frequency_domain | combined | custom
preprocessing:
  - _target_: hand_gesture_recognition.preprocessing.learned.PerChannelStandardScaler
model:
  _target_: hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator
  n_estimators: 100
  random_state: 42
```

The handcrafted feature representation is chosen with a single `feature_type`
knob (select from the CLI with `classical.feature_type=<name>`):

| `feature_type` | feature extractors applied |
|---|---|
| `time_domain` | `TimeDomainFeatures` |
| `frequency_domain` | `FrequencyDomainFeatures` (sampling rate auto-injected from the dataset) |
| `combined` | both, concatenated column-wise |
| `custom` | an explicit `classical.features:` list of `_target_` entries (for any other extractor) |

Swap the model by editing `model._target_` (e.g. to a `SVC` or
`LogisticRegression` `_target_`), or add a new file under `configs/classical/`
and select it with `classical=<name>`. The classical runner applies
`preprocessing` → `features` → `model`, all via `instantiate`. When several
extractors are listed they are applied independently and concatenated
column-wise (FeatureUnion), so `combined` composes time + frequency into one
feature vector.

**To add a feature extractor**, subclass
`sklearn.base.BaseEstimator, TransformerMixin` with `transform(X) -> (n_windows, features)`
(see `features/time_domain.py`) and reference it via `feature_type: custom` +
the `features:` list.

Built-in extractors:

- `features/time_domain.py` — `TimeDomainFeatures` (RMS, std, min/max, ...).
- `features/frequency_domain.py` — `FrequencyDomainFeatures` (total power, peak/
  mean/median frequency per channel). Requires a `sampling_rate` kwarg (injected
  automatically from `dataset.sampling_rate` by `ClassicalConfig.resolve_features`);
  it also accepts `use_hann` / `remove_dc` flags.

## Switching a neural model

Neural models are end-to-end over the preprocessed signal; the handcrafted
time/frequency features are a *separate* representation used by the classical
pipeline (or explicitly fused with neural representations later).

`configs/neural/lstm.yaml` (mirrors `dnn.yaml`):

```yaml
# @package neural
datamodule:
  _target_: hand_gesture_recognition.neural.datamodules.emg.EMGDataModule
  batch_size: 32
  num_workers: 0
model:
  _target_: hand_gesture_recognition.neural.models.lstm.LSTMModule
  hidden_size: 64
  num_layers: 2
  lr: 0.001
trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 5
  accelerator: auto   # GPU when available, else CPU
  enable_checkpointing: false
```

Built-in neural architectures (select with `neural=<name>`):

- `dnn` — `DNNModule`: MLP over the flattened window.
- `lstm` — `LSTMModule`: LSTM over the time axis.
- `cnn` — `CNNModule`: 1-D CNN over the time axis + global pooling.
- `cnn_lstm` — `CNNLSTMModule`: composite CNN (local temporal features) → LSTM
  (sequence model) → dense head.

Switch with `neural=dnn` / `neural=lstm` from the CLI.

**To add a new neural model** (e.g. a CNN), subclass
`hand_gesture_recognition.neural.models.base.ClassifierModule` (it provides
the loss, training/validation/test steps, and optimizer — you only implement
`forward(x) -> logits`) and add a `configs/neural/<name>.yaml` with its
`_target_`. The neural runner passes `n_channels` and `n_classes` to the model.

## Adding a reporter

Reporters consume `EvaluationResult`. Add one in `reporting/reporters.py`
(a class with `write(report: dict, path: str)`), then it can be wired into
`runner/run.py`'s `_persist_artifacts` (or a future reporter registry).

## The general recipe

1. **Write the component** implementing the relevant interface (a loader, a
   transformer, an estimator, a `LightningModule`, a reporter, ...).
2. **Add a `_target_` config** under the matching `configs/<group>/` directory.
3. **Select it** from the CLI (`<group>=<name>`), or make it the default in
   `configs/config.yaml`.

Because the runner wires everything through `instantiate`, no orchestration
code needs to change when you add or swap a component.
