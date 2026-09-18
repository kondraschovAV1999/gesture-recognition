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
features:
  - _target_: hand_gesture_recognition.features.time_domain.TimeDomainFeatures
preprocessing:
  - _target_: sklearn.preprocessing.StandardScaler
model:
  _target_: hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator
  n_estimators: 100
  random_state: 42
```

Swap the model by editing `model._target_` (e.g. to a `SVC` or `LogisticRegression`
`_target_`), or add a new file under `configs/classical/` and select it with
`classical=<name>`. The classical runner applies `features` → `preprocessing` →
`model` in order, all via `instantiate`.

**To add a feature extractor**, subclass
`sklearn.base.BaseEstimator, TransformerMixin` with `transform(X) -> (n_windows, features)`
(see `features/time_domain.py`) and reference it in `features:`.

## Switching a neural model

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
  accelerator: cpu
  enable_checkpointing: false
```

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
