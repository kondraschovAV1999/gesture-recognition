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
├── config.yaml              # defaults list + inlined experiment/task/split
├── dataset/                 # dataset loaders
├── cv/                      # cross-validation (none/kfold/loso)
├── windowing/               # window size/stride
├── signal_processing/       # deterministic signal transforms (before split)
├── gesture_crop/            # crop trials to the gesture segment
├── baseline/                # global rest-baseline removal
├── feature_extraction/      # handcrafted feature representation
├── classical/               # classical preprocessing/model
├── neural/                  # neural datamodule/optimizer/model/trainer
├── evaluation/              # metrics
├── analytics/               # analytics flags
└── visualization/           # plot selection/scale/stage/feature plots
```

`config.yaml` picks the default for each group:

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
  _target_: hand_gesture_recognition.pipeline.data_loaders.epn612.EPN612Loader
  subjects: null      # override with dataset.loader.subjects=[train:user1,...]
```

**To add a new loader** (e.g. a CSV loader), implement the interface:

```python
# src/hand_gesture_recognition/pipeline/data_loaders/csv_loader.py
from hand_gesture_recognition.utils.datasets import Dataset
from hand_gesture_recognition.pipeline.data_loaders.base import DatasetLoader

class CsvTimeSeriesLoader(DatasetLoader):
    def load(self, path: str) -> Dataset:
        ...  # build a Dataset of Users/Trials (each Trial: emg, gesture_name,
             # start/end bounds, user_id/trial_id), plus metadata
        return dataset
```

Then add `configs/dataset/csv.yaml` and select it with `dataset=csv`.

## Switching the split strategy

`split` is now inlined in `config.yaml` (there is no `configs/split/` preset
group); its `strategy` field defaults to `group_wise`. Built-in strategies:
`random`, `stratified`, `group_wise`, `session_wise`, `predefined`.

```bash
python -m hand_gesture_recognition --mode compose split.strategy=predefined
```

**To add a new strategy**, add a class to `pipeline/splits/strategy.py` (with a
`.split(dataset, config) -> DataSplit` method) and register it with
`@register_split("<name>")`, then add the name to the allowed set in
`SplitConfig.validate()`.

## Signal processing

`configs/signal_processing/` holds presets (e.g. `emg.yaml` = bandpass + notch
+ rectify). Select with `signal_processing=emg` (default `none`). Components
live in `src/hand_gesture_recognition/pipeline/preprocessing/signal.py`
(`RectifyTransformer`, `ButterworthFilter`, `EnvelopeExtractor`); learned
preprocessing lives in
`src/hand_gesture_recognition/pipeline/preprocessing/learned.py`
(`PerChannelStandardScaler`). To add a signal transform, subclass
`sklearn.base.BaseEstimator, TransformerMixin` with `transform(X) -> X`
(operating on `(time, channels)` or `(windows, time, channels)` along
`axis=-2`) and reference it in the `signal_processing` list.

## Switching a classical model

`configs/classical/random_forest.yaml`:

```yaml
# @package classical
preprocessing:
  - _target_: hand_gesture_recognition.pipeline.preprocessing.learned.PerChannelStandardScaler
model:
  _target_: hand_gesture_recognition.pipeline.models.classical.random_forest.RandomForestEstimator
  n_estimators: 100
  random_state: 42
```

The handcrafted feature representation is chosen with a single `feature_type`
knob (select from the CLI with `feature_extraction.feature_type=<name>`):

| `feature_type` | feature extractors applied |
|---|---|
| `time_domain` | `TimeDomainFeatures` |
| `frequency_domain` | `FrequencyDomainFeatures` (sampling rate auto-injected from the dataset) |
| `combined` | both, concatenated column-wise |

A subset of features can be requested by name with
`feature_extraction.features=[rms, wl, ...]` (empty = all). The canonical
feature-name catalogs live in
`configuration/schema/feature_extraction.py`
(`TIME_DOMAIN_FEATURE_NAMES` / `FREQUENCY_DOMAIN_FEATURE_NAMES`) and are
imported by the extractors as their `FEATURE_NAMES`.

Swap the model by editing `model._target_` (e.g. to a `SVC` or
`LogisticRegression` `_target_`), or add a new file under `configs/classical/`
and select it with `classical=<name>`. The classical runner applies
`preprocessing` → `features` → `model`, all via `instantiate`. When several
extractors are listed they are applied independently and concatenated
column-wise (FeatureUnion), so `combined` composes time + frequency into one
feature vector.

To add a new handcrafted feature, add its name to the appropriate catalog and
add a compute function to the extractor's `_COMPUTE` dispatch map (see
`pipeline/feature_extraction/extraction_strategies/time_domain.py`).

Built-in extractors:

- `pipeline/feature_extraction/extraction_strategies/time_domain.py` — `TimeDomainFeatures` (rms, std, min, max, mean, max_over_mean, min_over_max, mav, iemg, variance, wl per channel).
- `pipeline/feature_extraction/extraction_strategies/frequency_domain.py` — `FrequencyDomainFeatures` (total power, peak/mean/median frequency per channel). Requires a `sampling_rate` kwarg (injected automatically from `dataset.sampling_rate` by `FeatureExtractionConfig.resolve_features`); it also accepts `use_hann` / `remove_dc` flags.

## Switching a neural model

Neural models are end-to-end over the preprocessed signal; the handcrafted
time/frequency features are a *separate* representation used by the classical
pipeline (or explicitly fused with neural representations later).

`configs/neural/lstm.yaml` (mirrors `dnn.yaml`):

```yaml
# @package neural
datamodule:
  _target_: hand_gesture_recognition.pipeline.models.neural.datamodule.EMGDataModule
  batch_size: 32
  num_workers: 0
model:
  _target_: hand_gesture_recognition.pipeline.models.neural.lstm.LSTMModule
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
`hand_gesture_recognition.pipeline.models.neural.base.ClassifierModule` (it provides
the loss, training/validation/test steps, and optimizer — you only implement
`forward(x) -> logits`) and add a `configs/neural/<name>.yaml` with its
`_target_`. The neural runner passes `n_channels` and `n_classes` to the model.

## Adding a reporter

Reporters consume `EvaluationResult`. Add one in
`utils/observability/reporting.py` (a class with `write(report: dict, path:
str)`), then it can be wired into `runner/pipeline.py`'s `_persist_artifacts`
(or a future reporter registry).

## The general recipe

1. **Write the component** implementing the relevant interface (a loader, a
   transformer, an estimator, a `LightningModule`, a reporter, ...).
2. **Add a `_target_` config** under the matching `configs/<group>/` directory.
3. **Select it** from the CLI (`<group>=<name>`), or make it the default in
   `configs/config.yaml`.

Because the runner wires everything through `instantiate`, no orchestration
code needs to change when you add or swap a component.
