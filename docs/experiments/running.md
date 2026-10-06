# Running Experiments

## CLI

```bash
# compose mode (default): Hydra groups + CLI overrides
python -m hand_gesture_recognition --mode compose [OVERRIDES...]

# replay mode: reproduce a run from a previously persisted resolved config
python -m hand_gesture_recognition --mode replay --resolved output/evaluation/<run_id>/config/resolved.yaml
```

The runner prints stage logs + evaluation metrics to the console, and writes
artifacts under `output/`:

```
output/
├── analytics/<dataset_id>/        dataset_summary.json, quality_checks.json, class_distribution.png
├── evaluation/<run_id>/           config/resolved.yaml, report.json, report.html
└── logs/<run_id>/                 application.log
```

Runs are also tracked in MLflow (SQLite):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Common overrides

| Override | Effect |
|---|---|
| `dataset.name=synthetic` | use the fast synthetic fixture |
| `dataset.loader.subjects=[train:user1,train:user2,...]` | load a subset of the real EPN612 data |
| `split.strategy=subject_wise` / `random` / `predefined` | choose the split strategy |
| `backend=classical` / `neural` | choose the pipeline |
| `classical.feature_type=time_domain` / `frequency_domain` / `combined` / `custom` | choose the handcrafted feature representation (classical) |
| `neural=dnn` / `lstm` / `cnn` / `cnn_lstm` | choose the neural model (incl. the composite CNN + LSTM) |
| `neural.trainer.max_epochs=10` | set training epochs |
| `windowing.window_size_seconds=0.2` `windowing.stride_seconds=0.05` | window size / stride |
| `reporting.formats=[json,html]` | report formats |
| `dataset.include_no_gesture=false` | drop the `noGesture`/rest class: 5-class (gesture-only) pipeline |

### Examples

```bash
# Classical Random Forest, synthetic, subject-wise split
python -m hand_gesture_recognition --mode compose dataset.name=synthetic split.strategy=subject_wise

# Neural LSTM, synthetic, 3 epochs
python -m hand_gesture_recognition --mode compose dataset.name=synthetic backend=neural neural=lstm neural.trainer.max_epochs=3

# Classical on a real-data subset using the EPN612 A1 split
python -m hand_gesture_recognition --mode compose \
  dataset.loader.subjects=[train:user1,train:user2,train:user3,train:user4] \
  split.strategy=predefined

# Neural DNN on the same subset
python -m hand_gesture_recognition --mode compose \
  dataset.loader.subjects=[train:user1,train:user2,train:user3,train:user4] \
  split.strategy=predefined backend=neural neural=dnn

# 5-class (no noGesture/rest class): classify only gesture-active windows
python -m hand_gesture_recognition --mode compose \
  dataset.include_no_gesture=false gesture_crop=none
```

## Python API (running from a script/notebook)

The same pipeline is available programmatically:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner

cfg = load_config(
    mode="compose",
    overrides=["dataset.name=synthetic", "backend=classical"],
)
runner = ExperimentRunner(cfg)
result = runner.run()          # -> EvaluationResult
print(result.metrics)          # {'accuracy': ..., 'balanced_accuracy': ..., 'macro_f1': ...}
print(result.confusion_matrix)
print(result.per_class)        # per-class precision/recall/f1 + tp/fp/tn/fn
```

### Lower-level building blocks

Each stage is independently usable:

```python
from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader
from hand_gesture_recognition.data.splits.base import split_dataset, assert_no_group_leakage
from hand_gesture_recognition.data.windowing.base import window_dataset
from hand_gesture_recognition.configuration.schema import SplitConfig, WindowingConfig

# 1) load
ds = EPN612Loader(subjects=["train:user1", "train:user2"]).load("resources/data/EPN612")
ds.subject_info          # pandas DataFrame of per-subject demographics
ds.gesture_start         # per-trial gesture start indices

# 2) split (A1 predefined: train=trainingJSON.trainingSamples, ...)
split = split_dataset(ds, SplitConfig(strategy="predefined"))
assert_no_group_leakage(split)

# 3) window (shared, per-window labels)
wd = window_dataset(split.train, WindowingConfig(window_size_seconds=0.2, stride_seconds=0.05))
wd.X, wd.y               # (n_windows, time, channels), per-window labels
```

## Reproducing a run

Every run persists its full resolved config. To reproduce it exactly:

```bash
python -m hand_gesture_recognition --mode replay \
  --resolved output/evaluation/<run_id>/config/resolved.yaml
```

The config loader re-validates it against the current schema and reconstructs
the same `ExperimentConfig` (failing loudly if the config is no longer valid),
without re-applying newer defaults.
