# Hand Gesture Recognition — ML Research Framework

A reusable, reproducible research/benchmarking framework for time-series and
tabular ML experiments, with a primary use case of EMG hand-gesture
recognition (the EMG-EPN612 dataset).

## Features

- **Two backends, one pipeline** — classical ML (scikit-learn) and neural
  networks (PyTorch Lightning), sharing the same windowing, `Prediction`
  schema, evaluation, tracking, and reporting.
- **Fully config-driven** — Hydra hierarchical configs; every pluggable piece
  (dataset loader, split strategy, preprocessing, features, model, datamodule,
  trainer) is a `_target_` you can swap from the command line.
- **Shared, backend-agnostic windowing** — one implementation, applied after
  splitting, with per-window labels from gesture ground-truth timing.
- **Reproducible** — config compose + replay mode, dataset content hash,
  `seed_everything`, MLflow tracking.
- **Rich evaluation + tracking** — aggregate + per-class metrics (precision,
  recall, F1, TP/FP/TN/FN), confusion-matrix artifact, SQLite-backed MLflow.

## Installation

```bash
conda create -n hand-gesture-recognition python=3.10 -y
conda activate hand-gesture-recognition
pip install -e .
```

## Quick start

```bash
# Classical (Random Forest) on synthetic data
python -m hand_gesture_recognition --mode compose dataset.name=synthetic

# Neural (LSTM) on synthetic data
python -m hand_gesture_recognition --mode compose dataset.name=synthetic backend=neural neural=lstm

# Classical on a small subset of the real EMG-EPN612 data
python -m hand_gesture_recognition --mode compose \
  dataset.loader.subjects=[train:user1,train:user2,train:user3,train:user4] \
  split.strategy=predefined
```

Each run prints stage logs + evaluation metrics, and writes artifacts under
`output/<run_id>/`. Browse runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Documentation

| Doc | What it covers |
|---|---|
| [Architecture & data flow](docs/architecture/overview.md) | Component layers, the pipeline, data flow, design decisions |
| [Configuring the project](docs/experiments/configuration.md) | Every config group and field: dataset, split, CV, windowing, signal processing, models, visualization |
| [Running experiments](docs/experiments/running.md) | CLI, config overrides, Python API, replay, MLflow |
| [Switching & extending components](docs/development/extending.md) | How to add/swap loaders, models, features, reporters |
| [EMG-EPN612 dataset](docs/datasets/epn612.md) | Dataset structure, fields, and the A1 split strategy |
| [Splitting strategies](docs/experiments/splitting.md) | Trial-level vs subject-wise splits, leakage, and choosing a strategy |

## Project layout

```
src/hand_gesture_recognition/
├── main.py             # CLI entry point (`python -m hand_gesture_recognition`)
├── configuration/      # Hydra schema + Configuration Loading Module (compose/replay)
├── runner/             # ExperimentRunner facade + Pipeline/PipelineBuilder orchestration
├── utils/
│   ├── datasets.py     # nested Dataset → User → Trial → emg model
│   ├── class_selection.py, fixtures.py, hashing.py, paths.py, compare.py
│   ├── analytics/      # dataset statistics, QC checks, plots
│   └── observability/  # MLflow tracking, logging, tracing, reporting, visualization
└── pipeline/
    ├── data_loaders/   # dataset loaders (EPN612 + loader interface)
    ├── preprocessing/  # signal transforms, learned preprocessing, PreprocessingPipeline
    ├── splits/         # split + CV strategies (registered by name)
    ├── feature_extraction/
    │   ├── extraction_strategies/   # time/frequency-domain feature extractors
    │   └── windowing/               # shared windowing + windowing strategies
    ├── evaluation/     # backend-independent metrics + shared Prediction schema
    └── models/         # classical/ and neural/ backends + training strategies
configs/                # Hydra config groups (dataset, windowing, ...)
tests/                  # unit, integration, dataset tests
docs/                   # this documentation
```
