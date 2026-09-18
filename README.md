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
| [Running experiments](docs/experiments/running.md) | CLI, config overrides, Python API, replay, MLflow |
| [Switching & extending components](docs/development/extending.md) | How to add/swap loaders, models, features, reporters |
| [EMG-EPN612 dataset](docs/datasets/epn612.md) | Dataset structure, fields, and the A1 split strategy |

## Project layout

```
src/hand_gesture_recognition/
├── configuration/   # Hydra schema + Configuration Loading Module (compose/replay)
├── data/            # Dataset, loaders, splits, shared windowing, hashing
├── preprocessing/   # (pluggable signal preprocessing)
├── features/        # handcrafted feature extractors
├── prediction/      # shared Prediction schema + adapter interface
├── classical/       # sklearn pipeline, models, Prediction adapter
├── neural/          # Lightning datamodules, models, Prediction adapter
├── analytics/       # dataset statistics, QC checks, plots
├── evaluation/      # backend-independent metrics
├── reporting/       # JSON/HTML reporters
├── tracking/        # MLflow tracker
├── logging_setup/   # centralized logging
├── tracing/         # stage-level tracing
└── runner/          # Experiment Runner orchestration
configs/             # Hydra config groups (dataset, split, windowing, ...)
tests/               # unit, integration, dataset tests
docs/                # this documentation
```
