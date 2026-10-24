# Package Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `src/hand_gesture_recognition/` into `configuration/`, `runner/`, `utils/`, `pipeline/`, and `main.py`, moving every module to its target home and rewriting all imports + Hydra `_target_` strings + docs.

**Architecture:** `pipeline/` holds the ML pipeline (data_loaders, preprocessing, feature_extraction, splits, evaluation, models); `utils/` holds shared data/helpers/observability; `configuration/` and `runner/` are unchanged homes. Purely mechanical: no behavior changes.

**Tech Stack:** Python 3.11+, Hydra (`_target_` strings), pytest.

## Global Constraints

- **Behavior-preserving:** no logic changes — only file moves, import rewrites, and `_target_` string updates.
- **Test suite must stay green** after each phase (`python -m pytest tests/unit tests/integration -q`).
- **Three places reference module paths and ALL must be updated together** for any moved module:
  1. `from hand_gesture_recognition.X import ...` / `import hand_gesture_recognition.X` in `src/` and `tests/`.
  2. Hydra `_target_: hand_gesture_recognition.X.Class` in `configs/*.yaml`.
  3. Hardcoded `_target_` dicts in `configuration/schema.py` (`_TIME_DOMAIN_FEATURE`, `_frequency_domain_feature`, `ClassicalConfig`, `NeuralConfig`, `DatasetConfig.loader`, `FeatureExtractionConfig.resolve_features`).
- Keep the public entry point working (`python -m hand_gesture_recognition`).

## Target layout

```
src/hand_gesture_recognition/
├── main.py                       # was __main__.py
├── __init__.py
├── configuration/                # unchanged
├── runner/                       # run.py, pipeline.py
├── utils/
│   ├── datasets.py               # was data/datasets/base.py
│   ├── class_selection.py        # was data/class_selection.py
│   ├── fixtures.py               # was data/fixtures.py
│   ├── hashing.py                # was data/hashing.py
│   ├── paths.py                  # was paths.py
│   ├── compare.py                # was compare.py
│   ├── analytics/                # was analytics/
│   └── observability/            # reporting + tracking + tracing + logging_setup
│       ├── reporting.py          # was reporting/reporters.py
│       ├── visualization.py      # was reporting/visualization.py
│       ├── comparison.py         # was reporting/comparison.py
│       ├── tracking.py           # was tracking/mlflow_tracker.py
│       ├── tracing.py            # was tracing/stage.py
│       └── logging.py            # was logging_setup/config.py
└── pipeline/
    ├── data_loaders/             # was data/loaders/
    ├── preprocessing/            # signal, baseline, learned, gesture_crop, pipeline
    ├── splits/                   # was data/splits/
    ├── feature_extraction/
    │   ├── extraction_strategies/ # was features/ (time_domain, frequency_domain)
    │   └── windowing/            # was data/windowing/ (base, gesture, strategy)
    ├── evaluation/               # was evaluation/ + prediction/
    │   ├── base.py, aggregate.py, grouped.py
    │   └── prediction.py         # was prediction/base.py
    └── models/
        ├── classical/            # random_forest, adapter (sklearn), legacy windowing transformer
        └── neural/               # dnn, lstm, cnn, cnn_lstm, base, datamodule, adapter (lightning)
```

## Module path mapping (old → new) — apply to imports, `_target_` strings, and `resolve_features`

| Old module | New module |
|---|---|
| `data.datasets.base` | `utils.datasets` |
| `data.loaders` (base, epn612) | `pipeline.data_loaders` |
| `data.splits` (base, folds, strategy) | `pipeline.splits` |
| `data.windowing` (base, gesture, strategy) | `pipeline.feature_extraction.windowing` |
| `data.class_selection` | `utils.class_selection` |
| `data.gesture_crop` | `pipeline.preprocessing.gesture_crop` |
| `data.fixtures` | `utils.fixtures` |
| `data.hashing` | `utils.hashing` |
| `features.time_domain` / `features.frequency_domain` | `pipeline.feature_extraction.extraction_strategies.time_domain` / `.frequency_domain` |
| `preprocessing.signal` / `.baseline` / `.learned` / `.pipeline` | `pipeline.preprocessing.signal` / `.baseline` / `.learned` / `.pipeline` |
| `classical.adapters.sklearn` | `pipeline.models.classical.adapter` |
| `classical.models.random_forest` | `pipeline.models.classical.random_forest` |
| `classical.pipelines.windowing` | `pipeline.models.classical.windowing` |
| `neural.adapters.lightning` | `pipeline.models.neural.adapter` |
| `neural.datamodules.emg` | `pipeline.models.neural.datamodule` |
| `neural.models.base` / `.dnn` / `.lstm` / `.cnn` / `.cnn_lstm` | `pipeline.models.neural.base` / `.dnn` / `.lstm` / `.cnn` / `.cnn_lstm` |
| `evaluation.base` / `.aggregate` / `.grouped` | `pipeline.evaluation.base` / `.aggregate` / `.grouped` |
| `prediction.base` | `pipeline.evaluation.prediction` |
| `analytics.stats` / `.qc` / `.plots` | `utils.analytics.stats` / `.qc` / `.plots` |
| `reporting.reporters` | `utils.observability.reporting` |
| `reporting.visualization` | `utils.observability.visualization` |
| `reporting.comparison` | `utils.observability.comparison` |
| `tracking.mlflow_tracker` | `utils.observability.tracking` |
| `tracing.stage` | `utils.observability.tracing` |
| `logging_setup.config` | `utils.observability.logging` |
| `paths` | `utils.paths` |
| `compare` | `utils.compare` |

(Also rename `training/` → `pipeline/models/` already done — the `training/` package from the refactor holds `base.py`, `features.py`, `classical.py`, `neural.py`; map `training.base` → `pipeline.models.training_base`? NO — see below.)

## Note on the `training/` package (from the prior refactor)

The prior refactor created `src/hand_gesture_recognition/training/` (`base.py` = TrainingStrategy registry, `features.py` = FeatureExtractor, `classical.py` = ClassicalTrainingStrategy, `neural.py` = NeuralTrainingStrategy). These are strategy/plumbing, not "models". Map them:
- `training.base` → `pipeline.models.base` (TrainingStrategy + registry)
- `training.features` → `pipeline.feature_extraction.features` (FeatureExtractor)
- `training.classical` → `pipeline.models.classical.strategy`
- `training.neural` → `pipeline.models.neural.strategy`

(This keeps the "models contain classical and neural" intent: each backend folder holds its model + adapter + datamodule + strategy.)

## Phases (each verified by full test suite before proceeding)

### Phase 1 — Move `pipeline/`-bound modules
Move data_loaders, splits, feature_extraction (features + windowing), preprocessing, evaluation(+prediction), models (classical/neural), and training→models. Rewrite ALL imports/`_target_`s per the mapping. Run `python -m pytest tests/unit tests/integration -q`.

### Phase 2 — Move `utils/`-bound modules
Move datasets, class_selection, fixtures, hashing, paths, compare, analytics, observability (reporting/visualization/comparison/tracking/tracing/logging). Rewrite imports/`_target_`s. Run the full suite.

### Phase 3 — `main.py` + entry point + docs
Rename `__main__.py` → `main.py` (keep `__main__.py` as a shim or update `pyproject.toml`/`setup.cfg`). Update `docs/architecture/*` references. Run the full suite.

## Self-review notes
- Import rewrite must be applied to `src/**`, `tests/**`, and `configs/**/*.yaml` + `configuration/schema.py` `_target_` dicts.
- `tests/datasets/` and `tests/fixtures/` also import modules — include them.
- After each phase, `python -m pytest tests/unit tests/integration -q` must be green (the suite was green at commit `d0bdaec` before this restructuring).
