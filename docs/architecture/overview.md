# Architecture & Data Flow

## High-level pipeline

```text
                  HYDRA CONFIGURATION
                          |
                          v
        Configuration Loading Module (compose / replay)
                          |
                          v
                    Experiment Runner
                          |
                          v
                     Dataset Loader
                          |
                          v
                        Dataset ───────────> Analytics (raw)
                          |                        |
                          v                        v
              Train / Val / Test Split      stats / QC / plots
                          |
                          v
        Windowing (shared, applied per partition)
                          |
               +----------+----------+
               |                     |
               v                     v
         Classical ML            Neural Network
         (sklearn Pipeline)      (Lightning)
               |                     |
               v                     v
       Classical Adapter       Neural Adapter
               |                     |
               +----------+----------+
                          |
                          v
                 Prediction (shared schema)
                          |
                          v
                       Evaluation
                          |
                          v
                   EvaluationResult
                          |
               +----------+----------+
               |                     |
               v                     v
             MLflow               Reporting
```

## Component layers

| Layer | Package | Responsibility |
|---|---|---|
| **Configuration** | `configuration/` | Hydra schema (`ExperimentConfig` dataclasses) + the *Configuration Loading Module* — the only code allowed to call Hydra's composition API. Resolves config (compose or replay) into a validated, typed `ExperimentConfig`. |
| **Data** | `data/` | `Dataset` representation, dataset loaders, [split strategies](docs/experiments/splitting.md), the **shared** windowing, and content hashing. |
| **Prediction** | `prediction/` | The shared `Prediction` dataclass + `PredictionAdapter` interface. |
| **Classical** | `classical/` | sklearn pipeline (windowing transformer, features, model) + the classical adapter. |
| **Neural** | `neural/` | Lightning `DataModule`/`LightningModule`s (DNN, LSTM) + the neural adapter. |
| **Analytics** | `analytics/` | Dataset statistics, quality-control checks, and plots (a diagnostic consumer of the raw `Dataset`). |
| **Evaluation** | `evaluation/` | Backend-independent metric computation (consumes `Prediction` only). |
| **Tracking / Logging / Tracing** | `tracking/`, `logging_setup/`, `tracing/` | MLflow run tracking, centralized logging, and stage-level tracing. |
| **Reporting** | `reporting/` | JSON/HTML reporters (consumers of `EvaluationResult`). |
| **Runner** | `runner/` | Thin orchestrator that wires the stages together. |

## Data flow, stage by stage

1. **Configuration resolution** — `load_config(mode="compose"|"replay", ...)` returns a validated `ExperimentConfig` (compose = Hydra groups + CLI overrides; replay = re-load a previously persisted `resolved.yaml`).
2. **Load** — the loader converts a dataset-specific source (e.g. EPN612 JSON) into the framework `Dataset`: a list of continuous, *unwindowed* trials with aligned `labels`, `subject_ids`, `trial_ids`, `gesture_start`/`gesture_end`, per-subject `subject_info` (pandas DataFrame), and `metadata` (sampling rate, channels, class names, content hash).
3. **Signal processing** (deterministic, before split) — the `signal_processing` config list (e.g. bandpass/notch Butterworth filters, rectification, envelope) transforms the raw continuous signal. These steps learn nothing, so they are safe before splitting.
4. **Split** — `split_dataset` partitions the processed `Dataset` at the **user/recording level** (random, stratified, subject-wise, session-wise, or predefined). Windows are *never* split across partitions; subject/session leakage is checked for group-based strategies.
5. **Windowing** — the **single shared** `window_dataset` slides windows over each partition, assigning a per-window label (`0` = rest, `1..5` = gesture) from the gesture bounds. Windows stay in **temporal order** (no window shuffling).
6. **Train + predict** — branches on `cfg.backend`:
   - *classical*: windowing → **learned preprocessing** (fit on train only) → feature extraction → sklearn estimator.
   - *neural*: `seed_everything` → `LightningDataModule` (calls the shared windowing) → `Trainer.fit` → `Trainer.predict`.
7. **Prediction normalization** — the backend adapter (classical or neural) wraps native output into the shared `Prediction` (same schema, `backend="classical"|"neural"`).
8. **Evaluate** — the backend-independent evaluator computes aggregate + per-class metrics (precision/recall/F1, TP/FP/TN/FN) and the confusion matrix.
9. **Track + report** — MLflow records params/metrics/artifacts (confusion-matrix PNG, resolved config, auto-logged sklearn model); JSON/HTML reports are written to `output/evaluation/<run_id>/`.

## Stage terminology

Three distinct concepts (not "everything is preprocessing"):

- **Signal processing** — deterministic transforms of the physical signal (bandpass, notch, rectify, envelope). Learns nothing → safe before split.
- **Preprocessing** — learned transforms that make ML inputs suitable (e.g. per-channel standardization). **Fit on train only**, transform all.
- **Feature extraction** — converting a window into a feature vector (RMS, std, min/max, frequency, …).

Config mapping: `signal_processing` (deterministic, before split), `preprocessing` (learned, after windowing), `features` (window → feature vector).

## Key design decisions

1. **One shared windowing implementation** (`data/windowing/`) — no per-backend duplication; splitting always happens before windowing.
2. **`Prediction` normalization** — every backend output goes through an adapter, so evaluation never sees sklearn/Lightning objects.
3. **Single configuration entry point** — only `configuration/loader.py` calls Hydra; everything else consumes a typed `ExperimentConfig`.
4. **Pluggable components via `_target_`** — models, loaders, features, preprocessing, and the trainer are instantiated with `hydra.utils.instantiate`, so adding one is a config entry, not a code change.
5. **Reproducibility** — dataset content hash (location-independent), explicit seeds, config replay mode.
