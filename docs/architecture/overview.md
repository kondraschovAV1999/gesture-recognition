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
| **Data** | `pipeline/data_loaders/`, `utils/datasets.py` | The nested `Dataset → User → Trial → emg` model, dataset loaders, and content hashing (`utils/hashing.py`). |
| **Splits** | `pipeline/splits/` | [Split strategies](../experiments/splitting.md) + cross-validation strategies, registered by name. |
| **Preprocessing** | `pipeline/preprocessing/` | The `PreprocessingPipeline` (pre-split + post-split stages), signal transforms, learned preprocessing, gesture crop, and baseline removal. |
| **Windowing** | `pipeline/feature_extraction/windowing/` | The **shared** windowing implementation + windowing strategies (`recognition`/`classification`/`two_level`), registered by name. |
| **Features** | `pipeline/feature_extraction/` | Handcrafted feature extractors (`extraction_strategies/`) + feature resolution. |
| **Prediction** | `pipeline/evaluation/prediction.py` | The shared `Prediction` dataclass + backend adapters. |
| **Classical** | `pipeline/models/classical/` | sklearn pipeline (windowing transformer, features, model), the classical adapter, and the classical training strategy. |
| **Neural** | `pipeline/models/neural/` | Lightning `DataModule`/`LightningModule`s (DNN, LSTM, CNN, CNN+LSTM), the neural adapter, and the neural training strategy. |
| **Evaluation** | `pipeline/evaluation/` | Backend-independent metric computation (consumes `Prediction` only). |
| **Analytics** | `utils/analytics/` | Dataset statistics, quality-control checks, and plots (a diagnostic consumer of the raw `Dataset`). |
| **Observability** | `utils/observability/` | MLflow run tracking, centralized logging, stage-level tracing, JSON/HTML reporters, and visualization. |
| **Runner** | `runner/` | `ExperimentRunner` facade (owns tracker lifecycle) + the `Pipeline`/`PipelineBuilder` Template-Method orchestration. |

## Data flow, stage by stage

The `Pipeline` (`runner/pipeline.py`) is a Template-Method orchestrator: it
resolves the split/CV, windowing, and training strategies once at construction,
then `run()` executes the ordered phases below. `ExperimentRunner`
(`runner/run.py`) is a thin facade that owns MLflow tracker lifecycle and
builds the `Pipeline` via `PipelineBuilder`.

1. **Configuration resolution** — `load_config(mode="compose"|"replay", ...)` returns a validated `ExperimentConfig` (compose = Hydra groups + CLI overrides; replay = re-load a previously persisted `resolved.yaml`).
2. **Load** — the loader converts a dataset-specific source (e.g. EPN612 JSON) into the nested framework `Dataset`: a `Dataset` holds `metadata` plus a list of `User`s; each `User` owns a list of `Trial`s and optional `demographics`; each `Trial` carries the continuous, *unwindowed* `emg` signal `(time, channels)`, its `gesture_name`, gesture `start`/`end` bounds, and `user_id`/`trial_id`. Flat `trials`/`labels`/`subject_ids`/`trial_ids` views are derived, cached properties of `Dataset`.
3. **Analytics** — (optional) dataset summary, quality checks, and a class-distribution plot are written for the raw `Dataset` (`utils/analytics/`).
4. **Pre-split preprocessing** — `PreprocessingPipeline.transform_dataset` applies the stateless `pre_steps` once over the whole dataset (no fit). Currently this is gesture crop for the recognition task.
5. **Split / CV** — the split strategy (`pipeline/splits/`) partitions the dataset at the **user/recording level** (random, stratified, subject-wise, session-wise, or predefined); with `cv.enabled`, the CV strategy generates folds. Windows are *never* split across partitions; subject/session leakage is asserted for group-based strategies.
6. **Post-split preprocessing** — `PreprocessingPipeline.fit` + `transform` apply the `post_steps` (baseline removal + the `signal_processing` transforms) to each partition, fitting stateful steps on the **train partition only**.
7. **Windowing** — the shared windowing dispatches through the windowing strategy registry (`get_windowing_strategy`). The strategy is selected from `cfg.task.name` (or `two_level` when `windowing.mode=two_level`): `recognition` slides windows over each partition (`window_dataset`, per-window labels `0` = rest, `1..N` = gesture from the trial bounds); `classification` emits one padded/truncated sample per gesture trial (`window_gestures`) and drops rest. Windows stay in **temporal order** (no window shuffling). With `two_level`, the strategy segments each trial into big windows (signal-processed per big window), slides small windows inside each, and labels at the big-window level.
8. **Train + predict** — the training strategy registry (`get_training_strategy`) branches on `cfg.backend`:
   - *classical*: windowing → learned preprocessing (fit on train only) → feature extraction → sklearn estimator.
   - *neural*: `seed_everything` → `LightningDataModule` (calls the shared windowing) → `Trainer.fit` → `Trainer.predict`.
9. **Prediction normalization** — the backend adapter (classical or neural) wraps native output into the shared `Prediction` (same schema, `backend="classical"|"neural"`).
10. **Evaluate** — the backend-independent evaluator computes aggregate + per-class metrics (precision/recall/F1, TP/FP/TN/FN) and the confusion matrix.
11. **Track + report** — MLflow records params/metrics/artifacts (confusion-matrix PNG, resolved config, auto-logged sklearn model); JSON/HTML reports are written to `output/evaluation/<run_id>/`.

## Stage terminology

- **`pre_steps` (pre-split)** — stateless, whole-dataset transforms applied once before splitting (currently gesture crop for the recognition task).
- **`post_steps` (post-split)** — applied after splitting via `PreprocessingPipeline.fit` + `transform`; stateful (dataset-level) steps are fit on the **train partition only**. This is where baseline removal and the `signal_processing` transforms (bandpass/notch/rectify/envelope) run.
- **Feature extraction** — converting a window into a feature vector (RMS, std, min/max, frequency, …), selected by `feature_extraction.feature_type`; learned per-channel standardization is configured under `classical.preprocessing` and applied by the classical training strategy.

## Key design decisions

1. **One shared windowing implementation** (`pipeline/feature_extraction/windowing/`) — no per-backend duplication; splitting always happens before windowing.
2. **`Prediction` normalization** — every backend output goes through an adapter, so evaluation never sees sklearn/Lightning objects.
3. **Single configuration entry point** — only `configuration/loader.py` calls Hydra; everything else consumes a typed `ExperimentConfig`.
4. **Pluggable components via `_target_`** — models, loaders, features, preprocessing, and the trainer are instantiated with `hydra.utils.instantiate`, so adding one is a config entry, not a code change.
5. **Reproducibility** — dataset content hash (location-independent), explicit seeds, config replay mode.
6. **`task` toggle** — `classification` (one sample per gesture) vs `recognition`
   (sliding window) is a first-class config switch; both modes emit the same
   `WindowedDataset`, so every downstream stage is shared.
7. **Two-level windowing** — `windowing.mode=two_level` adds a coarse big-window
   level (per-big-window filtering + big-window classification) over the fine
   small-window level, sharing the same `WindowMetadata`/`evaluate` machinery.
