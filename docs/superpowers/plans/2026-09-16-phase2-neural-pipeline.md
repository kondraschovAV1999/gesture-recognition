# Phase 2 — Neural Pipeline (DNN + LSTM) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the neural backend (LightningDataModule reusing shared windowing, DNN + LSTM LightningModules, Trainer, neural Prediction adapter, cross-backend parity test) on top of Phase 1.

**Architecture:** `src/neural/` mirrors `src/classical/`. The runner branches on `cfg.backend`; the neural path seeds everything, instantiates the DataModule/model/Trainer via `_target_`, fits, predicts, normalizes through `LightningPredictionAdapter`, then reuses Phase 1's evaluate + MLflow + reporting.

**Tech Stack:** torch (CPU), lightning (2.x), plus Phase 1 stack.

## Global Constraints

- Package `hand_gesture_recognition`; conda env `hand_gesture_recognition`; `python -m pytest`; single-token commits; no `python -c`.
- Import lightning as `import lightning.pytorch as pl`; `_target_` uses `lightning.pytorch.*`.
- DataModule must call the shared `window_dataset` — no second windowing implementation.
- `Prediction` schema identical to classical (`backend="neural"`).
- All pluggable pieces via `hydra.utils.instantiate`.

---

### Task 1: Schema + config for the neural backend

**Files:** `pyproject.toml`, `src/.../configuration/schema.py`, `src/.../configuration/loader.py`, `configs/config.yaml`, `configs/neural/*.yaml`.

- [ ] Add `torch>=2.0` and `lightning>=2.0` to `pyproject.toml` dependencies.
- [ ] In `schema.py`: add `NeuralConfig` (datamodule/model/trainer dicts), set `SUPPORTED_BACKENDS = ("classical", "neural")`, add `neural: NeuralConfig = field(default_factory=NeuralConfig)` to `ExperimentConfig`.
- [ ] In `loader.py` `_coerce`: add `neural=NeuralConfig(**d.get("neural", {}))`.
- [ ] `configs/neural/dnn.yaml` / `lstm.yaml` (model `_target_` + arch params) and a `neural` default in `configs/config.yaml`.

### Task 2: `EMGDataModule` (neural/datamodules/emg.py)

`__init__(split, windowing, batch_size, num_workers)`; `setup()` windows each partition via `window_dataset` and wraps in `TensorDataset`; `train/val/test_dataloader()`.

### Task 3: Models (neural/models/base.py, dnn.py, lstm.py)

`ClassifierModule` base (CrossEntropy loss, train/val/test steps, `predict_step` returning logits, Adam). `DNNModule` (Flatten → LazyLinear → dense → logits), `LSTMModule` (LSTM → last hidden → dense → logits).

### Task 4: `LightningPredictionAdapter` (neural/adapters/lightning.py)

Concatenate `Trainer.predict()` logits → softmax → `y_score`, argmax → `y_pred` → `Prediction(backend="neural")`.

### Task 5: Runner neural path (runner/run.py)

Branch on `cfg.backend`; neural: `pl.seed_everything` → DataModule → Trainer.fit → predict → adapter → same evaluate/MLflow.

### Task 6: Tests

- Unit: DataModule tensor shapes, model forward shapes, adapter softmax/argmax.
- Parity test (`tests/integration/test_cross_backend_parity.py`): synthetic → both backends → identical `Prediction`/`EvaluationResult` fields + shapes + same `y_true`/`sample_ids`.
- Extend smoke test / add neural smoke.
