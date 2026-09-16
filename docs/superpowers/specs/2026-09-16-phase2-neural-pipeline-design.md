# Phase 2 — Neural Pipeline (DNN + LSTM) — Design

**Date:** 2026-09-16
**Status:** Approved

## Goal

Add the neural-network backend to the framework: a `LightningDataModule`
that reuses the shared windowing, two `LightningModule`s (DNN and LSTM),
`Trainer` integration with `seed_everything`, a neural `Prediction` adapter,
and a cross-backend `Prediction`/`EvaluationResult` schema-parity test — all
reusing Phase 1's evaluation, MLflow tracking, logging, tracing, and reporting.

## Requirements

1. **Reuse shared windowing** — the DataModule calls `data/windowing`; no
   second windowing implementation (spec §6, §9.2).
2. **Two models** — DNN and LSTM, selectable via `_target_` config (ADR #18).
3. **Neural adapter** — normalize `Trainer.predict()` output into the shared
   `Prediction` schema (`backend="neural"`), identical to the classical one.
4. **Cross-backend parity test** — same synthetic dataset through both
   backends at the same windowing config yields `Prediction`/`EvaluationResult`
   with identical fields/shapes (spec §27).
5. **Same evaluator + MLflow** as Phase 1.

## Design

### 1. Dependencies

Install `torch` (CPU) + `lightning`; add both to `pyproject.toml`.

### 2. `src/neural/` package

- `neural/datamodules/emg.py` — `EMGDataModule(pl.LightningDataModule)`:
  - `__init__(split: DataSplit, windowing: WindowingConfig, batch_size: int,
    num_workers: int = 0)`.
  - `setup()`: for each partition (train/val/test) call the shared
    `window_dataset(...)`, then wrap `(X, y)` in `torch.utils.data.TensorDataset`.
  - `train_dataloader` / `val_dataloader` / `test_dataloader` return
    `torch.utils.data.DataLoader`s. Model-agnostic: always yields
    `(batch, time, channels)`.

- `neural/models/dnn.py` — `DNNModule(pl.LightningModule)`:
  `forward` flattens `(batch, time, channels)` → dense (ReLU) layers →
  `n_classes` logits.

- `neural/models/lstm.py` — `LSTMModule(pl.LightningModule)`:
  `forward` runs `(batch, time, channels)` through an LSTM, takes the last
  hidden state → dense → `n_classes` logits.

- Both modules share: CrossEntropy loss, `training/validation/test_step`
  (log loss + accuracy), `configure_optimizers` (Adam, lr from config).

- `neural/adapters/lightning.py` — `LightningPredictionAdapter`:
  concatenate `Trainer.predict()` batches → softmax → `y_score`; argmax →
  `y_pred`; wrap into `Prediction(backend="neural")` using the test
  `WindowedDataset` metadata (sample_ids, y_true) exactly like the classical
  adapter.

### 3. Schema + config

- Add `NeuralConfig` to `configuration/schema.py`:
  - `datamodule: Dict` (`_target_` = `...EMGDataModule`, `batch_size`, `num_workers`)
  - `model: Dict` (`_target_` = `...DNNModule` or `...LSTMModule`, plus arch params, `lr`)
  - `trainer: Dict` (`_target_` = `lightning.pytorch.Trainer`, `max_epochs`, `accelerator`, `enable_progress_bar`)
- `SUPPORTED_BACKENDS = ("classical", "neural")`.
- `ExperimentConfig` gains `neural: NeuralConfig`.
- `configs/neural/` group: `dnn.yaml`, `lstm.yaml` (model choices), and
  defaults for `datamodule`/`trainer`; `configs/config.yaml` gains a
  `- neural: dnn` default.

### 4. Runner

Branch on `cfg.backend`:

- `"classical"` → existing path.
- `"neural"` →
  1. `pl.seed_everything(cfg.seed, workers=True)`.
  2. `datamodule = instantiate(cfg.neural.datamodule, split=split, windowing=cfg.windowing)`.
  3. `model = instantiate(cfg.neural.model, n_channels=..., n_classes=...)`.
  4. `trainer = instantiate(cfg.neural.trainer)`.
  5. `trainer.fit(model, datamodule=datamodule)`.
  6. `raw = trainer.predict(model, datamodule=datamodule)`.
  7. `LightningPredictionAdapter(...).to_prediction(...)`.
  8. Same `evaluate` + `track_and_report` as classical.

`n_channels`/`n_classes` are derived from the loaded `Dataset` metadata.

### 5. Testing

- Unit: `EMGDataModule.setup` windows via the shared implementation (tensor
  shapes); `DNNModule`/`LSTMModule.forward` output shape = `(batch, n_classes)`;
  adapter softmax/argmax correctness.
- **Parity test** (`tests/integration/test_cross_backend_parity.py`): tiny
  synthetic dataset → classical and neural pipelines → assert
  `Prediction` and `EvaluationResult` have identical field names and
  `y_pred`/`y_score`/`confusion_matrix` shapes (values may differ).
- Extend the smoke test to run the neural backend end-to-end on synthetic
  data.

## Non-goals

CNN/Transformer models, HPO, GPU/CUDA, and hyperparameter search — all Phase 3.
