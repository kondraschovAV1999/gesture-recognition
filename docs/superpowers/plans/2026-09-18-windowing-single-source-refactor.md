# Windowing Single-Source-of-Truth Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the duplicate windowing between `EMGDataModule` and the neural runner, and make each partition's `WindowedDataset` the single authoritative representation — windowed exactly once, carrying a unique `window_id` per window, consumed by both backends and the prediction adapters.

**Architecture:** `window_dataset()` gains `window_ids`. A `WindowedPartitions` container holds the three windowed partitions. The runner windows each partition once and passes `WindowedPartitions` to both backends. `EMGDataModule` stops windowing (takes `WindowedPartitions` + a metadata-preserving `EMGWindowDataset`), `predict_step` returns logits + per-window metadata, and `LightningPredictionAdapter` reads metadata from the predict output instead of re-windowing.

**Tech Stack:** Python 3.10, numpy, PyTorch, PyTorch Lightning, dataclasses.

## Global Constraints

- Package name `hand_gesture_recognition`; source under `src/hand_gesture_recognition/`.
- Windows shell: `python -m pytest` (never bare `pytest`); `git commit -m <single-token>`; never `python -c "..."` (quote-stripping) — write a `.py` file instead.
- **`window_dataset()` is called exactly once per partition** — in the runner's data-preparation step only. No prediction adapter calls `window_dataset()`.
- Keep the shared `Prediction` schema fields; only the MEANING of `sample_ids` changes from trial IDs to unique `window_ids`.
- **Readable names (mandatory):** the batch dict uses `x` (input signal) and `y_true` (label); `predict_step` output uses `logits` (raw model output, NOT probabilities) plus `y_true`/`window_id`/`trial_id`/`subject_id`. Do not use a generic key like `"output"` or `"y"`.
- `window_ids` must be deterministic, unique within the experiment, and traceable back to the source trial. Keep `trial_id` separate.
- Visualization (`reporting/visualization.py`) re-windowing for plotting is OUT OF SCOPE — it is not a prediction path.
- Refactor only windowing/prediction-identity plumbing. Do NOT touch filtering, preprocessing steps, model architecture, evaluation metrics, or experiment config.

---

### Task 1: `window_ids` + `WindowedPartitions` (additive)

**Files:**
- Modify: `src/hand_gesture_recognition/data/windowing/base.py`
- Test: `tests/unit/test_windowing.py` (append)

**Interfaces:**
- Produces: `WindowMetadata.window_ids: np.ndarray`; `WindowedPartitions(train, val, test)` — consumed by Tasks 2–4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_windowing.py` (add the `WindowedPartitions` import at the top):

```python
from hand_gesture_recognition.data.windowing.base import WindowedPartitions


def test_window_ids_are_unique_and_traceable():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    ids = wd.metadata.window_ids.tolist()
    assert len(ids) == 14
    assert len(set(ids)) == 14  # unique
    # traceable: first trial's windows embed the trial id "t1" and subject "s1"
    assert ids[0].startswith("s1_t1_")
    assert ids[6].startswith("s2_t2_")


def test_windowed_partitions_dataclass():
    ds = _ds(lengths=(100, 100), labels=(0, 1), subjects=("s1", "s2"))
    cfg = WindowingConfig(window_size_samples=40, stride_samples=10)
    wd = window_dataset(ds, cfg)
    wp = WindowedPartitions(train=wd, val=wd, test=wd)
    assert wp.train is wd
    assert wp.val.X.shape[0] == 14
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_windowing.py -q`
Expected: FAIL — `AttributeError: 'WindowMetadata' object has no attribute 'window_ids'` (and `ImportError` for `WindowedPartitions`).

- [ ] **Step 3: Implement**

In `src/hand_gesture_recognition/data/windowing/base.py`:

**(a)** Add `window_ids` to `WindowMetadata`:

```python
@dataclass
class WindowMetadata:
    window_start: np.ndarray  # (n_windows,)
    window_end: np.ndarray  # (n_windows,)
    subject_ids: np.ndarray  # (n_windows,)
    trial_ids: np.ndarray  # (n_windows,)
    window_ids: np.ndarray  # (n_windows,) unique, traceable to trial
```

**(b)** In `window_dataset()`, collect `window_ids` alongside `subjects`/`trials`. Add a `wins: List[str] = []` list; inside the inner loop append `wins.append(f"{subj}_{tid}_window{i}")`; and pass `window_ids=np.asarray(wins)` to `WindowMetadata(...)`.

**(c)** Add the container dataclass after `WindowedDataset`:

```python
@dataclass
class WindowedPartitions:
    """The single authoritative windowed representation of train/val/test."""

    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_windowing.py -q`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/data/windowing/base.py tests/unit/test_windowing.py
git commit -m feat-window-ids-partitions
```

---

### Task 2: Neural side — metadata-preserving DataModule + models + adapter

**Files:**
- Modify: `src/hand_gesture_recognition/neural/datamodules/emg.py`
- Modify: `src/hand_gesture_recognition/neural/models/base.py`
- Modify: `src/hand_gesture_recognition/neural/adapters/lightning.py`
- Test: `tests/unit/test_neural.py` (update)

**Interfaces:**
- Consumes: `WindowedPartitions` (Task 1).
- Produces:
  - `EMGDataModule(windowed: WindowedPartitions, batch_size=32, num_workers=0, shuffle=False)` — no windowing.
  - `EMGWindowDataset(windowed: WindowedDataset)` — returns dict batches.
  - `predict_step` returns `{"logits", "y_true", "window_id", "trial_id", "subject_id"}`.
  - `LightningPredictionAdapter().to_prediction(raw_predictions, run_id, duration_s) -> Prediction`.

- [ ] **Step 1: Write/update the failing tests**

Rewrite the affected parts of `tests/unit/test_neural.py`. Replace the imports/helpers and the `EMGDataModule`-using tests as follows (keep `test_dnn_forward_shape`, `test_lstm_forward_shape`, `test_optimizer_is_configurable` unchanged). **Delete `test_emg_datamodule_preserves_temporal_order`** — it asserts positional order, which this refactor explicitly makes irrelevant (metadata now travels with each sample instead of relying on array position).

```python
from hand_gesture_recognition.data.windowing.base import (
    WindowedPartitions,
    WindowMetadata,
    WindowedDataset,
    window_dataset,
)
from hand_gesture_recognition.neural.datamodules.emg import EMGDataModule, EMGWindowDataset


def _windowed_partitions():
    ds = make_synthetic_dataset(n_subjects=4, n_trials_per_subject=4, length=100, n_channels=8, n_classes=3, seed=0)
    split = split_dataset(ds, SplitConfig(strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0))
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)
    return WindowedPartitions(
        train=window_dataset(split.train, windowing),
        val=window_dataset(split.val, windowing),
        test=window_dataset(split.test, windowing),
    )


def test_emg_datamodule_does_not_re_window():
    wp = _windowed_partitions()
    dm = EMGDataModule(windowed=wp, batch_size=8)
    dm.setup()
    X = next(iter(dm.train_dataloader()))
    # batch is a dict carrying metadata
    assert isinstance(X, dict)
    assert set(X.keys()) == {"x", "y_true", "window_id", "trial_id", "subject_id", "window_start", "window_end"}
    assert X["x"].shape[1:] == (40, 8)


def test_emg_window_dataset_returns_metadata_dict():
    wp = _windowed_partitions()
    ds = EMGWindowDataset(wp.train)
    item = ds[0]
    assert item["x"].shape == (40, 8)
    assert item["y_true"].ndim == 0
    assert isinstance(item["window_id"], str)
    assert isinstance(item["trial_id"], str)


def test_lightning_adapter_reads_metadata_from_predict_output():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    y = torch.tensor([0, 1, 2])
    raw = [
        {"logits": logits[:2], "y_true": y[:2], "window_id": ("w0", "w1"), "trial_id": ("t0", "t1"), "subject_id": ("s1", "s1")},
        {"logits": logits[2:], "y_true": y[2:], "window_id": ("w2",), "trial_id": ("t0",), "subject_id": ("s1",)},
    ]
    pred = LightningPredictionAdapter().to_prediction(raw_predictions=raw, run_id="r", duration_s=0.1)
    assert pred.backend == "neural"
    assert pred.sample_ids.tolist() == ["w0", "w1", "w2"]
    assert pred.y_true.tolist() == [0, 1, 2]
    assert pred.y_pred.tolist() == [0, 1, 2]
    assert pred.y_score.shape == (3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_neural.py -q`
Expected: FAIL — `TypeError`/`ImportError` on the new `EMGDataModule(windowed=...)` and `to_prediction(raw_predictions=...)` signatures.

- [ ] **Step 3: Implement**

**(a)** `src/hand_gesture_recognition/neural/datamodules/emg.py` — replace entirely:

```python
from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.data.windowing.base import WindowedDataset, WindowedPartitions


class EMGWindowDataset(Dataset):
    """Maps a `WindowedDataset` to a torch Dataset whose items carry metadata."""

    def __init__(self, windowed: WindowedDataset):
        self.windowed = windowed

    def __len__(self):
        return len(self.windowed.y)

    def __getitem__(self, idx):
        return {
            "x": torch.as_tensor(self.windowed.X[idx], dtype=torch.float32),
            "y_true": torch.as_tensor(self.windowed.y[idx], dtype=torch.long),
            "window_id": self.windowed.metadata.window_ids[idx],
            "trial_id": self.windowed.metadata.trial_ids[idx],
            "subject_id": self.windowed.metadata.subject_ids[idx],
            "window_start": self.windowed.metadata.window_start[idx],
            "window_end": self.windowed.metadata.window_end[idx],
        }


class EMGDataModule(LightningDataModule):
    """Wraps already-windowed partitions into DataLoaders (no windowing here)."""

    def __init__(
        self,
        windowed: WindowedPartitions,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        super().__init__()
        self.windowed = windowed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = EMGWindowDataset(self.windowed.train)
        self._val = EMGWindowDataset(self.windowed.val)
        self._test = EMGWindowDataset(self.windowed.test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
```

**(b)** `src/hand_gesture_recognition/neural/models/base.py` — update `_step` and `predict_step`:

```python
    def _step(self, batch, prefix: str) -> torch.Tensor:
        x = batch["x"]
        y = batch["y_true"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
        return loss

    # ... training_step/validation_step/test_step unchanged (they call _step) ...

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["x"]
        logits = self(x)
        return {
            "logits": logits,
            "y_true": batch["y_true"],
            "window_id": batch["window_id"],
            "trial_id": batch["trial_id"],
            "subject_id": batch["subject_id"],
        }
```

**(c)** `src/hand_gesture_recognition/neural/adapters/lightning.py` — replace the `to_prediction` method (drop `windowed` arg):

```python
    def to_prediction(self, raw_predictions, run_id: str, duration_s: float) -> Prediction:
        if raw_predictions:
            logits = torch.cat([p["logits"] for p in raw_predictions], dim=0)
            y_true = torch.cat([p["y_true"] for p in raw_predictions], dim=0).cpu().numpy()
            window_ids = np.concatenate([np.asarray(p["window_id"], dtype=object) for p in raw_predictions])
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = logits.argmax(dim=1).cpu().numpy()
        else:
            y_true = np.empty(0, dtype=int)
            window_ids = np.empty(0, dtype=object)
            probs = np.empty((0, 0), dtype=np.float32)
            y_pred = np.empty(0, dtype=int)

        return Prediction(
            run_id=run_id,
            sample_ids=window_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_score=probs,
            prediction_duration_s=duration_s,
            backend="neural",
        )
```

Remove the now-unused `windowed`/`WindowedDataset` references and update the docstring accordingly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_neural.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/neural/datamodules/emg.py src/hand_gesture_recognition/neural/models/base.py src/hand_gesture_recognition/neural/adapters/lightning.py tests/unit/test_neural.py
git commit -m refactor-neural-windowing
```

---

### Task 3: Runner + classical — window once, both backends consume it

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Modify: `src/hand_gesture_recognition/classical/adapters/sklearn.py`
- Test: `tests/unit/test_prediction.py` (update)

**Interfaces:**
- Consumes: `WindowedPartitions` (Task 1), `EMGDataModule(windowed=...)` (Task 2).
- Produces: runner windows each partition once; `SklearnPredictionAdapter.to_prediction(windowed, y_pred, y_score, run_id, duration_s)` sets `sample_ids = windowed.metadata.window_ids`.

- [ ] **Step 1: Update the failing test**

In `tests/unit/test_prediction.py`, update `_windowed` to include `window_ids` and change the assertion:

```python
def _windowed(n=10):
    return WindowedDataset(
        X=np.zeros((n, 40, 8), dtype=np.float32),
        y=np.arange(n) % 3,
        metadata=WindowMetadata(
            window_start=np.arange(n),
            window_end=np.arange(n) + 40,
            subject_ids=np.array([f"s{i % 2}" for i in range(n)]),
            trial_ids=np.array([f"t{i}" for i in range(n)]),
            window_ids=np.array([f"s{i % 2}_t{i}_window0" for i in range(n)]),
        ),
    )


def test_sklearn_adapter_builds_prediction():
    wd = _windowed()
    adapter = SklearnPredictionAdapter()
    pred = adapter.to_prediction(windowed=wd, y_pred=np.array([0] * 10), y_score=np.ones((10, 3)) / 3, run_id="r1", duration_s=0.1)
    assert pred.backend == "classical"
    assert pred.sample_ids.tolist() == [f"s{i % 2}_t{i}_window0" for i in range(10)]
```

(Add `window_ids` to the import from `data.windowing.base` and keep the rest of `_windowed` as-is.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_prediction.py -q`
Expected: FAIL — `sample_ids` currently equals `trial_ids` (assertion mismatch).

- [ ] **Step 3: Implement**

**(a)** `src/hand_gesture_recognition/classical/adapters/sklearn.py` — change `sample_ids`:

```python
            sample_ids=windowed.metadata.window_ids,
```

**(b)** `src/hand_gesture_recognition/runner/run.py` — introduce a windowing step in `run()` between `split` and `train`, and thread `windowed` into both backends. Change the current:

```python
            with stage(self.logger, "split"):
                split = split_dataset(dataset, cfg.split)
                ...
            with stage(self.logger, "train"):
                if cfg.backend == "classical":
                    predictions = self._train_and_predict_classical(split)
                else:
                    predictions = self._train_and_predict_neural(split)
```

to:

```python
            with stage(self.logger, "split"):
                split = split_dataset(dataset, cfg.split)
                ...
            with stage(self.logger, "windowing"):
                windowed = self._window(split)
            with stage(self.logger, "train"):
                if cfg.backend == "classical":
                    predictions = self._train_and_predict_classical(windowed)
                else:
                    predictions = self._train_and_predict_neural(windowed)
```

Add the `_window` helper (windows each partition exactly once):

```python
    def _window(self, split) -> WindowedPartitions:
        from hand_gesture_recognition.data.windowing.base import (
            WindowedPartitions,
            window_dataset,
        )

        return WindowedPartitions(
            train=window_dataset(split.train, self.cfg.windowing),
            val=window_dataset(split.val, self.cfg.windowing),
            test=window_dataset(split.test, self.cfg.windowing),
        )
```

**(c)** Rewrite `_train_and_predict_classical` to take `windowed` (drop the `WindowingTransformer` usage):

```python
    def _train_and_predict_classical(self, windowed: WindowedPartitions) -> PartitionPredictions:
        X_train, X_val, X_test = windowed.train.X, windowed.val.X, windowed.test.X
        for step_cfg in self.cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)
        for step_cfg in self.cfg.classical.features:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)

        model = instantiate(self.cfg.classical.model)
        t0 = time.perf_counter()
        model.fit(X_train, windowed.train.y)

        adapter = SklearnPredictionAdapter()

        def build(wd, X) -> Prediction:
            return adapter.to_prediction(
                windowed=wd,
                y_pred=model.predict(X),
                y_score=model.predict_proba(X),
                run_id=self.cfg.run_id,
                duration_s=0.0,
            )

        train_p = build(windowed.train, X_train)
        val_p = build(windowed.val, X_val)
        test_p = build(windowed.test, X_test)
        duration = time.perf_counter() - t0
        train_p.prediction_duration_s = duration
        val_p.prediction_duration_s = duration
        test_p.prediction_duration_s = duration
        return PartitionPredictions(train=train_p, val=val_p, test=test_p)
```

**(d)** Rewrite the tail of `_train_and_predict_neural` — remove the three `window_dataset` calls and the separate `window_dataset` import; build the DataModule from the shared `windowed`:

```python
        dm_partial = instantiate(self.cfg.neural.datamodule, _partial_=True)
        datamodule = dm_partial(windowed=windowed)
        model_partial = instantiate(self.cfg.neural.model, n_channels=n_channels, n_classes=n_classes, _partial_=True)
        model = model_partial(optimizer=self.cfg.neural.optimizer)
        callbacks = None
        if self.cfg.neural.early_stopping:
            callbacks = [instantiate(self.cfg.neural.early_stopping)]
        trainer_partial = instantiate(self.cfg.neural.trainer, _partial_=True)
        trainer = trainer_partial(callbacks=callbacks)

        t0 = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        train_raw = trainer.predict(model, dataloaders=datamodule.train_dataloader())
        val_raw = trainer.predict(model, dataloaders=datamodule.val_dataloader())
        test_raw = trainer.predict(model, dataloaders=datamodule.test_dataloader())
        duration = time.perf_counter() - t0

        adapter = LightningPredictionAdapter()
        return PartitionPredictions(
            train=adapter.to_prediction(raw_predictions=train_raw, run_id=self.cfg.run_id, duration_s=duration),
            val=adapter.to_prediction(raw_predictions=val_raw, run_id=self.cfg.run_id, duration_s=duration),
            test=adapter.to_prediction(raw_predictions=test_raw, run_id=self.cfg.run_id, duration_s=duration),
        )
```

Note: the neural config `datamodule` dict in `configs/neural/dnn.yaml`/`lstm.yaml` currently contains `batch_size`, `num_workers`, `shuffle` — these are fine (they map to `EMGDataModule` constructor params). The `datamodule` `_target_` stays `EMGDataModule`. `n_channels`/`n_classes` are still derived from `split`; keep those lines.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_prediction.py -q`
Expected: PASS.

Then run the full suite: `python -m pytest -q`
Expected: green (smoke + partition-evaluation + parity integration tests now run end-to-end through the single-windowing path).

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py src/hand_gesture_recognition/classical/adapters/sklearn.py tests/unit/test_prediction.py
git commit -m refactor-runner-single-windowing
```

---

### Task 4: Regression + alignment tests

**Files:**
- Modify: `tests/integration/test_cross_backend_parity.py` (update local helpers to the new interfaces)
- Modify: `tests/unit/test_neural.py` or new `tests/unit/test_windowing_alignment.py` (add alignment tests)
- Test: `tests/unit/test_windowing_alignment.py` (new)

**Interfaces:**
- Consumes: `window_dataset` (with `window_ids`), `WindowedPartitions`, `EMGDataModule`, both adapters (Tasks 1–3).

- [ ] **Step 1: Update `test_cross_backend_parity.py` helpers**

Rewrite `_classical_prediction` and `_neural_prediction` to use the new interfaces:

```python
def _classical_prediction(split, windowing):
    from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
    from hand_gesture_recognition.classical.models.random_forest import RandomForestEstimator
    from hand_gesture_recognition.data.windowing.base import window_dataset
    from hand_gesture_recognition.features.time_domain import TimeDomainFeatures

    train_wd = window_dataset(split.train, windowing)
    test_wd = window_dataset(split.test, windowing)
    feat = TimeDomainFeatures()
    X_train = feat.fit_transform(train_wd.X)
    X_test = feat.transform(test_wd.X)
    model = RandomForestEstimator(n_estimators=20, random_state=0)
    model.fit(X_train, train_wd.y)
    return SklearnPredictionAdapter().to_prediction(
        windowed=test_wd, y_pred=model.predict(X_test), y_score=model.predict_proba(X_test),
        run_id="parity", duration_s=0.0,
    )


def _neural_prediction(split, windowing):
    import lightning.pytorch as pl
    from hand_gesture_recognition.data.windowing.base import WindowedPartitions, window_dataset
    from hand_gesture_recognition.neural.adapters.lightning import LightningPredictionAdapter
    from hand_gesture_recognition.neural.datamodules.emg import EMGDataModule
    from hand_gesture_recognition.neural.models.dnn import DNNModule

    pl.seed_everything(0, workers=True)
    n_channels = len(split.train.metadata.channels)
    n_classes = len(split.train.metadata.class_names)
    windowed = WindowedPartitions(
        train=window_dataset(split.train, windowing),
        val=window_dataset(split.val, windowing),
        test=window_dataset(split.test, windowing),
    )
    dm = EMGDataModule(windowed=windowed, batch_size=32)
    model = DNNModule(n_channels=n_channels, n_classes=n_classes, hidden_dims=(16,))
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False,
                         enable_model_summary=False, enable_checkpointing=False, logger=False)
    trainer.fit(model, datamodule=dm)
    raw = trainer.predict(model, dataloaders=dm.test_dataloader())
    return LightningPredictionAdapter().to_prediction(raw_predictions=raw, run_id="parity", duration_s=0.0)
```

The existing assertions (`set(classical.__dict__.keys()) == set(neural.__dict__.keys())`, `np.array_equal(classical.y_true, neural.y_true)`, `np.array_equal(classical.sample_ids, neural.sample_ids)`) must still hold — both now carry `window_ids`.

- [ ] **Step 2: Add alignment tests**

Create `tests/unit/test_windowing_alignment.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.base import window_dataset


def _ds(lengths=(100, 60)):
    trials = [np.arange(l, dtype=np.float32)[:, None].repeat(2, axis=1) for l in lengths]
    return Dataset(
        trials=trials,
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s2"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.zeros(2, dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
        metadata=DatasetMetadata(name="x", version="0", sampling_rate=200.0,
                                 channels=["c1", "c2"], class_names=["a", "b"], content_hash="h"),
    )


def test_window_count_matches_X():
    ds = _ds(lengths=(100, 100))
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))
    assert wd.X.shape[0] == 14
    assert len(wd.metadata.window_ids) == 14


def test_metadata_alignment_per_window():
    ds = _ds(lengths=(100, 100))
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))
    for i in range(wd.X.shape[0]):
        assert wd.metadata.window_start[i] == i * 10
        assert wd.metadata.window_end[i] == i * 10 + 40
        assert wd.metadata.window_ids[i].endswith(f"_window{i % 7}")
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_windowing_alignment.py tests/integration/test_cross_backend_parity.py -q`
Expected: PASS.

Then full suite: `python -m pytest -q`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_cross_backend_parity.py tests/unit/test_windowing_alignment.py
git commit -m test-windowing-alignment
```

---

## Self-Review Notes (for the implementer)

- Spec coverage: window_ids (Task 1), WindowedPartitions (Task 1), EMGDataModule no-windowing + metadata-preserving Dataset (Task 2), predict_step metadata (Task 2), adapter no-windowed (Task 2), runner single windowing + classical (Task 3), regression/alignment tests (Task 4). All "Definition of done" items covered.
- Full suite is expected to be red after Task 2 (runner still uses the old `EMGDataModule`/adapter interfaces) and green again after Task 3. This is a tightly-coupled refactor; the final green is verified in Task 4.
- `WindowingTransformer` remains in `classical/pipelines/windowing.py` (used by `tests/datasets/test_classical_full.py` and `test_classical_pipeline.py`) but is no longer called by the runner — it is not a "separate windowing implementation", just an unused-by-runner wrapper.
