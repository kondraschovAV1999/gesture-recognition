# Parallel-Conv Neural Feature Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 2D Inception-style parallel-conv neural feature extractor (`ParallelConvModule`) with an FC prediction head, consuming handcrafted small-window features reshaped to `(channels, small_windows, features)`, wired into the two-level neural path.

**Architecture:** `ParallelConvLayer` + `FeatureExtractionModule` (2D backbone) + `ParallelConvModule(ClassifierModule)` (backbone → `AdaptiveAvgPool2d(1)` → `Linear(4*out_channels, n_classes)`). A dedicated `ParallelConvGroupedEMGDataset`/`DataModule` reshapes `(n_small, F*C) → (C, n_small, F)`. `NeuralTrainingStrategy` selects this datamodule (and `n_channels = C`) when the model `_target_` is `ParallelConvModule`.

**Tech Stack:** Python 3.10, PyTorch, Lightning, hydra, pytest.

## Global Constraints

- Feature-major reshape: flattened features are ordered `[feat1_ch1…feat1_chC, feat2_ch1…]`, so `(n_small, F*C).reshape(n_small, F, C).transpose(2, 0, 1)` yields `(C, n_small, F)`.
- `F = n_features // n_channels` (features per small window per channel).
- `ParallelConvModule` subclasses `ClassifierModule`, calling `super().__init__(n_classes=n_classes, optimizer=optimizer)` (the base now requires `n_classes`).
- Input to the model is `(batch, C, H=small_windows, W=features)`; the model is H/W-agnostic via `Conv2d` + `AdaptiveAvgPool2d`.
- Do NOT commit `configs/*.yaml` that are user WIP (`config.yaml`, `feature_extraction/default.yaml`, `neural/lstm.yaml`, `neural/*.yaml`, `windowing/*.yaml`, `signal_processing/emg.yaml`). The NEW `configs/neural/parallel_conv.yaml` is safe to commit (it is a new file, not user WIP).

---

### Task 1: `parallel_conv.py` model + unit tests

**Files:**
- Create: `src/hand_gesture_recognition/pipeline/models/neural/parallel_conv.py`
- Test: `tests/unit/test_parallel_conv.py` (new)

**Interfaces:**
- Consumes: `ClassifierModule` from `hand_gesture_recognition.pipeline.models.neural.base`.
- Produces: `ParallelConvLayer(in_channels, bottleneck_channels=16, out_channels=18)`, `FeatureExtractionModule(in_channels, out_channels=18, lrn_size=5, alpha=1e-4, beta=0.75, k=1.0)`, `ParallelConvModule(n_channels, n_classes, out_channels=18, optimizer=None)`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_parallel_conv.py`:

```python
import torch

from hand_gesture_recognition.pipeline.models.neural.parallel_conv import (
    ParallelConvModule,
    ParallelConvLayer,
    FeatureExtractionModule,
)


def test_parallel_conv_layer_output_channels():
    layer = ParallelConvLayer(in_channels=8)
    x = torch.randn(4, 8, 23, 4)
    out = layer(x)
    assert out.shape == (4, 4 * 18, 23, 4)  # 4 branches x 18 = 72 channels


def test_feature_extraction_module_shape():
    m = FeatureExtractionModule(in_channels=8)
    x = torch.randn(4, 8, 23, 4)
    out = m(x)
    assert out.shape == (4, 72, 23, 4)


def test_parallel_conv_module_forward_logits():
    model = ParallelConvModule(n_channels=8, n_classes=3)
    x = torch.randn(4, 8, 23, 4)
    logits = model(x)
    assert logits.shape == (4, 3)


def test_parallel_conv_module_has_f1_metric():
    model = ParallelConvModule(n_channels=8, n_classes=3)
    assert model.val_f1 is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_parallel_conv.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the module**

Create `src/hand_gesture_recognition/pipeline/models/neural/parallel_conv.py`:

```python
from __future__ import annotations

from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_gesture_recognition.pipeline.models.neural.base import ClassifierModule


class ParallelConvLayer(nn.Module):
    """Inception-style parallel conv layer: four branches concatenated on channels.

    Branches: 1x1->ReLU->3x3, 1x1, 3x3 MaxPool(stride=1)->1x1, 1x1->ReLU->5x5.
    All preserve spatial dims; output channels = 4 * out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int = 16,
        out_channels: int = 18,
    ):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, padding=1),
        )
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class FeatureExtractionModule(nn.Module):
    """Stacked parallel-conv feature extractor with residual connections + LRN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 18,
        lrn_size: int = 5,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
    ):
        super().__init__()
        width = 4 * out_channels
        self.block1 = ParallelConvLayer(in_channels=in_channels, out_channels=out_channels)
        self.block2 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block3 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block4 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block5 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.block6 = ParallelConvLayer(in_channels=width, out_channels=out_channels)
        self.cross_channel_norm = nn.LocalResponseNorm(
            size=lrn_size, alpha=alpha, beta=beta, k=k
        )
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.block1(x))
        res1 = x
        x = F.relu(self.block2(x))
        x = self.block3(x)
        x = F.relu(x + res1)
        res2 = x
        x = F.relu(self.block4(x))
        x = self.block5(x)
        x = F.relu(x + res2)
        x = self.block6(x)
        return self.cross_channel_norm(x)


class ParallelConvModule(ClassifierModule):
    """2D parallel-conv feature extractor + FC head.

    Input ``(batch, channels, small_windows, features)``; output logits
    ``(batch, n_classes)``. The shared ``ClassifierModule`` base provides
    cross-entropy loss, argmax accuracy, macro-F1, optimizer, and early-stopping.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        out_channels: int = 18,
        optimizer: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(n_classes=n_classes, optimizer=optimizer)
        self.features = FeatureExtractionModule(
            in_channels=n_channels, out_channels=out_channels
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (N, 4*out_channels, H, W)
        x = self.pool(x).flatten(1)   # (N, 4*out_channels)
        return self.head(x)           # (N, n_classes)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_parallel_conv.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/models/neural/parallel_conv.py tests/unit/test_parallel_conv.py
git commit -m "feat-parallel-conv-neural-feature-extractor-model"
```

---

### Task 2: 2D grouped datamodule + unit tests

**Files:**
- Modify: `src/hand_gesture_recognition/pipeline/models/neural/datamodule.py`
- Test: `tests/unit/test_parallel_conv_datamodule.py` (new)

**Interfaces:**
- Consumes: `FeatureExtractor` from `hand_gesture_recognition.pipeline.feature_extraction.features`.
- Produces: `ParallelConvGroupedEMGDataset(grouped, feature_specs, n_features, n_channels, max_small_per_big)` and `ParallelConvGroupedEMGDataModule(windowed, feature_specs, n_features, n_channels, max_small_per_big, batch_size=32, num_workers=0)`.

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_parallel_conv_datamodule.py`:

```python
import numpy as np

from hand_gesture_recognition.configuration.schema import (
    Dataset, DatasetMetadata, Trial, User, WindowingConfig,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    ParallelConvGroupedEMGDataset,
)

_TIME_SPEC = [
    {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
    }
]


def _grouped():
    meta = DatasetMetadata("x", "0", 200.0, ["c1", "c2"], ["noGesture", "fist"], "h")
    u = User(
        "s1",
        [
            Trial("t1", "s1", "fist", 0, 100, np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1)),
            Trial("t2", "s1", "noGesture", -1, -1, np.zeros((100, 2), np.float32)),
        ],
    )
    ds = Dataset(meta, [u])
    cfg = WindowingConfig(
        mode="two_level", window_size_samples=20, stride_samples=10,
        big_window_size_samples=60, big_stride_samples=30,
    )
    return get_windowing_strategy("two_level").window(ds, cfg, signal_steps=[])


def test_parallel_conv_dataset_reshapes_to_channels_first():
    grouped = _grouped()
    n_channels = grouped.big.X.shape[-1]  # 2
    max_small = max(len(s) for s in grouped.small_windows)
    n_features = 11 * n_channels  # 11 time-domain features x 2 channels = 22
    ds = ParallelConvGroupedEMGDataset(
        grouped, _TIME_SPEC, n_features=n_features,
        n_channels=n_channels, max_small_per_big=max_small,
    )
    item = ds[0]
    assert item["x"].shape == (n_channels, max_small, 11)  # (C, H, F)
    assert item["x"].dtype == np.float32 or item["x"].dtype == __import__("torch").float32
    assert int(item["y_true"].item()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_parallel_conv_datamodule.py -q`
Expected: FAIL (name not defined).

- [ ] **Step 3: Write the dataset + datamodule**

In `src/hand_gesture_recognition/pipeline/models/neural/datamodule.py`, add the import of `torch` (already present) and append after `GroupedEMGDataModule`:

```python
class ParallelConvGroupedEMGDataset(Dataset):
    """One item per big window: small-window features reshaped to (C, H, F).

    The handcrafted extractor emits feature-major ``(n_small, F*C)``; this
    reshapes to ``(channels, small_windows, features)`` so a 2D conv can treat
    the small-window axis as height and the feature axis as width.
    """

    def __init__(self, grouped, feature_specs, n_features, n_channels, max_small_per_big):
        self.y_true = grouped.y
        self.window_ids = grouped.metadata.window_ids
        self.trial_ids = grouped.metadata.trial_ids
        self.subject_ids = grouped.metadata.subject_ids
        self.n_channels = n_channels
        self.max_small_per_big = max_small_per_big
        self.f_per_channel = n_features // n_channels
        self.sequences = FeatureExtractor(feature_specs).transform_grouped(
            grouped.small_windows, max_small_per_big
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feats = self.sequences[idx]  # (max_small, F*C) feature-major
        x = feats.reshape(self.max_small_per_big, self.f_per_channel, self.n_channels)
        x = x.transpose(2, 0, 1)  # (C, max_small, F)
        return {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "y_true": torch.as_tensor(self.y_true[idx], dtype=torch.long),
            "window_id": self.window_ids[idx],
            "trial_id": self.trial_ids[idx],
            "subject_id": self.subject_ids[idx],
        }


class ParallelConvGroupedEMGDataModule(LightningDataModule):
    """Wraps two-level partitions into DataLoaders of (C, H, F) feature images."""

    def __init__(
        self,
        windowed,
        feature_specs,
        n_features,
        n_channels,
        max_small_per_big,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self.windowed = windowed
        self.feature_specs = feature_specs
        self.n_features = n_features
        self.n_channels = n_channels
        self.max_small_per_big = max_small_per_big
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train = None
        self._val = None
        self._test = None

    def _make(self, grouped):
        return ParallelConvGroupedEMGDataset(
            grouped, self.feature_specs, self.n_features, self.n_channels,
            self.max_small_per_big,
        )

    def setup(self, stage=None):
        self._train = self._make(self.windowed.train)
        self._val = self._make(self.windowed.val)
        self._test = self._make(self.windowed.test)

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self._test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return self.test_dataloader()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_parallel_conv_datamodule.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/models/neural/datamodule.py tests/unit/test_parallel_conv_datamodule.py
git commit -m "feat-parallel-conv-grouped-datamodule"
```

---

### Task 3: Strategy wiring + config + integration test + docs

**Files:**
- Modify: `src/hand_gesture_recognition/pipeline/models/neural/strategy.py`
- Create: `configs/neural/parallel_conv.yaml`
- Test: `tests/integration/test_parallel_conv.py` (new)
- Modify: `docs/experiments/configuration.md`

**Interfaces:**
- Consumes: `ParallelConvGroupedEMGDataModule` (Task 2), `ParallelConvModule` (Task 1).

- [ ] **Step 1: Wire the strategy**

In `src/hand_gesture_recognition/pipeline/models/neural/strategy.py`, add the import:

```python
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    GroupedEMGDataModule,
    ParallelConvGroupedEMGDataModule,
)
```

and replace the `if isinstance(windowed, GroupedWindowedPartitions):` block's body so it branches on the model target:

```python
        if isinstance(windowed, GroupedWindowedPartitions):
            n_features = self._compute_n_features(feature_specs, windowed)
            max_small_per_big = max(
                (
                    len(smalls)
                    for part in ("train", "val", "test")
                    for smalls in getattr(windowed, part).small_windows
                ),
                default=1,
            )
            model_target = cfg.neural.model.get("_target_", "")
            if "ParallelConvModule" in model_target:
                # 2D conv consumes (C, H, F): in_channels = EMG channels.
                n_channels = windowed.train.big.X.shape[-1]
                datamodule = ParallelConvGroupedEMGDataModule(
                    windowed=windowed,
                    feature_specs=feature_specs,
                    n_features=n_features,
                    n_channels=n_channels,
                    max_small_per_big=max_small_per_big,
                    batch_size=cfg.neural.datamodule.get("batch_size", 32),
                    num_workers=cfg.neural.datamodule.get("num_workers", 0),
                )
            else:
                datamodule = GroupedEMGDataModule(
                    windowed=windowed,
                    feature_specs=feature_specs,
                    n_features=n_features,
                    max_small_per_big=max_small_per_big,
                    batch_size=cfg.neural.datamodule.get("batch_size", 32),
                    num_workers=cfg.neural.datamodule.get("num_workers", 0),
                )
                n_channels = n_features
        else:
            n_channels = windowed.train.X.shape[-1]
            dm_partial = instantiate(cfg.neural.datamodule, _partial_=True)
            datamodule = dm_partial(windowed=windowed)
```

- [ ] **Step 2: Create the config**

Create `configs/neural/parallel_conv.yaml`:

```yaml
# @package neural
datamodule:
  _target_: hand_gesture_recognition.pipeline.models.neural.datamodule.EMGDataModule
  batch_size: 32
  num_workers: 0
model:
  _target_: hand_gesture_recognition.pipeline.models.neural.parallel_conv.ParallelConvModule
  out_channels: 18
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 0.0
early_stopping:
  _target_: lightning.pytorch.callbacks.EarlyStopping
  monitor: val_f1
  patience: 5
  mode: max
trainer:
  _target_: lightning.pytorch.Trainer
  max_epochs: 100
  accelerator: auto
  enable_progress_bar: true
  enable_model_summary: true
  enable_checkpointing: false
  logger: true
```

- [ ] **Step 3: Write the integration test**

Create `tests/integration/test_parallel_conv.py`:

```python
from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_parallel_conv_two_level_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "backend=neural",
            "neural=parallel_conv",
            "windowing=default",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
```

Note: the `parallel_conv` model only applies in two-level mode (the strategy branch). Since the smoke test above uses `windowing=default` (single), it exercises the single-level path with `ParallelConvModule` — which still works because the model input in single-level is `(batch, time, channels)` (3D), NOT `(batch, C, H, W)` (4D). To avoid a shape mismatch, run the smoke in **two-level** mode instead:

```python
def test_parallel_conv_two_level_smoke(tmp_path, monkeypatch):
    cfg = load_config(
        mode="compose",
        overrides=[
            "dataset.name=synthetic",
            "baseline=none",
            "split.strategy=random",
            "backend=neural",
            "neural=parallel_conv",
            "windowing=two_level",
            "windowing.big_window_size_samples=60",
            "windowing.big_stride_samples=30",
            "windowing.window_size_samples=20",
            "windowing.stride_samples=10",
            "neural.trainer.max_epochs=1",
            "reporting.formats=[json]",
        ],
    )
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    monkeypatch.setenv("HAND_GESTURE_RECOGNITION_OUTPUT", str(tmp_path / "output"))
    runner = ExperimentRunner(cfg)
    result = runner.run()
    assert "accuracy" in result.metrics
```

(Use the two-level variant. The synthetic dataset length 100 with big_ws=60/stride=30 gives enough big windows.)

- [ ] **Step 4: Run the integration test**

Run: `python -m pytest tests/integration/test_parallel_conv.py -q`
Expected: PASS.

- [ ] **Step 5: Update docs**

In `docs/experiments/configuration.md`, in the `neural` section (near the model list `dnn/lstm/cnn/cnn_lstm`), add one line noting `parallel_conv` (2D parallel-conv feature extractor over handcrafted small-window features, two-level only).

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/pipeline/models/neural/strategy.py configs/neural/parallel_conv.yaml tests/integration/test_parallel_conv.py docs/experiments/configuration.md
git commit -m "feat-wire-parallel-conv-neural-model"
```

---

## Self-Review

- **Spec coverage:** model → Task 1; datamodule → Task 2; strategy/config/integration/docs → Task 3.
- **Placeholder scan:** no TBD; all code specified.
- **Type consistency:** `ParallelConvModule(n_channels, n_classes, out_channels=18, optimizer=None)` (Task 1) matches strategy instantiation `n_channels=..., n_classes=...` (Task 3). `ParallelConvGroupedEMGDataModule(windowed, feature_specs, n_features, n_channels, max_small_per_big, batch_size, num_workers)` identical in Tasks 2 and 3. `F = n_features // n_channels` consistent throughout.
