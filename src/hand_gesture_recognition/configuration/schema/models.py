from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.pipeline.models.classical.random_forest.RandomForestEstimator",
            "n_estimators": 100,
            "random_state": 42,
        }
    )


@dataclass
class NeuralConfig:
    datamodule: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.pipeline.models.neural.datamodule.EMGDataModule",
            "batch_size": 32,
            "num_workers": 0,
        }
    )
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.pipeline.models.neural.dnn.DNNModule",
            "hidden_dims": [128, 64],
        }
    )
    optimizer: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "torch.optim.Adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
        }
    )
    early_stopping: Optional[Dict[str, Any]] = None
    trainer: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "lightning.pytorch.Trainer",
            "max_epochs": 5,
            "accelerator": "auto",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "enable_checkpointing": False,
            "logger": False,
        }
    )
