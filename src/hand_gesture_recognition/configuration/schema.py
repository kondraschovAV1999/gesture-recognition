from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hand_gesture_recognition.configuration.errors import ConfigValidationError

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("classical", "neural")


@dataclass
class WindowingConfig:
    window_size_samples: Optional[int] = None
    window_size_seconds: Optional[float] = None
    stride_samples: Optional[int] = None
    stride_seconds: Optional[float] = None
    overlap_percent: Optional[float] = None
    label_granularity: str = "per_window"  # per_window | sequence
    padding: str = "none"  # none | zero

    def resolve(self, sampling_rate: float) -> Tuple[int, int]:
        """Return (window_size_samples, stride_samples), deriving/validating.

        Fails loudly if two representations of the same quantity disagree.
        """
        if sampling_rate <= 0:
            raise ConfigValidationError(
                f"sampling_rate must be > 0, got {sampling_rate}"
            )

        ws = self.window_size_samples
        if self.window_size_seconds is not None:
            derived = round(self.window_size_seconds * sampling_rate)
            if ws is not None and ws != derived:
                raise ConfigValidationError(
                    f"window_size_samples={ws} != window_size_seconds="
                    f"{self.window_size_seconds}s * {sampling_rate}Hz = {derived}"
                )
            ws = derived
        if ws is None:
            raise ConfigValidationError(
                "window size must be given as window_size_samples or window_size_seconds"
            )
        if ws < 1:
            raise ConfigValidationError(f"window_size_samples must be >= 1, got {ws}")

        stride = self.stride_samples
        if self.stride_seconds is not None:
            derived = round(self.stride_seconds * sampling_rate)
            if stride is not None and stride != derived:
                raise ConfigValidationError(
                    f"stride_samples={stride} != stride_seconds="
                    f"{self.stride_seconds}s * {sampling_rate}Hz = {derived}"
                )
            stride = derived
        if stride is None:
            if self.overlap_percent is not None:
                stride = round(ws * (1.0 - self.overlap_percent / 100.0))
            else:
                stride = ws  # non-overlapping default
        if stride < 1:
            raise ConfigValidationError(f"stride must be >= 1, got {stride}")
        return ws, stride


@dataclass
class GestureCropConfig:
    enabled: bool = False
    pad_samples: int = 100  # symmetric context kept before AND after the gesture
    no_gesture_max_length_samples: Optional[int] = None  # None = auto-derive


@dataclass
class DatasetConfig:
    name: str = "epn612"
    path: str = "resources/data/EPN612"
    sampling_rate: float = 200.0
    loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.data.loaders.epn612.EPN612Loader"
        }
    )


@dataclass
class SplitConfig:
    strategy: str = (
        "subject_wise"  # random | stratified | subject_wise | session_wise | predefined
    )
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    split_on: str = "subject_ids"  # which Dataset field carries group identity

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.0 < self.train_ratio < 1.0):
            raise ConfigValidationError(
                f"train_ratio must be in (0,1), got {self.train_ratio}"
            )
        if abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(f"split ratios must sum to 1.0, got {total}")
        if self.strategy not in (
            "random",
            "stratified",
            "subject_wise",
            "session_wise",
            "predefined",
        ):
            raise ConfigValidationError(f"unknown split strategy: {self.strategy}")


@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    features: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator",
            "n_estimators": 100,
            "random_state": 42,
        }
    )


@dataclass
class NeuralConfig:
    datamodule: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.neural.datamodules.emg.EMGDataModule",
            "batch_size": 32,
            "num_workers": 0,
        }
    )
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.neural.models.dnn.DNNModule",
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


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "balanced_accuracy", "macro_f1"]
    )
    average: str = "macro"
    exclude_classes: List[int] = field(default_factory=list)


@dataclass
class AnalyticsConfig:
    enabled: bool = True
    run_qc: bool = True


@dataclass
class TrackingConfig:
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "hand_gesture_recognition"


@dataclass
class ReportingConfig:
    formats: List[str] = field(default_factory=lambda: ["json", "html"])
    visualize: bool = True


@dataclass
class ExperimentConfig:
    run_id: Optional[str] = None
    backend: str = "classical"
    seed: int = 42
    deterministic: bool = False
    schema_version: str = SCHEMA_VERSION
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    signal_processing: List[Dict[str, Any]] = field(default_factory=list)
    gesture_crop: GestureCropConfig = field(default_factory=GestureCropConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.backend not in SUPPORTED_BACKENDS:
        raise ConfigValidationError(
            f"backend={cfg.backend!r} not supported (expected one of {SUPPORTED_BACKENDS})"
        )
    if cfg.seed is None:
        raise ConfigValidationError("seed must be set for reproducibility")
    cfg.split.validate()
    # Validate windowing consistency against the configured sampling rate.
    cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)
    if cfg.schema_version != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"schema_version={cfg.schema_version!r} != {SCHEMA_VERSION!r}"
        )
    return cfg
