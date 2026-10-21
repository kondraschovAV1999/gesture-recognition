from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hand_gesture_recognition.configuration.errors import ConfigValidationError

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("classical", "neural")
SUPPORTED_TASKS = ("classification", "recognition")


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
class BaselineConfig:
    enabled: bool = False


@dataclass
class DatasetConfig:
    name: str = "epn612"
    path: str = "resources/data/EPN612"
    sampling_rate: float = 200.0
    include_no_gesture: bool = True
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
            "trial_wise",
            "predefined",
        ):
            raise ConfigValidationError(f"unknown split strategy: {self.strategy}")


@dataclass
class CrossValidationConfig:
    enabled: bool = False
    method: str = "kfold"  # kfold | loso
    n_splits: int = 5  # kfold only
    group_folds: bool = True  # kfold only: subject-wise vs trial-level
    include_val: bool = True
    val_ratio: float = 0.2  # fraction of the training fold carved into val
    seed: int = 42
    report_per_fold: bool = True

    def validate(self) -> None:
        if self.method not in ("kfold", "loso"):
            raise ConfigValidationError(f"unknown cv method: {self.method!r}")
        if self.method == "kfold" and self.n_splits < 2:
            raise ConfigValidationError(
                f"cv.n_splits must be >= 2, got {self.n_splits}"
            )
        if not (0.0 < self.val_ratio < 1.0):
            raise ConfigValidationError(
                f"cv.val_ratio must be in (0,1), got {self.val_ratio}"
            )


@dataclass
class VisualizationSelectionConfig:
    gestures: List[str] = field(default_factory=list)  # [] = all classes
    channels: List[int] = field(default_factory=list)  # [] = all channels (1-based)
    by_gesture: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class FeatureOverWindowsConfig:
    enabled: bool = True
    features: List[str] = field(default_factory=list)  # names; [] = all
    max_windows: int = 0  # 0 = all


@dataclass
class VisualizationConfig:
    enabled: bool = True
    consistent_scale: bool = True
    selection: VisualizationSelectionConfig = field(
        default_factory=VisualizationSelectionConfig
    )
    stage_progression: bool = True
    feature_over_windows: FeatureOverWindowsConfig = field(
        default_factory=FeatureOverWindowsConfig
    )


_CLASSICAL_FEATURE_TYPES = ("time_domain", "frequency_domain", "combined", "custom")

_TIME_DOMAIN_FEATURE = {
    "_target_": "hand_gesture_recognition.features.time_domain.TimeDomainFeatures",
}


def _frequency_domain_feature(sampling_rate: float) -> Dict[str, Any]:
    return {
        "_target_": "hand_gesture_recognition.features.frequency_domain.FrequencyDomainFeatures",
        "sampling_rate": sampling_rate,
        "use_hann": True,
        "remove_dc": True,
    }


@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    feature_type: str = (
        "time_domain"  # time_domain | frequency_domain | combined | custom
    )
    # Only used when feature_type == "custom" (an explicit list of _target_ entries).
    features: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator",
            "n_estimators": 100,
            "random_state": 42,
        }
    )

    def resolve_features(self, sampling_rate: float) -> List[Dict[str, Any]]:
        """Return the concrete feature-extractor specs for this config.

        ``custom`` defers to the explicit ``features`` list; the other presets
        are built here so a single ``feature_type`` knob selects the
        representation (no per-combination config files needed). The sampling
        rate is injected from the dataset so the frequency extractor never
        needs it hardcoded in YAML.
        """
        if self.feature_type == "custom":
            return list(self.features)
        if self.feature_type == "time_domain":
            return [_TIME_DOMAIN_FEATURE]
        if self.feature_type == "frequency_domain":
            return [_frequency_domain_feature(sampling_rate)]
        if self.feature_type == "combined":
            return [_TIME_DOMAIN_FEATURE, _frequency_domain_feature(sampling_rate)]
        raise ConfigValidationError(
            f"classical.feature_type={self.feature_type!r} not in {_CLASSICAL_FEATURE_TYPES}"
        )

    def validate(self) -> None:
        if self.feature_type not in _CLASSICAL_FEATURE_TYPES:
            raise ConfigValidationError(
                f"classical.feature_type={self.feature_type!r} not in {_CLASSICAL_FEATURE_TYPES}"
            )
        if self.feature_type == "custom" and not self.features:
            raise ConfigValidationError(
                "classical.feature_type='custom' requires a non-empty classical.features list"
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


@dataclass
class TaskConfig:
    name: str = "recognition"  # "classification" | "recognition"


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
    task: TaskConfig = field(default_factory=TaskConfig)
    signal_processing: List[Dict[str, Any]] = field(default_factory=list)
    gesture_crop: GestureCropConfig = field(default_factory=GestureCropConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def validate_experiment_config(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.backend not in SUPPORTED_BACKENDS:
        raise ConfigValidationError(
            f"backend={cfg.backend!r} not supported (expected one of {SUPPORTED_BACKENDS})"
        )
    if cfg.task.name not in SUPPORTED_TASKS:
        raise ConfigValidationError(
            f"task.name={cfg.task.name!r} not in {SUPPORTED_TASKS}"
        )
    if cfg.task.name == "classification" and cfg.dataset.include_no_gesture:
        warnings.warn(
            "task=classification always excludes the no-gesture class; "
            "dataset.include_no_gesture is ignored in classification mode.",
            UserWarning,
            stacklevel=2,
        )
    if cfg.seed is None:
        raise ConfigValidationError("seed must be set for reproducibility")
    cfg.split.validate()
    cfg.classical.validate()
    cfg.cv.validate()
    # Validate windowing consistency against the configured sampling rate.
    cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)
    if not cfg.dataset.include_no_gesture and 0 in cfg.evaluation.exclude_classes:
        raise ConfigValidationError(
            "evaluation.exclude_classes=[0] is ambiguous in 5-class mode "
            "(label 0 is now 'fist'); remove it or set dataset.include_no_gesture=true"
        )
    if cfg.schema_version != SCHEMA_VERSION:
        raise ConfigValidationError(
            f"schema_version={cfg.schema_version!r} != {SCHEMA_VERSION!r}"
        )
    return cfg
