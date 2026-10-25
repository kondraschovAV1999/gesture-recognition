from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hand_gesture_recognition.configuration.errors import ConfigValidationError
from hand_gesture_recognition.configuration.schema.dataset import DatasetConfig
from hand_gesture_recognition.configuration.schema.evaluation import EvaluationConfig
from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FeatureExtractionConfig,
)
from hand_gesture_recognition.configuration.schema.models import (
    ClassicalConfig,
    NeuralConfig,
)
from hand_gesture_recognition.configuration.schema.observability import (
    AnalyticsConfig,
    ReportingConfig,
    TrackingConfig,
    VisualizationConfig,
)
from hand_gesture_recognition.configuration.schema.preprocessing import (
    BaselineConfig,
    GestureCropConfig,
)
from hand_gesture_recognition.configuration.schema.splitting import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.configuration.schema.task import TaskConfig
from hand_gesture_recognition.configuration.schema.windowing import WindowingConfig

SCHEMA_VERSION = "1.0"
SUPPORTED_BACKENDS = ("classical", "neural")
SUPPORTED_TASKS = ("classification", "recognition")


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
    feature_extraction: FeatureExtractionConfig = field(
        default_factory=FeatureExtractionConfig
    )
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
    if cfg.baseline.mode not in ("global", "per_user"):
        raise ConfigValidationError(
            f"baseline.mode={cfg.baseline.mode!r} not in ('global', 'per_user')"
        )
    if (
        cfg.baseline.enabled
        and cfg.baseline.mode == "per_user"
        and cfg.split.strategy in ("group_wise", "session_wise", "predefined")
    ):
        raise ConfigValidationError(
            "baseline.mode=per_user requires a trial-level split "
            "(random/stratified/trial_wise)"
        )
    if cfg.seed is None:
        raise ConfigValidationError("seed must be set for reproducibility")
    cfg.split.validate()
    cfg.feature_extraction.validate()
    cfg.cv.validate()
    # Validate windowing consistency against the configured sampling rate.
    if cfg.windowing.mode not in ("single", "two_level"):
        raise ConfigValidationError(
            f"windowing.mode={cfg.windowing.mode!r} not in ('single', 'two_level')"
        )
    small_size, _ = cfg.windowing.resolve(sampling_rate=cfg.dataset.sampling_rate)
    if cfg.windowing.mode == "two_level":
        big_size, _ = cfg.windowing.resolve_big(sampling_rate=cfg.dataset.sampling_rate)
        if big_size < small_size:
            raise ConfigValidationError(
                f"two_level requires big window size ({big_size}) >= "
                f"small window size ({small_size})"
            )
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
