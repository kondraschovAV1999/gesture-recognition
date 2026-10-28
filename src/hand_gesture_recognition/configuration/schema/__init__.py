"""Consolidated schema package: config + data-contract dataclasses.

This package is a *leaf* of the dependency graph. Submodules may import only
from the stdlib, ``numpy``, ``hand_gesture_recognition.configuration.errors``,
and sibling modules within this package. Functional modules (``pipeline`` /
``utils`` / ``runner``) import dataclasses from here, never the reverse, so the
schema can never introduce an import cycle with the rest of the framework.

The ``__init__`` re-exports every public name so a single import path works::

    from hand_gesture_recognition.configuration.schema import Dataset, Prediction
"""

from hand_gesture_recognition.configuration.schema.dataset import (
    Dataset,
    DatasetConfig,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.configuration.schema.evaluation import (
    EvaluationConfig,
    EvaluationResult,
)
from hand_gesture_recognition.configuration.schema.experiment import (
    SCHEMA_VERSION,
    SUPPORTED_BACKENDS,
    SUPPORTED_TASKS,
    ExperimentConfig,
    validate_experiment_config,
)
from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
    FeatureExtractionConfig,
    TIME_DOMAIN_FEATURE_NAMES,
)
from hand_gesture_recognition.configuration.schema.models import (
    ClassicalConfig,
    NeuralConfig,
)
from hand_gesture_recognition.configuration.schema.observability import (
    AnalyticsConfig,
    FeatureOverWindowsConfig,
    ReportingConfig,
    TrackingConfig,
    VisualizationConfig,
    VisualizationSelectionConfig,
)
from hand_gesture_recognition.configuration.schema.prediction import (
    PartitionPredictions,
    Prediction,
)
from hand_gesture_recognition.configuration.schema.preprocessing import (
    BaselineConfig,
    GestureCropConfig,
)
from hand_gesture_recognition.configuration.schema.splitting import (
    CrossValidationConfig,
    DataSplit,
    SplitConfig,
)
from hand_gesture_recognition.configuration.schema.task import TaskConfig
from hand_gesture_recognition.configuration.schema.windowing import (
    GroupedWindowedDataset,
    GroupedWindowedPartitions,
    WindowingConfig,
    WindowMetadata,
    WindowedDataset,
    WindowedPartitions,
)

__all__ = [
    "AnalyticsConfig",
    "BaselineConfig",
    "ClassicalConfig",
    "CrossValidationConfig",
    "DataSplit",
    "Dataset",
    "DatasetConfig",
    "DatasetMetadata",
    "EvaluationConfig",
    "EvaluationResult",
    "ExperimentConfig",
    "FREQUENCY_DOMAIN_FEATURE_NAMES",
    "FeatureExtractionConfig",
    "FeatureOverWindowsConfig",
    "GestureCropConfig",
    "GroupedWindowedDataset",
    "GroupedWindowedPartitions",
    "NeuralConfig",
    "PartitionPredictions",
    "Prediction",
    "ReportingConfig",
    "SCHEMA_VERSION",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_TASKS",
    "SplitConfig",
    "TaskConfig",
    "TIME_DOMAIN_FEATURE_NAMES",
    "TrackingConfig",
    "Trial",
    "User",
    "VisualizationConfig",
    "VisualizationSelectionConfig",
    "WindowingConfig",
    "WindowMetadata",
    "WindowedDataset",
    "WindowedPartitions",
    "validate_experiment_config",
]
