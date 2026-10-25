from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
