from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hand_gesture_recognition.configuration.errors import (
    ConfigReplayError,
    ConfigValidationError,
)
from hand_gesture_recognition.configuration.schema import (
    AnalyticsConfig,
    ClassicalConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ReportingConfig,
    SCHEMA_VERSION,
    SplitConfig,
    TrackingConfig,
    WindowingConfig,
    validate_experiment_config,
)

_CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs")
)


def _coerce(d: Dict[str, Any]) -> ExperimentConfig:
    """Build a fully-typed ExperimentConfig from a plain resolved dict.

    Nested dataclasses are constructed explicitly so the rest of the framework
    always receives typed sub-configs (DatasetConfig, SplitConfig, ...) rather
    than plain dicts.
    """
    return ExperimentConfig(
        run_id=d.get("run_id"),
        backend=d.get("backend", "classical"),
        seed=d.get("seed", 42),
        deterministic=d.get("deterministic", False),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
        dataset=DatasetConfig(**d.get("dataset", {})),
        split=SplitConfig(**d.get("split", {})),
        windowing=WindowingConfig(**d.get("windowing", {})),
        classical=ClassicalConfig(**d.get("classical", {})),
        evaluation=EvaluationConfig(**d.get("evaluation", {})),
        analytics=AnalyticsConfig(**d.get("analytics", {})),
        tracking=TrackingConfig(**d.get("tracking", {})),
        reporting=ReportingConfig(**d.get("reporting", {})),
    )


def _compose(overrides: List[str]) -> ExperimentConfig:
    with initialize_config_dir(version_base="1.3", config_dir=_CONFIGS_DIR):
        raw = compose(config_name="config", overrides=overrides)
    d = OmegaConf.to_container(raw, resolve=True)
    return validate_experiment_config(_coerce(d))


def _replay(resolved_path: str) -> ExperimentConfig:
    if not os.path.exists(resolved_path):
        raise ConfigReplayError(f"resolved config not found: {resolved_path}")
    raw = OmegaConf.load(resolved_path)
    d = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(d, dict):
        raise ConfigReplayError(f"resolved config {resolved_path} is not a mapping")
    return validate_experiment_config(_coerce(d))


def load_config(
    mode: str = "compose",
    overrides: Optional[List[str]] = None,
    resolved_path: Optional[str] = None,
) -> ExperimentConfig:
    """Single entry point for configuration resolution (compose or replay)."""
    overrides = overrides or []
    if mode == "compose":
        return _compose(overrides)
    if mode == "replay":
        if resolved_path is None:
            raise ConfigReplayError("replay mode requires resolved_path")
        return _replay(resolved_path)
    raise ConfigValidationError(f"unknown config load mode: {mode!r}")


def save_resolved_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), path)
