import pytest
from hand_gesture_recognition.configuration.schema import (
    DatasetConfig,
    ExperimentConfig,
    WindowingConfig,
    validate_experiment_config,
)
from hand_gesture_recognition.configuration.errors import ConfigValidationError


def _base_experiment() -> ExperimentConfig:
    return ExperimentConfig(
        run_id="test",
        backend="classical",
        seed=42,
        deterministic=False,
        windowing=WindowingConfig(window_size_samples=40, stride_samples=10),
    )


def test_window_size_samples_and_seconds_consistent():
    cfg = WindowingConfig(
        window_size_samples=40, window_size_seconds=0.2, stride_samples=10
    )
    resolved = cfg.resolve(sampling_rate=200)
    assert resolved == (40, 10)


def test_window_size_inconsistency_raises():
    cfg = WindowingConfig(window_size_samples=30, window_size_seconds=0.2)
    with pytest.raises(ConfigValidationError):
        cfg.resolve(sampling_rate=200)


def test_stride_seconds_derived_from_samples():
    cfg = WindowingConfig(window_size_samples=40, stride_seconds=0.05)
    resolved = cfg.resolve(sampling_rate=200)
    assert resolved == (40, 10)


def test_experiment_validation_passes_on_defaults():
    cfg = _base_experiment()
    out = validate_experiment_config(cfg)
    assert out.backend == "classical"
    assert out.schema_version == "1.0"


def test_experiment_validation_rejects_bad_backend():
    cfg = _base_experiment()
    cfg.backend = "unsupported"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_validation_uses_configured_sampling_rate():
    cfg = ExperimentConfig(
        run_id="test",
        backend="classical",
        seed=42,
        deterministic=False,
        dataset=DatasetConfig(sampling_rate=100.0),
        windowing=WindowingConfig(
            window_size_samples=40, window_size_seconds=0.4, stride_samples=10
        ),
    )
    # 0.4s * 100 Hz == 40 samples -> consistent ONLY at the configured 100 Hz;
    # at a hardcoded 200 Hz it would be 80 samples and this must raise.
    out = validate_experiment_config(cfg)
    assert out.windowing.resolve(sampling_rate=100.0) == (40, 10)
