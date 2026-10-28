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


def test_cv_config_defaults_disabled():
    from hand_gesture_recognition.configuration.schema import CrossValidationConfig

    cv = CrossValidationConfig()
    assert cv.enabled is False
    assert cv.method == "kfold"


def test_cv_config_validation_rejects_bad_method():
    from hand_gesture_recognition.configuration.schema import (
        CrossValidationConfig,
    )

    with pytest.raises(ConfigValidationError):
        CrossValidationConfig(method="bogus").validate()


def test_visualization_config_defaults():
    from hand_gesture_recognition.configuration.schema import (
        FeatureOverWindowsConfig,
        VisualizationConfig,
    )

    v = VisualizationConfig()
    assert v.enabled is True
    assert v.consistent_scale is True
    assert v.feature_over_windows.enabled is True
    assert v.feature_over_windows.features == []


def test_experiment_config_has_cv_and_visualization():
    cfg = _base_experiment()
    assert cfg.cv.enabled is False
    assert cfg.visualization.enabled is True


def test_task_defaults_to_recognition():
    cfg = _base_experiment()
    out = validate_experiment_config(cfg)
    assert out.task.name == "recognition"


def test_task_rejects_unknown_value():
    cfg = _base_experiment()
    cfg.task.name = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_task_classification_warns_when_include_no_gesture():
    cfg = _base_experiment()
    cfg.task.name = "classification"
    cfg.dataset.include_no_gesture = True
    with pytest.warns(UserWarning):
        validate_experiment_config(cfg)


def test_windowing_mode_defaults_single():
    from hand_gesture_recognition.configuration.schema import WindowingConfig

    assert WindowingConfig().mode == "single"


def test_windowing_rejects_bad_mode():
    cfg = _base_experiment()
    cfg.windowing.mode = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_two_level_requires_big_at_least_small():
    cfg = _base_experiment()
    cfg.windowing.mode = "two_level"
    cfg.windowing.window_size_samples = 40
    cfg.windowing.stride_samples = 10
    cfg.windowing.big_window_size_samples = 20  # < small
    cfg.windowing.big_stride_samples = 10
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_resolve_big_derives_and_validates():
    from hand_gesture_recognition.configuration.schema import WindowingConfig

    cfg = WindowingConfig(
        big_window_size_seconds=0.5, big_stride_seconds=0.1
    )  # 100 / 20 @ 200 Hz
    assert cfg.resolve_big(sampling_rate=200.0) == (100, 20)


def test_baseline_mode_defaults_global():
    from hand_gesture_recognition.configuration.schema import BaselineConfig

    assert BaselineConfig().mode == "global"


def test_baseline_rejects_bad_mode():
    cfg = _base_experiment()
    cfg.baseline.mode = "bogus"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_per_user_requires_trial_level_split():
    cfg = _base_experiment()
    cfg.baseline.enabled = True
    cfg.baseline.mode = "per_user"
    cfg.split.strategy = "group_wise"
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(cfg)


def test_feature_extraction_config_defaults():
    from hand_gesture_recognition.configuration.schema import (
        FeatureExtractionConfig,
        FREQUENCY_DOMAIN_FEATURE_NAMES,
        TIME_DOMAIN_FEATURE_NAMES,
    )

    f = FeatureExtractionConfig()
    assert f.feature_type == "time_domain"
    assert f.features == []
    assert TIME_DOMAIN_FEATURE_NAMES == [
        "rms",
        "std",
        "min",
        "max",
        "mean",
        "max_over_mean",
        "min_over_max",
        "mav",
        "iemg",
        "variance",
        "wl",
    ]
    assert FREQUENCY_DOMAIN_FEATURE_NAMES == [
        "total_power",
        "peak_frequency",
        "mean_frequency",
        "median_frequency",
    ]


def test_feature_extraction_validate_rejects_custom():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(feature_type="custom")
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_rejects_unknown_feature():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(feature_type="time_domain", features=["rms", "bogus"])
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_rejects_cross_domain_feature():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    # total_power is a frequency-domain feature, not time-domain.
    f = FeatureExtractionConfig(feature_type="time_domain", features=["total_power"])
    with pytest.raises(ConfigValidationError):
        f.validate()


def test_feature_extraction_validate_accepts_combined_names():
    from hand_gesture_recognition.configuration.schema import FeatureExtractionConfig

    f = FeatureExtractionConfig(
        feature_type="combined", features=["rms", "total_power"]
    )
    f.validate()  # must not raise


def test_windowing_label_threshold_default():
    from hand_gesture_recognition.configuration.schema import WindowingConfig

    assert WindowingConfig().label_threshold == 0.5


def test_evaluation_breakdown_by_parses_and_validates():
    from hand_gesture_recognition.configuration.schema import EvaluationConfig

    assert EvaluationConfig(breakdown_by=["subject_id"]).breakdown_by == ["subject_id"]
    cfg = _base_experiment()
    cfg.evaluation.breakdown_by = ["subject_id", "trial_id", "y_true"]
    out = validate_experiment_config(cfg)
    assert out.evaluation.breakdown_by == ["subject_id", "trial_id", "y_true"]
