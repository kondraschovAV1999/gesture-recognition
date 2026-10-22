import pytest

from hand_gesture_recognition.configuration.errors import (
    ConfigValidationError,
    ConfigReplayError,
)
from hand_gesture_recognition.configuration.loader import (
    load_config,
    save_resolved_config,
)


def test_compose_mode_returns_validated_config():
    cfg = load_config(mode="compose", overrides=["seed=42", "backend=classical"])
    assert cfg.backend == "classical"
    assert cfg.seed == 42
    assert cfg.schema_version == "1.0"


def test_compose_mode_rejects_invalid_override():
    with pytest.raises(ConfigValidationError):
        load_config(mode="compose", overrides=["backend=unsupported"])


def test_compose_mode_rejects_inconsistent_windowing():
    with pytest.raises(ConfigValidationError):
        load_config(
            mode="compose",
            overrides=[
                "windowing.window_size_samples=30",
                "windowing.window_size_seconds=0.2",
            ],
        )


def test_replay_mode_roundtrips(tmp_path):
    cfg = load_config(mode="compose", overrides=["seed=7"])
    p = tmp_path / "resolved.yaml"
    save_resolved_config(cfg, str(p))
    replayed = load_config(mode="replay", resolved_path=str(p))
    assert replayed.seed == 7
    assert replayed == cfg


def test_replay_mode_missing_file_raises(tmp_path):
    with pytest.raises(ConfigReplayError):
        load_config(mode="replay", resolved_path=str(tmp_path / "nope.yaml"))


def test_gesture_crop_config_default_enabled():
    cfg = load_config(mode="compose")
    assert cfg.gesture_crop.enabled is True
    assert cfg.gesture_crop.no_gesture_max_length_samples is None


def test_gesture_crop_config_can_be_disabled():
    cfg = load_config(mode="compose", overrides=["gesture_crop=none"])
    assert cfg.gesture_crop.enabled is False


def test_features_resolves_frequency():
    cfg = load_config(
        mode="compose",
        overrides=["features.feature_type=frequency_domain", "backend=classical"],
    )
    specs = cfg.features.resolve_features(cfg.dataset.sampling_rate)
    targets = [f["_target_"] for f in specs]
    assert any("FrequencyDomainFeatures" in t for t in targets)
    assert specs[0]["sampling_rate"] == cfg.dataset.sampling_rate


def test_features_resolves_combined():
    cfg = load_config(
        mode="compose",
        overrides=["features.feature_type=combined", "backend=classical"],
    )
    specs = cfg.features.resolve_features(cfg.dataset.sampling_rate)
    targets = [f["_target_"] for f in specs]
    assert any("TimeDomainFeatures" in t for t in targets)
    assert any("FrequencyDomainFeatures" in t for t in targets)


def test_features_rejects_unknown():
    with pytest.raises(ConfigValidationError):
        load_config(
            mode="compose",
            overrides=["features.feature_type=bogus", "backend=classical"],
        )


def test_features_defaults_to_time_domain():
    cfg = load_config(mode="compose")
    assert cfg.features.feature_type == "time_domain"


def test_compose_defaults_to_recognition_task():
    cfg = load_config(mode="compose")
    assert cfg.task.name == "recognition"


def test_compose_loads_classification_task():
    cfg = load_config(mode="compose", overrides=["task=classification"])
    assert cfg.task.name == "classification"


def test_compose_loads_two_level_windowing():
    cfg = load_config(mode="compose", overrides=["windowing=two_level"])
    assert cfg.windowing.mode == "two_level"
    assert cfg.windowing.big_window_size_samples is not None


def test_compose_loads_per_user_baseline():
    cfg = load_config(
        mode="compose", overrides=["baseline=per_user", "split.strategy=trial_wise"]
    )
    assert cfg.baseline.mode == "per_user"
    assert cfg.baseline.enabled is True
