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
        load_config(mode="compose", overrides=["backend=neural"])


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
    assert replayed.backend == "classical"
    assert replayed == cfg


def test_replay_mode_missing_file_raises(tmp_path):
    with pytest.raises(ConfigReplayError):
        load_config(mode="replay", resolved_path=str(tmp_path / "nope.yaml"))
