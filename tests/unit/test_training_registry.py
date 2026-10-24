from hand_gesture_recognition.pipeline.models.base import (
    TrainingStrategy,
    get_training_strategy,
    register_training,
)


def test_register_and_get_round_trip():
    @register_training("dummy")
    class Dummy(TrainingStrategy):
        pass

    strategy = get_training_strategy("dummy")
    assert isinstance(strategy, Dummy)


def test_get_unknown_raises_keyerror():
    try:
        get_training_strategy("unknown")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unregistered strategy")
