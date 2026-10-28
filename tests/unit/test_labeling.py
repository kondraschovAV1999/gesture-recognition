from hand_gesture_recognition.pipeline.feature_extraction.windowing.labeling import (
    ThresholdLabeling,
)


def test_majority_threshold_strict():
    l = ThresholdLabeling(0.5)
    assert l.is_rest(0, 4, 0, 10) is False  # 100% inside -> gesture
    assert l.is_rest(3, 7, 4, 10) is False  # 75% inside -> gesture
    assert l.is_rest(2, 6, 4, 10) is True  # 50% inside == threshold -> rest (strict)
    assert l.is_rest(0, 4, 8, 10) is True  # 0% inside -> rest


def test_threshold_zero_reproduces_overlap():
    l = ThresholdLabeling(0.0)
    assert l.is_rest(2, 6, 4, 10) is False  # any overlap -> gesture
    assert (
        l.is_rest(0, 4, 4, 10) is True
    )  # ends exactly at g_start -> no overlap -> rest


def test_rest_trial_always_rest():
    assert ThresholdLabeling(0.5).is_rest(0, 4, -1, -1) is True
    assert ThresholdLabeling(0.0).is_rest(0, 4, -1, -1) is True
