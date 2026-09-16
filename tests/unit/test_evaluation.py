import numpy as np

from hand_gesture_recognition.configuration.schema import EvaluationConfig
from hand_gesture_recognition.evaluation.base import evaluate
from hand_gesture_recognition.prediction.base import Prediction


def _pred():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    return Prediction(
        run_id="r1",
        sample_ids=np.arange(6),
        y_true=y_true,
        y_pred=y_pred,
        y_score=np.eye(3)[y_pred],
        prediction_duration_s=0.0,
        backend="classical",
    )


def test_evaluate_computes_accuracy_and_f1():
    cfg = EvaluationConfig(metrics=["accuracy", "macro_f1"])
    res = evaluate(_pred(), cfg, dataset_id="d1", split_info={"strategy": "random"})
    assert res.run_id == "r1"
    assert res.metrics["accuracy"] == 5 / 6
    assert 0.0 <= res.metrics["macro_f1"] <= 1.0
    assert res.confusion_matrix.shape == (3, 3)
    assert set(res.per_class.keys()) == {"0", "1", "2"}
