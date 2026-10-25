import numpy as np
import pytest

from hand_gesture_recognition.configuration.schema import EvaluationConfig
from hand_gesture_recognition.pipeline.evaluation.base import evaluate
from hand_gesture_recognition.pipeline.evaluation.grouped import evaluate_grouped
from hand_gesture_recognition.configuration.schema import Prediction


def _pred():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    return Prediction(
        run_id="r1",
        sample_ids=np.arange(6),
        subject_ids=np.array([""] * 6, dtype=object),
        trial_ids=np.array([""] * 6, dtype=object),
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


def test_evaluate_computes_per_class_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    pred = Prediction(
        run_id="r1",
        sample_ids=np.arange(6),
        subject_ids=np.array([""] * 6, dtype=object),
        trial_ids=np.array([""] * 6, dtype=object),
        y_true=y_true,
        y_pred=y_pred,
        y_score=np.eye(3)[y_pred],
        prediction_duration_s=0.0,
        backend="classical",
    )
    res = evaluate(
        pred,
        EvaluationConfig(metrics=["accuracy"]),
        dataset_id="d1",
        split_info={},
        class_names=["rest", "fist", "open"],
    )

    assert set(res.per_class.keys()) == {"rest", "fist", "open"}
    # Confusion matrix: [[2,1,0],[0,1,1],[0,0,1]]
    # class "rest" (0): TP=2, FP=0, FN=1, TN=3
    assert res.per_class["rest"]["tp"] == 2
    assert res.per_class["rest"]["fp"] == 0
    assert res.per_class["rest"]["fn"] == 1
    assert res.per_class["rest"]["tn"] == 3
    assert res.per_class["rest"]["precision"] == pytest.approx(1.0)  # 2/(2+0)
    assert res.per_class["rest"]["recall"] == pytest.approx(2 / 3)  # 2/(2+1)
    # class "open" (2): TP=1, FP=1, FN=0
    assert res.per_class["open"]["tp"] == 1
    assert res.per_class["open"]["fp"] == 1
    assert res.per_class["open"]["fn"] == 0


def test_evaluate_exclude_classes():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    pred = Prediction(
        run_id="r1",
        sample_ids=np.arange(6),
        subject_ids=np.array([""] * 6, dtype=object),
        trial_ids=np.array([""] * 6, dtype=object),
        y_true=y_true,
        y_pred=y_pred,
        y_score=np.eye(3)[y_pred],
        prediction_duration_s=0.0,
        backend="classical",
    )
    res = evaluate(
        pred,
        EvaluationConfig(metrics=["accuracy"], exclude_classes=[0]),
        dataset_id="d1",
        split_info={},
    )
    # true noGesture samples dropped -> y_true=[1,1,2], y_pred=[1,2,2]
    assert res.metrics["accuracy"] == pytest.approx(2 / 3)
    assert "0" not in res.per_class  # noGesture excluded from per-class


def test_evaluate_grouped_by_subject():
    p = Prediction(
        run_id="r",
        sample_ids=np.array(["a0", "a1", "b0"]),
        subject_ids=np.array(["A", "A", "B"]),
        trial_ids=np.array(["a", "a", "b"]),
        y_true=np.array([0, 0, 1]),
        y_pred=np.array([0, 1, 1]),
        y_score=None,
        prediction_duration_s=0.0,
        backend="classical",
    )
    out = evaluate_grouped(
        p,
        by="subject_id",
        config=EvaluationConfig(),
        dataset_id="d",
        class_names=["c0", "c1"],
    )
    assert set(out) == {"A", "B"}
    assert out["A"].metrics["accuracy"] == 0.5
    assert out["B"].metrics["accuracy"] == 1.0


def test_evaluate_grouped_empty_returns_empty_dict():
    p = Prediction(
        run_id="r",
        sample_ids=np.arange(0),
        subject_ids=np.array([], dtype=object),
        trial_ids=np.array([], dtype=object),
        y_true=np.array([], dtype=int),
        y_pred=np.array([], dtype=int),
        y_score=None,
        prediction_duration_s=0.0,
        backend="classical",
    )
    assert (
        evaluate_grouped(p, by="subject_id", config=EvaluationConfig(), dataset_id="d")
        == {}
    )


def test_evaluate_grouped_unknown_key_raises():
    with pytest.raises(ValueError):
        evaluate_grouped(_pred(), by="bogus", config=EvaluationConfig(), dataset_id="d")
