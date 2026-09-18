import pytest


def test_compare_runs(tmp_path):
    import mlflow

    from hand_gesture_recognition.reporting.comparison import compare_runs

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("hand_gesture_recognition")
    for name, f1 in [("run-a", 0.8), ("run-b", 0.6)]:
        with mlflow.start_run(run_name=name):
            mlflow.log_metric("macro_f1", f1)

    df = compare_runs(["run-a", "run-b"], tracking_uri=uri, metric="macro_f1")
    assert df["run_name"].tolist() == ["run-a", "run-b"]
    assert df["macro_f1"].tolist() == pytest.approx([0.8, 0.6])


def test_compare_runs_missing_run(tmp_path):
    import mlflow
    import pandas as pd

    from hand_gesture_recognition.reporting.comparison import compare_runs

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("hand_gesture_recognition")
    with mlflow.start_run(run_name="run-a"):
        mlflow.log_metric("macro_f1", 0.7)

    df = compare_runs(["run-a", "missing"], tracking_uri=uri, metric="macro_f1")
    assert df["run_name"].tolist() == ["run-a", "missing"]
    assert df["macro_f1"].iloc[0] == pytest.approx(0.7)
    assert pd.isna(df["macro_f1"].iloc[1])  # missing run -> NaN
