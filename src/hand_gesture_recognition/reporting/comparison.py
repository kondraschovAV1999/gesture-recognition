from __future__ import annotations

from typing import List, Optional

import pandas as pd


def compare_runs(
    run_names: List[str],
    tracking_uri: str = "sqlite:///mlflow.db",
    metric: str = "macro_f1",
    experiment_name: str = "hand_gesture_recognition",
    output: Optional[str] = None,
) -> pd.DataFrame:
    """Compare a metric across runs, queried from MLflow by run NAME.

    Args:
        run_names: framework run names (the timestamped `run_id`s, also the
            MLflow `run_name`).
        tracking_uri: MLflow tracking URI (default SQLite store).
        metric: metric to compare (e.g. "macro_f1", "balanced_accuracy").
        experiment_name: MLflow experiment to search.
        output: optional CSV path; also writes a PNG bar chart alongside it.

    Returns:
        DataFrame with columns ``run_name`` and ``<metric>`` (and any other
        metrics present on the runs).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment {experiment_name!r} not found")

    name_to_run = {
        r.info.run_name: r for r in client.search_runs([experiment.experiment_id])
    }

    rows = []
    for name in run_names:
        run = name_to_run.get(name)
        row = {"run_name": name}
        if run is not None:
            row[metric] = run.data.metrics.get(metric)
        else:
            row[metric] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    if output:
        df.to_csv(output, index=False)
        _plot(df, metric, output)

    return df


def _plot(df: pd.DataFrame, metric: str, csv_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["run_name"].astype(str), df[metric])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across runs")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    png_path = csv_path.rsplit(".", 1)[0] + ".png"
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
