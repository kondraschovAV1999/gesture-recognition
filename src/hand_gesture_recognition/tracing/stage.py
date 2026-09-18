from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


def _log_metric_to_mlflow(name: str, value: float) -> None:
    """Best-effort: record stage timing as an MLflow metric if a run is active."""
    try:
        import mlflow

        if mlflow.active_run():
            mlflow.log_metric(f"stage_{name}_seconds", value)
    except Exception:
        pass


@contextmanager
def stage(logger: logging.Logger, name: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info(f"[stage] {name} started")
    try:
        yield
    except Exception:
        logger.exception(f"[stage] {name} FAILED")
        raise
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"[stage] {name} finished in {elapsed:.3f}s")
        _log_metric_to_mlflow(name, elapsed)
