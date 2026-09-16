from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


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
