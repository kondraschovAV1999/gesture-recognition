from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str, run_id: Optional[str] = None, log_dir: str = "output/logs"
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if run_id:
        d = os.path.join(log_dir, run_id)
        os.makedirs(d, exist_ok=True)
        fh = logging.FileHandler(os.path.join(d, "application.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
