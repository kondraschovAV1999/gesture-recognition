from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskConfig:
    name: str = "recognition"  # "classification" | "recognition"
