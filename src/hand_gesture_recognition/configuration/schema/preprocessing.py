from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GestureCropConfig:
    enabled: bool = False
    pad_samples: int = 100  # symmetric context kept before AND after the gesture
    no_gesture_max_length_samples: Optional[int] = None  # None = auto-derive


@dataclass
class BaselineConfig:
    enabled: bool = False
    mode: str = "global"  # "global" | "per_user"
