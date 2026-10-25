from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hand_gesture_recognition.configuration.errors import ConfigValidationError


@dataclass
class WindowingConfig:
    mode: str = "single"  # "single" | "two_level"
    window_size_samples: Optional[int] = None
    window_size_seconds: Optional[float] = None
    stride_samples: Optional[int] = None
    stride_seconds: Optional[float] = None
    overlap_percent: Optional[float] = None
    label_granularity: str = "per_window"  # per_window | sequence
    padding: str = "none"  # none | zero
    # --- big window (used only when mode == "two_level") ---
    big_window_size_samples: Optional[int] = None
    big_window_size_seconds: Optional[float] = None
    big_stride_samples: Optional[int] = None
    big_stride_seconds: Optional[float] = None
    big_overlap_percent: Optional[float] = None

    def resolve(self, sampling_rate: float) -> Tuple[int, int]:
        """Return (window_size_samples, stride_samples), deriving/validating.

        Fails loudly if two representations of the same quantity disagree.
        """
        if sampling_rate <= 0:
            raise ConfigValidationError(
                f"sampling_rate must be > 0, got {sampling_rate}"
            )

        ws = self.window_size_samples
        if self.window_size_seconds is not None:
            derived = round(self.window_size_seconds * sampling_rate)
            if ws is not None and ws != derived:
                raise ConfigValidationError(
                    f"window_size_samples={ws} != window_size_seconds="
                    f"{self.window_size_seconds}s * {sampling_rate}Hz = {derived}"
                )
            ws = derived
        if ws is None:
            raise ConfigValidationError(
                "window size must be given as window_size_samples or window_size_seconds"
            )
        if ws < 1:
            raise ConfigValidationError(f"window_size_samples must be >= 1, got {ws}")

        stride = self.stride_samples
        if self.stride_seconds is not None:
            derived = round(self.stride_seconds * sampling_rate)
            if stride is not None and stride != derived:
                raise ConfigValidationError(
                    f"stride_samples={stride} != stride_seconds="
                    f"{self.stride_seconds}s * {sampling_rate}Hz = {derived}"
                )
            stride = derived
        if stride is None:
            if self.overlap_percent is not None:
                stride = round(ws * (1.0 - self.overlap_percent / 100.0))
            else:
                stride = ws  # non-overlapping default
        if stride < 1:
            raise ConfigValidationError(f"stride must be >= 1, got {stride}")
        return ws, stride

    def resolve_big(self, sampling_rate: float) -> Tuple[int, int]:
        """Return (big_window_size_samples, big_stride_samples), mirroring resolve()."""
        if sampling_rate <= 0:
            raise ConfigValidationError(
                f"sampling_rate must be > 0, got {sampling_rate}"
            )
        ws = self.big_window_size_samples
        if self.big_window_size_seconds is not None:
            derived = round(self.big_window_size_seconds * sampling_rate)
            if ws is not None and ws != derived:
                raise ConfigValidationError(
                    f"big_window_size_samples={ws} != big_window_size_seconds="
                    f"{self.big_window_size_seconds}s * {sampling_rate}Hz = {derived}"
                )
            ws = derived
        if ws is None:
            raise ConfigValidationError(
                "big window size must be given as big_window_size_samples "
                "or big_window_size_seconds"
            )
        if ws < 1:
            raise ConfigValidationError(
                f"big_window_size_samples must be >= 1, got {ws}"
            )
        stride = self.big_stride_samples
        if self.big_stride_seconds is not None:
            derived = round(self.big_stride_seconds * sampling_rate)
            if stride is not None and stride != derived:
                raise ConfigValidationError(
                    f"big_stride_samples={stride} != big_stride_seconds="
                    f"{self.big_stride_seconds}s * {sampling_rate}Hz = {derived}"
                )
            stride = derived
        if stride is None:
            if self.big_overlap_percent is not None:
                stride = round(ws * (1.0 - self.big_overlap_percent / 100.0))
            else:
                stride = ws
        if stride < 1:
            raise ConfigValidationError(f"big stride must be >= 1, got {stride}")
        return ws, stride


@dataclass
class WindowMetadata:
    window_start: np.ndarray  # (n_windows,)
    window_end: np.ndarray  # (n_windows,)
    subject_ids: np.ndarray  # (n_windows,)
    trial_ids: np.ndarray  # (n_windows,)
    window_ids: np.ndarray  # (n_windows,) unique, traceable to trial


@dataclass
class WindowedDataset:
    X: np.ndarray  # (n_windows, window_size, channels)
    y: np.ndarray  # (n_windows,)
    metadata: WindowMetadata


@dataclass
class WindowedPartitions:
    """The single authoritative windowed representation of train/val/test."""

    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset


@dataclass
class GroupedWindowedDataset:
    big: WindowedDataset  # one sample per big window (classical view)
    small_windows: List[np.ndarray]  # per big window: (n_small, small_ws, ch)
    big_window_id: np.ndarray  # (n_big,) links small groups to big windows
    y: np.ndarray
    metadata: object


@dataclass
class GroupedWindowedPartitions:
    train: GroupedWindowedDataset
    val: GroupedWindowedDataset
    test: GroupedWindowedDataset
