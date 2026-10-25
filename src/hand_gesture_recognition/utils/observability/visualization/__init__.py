"""Visualization package: EMG plotting and analysis helpers.

Split from the original single-module ``visualization.py`` into focused
submodules. Every public name is re-exported here so existing imports like
``from hand_gesture_recognition.utils.observability.visualization import
plot_raw_signal_by_class`` keep working unchanged.
"""

from .data_flow import plot_signal_data_flow
from .features import plot_feature_over_windows
from .predictions import plot_predictions_on_signal
from .processing import plot_signal_processing_comparison
from .raw import plot_raw_signal, plot_raw_signal_by_class
from .spectrum import plot_fft_spectrum
from .stages import compute_stages, plot_stage_progression, stage_label

from ._helpers import resolve_selection

__all__ = [
    "compute_stages",
    "plot_feature_over_windows",
    "plot_fft_spectrum",
    "plot_predictions_on_signal",
    "plot_raw_signal",
    "plot_raw_signal_by_class",
    "plot_signal_data_flow",
    "plot_signal_processing_comparison",
    "plot_stage_progression",
    "resolve_selection",
    "stage_label",
]
