from __future__ import annotations


class ThresholdLabeling:
    """Threshold-based window ground-truth labeling.

    A window is ``rest`` (labeled noGesture) unless the fraction of its samples
    that fall inside the gesture interval ``[g_start, g_end]`` strictly exceeds
    ``threshold``. With ``threshold=0.0`` this reproduces the old any-overlap
    behavior; the default ``0.5`` is majority overlap.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def is_rest(
        self, window_start: int, window_end: int, g_start: int, g_end: int
    ) -> bool:
        """True if the window should be labeled noGesture."""
        if g_start < 0:  # rest trial has no gesture bounds
            return True
        overlap = max(0, min(window_end, g_end) - max(window_start, g_start))
        return overlap / (window_end - window_start) <= self.threshold
