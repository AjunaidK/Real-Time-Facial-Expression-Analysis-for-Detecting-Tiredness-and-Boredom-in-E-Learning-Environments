"""
Light temporal smoothing of the landmark stream (Chapter 4.3).

Raw MediaPipe landmarks jitter frame-to-frame even on a still face, which adds
noise to derived features. An exponential moving average is cheap and adequate;
it is applied to the whole (N,3) landmark array before features are computed.
"""

from __future__ import annotations
import numpy as np


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.5):
        # alpha in (0,1]; higher = more responsive, lower = smoother
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def reset(self):
        self._prev = None

    def smooth(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None:
            # keep last known? No — a missing face should propagate as missing.
            self._prev = None
            return None
        if self._prev is None or self._prev.shape != landmarks.shape:
            self._prev = landmarks.copy()
            return landmarks
        smoothed = self.alpha * landmarks + (1.0 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed
