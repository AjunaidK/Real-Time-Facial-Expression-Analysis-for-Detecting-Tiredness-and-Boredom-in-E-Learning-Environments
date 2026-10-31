"""
Temporal aggregation of per-frame features over a sliding window.

Tiredness and boredom are states that unfold over seconds, not single frames
(D'Mello & Graesser, 2012), so the classifier sees window-level statistics, not
raw frames. This module also implements event-based measures: blink detection
(EAR threshold + minimum-frame run), PERCLOS (proportion of window with eyes
closed), and yawn frequency (MAR threshold + minimum duration).

The same aggregation is mirrored in client/features.js for the browser build.
"""

from __future__ import annotations
from collections import deque
import numpy as np

# Thresholds — starting values; tune empirically on your data (Chapter 4).
EAR_BLINK_THRESHOLD = 0.21     # below this the eye is considered closed
EAR_CLOSED_THRESHOLD = 0.21    # for PERCLOS accounting
BLINK_MIN_FRAMES = 2           # consecutive closed frames to count one blink
MAR_YAWN_THRESHOLD = 0.6       # above this the mouth is considered wide open
YAWN_MIN_FRAMES = 8            # sustained frames to count a yawn (not speech)
GAZE_OFFSCREEN_THRESHOLD = 0.35  # |gaze| or |yaw|-based off-screen proxy


class TemporalAggregator:
    """Maintains a rolling window of per-frame feature vectors and produces the
    window-level feature vector consumed by the models.

    Parameters
    ----------
    fps : assumed processing frame rate (used only to report rates per second)
    window_seconds : length of the sliding window
    """

    def __init__(self, fps: float = 10.0, window_seconds: float = 5.0):
        self.fps = fps
        self.window_len = max(1, int(round(fps * window_seconds)))
        self.buffer: deque[dict] = deque(maxlen=self.window_len)

        # event state
        self._closed_run = 0
        self._yawn_run = 0
        self._blink_events: deque[int] = deque(maxlen=self.window_len)
        self._yawn_events: deque[int] = deque(maxlen=self.window_len)

    def reset(self):
        self.buffer.clear()
        self._closed_run = 0
        self._yawn_run = 0
        self._blink_events.clear()
        self._yawn_events.clear()

    def update(self, feat: dict, face_present: bool = True) -> None:
        """Add one frame. If no face is present, record a sentinel frame – this
        is itself a disengagement signal (Chapter 4.3)."""
        if not face_present:
            # push a 'no face' marker; treated as eyes-not-visible / off-screen
            self.buffer.append({**{k: 0.0 for k in feat}, "_noface": 1.0})
            self._blink_events.append(0)
            self._yawn_events.append(0)
            self._closed_run = 0
            self._yawn_run = 0
            return

        self.buffer.append({**feat, "_noface": 0.0})

        # blink detection (EAR)
        blink = 0
        if feat["ear"] < EAR_BLINK_THRESHOLD:
            self._closed_run += 1
        else:
            if self._closed_run >= BLINK_MIN_FRAMES:
                blink = 1
            self._closed_run = 0
        self._blink_events.append(blink)

        # yawn detection (MAR sustained)
        yawn = 0
        if feat["mar"] > MAR_YAWN_THRESHOLD:
            self._yawn_run += 1
        else:
            if self._yawn_run >= YAWN_MIN_FRAMES:
                yawn = 1
            self._yawn_run = 0
        self._yawn_events.append(yawn)

    def ready(self) -> bool:
        return len(self.buffer) >= self.window_len

    def window_features(self) -> np.ndarray:
        """Aggregate the current window into the model's feature vector.

        Returns a fixed-length vector (see WINDOW_FEATURE_ORDER).
        """
        if not self.buffer:
            return np.zeros(len(WINDOW_FEATURE_ORDER), dtype=np.float32)

        ear = np.array([f["ear"] for f in self.buffer])
        mar = np.array([f["mar"] for f in self.buffer])
        yaw = np.array([f["yaw"] for f in self.buffer])
        pitch = np.array([f["pitch"] for f in self.buffer])
        gaze = np.array([f["gaze"] for f in self.buffer])
        noface = np.array([f.get("_noface", 0.0) for f in self.buffer])

        n = len(self.buffer)
        secs = n / self.fps if self.fps > 0 else 1.0

        blink_rate = sum(self._blink_events) / secs          # blinks / second
        yawn_freq = sum(self._yawn_events) / secs            # yawns / second
        perclos = float(np.mean(ear < EAR_CLOSED_THRESHOLD)) # fraction closed
        offscreen = float(np.mean(
            (np.abs(gaze) > GAZE_OFFSCREEN_THRESHOLD) | (noface > 0.5)
        ))

        out = {
            "ear_mean": float(np.mean(ear)),
            "ear_var": float(np.var(ear)),
            "perclos": perclos,
            "blink_rate": float(blink_rate),
            "mar_mean": float(np.mean(mar)),
            "yawn_freq": float(yawn_freq),
            "offscreen_prop": offscreen,
            "yaw_var": float(np.var(yaw)),
            "pitch_var": float(np.var(pitch)),
            "noface_prop": float(np.mean(noface)),
        }
        return np.array([out[k] for k in WINDOW_FEATURE_ORDER], dtype=np.float32)

    def window_sequence(self) -> np.ndarray:
        """Return the raw per-frame feature sequence (window_len, n_frame_feats)
        for the LSTM tiredness model. Pads at the front if under-filled."""
        from .geometry import FRAME_FEATURE_ORDER
        seq = np.array(
            [[f.get(k, 0.0) for k in FRAME_FEATURE_ORDER] for f in self.buffer],
            dtype=np.float32,
        )
        if len(seq) < self.window_len:
            pad = np.zeros((self.window_len - len(seq), seq.shape[1]),
                           dtype=np.float32)
            seq = np.vstack([pad, seq]) if len(seq) else np.zeros(
                (self.window_len, len(FRAME_FEATURE_ORDER)), dtype=np.float32)
        return seq


# Ordered names of window-level features – KEEP IN SYNC with client/features.js
WINDOW_FEATURE_ORDER = [
    "ear_mean", "ear_var", "perclos", "blink_rate",
    "mar_mean", "yawn_freq", "offscreen_prop",
    "yaw_var", "pitch_var", "noface_prop",
]
