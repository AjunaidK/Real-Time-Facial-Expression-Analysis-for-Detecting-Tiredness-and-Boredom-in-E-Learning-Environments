"""
Thin wrapper around MediaPipe Face Mesh for landmark extraction.

Kept separate so that training, the server, and any offline scripts share one
extraction path. Returns an (N, 3) numpy array of normalised landmarks, or None
if no face is detected (which the temporal layer treats as a disengagement cue).
"""

from __future__ import annotations
import numpy as np


class LandmarkExtractor:
    def __init__(self, refine_landmarks: bool = True,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False):
        import mediapipe as mp
        self._mp = mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr) -> np.ndarray | None:
        """frame_bgr: an HxWx3 BGR image (OpenCV). Returns (N,3) or None."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
