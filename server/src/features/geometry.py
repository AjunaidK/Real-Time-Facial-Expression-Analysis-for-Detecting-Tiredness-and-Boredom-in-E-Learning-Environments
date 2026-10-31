"""
Geometric features derived from MediaPipe Face Mesh landmarks.

This module is the single source of truth for feature definitions. It is used by
the training pipeline (src/training) and by the server (server/). The browser
client re-implements the SAME formulas in JavaScript (client/features.js) so that
the client-side and server-side configurations are numerically comparable.

MediaPipe Face Mesh returns 468 landmarks, each with (x, y, z) normalised to the
image size (x, y in [0, 1]). See:
  Kartynnik et al. (2019), Real-time Facial Surface Geometry from Monocular Video.
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Landmark index groups (MediaPipe Face Mesh canonical topology, 468 points).
# The six-point eye model follows the Eye Aspect Ratio of Soukupova & Cech (2016),
# mapped onto MediaPipe indices. Order per eye: [p1, p2, p3, p4, p5, p6]
#   p1, p4  = horizontal corners
#   p2, p3, p5, p6 = upper/lower lid points
# ---------------------------------------------------------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth: outer-lip corners and top/bottom for the Mouth Aspect Ratio.
MOUTH = {
    "left": 61, "right": 291, "top": 13, "bottom": 14,
    # extra vertical pairs for a more stable MAR
    "top2": 0, "bottom2": 17,
}

# Iris centres (available when MediaPipe refine_landmarks=True; indices 468-477).
LEFT_IRIS = 468
RIGHT_IRIS = 473

# A stable subset used for solvePnP head-pose estimation, paired with a canonical
# 3D face model below (nose tip, chin, eye corners, mouth corners).
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# Canonical 3D model points (arbitrary mm units) matching POSE_LANDMARKS order.
# Standard values widely used for 6-point head-pose estimation.
CANONICAL_3D = np.array([
    [0.0,    0.0,    0.0],      # nose tip
    [0.0,  -63.6,  -12.5],      # chin
    [-43.3, 32.7,  -26.0],      # left eye corner
    [43.3,  32.7,  -26.0],      # right eye corner
    [-28.9,-28.9,  -24.1],      # left mouth
    [28.9, -28.9,  -24.1],      # right mouth
], dtype=np.float64)


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(landmarks: np.ndarray, eye_idx: list[int]) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|).  Falls toward 0 on closure."""
    p = [landmarks[i][:2] for i in eye_idx]
    vert = _euclid(p[1], p[5]) + _euclid(p[2], p[4])
    horiz = 2.0 * _euclid(p[0], p[3])
    return vert / horiz if horiz > 1e-6 else 0.0


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """MAR = vertical mouth opening / horizontal mouth width. Rises on a yawn."""
    top = landmarks[MOUTH["top"]][:2]
    bottom = landmarks[MOUTH["bottom"]][:2]
    top2 = landmarks[MOUTH["top2"]][:2]
    bottom2 = landmarks[MOUTH["bottom2"]][:2]
    left = landmarks[MOUTH["left"]][:2]
    right = landmarks[MOUTH["right"]][:2]
    vert = (_euclid(top, bottom) + _euclid(top2, bottom2)) / 2.0
    horiz = _euclid(left, right)
    return vert / horiz if horiz > 1e-6 else 0.0


def head_pose(landmarks: np.ndarray, image_w: int, image_h: int):
    """Estimate (pitch, yaw, roll) in degrees via solvePnP.

    Requires OpenCV. Returns (pitch, yaw, roll) or (0,0,0) if the solve fails.
    Landmarks are expected in pixel coordinates here (x*image_w, y*image_h).
    """
    import cv2  # local import so the module is importable without cv2 present

    image_points = np.array([
        landmarks[POSE_LANDMARKS["nose_tip"]][:2],
        landmarks[POSE_LANDMARKS["chin"]][:2],
        landmarks[POSE_LANDMARKS["left_eye_corner"]][:2],
        landmarks[POSE_LANDMARKS["right_eye_corner"]][:2],
        landmarks[POSE_LANDMARKS["left_mouth"]][:2],
        landmarks[POSE_LANDMARKS["right_mouth"]][:2],
    ], dtype=np.float64)
    image_points[:, 0] *= image_w
    image_points[:, 1] *= image_h

    focal = float(image_w)
    cam = np.array([[focal, 0, image_w / 2.0],
                    [0, focal, image_h / 2.0],
                    [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(CANONICAL_3D, image_points, cam, dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rot[2, 0], sy))
    yaw = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    roll = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    return float(pitch), float(yaw), float(roll)


def gaze_offset(landmarks: np.ndarray) -> float:
    """Coarse gaze proxy: horizontal offset of the iris centre within the eye,
    in [-1, 1] roughly (0 = looking at centre). Averaged across both eyes.
    Requires refined landmarks (iris). Returns 0.0 if iris points are absent.
    """
    if len(landmarks) <= RIGHT_IRIS:
        return 0.0

    def eye_ratio(eye_idx, iris_idx):
        left_corner = landmarks[eye_idx[0]][:2]
        right_corner = landmarks[eye_idx[3]][:2]
        iris = landmarks[iris_idx][:2]
        span = right_corner[0] - left_corner[0]
        if abs(span) < 1e-6:
            return 0.0
        # position of iris between corners, remapped to [-1, 1]
        rel = (iris[0] - left_corner[0]) / span
        return (rel - 0.5) * 2.0

    return float((eye_ratio(LEFT_EYE, LEFT_IRIS) +
                  eye_ratio(RIGHT_EYE, RIGHT_IRIS)) / 2.0)


def per_frame_features(landmarks: np.ndarray, image_w: int, image_h: int) -> dict:
    """Compute all per-frame features for one face. `landmarks` is (N, 3) with
    x, y normalised to [0,1]. Returns a dict of named scalars."""
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
           eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
    mar = mouth_aspect_ratio(landmarks)
    pitch, yaw, roll = head_pose(landmarks, image_w, image_h)
    gaze = gaze_offset(landmarks)
    return {
        "ear": ear, "mar": mar,
        "pitch": pitch, "yaw": yaw, "roll": roll,
        "gaze": gaze,
    }


# Ordered feature names – KEEP IN SYNC with client/features.js
FRAME_FEATURE_ORDER = ["ear", "mar", "pitch", "yaw", "roll", "gaze"]


def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([feat[k] for k in FRAME_FEATURE_ORDER], dtype=np.float32)
