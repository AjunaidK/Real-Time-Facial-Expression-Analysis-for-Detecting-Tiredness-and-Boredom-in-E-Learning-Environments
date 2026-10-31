"""
Preprocess the UTA Real-Life Drowsiness Dataset (UTA-RLDD) into windowed
per-frame feature sequences for the tiredness LSTM.

UTA-RLDD provides ~30h of RGB video from 60 subjects, labelled alert / low
vigilant / drowsy (Ghoddoosian, Galib & Athitsos, 2019). We collapse to a binary
target: {alert} -> 0 (not tired), {low vigilant, drowsy} -> 1 (tired).

Expected input layout (adapt to however you downloaded it):
  data/uta_rldd/<subject_id>/<label>/<video>.{mp4,avi,mov}
where <label> in {"0","5","10"} or {"alert","lowvigilant","drowsy"}.

Output:
  data/processed/uta_tiredness.npz  with arrays:
     X   (n_windows, window_len, n_frame_features)
     y   (n_windows,)  binary
     subj(n_windows,)  subject id per window  (for participant-independent split)

Run:
  python -m src.training.preprocess_uta --data data/uta_rldd --out data/processed
"""

from __future__ import annotations
import argparse
import os
import glob
import numpy as np

from ..features.landmarks import LandmarkExtractor
from ..features.smoothing import LandmarkSmoother
from ..features.geometry import per_frame_features, FRAME_FEATURE_ORDER
from ..features.temporal import TemporalAggregator

LABEL_MAP = {
    "0": 0, "alert": 0,
    "5": 1, "lowvigilant": 1, "low_vigilant": 1,
    "10": 1, "drowsy": 1,
}

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def iter_videos(root: str):
    for subject in sorted(os.listdir(root)):
        subj_dir = os.path.join(root, subject)
        if not os.path.isdir(subj_dir):
            continue
        for label_name in os.listdir(subj_dir):
            label_dir = os.path.join(subj_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            key = label_name.lower().strip()
            if key not in LABEL_MAP:
                continue
            for ext in VIDEO_EXTS:
                for vid in glob.glob(os.path.join(label_dir, ext)):
                    yield subject, LABEL_MAP[key], vid


def process_video(path, extractor, fps, window_seconds, stride_windows=1):
    """Yield (sequence, ) windows from one video."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # sample down to the target fps
    step = max(1, int(round(src_fps / fps)))

    agg = TemporalAggregator(fps=fps, window_seconds=window_seconds)
    smoother = LandmarkSmoother(alpha=0.5)
    frame_i = 0
    emit_counter = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_i % step != 0:
            frame_i += 1
            continue
        frame_i += 1
        h, w = frame.shape[:2]
        lm = smoother.smooth(extractor.extract(frame))
        if lm is None:
            agg.update({k: 0.0 for k in FRAME_FEATURE_ORDER}, face_present=False)
        else:
            feat = per_frame_features(lm, w, h)
            agg.update(feat, face_present=True)

        if agg.ready():
            emit_counter += 1
            if emit_counter % stride_windows == 0:
                yield agg.window_sequence()
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="UTA-RLDD root directory")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--stride", type=int, default=5,
                    help="emit one window every N ready windows (thinning)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    X, y, subj = [], [], []

    with LandmarkExtractor(static_image_mode=False) as extractor:
        for subject, label, vid in iter_videos(args.data):
            n = 0
            for seq in process_video(vid, extractor, args.fps, args.window,
                                     stride_windows=args.stride):
                X.append(seq); y.append(label); subj.append(subject); n += 1
            print(f"[{subject}] label={label} {os.path.basename(vid)} -> {n} windows")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    subj = np.asarray(subj)
    out_path = os.path.join(args.out, "uta_tiredness.npz")
    np.savez_compressed(out_path, X=X, y=y, subj=subj,
                        feature_order=np.array(FRAME_FEATURE_ORDER))
    print(f"Saved {X.shape[0]} windows -> {out_path}  (X shape {X.shape})")


if __name__ == "__main__":
    main()
