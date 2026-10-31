"""
Preprocess DAiSEE into (face-crop image, window-feature vector, label) samples
for the boredom model.

DAiSEE: 9,068 ten-second clips, 112 users, annotated for boredom / confusion /
engagement / frustration at 4 intensity levels (Gupta et al., 2016). We use the
BOREDOM label and binarise: levels {0,1} -> 0 (not bored), {2,3} -> 1 (bored).
(Adjust the threshold in binarise_boredom if you prefer a different cut.)

DAiSEE ships a Labels/*.csv mapping ClipID -> Boredom,Engagement,Confusion,
Frustration. Videos live under DataSet/{Train,Validation,Test}/<user>/<clip>/*.avi.
Adapt paths to your download.

For each clip we take a representative face crop (mid-clip frame) plus the
window-level geometric features aggregated over the clip.

Output: data/processed/daisee_boredom.npz
   images   (n, H, W, 3) uint8 face crops
   feats    (n, n_window_features) float32
   y        (n,) binary
   subj     (n,) user id  (participant-independent split)

Run:
  python -m src.training.preprocess_daisee --root data/daisee --out data/processed
"""

from __future__ import annotations
import argparse
import os
import csv
import glob
import numpy as np

from ..features.landmarks import LandmarkExtractor
from ..features.smoothing import LandmarkSmoother
from ..features.geometry import per_frame_features, FRAME_FEATURE_ORDER
from ..features.temporal import TemporalAggregator, WINDOW_FEATURE_ORDER

IMAGE_SIZE = 160


def binarise_boredom(level: int) -> int:
    return 1 if level >= 2 else 0


def load_labels(root: str) -> dict[str, int]:
    """Return {clip_id: boredom_level}. Merges all label CSVs found."""
    labels = {}
    for csv_path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fields = {k.lower().strip(): k for k in reader.fieldnames or []}
            clip_key = fields.get("clipid") or fields.get("clip")
            bored_key = fields.get("boredom")
            if not clip_key or not bored_key:
                continue
            for row in reader:
                clip = os.path.splitext(row[clip_key].strip())[0]
                try:
                    labels[clip] = int(float(row[bored_key]))
                except ValueError:
                    pass
    return labels


def clip_user(clip_path: str) -> str:
    # DAiSEE layout .../<user>/<clip_id>/<file>.avi
    parts = os.path.normpath(clip_path).split(os.sep)
    return parts[-3] if len(parts) >= 3 else "unknown"


def process_clip(path, extractor, fps, window_seconds):
    """Return (face_crop_bgr, window_feature_vector) for a clip, or (None, None)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mid = total // 2

    agg = TemporalAggregator(fps=fps, window_seconds=window_seconds)
    smoother = LandmarkSmoother(alpha=0.5)
    face_crop = None
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        take = (frame_i % step == 0)
        h, w = frame.shape[:2]
        if take:
            lm = smoother.smooth(extractor.extract(frame))
            if lm is None:
                agg.update({k: 0.0 for k in FRAME_FEATURE_ORDER},
                           face_present=False)
            else:
                agg.update(per_frame_features(lm, w, h), face_present=True)
                # grab a face crop near the middle of the clip
                if face_crop is None and abs(frame_i - mid) < step:
                    face_crop = crop_face(frame, lm, w, h)
        frame_i += 1
    cap.release()

    if face_crop is None:
        return None, None
    return face_crop, agg.window_features()


def crop_face(frame_bgr, landmarks, w, h, margin=0.25):
    xs = landmarks[:, 0] * w
    ys = landmarks[:, 1] * h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = max(0, int(x0 - mw)); y0 = max(0, int(y0 - mh))
    x1 = min(w, int(x1 + mw)); y1 = min(h, int(y1 + mh))
    import cv2
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DAiSEE root directory")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #clips")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    labels = load_labels(args.root)
    print(f"Loaded {len(labels)} boredom labels")

    videos = glob.glob(os.path.join(args.root, "**", "*.avi"), recursive=True)
    videos += glob.glob(os.path.join(args.root, "**", "*.mp4"), recursive=True)
    print(f"Found {len(videos)} clips")

    images, feats, ys, subj = [], [], [], []
    with LandmarkExtractor(static_image_mode=False) as extractor:
        for i, vid in enumerate(sorted(videos)):
            if args.limit and i >= args.limit:
                break
            clip_id = os.path.splitext(os.path.basename(vid))[0]
            if clip_id not in labels:
                continue
            crop, wfeat = process_clip(vid, extractor, args.fps, args.window)
            if crop is None:
                continue
            images.append(crop)
            feats.append(wfeat)
            ys.append(binarise_boredom(labels[clip_id]))
            subj.append(clip_user(vid))
            if (i + 1) % 50 == 0:
                print(f"  processed {i+1} clips, kept {len(ys)}")

    images = np.asarray(images, dtype=np.uint8)
    feats = np.asarray(feats, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.int64)
    subj = np.asarray(subj)
    out_path = os.path.join(args.out, "daisee_boredom.npz")
    np.savez_compressed(out_path, images=images, feats=feats, y=ys, subj=subj,
                        feature_order=np.array(WINDOW_FEATURE_ORDER))
    print(f"Saved {len(ys)} samples -> {out_path}  "
          f"(pos rate {ys.mean():.3f})")


if __name__ == "__main__":
    main()
