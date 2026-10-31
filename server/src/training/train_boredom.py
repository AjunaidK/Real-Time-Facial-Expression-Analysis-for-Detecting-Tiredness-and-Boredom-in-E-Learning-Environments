"""
Train the boredom fusion model (MobileNetV2 + geometric features) on DAiSEE.

Methodology (Chapters 3.6 / 4.5):
  * binarised boredom target
  * class weighting + image augmentation to counter DAiSEE imbalance
  * participant-independent split
  * two-phase transfer learning: (1) frozen backbone, (2) optional fine-tune

Run:
  python -m src.training.train_boredom \
      --data data/processed/daisee_boredom.npz --out models_out/boredom
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np

from ..models.boredom_fusion import build_boredom_model
from .utils import participant_split_multi, class_weights, evaluate_binary_fusion


def make_augment():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models_out/boredom")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data, allow_pickle=True)
    images, feats, y, subj = d["images"], d["feats"], d["y"], d["subj"]
    print(f"Loaded images={images.shape} feats={feats.shape} "
          f"pos_rate={y.mean():.3f} subjects={len(set(subj.tolist()))}")

    import tensorflow as tf
    from ..models.boredom_fusion import preprocess_face  # noqa

    # participant-independent split across both inputs + label
    (tr, te) = participant_split_multi(
        {"images": images, "feats": feats}, y, subj,
        test_frac=args.test_frac, seed=args.seed)
    Xtr_img, Xtr_feat, ytr = tr["images"], tr["feats"], tr["y"]
    Xte_img, Xte_feat, yte = te["images"], te["feats"], te["y"]
    print(f"Train {len(ytr)}  Test {len(yte)}")

    # preprocess images for MobileNetV2 ([-1,1]); augment train only
    augment = make_augment()

    def prep(img_batch, training):
        x = tf.cast(img_batch, tf.float32)
        if training:
            x = augment(x, training=True)
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    model = build_boredom_model(
        image_size=images.shape[1], n_window_features=feats.shape[1],
        train_backbone=False)
    model.summary()

    cw = class_weights(ytr)
    cb = [tf.keras.callbacks.EarlyStopping(
        monitor="val_recall", mode="max", patience=6,
        restore_best_weights=True)]

    # --- phase 1: frozen backbone ---
    hist1 = model.fit(
        {"image": prep(Xtr_img, True), "features": Xtr_feat}, ytr,
        validation_data=(
            {"image": prep(Xte_img, False), "features": Xte_feat}, yte),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=cw, callbacks=cb, verbose=2)

    # --- phase 2: optional fine-tune of top backbone layers ---
    hist2 = {}
    if args.finetune_epochs > 0:
        for layer in model.layers:
            if layer.name.startswith("mobilenetv2"):
                layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # low LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy",
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")])
        h = model.fit(
            {"image": prep(Xtr_img, True), "features": Xtr_feat}, ytr,
            validation_data=(
                {"image": prep(Xte_img, False), "features": Xte_feat}, yte),
            epochs=args.finetune_epochs, batch_size=args.batch,
            class_weight=cw, callbacks=cb, verbose=2)
        hist2 = h.history

    metrics = evaluate_binary_fusion(
        model, prep(Xte_img, False), Xte_feat, yte)
    print("Test metrics:", json.dumps(metrics, indent=2))

    model.save(os.path.join(args.out, "boredom_keras.keras"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    hist = {"phase1": {k: [float(v) for v in vs]
                       for k, vs in hist1.history.items()},
            "phase2": {k: [float(v) for v in vs] for k, vs in hist2.items()}}
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print("Saved model + metrics + history to", args.out)


if __name__ == "__main__":
    main()
