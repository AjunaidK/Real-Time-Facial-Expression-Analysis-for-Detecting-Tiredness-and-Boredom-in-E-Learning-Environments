"""
Train the tiredness LSTM on preprocessed UTA-RLDD windows.

Key methodology points (Chapter 3.6):
  * participant-independent split — no subject in both train and test
  * class weighting to counter imbalance
  * report accuracy / precision / recall / F1 and a confusion matrix

Run:
  python -m src.training.train_tiredness \
      --data data/processed/uta_tiredness.npz --out models_out/tiredness
"""

from __future__ import annotations
import argparse
import os
import json
import numpy as np

from ..models.tiredness_lstm import build_tiredness_model
from .utils import participant_split, class_weights, evaluate_binary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models_out/tiredness")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.data, allow_pickle=True)
    X, y, subj = d["X"], d["y"], d["subj"]
    print(f"Loaded X={X.shape} y={y.shape} pos_rate={y.mean():.3f} "
          f"subjects={len(set(subj.tolist()))}")

    (Xtr, ytr), (Xte, yte) = participant_split(
        X, y, subj, test_frac=args.test_frac, seed=args.seed)
    print(f"Train {Xtr.shape[0]}  Test {Xte.shape[0]}")

    import tensorflow as tf
    model = build_tiredness_model(window_len=X.shape[1], n_features=X.shape[2])
    model.summary()

    cw = class_weights(ytr)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", mode="max", patience=8,
            restore_best_weights=True),
    ]
    history = model.fit(
        Xtr, ytr, validation_data=(Xte, yte),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=cw, callbacks=callbacks, verbose=2,
    )

    metrics = evaluate_binary(model, Xte, yte)
    print("Test metrics:", json.dumps(metrics, indent=2))

    model.save(os.path.join(args.out, "tiredness_keras.keras"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # save history for the Figure 4.2 curves
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print("Saved model + metrics + history to", args.out)


if __name__ == "__main__":
    main()
