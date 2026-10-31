"""
Shared training utilities: participant-independent splitting, class weights, and
evaluation (accuracy / precision / recall / F1 / confusion matrix).

Participant-independent splitting is essential (Chapter 3.6): if the same face
appears in train and test, reported accuracy measures memorisation, not
generalisation.
"""

from __future__ import annotations
import numpy as np


def participant_split(X, y, subj, test_frac=0.2, seed=42):
    """Split arrays so that whole subjects go to either train or test."""
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(set(subj.tolist())))
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    test_subjects = set(subjects[:n_test].tolist())

    test_mask = np.array([s in test_subjects for s in subj])
    train_mask = ~test_mask
    return (X[train_mask], y[train_mask]), (X[test_mask], y[test_mask])


def participant_split_multi(inputs: dict, y, subj, test_frac=0.2, seed=42):
    """Like participant_split but for multiple parallel input arrays."""
    rng = np.random.default_rng(seed)
    subjects = np.array(sorted(set(subj.tolist())))
    rng.shuffle(subjects)
    n_test = max(1, int(round(len(subjects) * test_frac)))
    test_subjects = set(subjects[:n_test].tolist())
    test_mask = np.array([s in test_subjects for s in subj])
    train_mask = ~test_mask

    tr = {k: v[train_mask] for k, v in inputs.items()}
    te = {k: v[test_mask] for k, v in inputs.items()}
    tr["y"] = y[train_mask]
    te["y"] = y[test_mask]
    return tr, te


def class_weights(y):
    """Inverse-frequency class weights for a binary target -> dict for Keras."""
    y = np.asarray(y)
    n = len(y)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, n - n_pos)
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def _metrics_from_preds(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "threshold": threshold,
    }


def evaluate_binary(model, X, y, threshold=0.5):
    y_prob = model.predict(X, verbose=0)
    return _metrics_from_preds(y, y_prob, threshold)


def evaluate_binary_fusion(model, X_img, X_feat, y, threshold=0.5):
    y_prob = model.predict({"image": X_img, "features": X_feat}, verbose=0)
    return _metrics_from_preds(y, y_prob, threshold)
