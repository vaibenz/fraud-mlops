"""Metrics for heavily imbalanced classification.

Accuracy is meaningless here (predicting "never fraud" scores 99.8%), so models
are compared on PR AUC, and the alert threshold is tuned on the validation set
then reported, untouched, on the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def pick_threshold(y: np.ndarray, p: np.ndarray, min_precision: float | None = None) -> float:
    """Threshold that maximises F1 on validation data, or the lowest threshold
    reaching `min_precision` if an alert budget is set."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    if min_precision is not None:
        ok = np.where(prec >= min_precision)[0]
        if len(ok):
            return float(thr[ok[0]])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alerts_per_10k": float(10_000 * (tp + fp) / len(y)),
    }
