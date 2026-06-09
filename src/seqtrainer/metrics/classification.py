"""Classification metrics used by benchmark workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _default_mcc_thresholds(scores: np.ndarray) -> np.ndarray:
    """Build MCC threshold candidates from fixed grid values and observed scores."""
    finite_scores = np.sort(np.unique(scores[np.isfinite(scores)]))
    if finite_scores.size == 0:
        raise ValueError("Cannot select a threshold when all scores are non-finite")

    candidates = [np.linspace(0.0, 1.0, 201)]
    candidates.append(finite_scores)
    if finite_scores.size > 1:
        candidates.append((finite_scores[:-1] + finite_scores[1:]) / 2.0)

    candidates.append(np.array([np.nextafter(finite_scores[0], -np.inf)]))
    candidates.append(np.array([np.nextafter(finite_scores[-1], np.inf)]))
    return np.unique(np.concatenate(candidates))


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the shared binary-classification metric suite."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] == 0:
        raise ValueError("Cannot compute metrics for an empty label array")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    positives = tp + fn
    negatives = tn + fp
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(tp / positives) if positives else None,
        "specificity": float(tn / negatives) if negatives else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    return metrics


def best_threshold_by_mcc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose a binary threshold by maximizing MCC on validation data."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] == 0:
        raise ValueError("Cannot select a threshold for an empty label array")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    candidates = np.asarray(thresholds, dtype=float) if thresholds is not None else _default_mcc_thresholds(scores)
    best_threshold, best_mcc = 0.5, float("-inf")
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        score = float(matthews_corrcoef(labels, predictions))
        if score > best_mcc:
            best_threshold = float(threshold)
            best_mcc = score

    return best_threshold, best_mcc
