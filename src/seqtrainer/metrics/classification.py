"""Classification metrics used by benchmark workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
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


def threshold_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Convert class-one scores into binary predictions."""
    scores = np.asarray(y_score, dtype=float)
    return (scores >= threshold).astype(int)


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

    predictions = threshold_predictions(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    positives = tp + fn
    negatives = tn + fp
    sensitivity = float(tp / positives) if positives else None
    specificity = float(tn / negatives) if negatives else None
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": _balanced_accuracy(sensitivity, specificity),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": _mcc_from_counts(tn, fp, fn, tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
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
        metrics["warning"] = "AUROC/AUPRC undefined because the split contains one observed class."

    return metrics


def binary_classification_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    """Compute the shared metric suite when only hard predictions are available."""
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    if labels.shape[0] == 0:
        raise ValueError("Cannot compute metrics for an empty label array")
    if labels.shape[0] != predictions.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    positives = tp + fn
    negatives = tn + fp
    sensitivity = float(tp / positives) if positives else None
    specificity = float(tn / negatives) if negatives else None
    metrics: dict[str, Any] = {
        "threshold": float(threshold) if threshold is not None else None,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": _balanced_accuracy(sensitivity, specificity),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": _mcc_from_counts(tn, fp, fn, tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "auroc": None,
        "auprc": None,
    }
    if warning:
        metrics["warning"] = warning
    return metrics


def best_threshold_by_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "mcc",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose a binary threshold by maximizing a validation metric."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] == 0:
        raise ValueError("Cannot select a threshold for an empty label array")
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    metric_key = metric.lower()
    if metric_key not in {"mcc", "f1", "balanced_accuracy"}:
        raise ValueError("metric must be one of: mcc, f1, balanced_accuracy")

    candidates = np.asarray(thresholds, dtype=float) if thresholds is not None else _default_mcc_thresholds(scores)
    best_threshold, best_score = 0.5, float("-inf")
    for threshold in candidates:
        predictions = threshold_predictions(scores, float(threshold))
        score = _threshold_metric(labels, predictions, metric_key)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)

    return best_threshold, best_score


def best_threshold_by_mcc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose a binary threshold by maximizing MCC on validation data."""
    return best_threshold_by_metric(y_true, y_score, metric="mcc", thresholds=thresholds)


def _threshold_metric(labels: np.ndarray, predictions: np.ndarray, metric: str) -> float:
    if metric == "mcc":
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        return _mcc_from_counts(tn, fp, fn, tp)
    if metric == "f1":
        return float(f1_score(labels, predictions, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else None
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    return float(_balanced_accuracy(sensitivity, specificity) or 0.0)


def _balanced_accuracy(sensitivity: float | None, specificity: float | None) -> float | None:
    observed_rates = [value for value in (sensitivity, specificity) if value is not None]
    if not observed_rates:
        return None
    return float(sum(observed_rates) / len(observed_rates))


def _mcc_from_counts(tn: int, fp: int, fn: int, tp: int) -> float:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / np.sqrt(denominator))
