"""Window- and merged-feature evaluation for labelled external plasmids."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from seqtrainer.metrics.classification import binary_classification_metrics

from .ground_truth import GroundTruthPromoter


def window_gold_labels(
    windows: Iterable[Any],
    gold: Iterable[GroundTruthPromoter],
    sequence_length: int,
) -> tuple[list[int], list[str]]:
    """Label a window positive when its centre is inside same-strand gold."""
    gold_list = list(gold)
    labels: list[int] = []
    ids: list[str] = []
    for window in windows:
        centre = (int(window.start) + (int(window.end) - int(window.start)) // 2) % sequence_length
        matches = [item for item in gold_list if _strand_matches(window.strand, item.strand) and _point_in_promoter(centre, item, sequence_length)]
        labels.append(int(bool(matches)))
        ids.append(";".join(item.gold_id for item in matches))
    return labels, ids


def evaluate_windows(
    windows: Iterable[Any],
    scores: Iterable[float],
    gold: Iterable[GroundTruthPromoter],
    *,
    threshold: float,
    sequence_length: int,
    plasmid_id: str,
    predictor_method: str,
    model_version: str | None,
    completeness: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    windows_list = list(windows)
    score_list = [float(score) for score in scores]
    labels, overlapping = window_gold_labels(windows_list, gold, sequence_length)
    rows = []
    for window, score, label, gold_ids in zip(windows_list, score_list, labels, overlapping):
        rows.append({
            "plasmid_id": plasmid_id, "record_id": plasmid_id,
            "start": int(window.start), "end": int(window.end % sequence_length if window.is_circular_boundary_window else window.end),
            "strand": window.strand, "wraps_origin": bool(window.is_circular_boundary_window),
            "sequence": window.sequence, "score": score, "threshold": float(threshold),
            "predicted_label": int(score >= threshold), "gold_label": label,
            "overlapping_gold_ids": gold_ids, "predictor_method": predictor_method,
            "model_version": model_version or "unknown",
        })
    frame = pd.DataFrame(rows)
    if completeness == "verified_complete" and len(frame):
        metrics = binary_classification_metrics(frame["gold_label"].to_numpy(), frame["score"].to_numpy(), threshold)
    else:
        metrics = {"warning": "Partial-label metrics: unlabeled windows are not confirmed biological negatives.", "labelled_promoter_recovery": _recovery(frame, gold)}
    return frame, metrics


def evaluate_merged_features(
    predictions: Iterable[Any],
    gold: Iterable[GroundTruthPromoter],
    *,
    sequence_length: int,
    plasmid_id: str,
    iou_thresholds: tuple[float, ...] = (0.10, 0.25, 0.50),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predicted = list(predictions)
    gold_list = list(gold)
    candidate_pairs = sorted(
        (
            _iou(prediction, gold_item, sequence_length),
            prediction_index,
            gold_index,
        )
        for prediction_index, prediction in enumerate(predicted)
        for gold_index, gold_item in enumerate(gold_list)
        if _strand_matches(prediction.strand, gold_item.strand)
    )
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_predictions: set[int] = set()
    matched_gold: set[int] = set()
    assignments: dict[int, tuple[int, float]] = {}
    for score, prediction_index, gold_index in candidate_pairs:
        if prediction_index in matched_predictions or gold_index in matched_gold:
            continue
        matched_predictions.add(prediction_index)
        matched_gold.add(gold_index)
        assignments[gold_index] = (prediction_index, score)
    rows = []
    for gold_index, item in enumerate(gold_list):
        assignment = assignments.get(gold_index)
        pred_index, score = assignment if assignment else (None, 0.0)
        prediction_id = getattr(predicted[pred_index], "region_id", "") if pred_index is not None else ""
        rows.append({
            "plasmid_id": plasmid_id, "gold_id": item.gold_id, "best_prediction_id": prediction_id,
            "same_strand": bool(pred_index is not None), "intersection_bp": _intersection(predicted[pred_index], item, sequence_length) if pred_index is not None else 0,
            "union_bp": _union(predicted[pred_index], item, sequence_length) if pred_index is not None else 0,
            "iou": score, "start_error_bp": _boundary_error(predicted[pred_index], item, sequence_length, "start") if pred_index is not None else None,
            "end_error_bp": _boundary_error(predicted[pred_index], item, sequence_length, "end") if pred_index is not None else None,
            **{f"matched_at_{str(t).replace('.', '_')}": bool(score >= t) for t in iou_thresholds},
        })
    frame = pd.DataFrame(rows)
    at_half = int(frame["matched_at_0_5"].sum()) if len(frame) and "matched_at_0_5" in frame else 0
    scores = frame["iou"].to_numpy(dtype=float) if len(frame) else np.array([])
    boundary_values = [value for value in frame[["start_error_bp", "end_error_bp"]].to_numpy().ravel() if pd.notna(value)] if len(frame) else []
    metrics = {
        "gold_promoter_count": len(gold_list), "predicted_promoter_count": len(predicted),
        "matched_promoter_count": at_half, "labelled_promoter_recall": float(at_half / len(gold_list)) if gold_list else None,
        "mean_best_iou": float(scores.mean()) if len(scores) else 0.0,
        "median_best_iou": float(np.median(scores)) if len(scores) else 0.0,
        "median_boundary_error": float(np.median(boundary_values)) if boundary_values else None,
        "predictions_per_kilobase": float(len(predicted) / sequence_length * 1000) if sequence_length else None,
        "iou_thresholds": list(iou_thresholds),
    }
    return frame, metrics


def _strand_matches(first: str | int | None, second: str | int | None) -> bool:
    first_value = 1 if first in ("+", 1) else -1 if first in ("-", -1) else None
    return first_value == second


def _point_in_promoter(point: int, promoter: GroundTruthPromoter, sequence_length: int) -> bool:
    if promoter.wraps_origin or promoter.start > promoter.end:
        return point >= promoter.start or point < promoter.end
    return promoter.start <= point < promoter.end


def _intervals(item: Any, sequence_length: int) -> list[tuple[int, int]]:
    start, end = int(item.start), int(item.end)
    if getattr(item, "wraps_origin", False) or start > end or end > sequence_length:
        return [(start, sequence_length), (0, end % sequence_length)]
    return [(start, end)]


def _iou(first: Any, second: Any, sequence_length: int) -> float:
    inter = _intersection(first, second, sequence_length)
    union = _union(first, second, sequence_length)
    return float(inter / union) if union else 0.0


def _intersection(first: Any, second: Any, sequence_length: int) -> int:
    return sum(max(0, min(a[1], b[1]) - max(a[0], b[0])) for a in _intervals(first, sequence_length) for b in _intervals(second, sequence_length))


def _union(first: Any, second: Any, sequence_length: int) -> int:
    intervals = sorted(_intervals(first, sequence_length) + _intervals(second, sequence_length))
    total = 0
    current_end = None
    for start, end in intervals:
        if current_end is None or start > current_end:
            total += end - start
        else:
            total += max(0, end - current_end)
        current_end = max(current_end or 0, end)
    return total


def _boundary_error(prediction: Any, gold: Any, sequence_length: int, field: str) -> int:
    return abs(int(getattr(prediction, field)) - int(getattr(gold, field)))


def _recovery(frame: pd.DataFrame, gold: Iterable[GroundTruthPromoter]) -> dict[str, Any]:
    gold_list = list(gold)
    if not gold_list:
        return {"gold_promoter_count": 0, "recovered": 0, "recall": None}
    recovered_ids: set[str] = set()
    if len(frame):
        for value in frame.loc[(frame["predicted_label"] == 1) & (frame["gold_label"] == 1), "overlapping_gold_ids"]:
            recovered_ids.update(item for item in str(value).split(";") if item)
    recovered = len(recovered_ids)
    return {"gold_promoter_count": len(gold_list), "recovered": recovered, "recovered_gold_ids": sorted(recovered_ids), "recall": float(recovered / len(gold_list))}
