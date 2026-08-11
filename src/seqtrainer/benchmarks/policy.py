"""Shared benchmark policies for class balance and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImbalancePolicy:
    """Decision about whether training-time imbalance handling is warranted."""

    apply_to_training: bool
    strategy: str
    class_counts: dict[str, int]
    imbalance_ratio: float | None
    reason: str


def decide_imbalance_policy(
    split_summary: dict[str, Any],
    *,
    split: str = "train",
    ratio_threshold: float = 1.5,
    strategy: str = "class_weighting",
) -> ImbalancePolicy:
    """Decide whether to use training-only imbalance handling.

    The policy intentionally uses only the training split. Validation/test
    distributions are recorded for interpretation, but they should not drive
    training-time weighting or sampling decisions.
    """
    if split != "train":
        raise ValueError("Imbalance handling policy may only be decided from the training split")

    class_counts = {
        str(label): int(count)
        for label, count in split_summary.get(split, {}).get("class_counts", {}).items()
    }
    if len(class_counts) < 2:
        return ImbalancePolicy(
            apply_to_training=False,
            strategy="none",
            class_counts=class_counts,
            imbalance_ratio=None,
            reason="Training split has fewer than two observed classes.",
        )

    nonzero_counts = [count for count in class_counts.values() if count > 0]
    if len(nonzero_counts) < 2:
        return ImbalancePolicy(
            apply_to_training=False,
            strategy="none",
            class_counts=class_counts,
            imbalance_ratio=None,
            reason="At least one class has zero training examples.",
        )

    imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)
    if imbalance_ratio >= ratio_threshold:
        return ImbalancePolicy(
            apply_to_training=True,
            strategy=strategy,
            class_counts=class_counts,
            imbalance_ratio=float(imbalance_ratio),
            reason=f"Training class ratio {imbalance_ratio:.3f} exceeds threshold {ratio_threshold:.3f}.",
        )

    return ImbalancePolicy(
        apply_to_training=False,
        strategy="none",
        class_counts=class_counts,
        imbalance_ratio=float(imbalance_ratio),
        reason=f"Training class ratio {imbalance_ratio:.3f} is below threshold {ratio_threshold:.3f}.",
    )


def threshold_metric_from_strategy(strategy: str) -> str | None:
    """Map config threshold strategies to validation metric names."""
    mapping = {
        "validation_mcc": "mcc",
        "validation_f1": "f1",
        "validation_balanced_accuracy": "balanced_accuracy",
        "fixed_0_5": None,
        "none": None,
    }
    if strategy not in mapping:
        raise ValueError(f"Unsupported threshold strategy: {strategy}")
    return mapping[strategy]
