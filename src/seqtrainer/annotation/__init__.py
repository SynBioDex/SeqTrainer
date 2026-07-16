"""Promoter annotation workflows for SeqTrainer."""

from .promoter_inference import PromoterAnnotationConfig, run_promoter_annotation
from .ground_truth import GroundTruthPromoter, extract_ground_truth_promoters, write_gold_promoters

__all__ = [
    "PromoterAnnotationConfig",
    "GroundTruthPromoter",
    "extract_ground_truth_promoters",
    "run_promoter_annotation",
    "write_gold_promoters",
]

