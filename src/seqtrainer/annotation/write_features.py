"""Write predicted promoter features into GenBank records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromoterRegion:
    """Merged predicted promoter region."""

    region_id: str
    start: int
    end: int
    strand: str
    score: float
    source_window_ids: tuple[str, ...]
    crosses_boundary: bool = False


def add_predicted_promoter_features(
    record: Any,
    regions: list[PromoterRegion],
    *,
    model_family: str,
    threshold: float,
    window_size: int,
    step_size: int,
) -> tuple[int, int]:
    """Append predicted promoter SeqFeatures to a record.

    Returns ``(features_added, circular_boundary_features_written)``.
    """
    try:
        from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "GenBank annotation requires Biopython. Install with `pip install -e \".[annotation]\"`."
        ) from exc

    seq_len = len(record.seq)
    added = 0
    boundary_written = 0
    for region in regions:
        strand_value = 1 if region.strand == "+" else -1
        if region.crosses_boundary:
            first = FeatureLocation(region.start, seq_len, strand=strand_value)
            second = FeatureLocation(0, region.end % seq_len, strand=strand_value)
            location = CompoundLocation([first, second])
            boundary_written += 1
        else:
            location = FeatureLocation(region.start, min(region.end, seq_len), strand=strand_value)

        score = f"{region.score:.6f}"
        qualifiers = {
            "label": ["predicted_promoter"],
            "note": [
                "SeqTrainer predicted promoter; "
                f"model={model_family}; score={score}; threshold={threshold:.6f}; "
                "computational prediction only"
            ],
            "inference": ["SeqTrainer promoter inference"],
            "seqtrainer_model_family": [model_family],
            "seqtrainer_score": [score],
            "seqtrainer_threshold": [f"{threshold:.6f}"],
            "seqtrainer_window_size": [str(window_size)],
            "seqtrainer_step_size": [str(step_size)],
            "seqtrainer_region_id": [region.region_id],
        }
        record.features.append(SeqFeature(location=location, type="promoter", qualifiers=qualifiers))
        added += 1
    return added, boundary_written

