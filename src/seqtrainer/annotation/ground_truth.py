"""Auditable extraction of explicitly labelled promoter ground truth."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class GroundTruthPromoter:
    plasmid_id: str
    record_id: str
    start: int
    end: int
    strand: int | None
    wraps_origin: bool
    label: str
    feature_type: str
    evidence_tier: str
    evidence_rule: str
    raw_qualifiers: str
    source_url: str | None = None
    gold_id: str = ""


def extract_ground_truth_promoters(
    record: Any,
    *,
    plasmid_id: str | None = None,
    source_url: str | None = None,
    label_mode: str = "labelled",
) -> list[GroundTruthPromoter]:
    """Extract only explicit promoter annotations from one Biopython record.

    Coordinates are 0-based/end-exclusive. Tier A uses feature type, explicit
    regulatory class, or SO:0000167. Tier B requires a standalone ``promoter``
    term in a label/name/note and rejects ``promoterless``/``no promoter``.
    """
    if label_mode not in {"strict", "labelled"}:
        raise ValueError("label_mode must be 'strict' or 'labelled'")
    plasmid = plasmid_id or str(getattr(record, "id", "record"))
    record_id = str(getattr(record, "id", "record"))
    result: list[GroundTruthPromoter] = []
    for feature_index, feature in enumerate(getattr(record, "features", [])):
        match = _promoter_match(feature, strict=label_mode == "strict")
        if match is None:
            continue
        intervals = _location_intervals(feature)
        if not intervals:
            continue
        start = intervals[0][0]
        end = intervals[-1][1]
        wraps = len(intervals) > 1 and intervals[0][0] > intervals[-1][1]
        label = _first_qualifier(feature, ("label", "name", "gene", "note")) or feature.type
        raw = json.dumps(dict(getattr(feature, "qualifiers", {})), sort_keys=True, default=str)
        result.append(
            GroundTruthPromoter(
                plasmid_id=plasmid,
                record_id=record_id,
                gold_id=f"{plasmid}:gold_promoter_{len(result):04d}",
                start=int(start),
                end=int(end),
                strand=getattr(feature.location, "strand", None),
                wraps_origin=wraps,
                label=label,
                feature_type=str(feature.type),
                evidence_tier=match[0],
                evidence_rule=match[1],
                raw_qualifiers=raw,
                source_url=source_url,
            )
        )
    return result


def write_gold_promoters(promoters: Iterable[GroundTruthPromoter], path: str | Path) -> Path:
    """Write the stable gold-promoter CSV schema."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "plasmid_id", "record_id", "gold_id", "start", "end", "strand", "wraps_origin",
        "label", "feature_type", "evidence_tier", "evidence_rule", "raw_qualifiers", "source_url",
    ]
    rows = [asdict(promoter) for promoter in promoters]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _promoter_match(feature: Any, *, strict: bool) -> tuple[str, str] | None:
    feature_type = str(getattr(feature, "type", "")).lower()
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    regulatory = " ".join(_qualifier_values(qualifiers, "regulatory_class")).lower()
    xrefs = " ".join(_qualifier_values(qualifiers, "db_xref")).lower()
    if feature_type == "promoter":
        return "A", "feature_type=promoter"
    if feature_type == "regulatory" and "promoter" in regulatory:
        return "A", "regulatory_class_contains=promoter"
    if "so:0000167" in xrefs or "so_0000167" in xrefs:
        return "A", "db_xref=SO:0000167"
    if strict:
        return None
    candidates = _qualifier_values(qualifiers, "label") + _qualifier_values(qualifiers, "name") + _qualifier_values(qualifiers, "note")
    for value in candidates:
        text = value.lower()
        if "promoterless" in text or "no promoter" in text:
            continue
        words = text.replace("_", " ").replace("-", " ").split()
        if "promoter" in words:
            return "B", "explicit_label_contains_standalone_promoter"
    return None


def _location_intervals(feature: Any) -> list[tuple[int, int]]:
    location = getattr(feature, "location", None)
    if location is None:
        return []
    parts = getattr(location, "parts", None)
    locations = list(parts) if parts is not None else [location]
    return [(int(part.start), int(part.end)) for part in locations]


def _qualifier_values(qualifiers: dict[str, Any], key: str) -> list[str]:
    values = qualifiers.get(key, [])
    if not isinstance(values, (list, tuple)):
        values = [values]
    return [str(value) for value in values]


def _first_qualifier(feature: Any, keys: tuple[str, ...]) -> str:
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    for key in keys:
        values = _qualifier_values(qualifiers, key)
        if values:
            return values[0]
    return ""
