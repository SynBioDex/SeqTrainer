"""SBOL3 export for deposited and SeqTrainer-predicted plasmid features."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

from .coordinate_conversion import sbol_orientation, sbol_ranges_for_location
from .ground_truth import GroundTruthPromoter
from .provenance import stable_identity
from .write_features import PromoterRegion


_ROLE_MAP = {
    "promoter": "SO:0000167",
    "cds": "SO:0000316",
    "rbs": "SO:0000552",
    "terminator": "SO:0000141",
    "origin": "SO:0000296",
    "operator": "SO:0000057",
}


def export_sbol3(
    record: Any,
    *,
    gold_promoters: Iterable[GroundTruthPromoter],
    predicted_regions: Iterable[PromoterRegion],
    output_path: str | Path,
    validation_path: str | Path | None = None,
    namespace: str = "https://seqtrainer.org/designs",
    source_url: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create, validate, write, and round-trip one deterministic SBOL3 file."""
    try:
        import sbol3
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            'SBOL3 export requires sbol3 and tyto. Install with `pip install -e ".[annotation]"`.'
        ) from exc

    sbol3.set_namespace(namespace)
    doc = sbol3.Document()
    plasmid_id = _safe_id(str(getattr(record, "id", "plasmid")))
    component_identity = stable_identity(namespace, "plasmids", plasmid_id)
    sequence_identity = stable_identity(namespace, "sequences", plasmid_id)
    roles = [sbol3.SO_DOUBLE_STRANDED]
    if str(getattr(record, "annotations", {}).get("topology", "")).lower() == "circular":
        roles.append(sbol3.SO_CIRCULAR)
    component = sbol3.Component(component_identity, sbol3.SBO_DNA, roles=roles, name=plasmid_id)
    sequence = sbol3.Sequence(sequence_identity, elements=str(record.seq).upper(), encoding=sbol3.IUPAC_DNA_ENCODING)
    doc.add(component)
    doc.add(sequence)
    component.sequences = [sequence]

    activity = sbol3.Activity(
        stable_identity(namespace, "activities", "seqtrainer_annotation"),
        name="SeqTrainer promoter annotation",
        description="One inference activity for this deterministic external evaluation run.",
    )
    doc.add(activity)
    if source_url:
        activity.derived_from = [source_url]

    for index, feature in enumerate(getattr(record, "features", [])):
        if "seqtrainer_model_family" in (getattr(feature, "qualifiers", {}) or {}):
            continue
        locations = _feature_locations(sbol3, sequence, feature, len(record.seq), namespace, plasmid_id, index)
        if not locations:
            continue
        feature_type = str(getattr(feature, "type", "unknown")).lower()
        role = _role_for_feature(sbol3, feature_type)
        kwargs: dict[str, Any] = {
            "locations": locations,
            "name": _feature_label(feature) or f"original_feature_{index:04d}",
            "description": f"Depositor-provided GenBank feature; source type={feature.type}.",
            "derived_from": [source_url] if source_url else None,
        }
        if role:
            kwargs["roles"] = [role]
        component.features.append(sbol3.SequenceFeature(**kwargs))

    for index, promoter in enumerate(gold_promoters):
        locations = _ground_truth_locations(sbol3, sequence, promoter, len(record.seq), namespace, plasmid_id, index)
        feature = sbol3.SequenceFeature(
            locations=locations,
            roles=[sbol3.SO_PROMOTER],
            name=promoter.label or f"deposited_promoter_{index:04d}",
            description=f"Depositor-provided promoter; evidence tier={promoter.evidence_tier}; rule={promoter.evidence_rule}.",
            derived_from=[source_url] if source_url else None,
        )
        component.features.append(feature)

    for index, promoter in enumerate(predicted_regions):
        locations = _region_locations(sbol3, sequence, promoter, len(record.seq), namespace, plasmid_id, index)
        details = provenance or {}
        description = (
            "SeqTrainer model-predicted promoter; "
            f"score={promoter.score:.6f}; threshold={details.get('threshold', 'unknown')}; "
            f"model={details.get('model_family', 'unknown')}"
        )
        feature = sbol3.SequenceFeature(
            locations=locations,
            roles=[sbol3.SO_PROMOTER],
            name=f"seqtrainer_predicted_promoter_{index:04d}",
            description=description,
            generated_by=[activity],
        )
        component.features.append(feature)

    report = doc.validate()
    validation = _validation_json(report)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if bool(report):
        if validation_path:
            Path(validation_path).write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        raise ValueError(f"SBOL3 validation failed; see {validation_path or 'validation report'}")
    doc.write(str(out))
    round_trip = sbol3.Document()
    round_trip.read(str(out))
    if validation_path:
        Path(validation_path).parent.mkdir(parents=True, exist_ok=True)
        Path(validation_path).write_text(json.dumps({**validation, "round_trip_objects": len(round_trip.objects)}, indent=2) + "\n", encoding="utf-8")
    return {"path": str(out), "validation": validation, "round_trip_objects": len(round_trip.objects), "sbol3_version": getattr(sbol3, "__version__", None), "python": platform.python_version(), "seqtrainer_python": sys.version}


def _feature_locations(sbol3: Any, sequence: Any, feature: Any, length: int, namespace: str, plasmid: str, index: int) -> list[Any]:
    ranges = sbol_ranges_for_location(feature.location, length)
    return [sbol3.Range(sequence, item["start"], item["end"], orientation=sbol_orientation(item["orientation"]), order=order) for order, item in enumerate(ranges)]


def _ground_truth_locations(sbol3: Any, sequence: Any, promoter: GroundTruthPromoter, length: int, namespace: str, plasmid: str, index: int) -> list[Any]:
    return _interval_locations(sbol3, sequence, promoter.start, promoter.end, promoter.strand, promoter.wraps_origin, length, namespace, plasmid, f"gold_{index}")


def _region_locations(sbol3: Any, sequence: Any, promoter: PromoterRegion, length: int, namespace: str, plasmid: str, index: int) -> list[Any]:
    strand = 1 if promoter.strand == "+" else -1 if promoter.strand == "-" else None
    return _interval_locations(sbol3, sequence, promoter.start, promoter.end, strand, promoter.crosses_boundary, length, namespace, plasmid, f"predicted_{index}")


def _interval_locations(sbol3: Any, sequence: Any, start: int, end: int, strand: int | None, wraps: bool, length: int, namespace: str, plasmid: str, key: str) -> list[Any]:
    intervals = [(start, length), (0, end % length)] if wraps or start > end else [(start, end)]
    orientation = sbol_orientation(strand)
    return [sbol3.Range(sequence, left + 1, right, orientation=orientation, order=index) for index, (left, right) in enumerate(intervals)]


def _role_for_feature(sbol3: Any, feature_type: str) -> str | None:
    value = _ROLE_MAP.get(feature_type)
    return {"SO:0000167": sbol3.SO_PROMOTER, "SO:0000316": "http://identifiers.org/so/SO:0000316", "SO:0000552": "http://identifiers.org/so/SO:0000552", "SO:0000141": "http://identifiers.org/so/SO:0000141", "SO:0000296": "http://identifiers.org/so/SO:0000296", "SO:0000057": "http://identifiers.org/so/SO:0000057"}.get(value) if value else None


def _feature_label(feature: Any) -> str:
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    for key in ("label", "gene", "product", "locus_tag", "note"):
        values = qualifiers.get(key)
        if values:
            return str(values[0])
    return ""


def _safe_id(value: str) -> str:
    clean = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    clean = clean.strip("_") or "plasmid"
    if not clean[0].isalpha():
        clean = f"plasmid_{clean}"
    return clean


def _validation_json(report: Any) -> dict[str, Any]:
    return {"valid": not bool(report), "errors": [str(item) for item in getattr(report, "errors", [])], "warnings": [str(item) for item in getattr(report, "warnings", [])]}
