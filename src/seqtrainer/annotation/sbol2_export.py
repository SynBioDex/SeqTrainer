"""SBOL2 RDF/XML export for graphical tools such as SBOLCanvas."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Iterable

from .coordinate_conversion import sbol_ranges_for_location
from .ground_truth import GroundTruthPromoter
from .write_features import PromoterRegion


_ROLE_MAP = {
    # SBOLCanvas assumes every child ComponentDefinition has at least one role
    # while constructing a visual glyph. Use the Sequence Ontology's generic
    # sequence-feature role for GenBank features that do not map more narrowly.
    "sequence_feature": "http://identifiers.org/so/SO:0000110",
    "promoter": "http://identifiers.org/so/SO:0000167",
    "cds": "http://identifiers.org/so/SO:0000316",
    "rbs": "http://identifiers.org/so/SO:0000552",
    "terminator": "http://identifiers.org/so/SO:0000141",
    "origin": "http://identifiers.org/so/SO:0000296",
    "operator": "http://identifiers.org/so/SO:0000057",
}


def export_sbol2(
    record: Any,
    *,
    gold_promoters: Iterable[GroundTruthPromoter],
    predicted_regions: Iterable[PromoterRegion],
    output_path: str | Path,
    namespace: str = "https://seqtrainer.org/designs",
    source_url: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an SBOL2 RDF/XML document with Canvas-renderable components."""
    try:
        import sbol2
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime
        raise ModuleNotFoundError(
            'SBOL2 export requires pySBOL2. Install with `pip install -e ".[annotation]"`.'
        ) from exc

    sbol2.setHomespace(namespace)
    doc = sbol2.Document()
    plasmid_id = _safe_id(str(getattr(record, "id", "plasmid")))
    parent = sbol2.ComponentDefinition(plasmid_id, component_type=sbol2.BIOPAX_DNA)
    parent.name = plasmid_id
    parent.description = "SeqTrainer annotated DNA design."
    if str(getattr(record, "annotations", {}).get("topology", "")).lower() == "circular":
        parent.addRole(sbol2.SO_CIRCULAR)

    sequence = sbol2.Sequence(
        f"{plasmid_id}_sequence",
        elements=str(record.seq).upper(),
        encoding=sbol2.SBOL_ENCODING_IUPAC,
    )
    parent.sequences = [sequence]
    doc.add(parent)
    doc.add(sequence)

    used_ids: set[str] = set()
    source_count = 0
    gold_count = 0
    predicted_count = 0

    for index, feature in enumerate(getattr(record, "features", [])):
        qualifiers = getattr(feature, "qualifiers", {}) or {}
        if "seqtrainer_model_family" in qualifiers:
            continue
        locations = sbol_ranges_for_location(feature.location, len(record.seq))
        if not locations:
            continue
        label = _feature_label(feature) or f"original_feature_{index:04d}"
        feature_id = _unique_id(label, used_ids)
        _add_feature(
            doc,
            parent,
            feature_id=feature_id,
            label=label,
            locations=locations,
            role=_role_for_feature(str(getattr(feature, "type", "")).lower(), label),
            description=f"Depositor-provided GenBank feature; source type={feature.type}.",
            metadata={"source": "genbank", "source_url": source_url},
        )
        source_count += 1

    for index, promoter in enumerate(gold_promoters):
        feature_id = _unique_id(promoter.label or f"deposited_promoter_{index:04d}", used_ids)
        _add_feature(
            doc,
            parent,
            feature_id=feature_id,
            label=promoter.label or feature_id,
            locations=_interval_locations(promoter.start, promoter.end, promoter.strand, promoter.wraps_origin, len(record.seq)),
            role=sbol2.SO_PROMOTER,
            description=f"Depositor-provided promoter; evidence tier={promoter.evidence_tier}.",
            metadata={"source": "ground_truth", "evidence_tier": promoter.evidence_tier, "source_url": source_url},
        )
        gold_count += 1

    details = provenance or {}
    for index, promoter in enumerate(predicted_regions):
        feature_id = _unique_id(f"seqtrainer_predicted_promoter_{index:04d}", used_ids)
        _add_feature(
            doc,
            parent,
            feature_id=feature_id,
            label="predicted_promoter",
            locations=_interval_locations(promoter.start, promoter.end, promoter.strand, promoter.crosses_boundary, len(record.seq)),
            role=sbol2.SO_PROMOTER,
            description=(
                f"SeqTrainer predicted promoter; score={promoter.score:.6f}; "
                f"threshold={details.get('threshold', 'unknown')}; "
                f"model={details.get('model_family', 'unknown')}."
            ),
            metadata={
                "source": "seqtrainer_prediction",
                "score": promoter.score,
                "threshold": details.get("threshold"),
                "model_family": details.get("model_family"),
                "region_id": promoter.region_id,
            },
        )
        predicted_count += 1

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.write(str(out))
    round_trip = sbol2.Document()
    round_trip.read(str(out))
    return {
        "path": str(out),
        "format": "SBOL2 RDF/XML",
        "round_trip_objects": len(round_trip),
        "source_feature_count": source_count,
        "gold_promoter_count": gold_count,
        "predicted_promoter_count": predicted_count,
        "python": platform.python_version(),
        "seqtrainer_python": sys.version,
    }


def _add_feature(
    doc: Any,
    parent: Any,
    *,
    feature_id: str,
    label: str,
    locations: list[dict[str, Any]],
    role: str | None,
    description: str,
    metadata: dict[str, Any],
) -> None:
    import sbol2

    definition = sbol2.ComponentDefinition(feature_id, component_type=sbol2.BIOPAX_DNA)
    definition.name = label
    metadata_text = "; ".join(f"{key}={value}" for key, value in metadata.items() if value is not None)
    definition.description = f"{description} {metadata_text}".strip()
    if role:
        definition.addRole(role)
    doc.add(definition)

    component = sbol2.Component(feature_id, definition=definition.identity)
    parent.components.add(component)
    # pySBOL2 rewrites the component identity when it becomes owned by the
    # parent, and its constructor does not reliably retain the definition URI.
    # Reassign after ownership so SBOL2 rule sbol-10602 is satisfied.
    component.definition = definition.identity
    annotation = sbol2.SequenceAnnotation(f"{feature_id}_annotation")
    annotation.component = component
    for order, item in enumerate(locations):
        location = sbol2.Range(f"{feature_id}_range_{order}", int(item["start"]), int(item["end"]))
        location.orientation = (
            sbol2.SBOL_ORIENTATION_REVERSE_COMPLEMENT
            if item.get("orientation") == -1
            else sbol2.SBOL_ORIENTATION_INLINE
        )
        annotation.locations.add(location)
    parent.sequenceAnnotations.add(annotation)


def _interval_locations(start: int, end: int, strand: str | int | None, wraps: bool, length: int) -> list[dict[str, Any]]:
    intervals = [(start, length), (0, end % length)] if wraps or start > end else [(start, end)]
    orientation = -1 if strand in ("-", -1) else 1
    return [{"start": left + 1, "end": right, "orientation": orientation} for left, right in intervals]


def _role_for_feature(feature_type: str, label: str) -> str | None:
    if feature_type in _ROLE_MAP:
        return _ROLE_MAP[feature_type]
    if "promoter" in label.lower():
        return _ROLE_MAP["promoter"]
    return _ROLE_MAP["sequence_feature"]


def _feature_label(feature: Any) -> str:
    qualifiers = getattr(feature, "qualifiers", {}) or {}
    for key in ("label", "gene", "product", "locus_tag", "note"):
        values = qualifiers.get(key)
        if values:
            return str(values[0])
    return ""


def _unique_id(value: str, used: set[str]) -> str:
    base = _safe_id(value)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _safe_id(value: str) -> str:
    clean = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    clean = clean.strip("_") or "feature"
    if not clean[0].isalpha():
        clean = f"feature_{clean}"
    return clean
