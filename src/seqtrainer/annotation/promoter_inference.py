"""End-to-end promoter annotation MVP."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .genbank_io import read_genbank, record_topology, write_genbank
from .predictors import PromoterPredictor, build_predictor
from .windows import SequenceWindow, generate_sliding_windows
from .write_features import PromoterRegion, add_predicted_promoter_features


@dataclass(frozen=True)
class PromoterAnnotationConfig:
    """Configuration for promoter annotation over one GenBank record."""

    input_file: Path
    output_file: Path | None = None
    predictions_csv: Path | None = None
    manifest: Path | None = None
    model_family: str = "dummy"
    checkpoint: Path | None = None
    benchmark_manifest: Path | None = None
    threshold: float | None = None
    window_size: int | None = None
    step_size: int = 25
    scan_both_strands: bool = True
    merge_distance: int = 25
    min_score: float | None = None
    preserve_existing_features: bool = True
    gold_csv: Path | None = None
    evaluation_dir: Path | None = None
    sbol_output: Path | None = None
    sbol_namespace: str = "https://seqtrainer.org/designs"
    promoter_label_mode: str = "labelled"
    annotation_completeness: str = "unknown"
    iou_threshold: float = 0.50
    source_url: str | None = None


def run_promoter_annotation(
    config: PromoterAnnotationConfig,
    *,
    predictor: PromoterPredictor | None = None,
) -> dict[str, Any]:
    """Annotate likely promoter regions in a GenBank plasmid record."""
    record = read_genbank(config.input_file)
    original_feature_count = len(record.features)
    if not config.preserve_existing_features:
        record.features = []

    manifest_data, manifest_warnings = _load_manifest(
        config.benchmark_manifest,
        allow_missing=config.threshold is not None and config.window_size is not None,
    )
    threshold, threshold_source = _resolve_threshold(config.threshold, manifest_data)
    window_size = _resolve_window_size(config.window_size, manifest_data)
    step_size = config.step_size or 25

    predictor = predictor or build_predictor(
        config.model_family,
        checkpoint=config.checkpoint,
        benchmark_manifest=config.benchmark_manifest,
    )

    gold_promoters = None
    if config.evaluation_dir is not None or config.sbol_output is not None or config.gold_csv is not None:
        from .ground_truth import extract_ground_truth_promoters

        gold_promoters = extract_ground_truth_promoters(
            record,
            plasmid_id=str(record.id),
            source_url=config.source_url,
            label_mode=config.promoter_label_mode,
        )

    windows = generate_sliding_windows(
        str(record.seq),
        window_size=window_size,
        step_size=step_size,
        circular=record_topology(record) == "circular",
        scan_both_strands=config.scan_both_strands,
    )
    scores = predictor.predict_proba([window.sequence for window in windows])
    if len(scores) != len(windows):
        raise ValueError("Predictor returned a different number of scores than input windows")

    rows = []
    passing: list[tuple[SequenceWindow, float]] = []
    for window, raw_score in zip(windows, scores):
        score = float(raw_score)
        passed = score >= threshold and (config.min_score is None or score >= config.min_score)
        overlaps = _overlap_summary(record, window.start, window.end, window.is_circular_boundary_window)
        row = {
            "sequence_id": record.id,
            "window_id": window.window_id,
            "start": int(window.start),
            "end": int(window.end % len(record.seq) if window.is_circular_boundary_window else window.end),
            "strand": window.strand,
            "score": score,
            "threshold": threshold,
            "passed_threshold": bool(passed),
            "merged_region_id": "",
            "overlaps_existing_feature": overlaps["overlaps_existing_feature"],
            "overlaps_existing_promoter": overlaps["overlaps_existing_promoter"],
            "overlapping_feature_labels": ";".join(overlaps["overlapping_feature_labels"]),
            "is_circular_boundary_window": window.is_circular_boundary_window,
            "window_sequence": window.sequence,
        }
        rows.append(row)
        if passed:
            passing.append((window, score))

    regions = _merge_passing_windows(passing, merge_distance=config.merge_distance)
    region_by_window = {
        window_id: region.region_id
        for region in regions
        for window_id in region.source_window_ids
    }
    for row in rows:
        row["merged_region_id"] = region_by_window.get(row["window_id"], "")

    added, boundary_written = add_predicted_promoter_features(
        record,
        regions,
        model_family=config.model_family,
        threshold=threshold,
        window_size=window_size,
        step_size=step_size,
    )

    output_file, predictions_csv, manifest_path = _resolve_outputs(config)
    write_genbank(record, output_file)
    prediction_frame = pd.DataFrame(rows)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_csv, index=False)

    evaluation_artifacts = _write_external_evaluation(
        record,
        config=config,
        windows=windows,
        scores=scores,
        regions=regions,
        threshold=threshold,
        threshold_source=threshold_source,
        predictor=predictor,
        gold_promoters=gold_promoters,
    )

    warnings = list(manifest_warnings)
    if config.model_family == "dummy":
        warnings.append("Dummy predictor used for smoke testing only; do not treat scores as biological evidence.")
    if threshold_source == "default":
        warnings.append("No threshold found in benchmark manifest; used default threshold 0.80.")

    manifest = {
        "input_file": str(config.input_file),
        "input_sha256": _file_sha256(config.input_file),
        "output_file": str(output_file),
        "predictions_csv": str(predictions_csv),
        "sequence_id": record.id,
        "sequence_length": len(record.seq),
        "topology": record_topology(record),
        "model_family": config.model_family,
        "checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "checkpoint_sha256": _file_sha256(config.checkpoint) if config.checkpoint else None,
        "benchmark_manifest": str(config.benchmark_manifest) if config.benchmark_manifest else None,
        "benchmark_manifest_sha256": _file_sha256(config.benchmark_manifest) if config.benchmark_manifest else None,
        "source_url": config.source_url,
        "window_size": window_size,
        "step_size": step_size,
        "scan_both_strands": config.scan_both_strands,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "merge_distance": config.merge_distance,
        "total_windows_scanned": len(windows),
        "windows_above_threshold": len(passing),
        "predicted_promoters_added": added,
        "existing_features_preserved": original_feature_count if config.preserve_existing_features else 0,
        "overlaps_existing_promoters_count": int(prediction_frame["overlaps_existing_promoter"].sum()) if "overlaps_existing_promoter" in prediction_frame else 0,
        "circular_boundary_windows_scanned": int(prediction_frame["is_circular_boundary_window"].sum()) if "is_circular_boundary_window" in prediction_frame else 0,
        "circular_boundary_features_written": boundary_written,
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "predictor_metadata": predictor.metadata(),
        "annotation_completeness": config.annotation_completeness,
        "sbol_namespace": config.sbol_namespace if config.sbol_output else None,
        "evaluation": evaluation_artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest_file": str(manifest_path)}


def _write_external_evaluation(
    record: Any,
    *,
    config: PromoterAnnotationConfig,
    windows: list[SequenceWindow],
    scores: list[float],
    regions: list[PromoterRegion],
    threshold: float,
    threshold_source: str,
    predictor: PromoterPredictor,
    gold_promoters: list[Any] | None,
) -> dict[str, Any]:
    """Write labelled-plasmid evaluation artifacts when evaluation is requested."""
    if config.evaluation_dir is None and config.sbol_output is None and config.gold_csv is None:
        return {}
    from .evaluation import evaluate_merged_features, evaluate_windows
    from .ground_truth import write_gold_promoters
    from .provenance import model_provenance

    evaluation_dir = Path(config.evaluation_dir or Path(config.predictions_csv or "outputs/annotations").parent)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    gold = list(gold_promoters or [])
    gold_path = write_gold_promoters(gold, config.gold_csv or evaluation_dir / "gold_promoters.csv")
    window_frame, window_metrics = evaluate_windows(
        windows,
        scores,
        gold,
        threshold=threshold,
        sequence_length=len(record.seq),
        plasmid_id=str(record.id),
        predictor_method=config.model_family,
        model_version=str(predictor.metadata().get("model_name", config.model_family)),
        completeness=config.annotation_completeness,
    )
    merged_frame, merged_metrics = evaluate_merged_features(
        regions,
        gold,
        sequence_length=len(record.seq),
        plasmid_id=str(record.id),
        iou_thresholds=(0.10, 0.25, config.iou_threshold),
    )
    window_path = evaluation_dir / "window_predictions.csv"
    merged_path = evaluation_dir / "merged_predictions.csv"
    matches_path = evaluation_dir / "promoter_matches.csv"
    window_frame.to_csv(window_path, index=False)
    merged_frame.to_csv(merged_path, index=False)
    merged_frame.to_csv(matches_path, index=False)
    metrics = {"window": window_metrics, "merged": merged_metrics, "annotation_completeness": config.annotation_completeness}
    (evaluation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n", encoding="utf-8")
    pd.DataFrame([{"scope": "window", **_flat_metrics(window_metrics)}, {"scope": "merged", **_flat_metrics(merged_metrics)}]).to_csv(evaluation_dir / "metrics.csv", index=False)

    sbol_artifacts = {}
    if config.sbol_output:
        from .sbol3_export import export_sbol3

        sbol_artifacts = export_sbol3(
            record,
            gold_promoters=gold,
            predicted_regions=regions,
            output_path=config.sbol_output,
            validation_path=evaluation_dir / "sbol_validation.json",
            namespace=config.sbol_namespace,
            source_url=config.source_url,
            provenance={**model_provenance(checkpoint=config.checkpoint, benchmark_manifest=config.benchmark_manifest, model_family=config.model_family, threshold=threshold, threshold_source=threshold_source), **predictor.metadata()},
        )
    return {"evaluation_dir": str(evaluation_dir), "gold_csv": str(gold_path), "metrics_json": str(evaluation_dir / "metrics.json"), "sbol": sbol_artifacts}


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for key, value in metrics.items():
        flat[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    return flat


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_manifest(path: Path | None, *, allow_missing: bool = False) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {}, []
    if not path.exists():
        if allow_missing:
            return {}, [
                (
                    f"Benchmark manifest not found at {path}; continued with explicit CLI "
                    "threshold/window-size values."
                )
            ]
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8")), []


def _resolve_threshold(explicit: float | None, manifest: dict[str, Any]) -> tuple[float, str]:
    if explicit is not None:
        return float(explicit), "cli"
    candidates = [
        manifest.get("evaluation", {}).get("selected_threshold"),
        manifest.get("threshold_selection", {}).get("threshold"),
    ]
    for value in candidates:
        if value is not None:
            return float(value), "benchmark_manifest"
    return 0.80, "default"


def _resolve_window_size(explicit: int | None, manifest: dict[str, Any]) -> int:
    if explicit is not None:
        return int(explicit)
    candidates = [
        manifest.get("preprocessing", {}).get("sequence_length"),
        manifest.get("model", {}).get("params", {}).get("max_length"),
        manifest.get("model", {}).get("params", {}).get("model_max_length"),
    ]
    for value in candidates:
        if value:
            return int(value)
    return 300


def _resolve_outputs(config: PromoterAnnotationConfig) -> tuple[Path, Path, Path]:
    out_dir = Path("outputs") / "annotations"
    stem = config.input_file.stem
    output_file = config.output_file or out_dir / f"{stem}_{config.model_family}_annotated.gb"
    predictions_csv = config.predictions_csv or out_dir / f"{stem}_{config.model_family}_predictions.csv"
    manifest = config.manifest or out_dir / f"{stem}_{config.model_family}_manifest.json"
    return output_file, predictions_csv, manifest


def _merge_passing_windows(
    passing: list[tuple[SequenceWindow, float]],
    *,
    merge_distance: int,
) -> list[PromoterRegion]:
    sorted_windows = sorted(passing, key=lambda item: (item[0].strand, item[0].start, item[0].end))
    regions: list[PromoterRegion] = []
    current: dict[str, Any] | None = None
    for window, score in sorted_windows:
        if (
            current is None
            or current["strand"] != window.strand
            or window.start > current["end"] + merge_distance
            or window.is_circular_boundary_window != current["crosses_boundary"]
        ):
            if current is not None:
                regions.append(_region_from_state(current, len(regions)))
            current = {
                "start": window.start,
                "end": window.end,
                "strand": window.strand,
                "score": score,
                "windows": [window.window_id],
                "crosses_boundary": window.is_circular_boundary_window,
            }
            continue
        current["end"] = max(current["end"], window.end)
        current["score"] = max(current["score"], score)
        current["windows"].append(window.window_id)
    if current is not None:
        regions.append(_region_from_state(current, len(regions)))
    return regions


def _region_from_state(state: dict[str, Any], idx: int) -> PromoterRegion:
    return PromoterRegion(
        region_id=f"predicted_promoter_{idx}",
        start=int(state["start"]),
        end=int(state["end"]),
        strand=str(state["strand"]),
        score=float(state["score"]),
        source_window_ids=tuple(state["windows"]),
        crosses_boundary=bool(state["crosses_boundary"]),
    )


def _overlap_summary(record: Any, start: int, end: int, crosses_boundary: bool) -> dict[str, Any]:
    intervals = _window_intervals(start, end, len(record.seq), crosses_boundary)
    labels: list[str] = []
    promoter_overlap = False
    for feature in record.features:
        feature_intervals = _feature_intervals(feature)
        if not any(_intervals_overlap(a, b) for a in intervals for b in feature_intervals):
            continue
        label = _feature_label(feature)
        if label:
            labels.append(label)
        if feature.type == "promoter" or "promoter" in label.lower():
            promoter_overlap = True
    return {
        "overlaps_existing_feature": bool(labels),
        "overlaps_existing_promoter": promoter_overlap,
        "overlapping_feature_labels": sorted(set(labels)),
    }


def _window_intervals(start: int, end: int, seq_len: int, crosses_boundary: bool) -> list[tuple[int, int]]:
    if crosses_boundary:
        return [(start, seq_len), (0, end % seq_len)]
    return [(start, min(end, seq_len))]


def _feature_intervals(feature: Any) -> list[tuple[int, int]]:
    parts = getattr(feature.location, "parts", None)
    locations = list(parts) if parts is not None else [feature.location]
    return [(int(location.start), int(location.end)) for location in locations]


def _intervals_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _feature_label(feature: Any) -> str:
    for key in ("label", "gene", "product", "note", "locus_tag"):
        values = feature.qualifiers.get(key)
        if values:
            return str(values[0])
    return feature.type


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
