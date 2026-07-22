from pathlib import Path

import pandas as pd

from seqtrainer.annotation import PromoterAnnotationConfig, run_promoter_annotation
from seqtrainer.annotation.genbank_io import read_genbank
from seqtrainer.annotation.windows import generate_sliding_windows, reverse_complement
from seqtrainer.cli.main import main


def _write_synthetic_genbank(path: Path) -> Path:
    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    record = SeqRecord(
        Seq("TA" + "C" * 20 + "TA"),
        id="synthetic_plasmid",
        name="synthetic_plasmid",
        description="Synthetic circular plasmid for SeqTrainer annotation tests",
    )
    record.annotations["molecule_type"] = "DNA"
    record.annotations["topology"] = "circular"
    record.features = [
        SeqFeature(
            FeatureLocation(2, 8, strand=1),
            type="promoter",
            qualifiers={"label": ["existing_exact_hit_promoter"]},
        ),
        SeqFeature(
            FeatureLocation(10, 18, strand=1),
            type="CDS",
            qualifiers={"label": ["existing_cds"]},
        ),
    ]
    SeqIO.write(record, str(path), "genbank")
    return path


def test_reverse_complement_and_circular_windows():
    assert reverse_complement("ACGTN") == "NACGT"

    windows = generate_sliding_windows(
        "TA" + "C" * 20 + "TA",
        window_size=8,
        step_size=4,
        circular=True,
        scan_both_strands=True,
    )

    assert any(window.strand == "-" for window in windows)
    assert any(window.is_circular_boundary_window for window in windows)
    assert any("TATA" in window.sequence for window in windows)


def test_promoter_annotation_dummy_preserves_features_and_writes_outputs(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")
    output_gb = tmp_path / "annotated.gb"
    predictions_csv = tmp_path / "predictions.csv"
    manifest_json = tmp_path / "manifest.json"

    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_gb,
            output_file=output_gb,
            predictions_csv=predictions_csv,
            manifest=manifest_json,
            model_family="dummy",
            threshold=0.80,
            window_size=8,
            step_size=4,
            scan_both_strands=True,
            merge_distance=0,
        )
    )

    assert output_gb.exists()
    assert predictions_csv.exists()
    assert manifest_json.exists()
    assert manifest["topology"] == "circular"
    assert manifest["existing_features_preserved"] == 2
    assert manifest["predicted_promoters_added"] >= 1
    assert manifest["circular_boundary_windows_scanned"] >= 1

    annotated = read_genbank(output_gb)
    labels = [feature.qualifiers.get("label", [""])[0] for feature in annotated.features]
    assert "existing_exact_hit_promoter" in labels
    assert "predicted_promoter" in labels
    assert len(annotated.features) > 2

    predictions = pd.read_csv(predictions_csv)
    required_columns = {
        "sequence_id",
        "window_id",
        "start",
        "end",
        "strand",
        "score",
        "threshold",
        "passed_threshold",
        "merged_region_id",
        "overlaps_existing_feature",
        "overlaps_existing_promoter",
        "overlapping_feature_labels",
        "is_circular_boundary_window",
        "window_sequence",
    }
    assert required_columns.issubset(predictions.columns)
    assert predictions["passed_threshold"].any()
    assert predictions["is_circular_boundary_window"].any()


def test_annotation_cli_dummy_smoke(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")
    output_gb = tmp_path / "cli_annotated.gb"
    predictions_csv = tmp_path / "cli_predictions.csv"
    manifest_json = tmp_path / "cli_manifest.json"

    exit_code = main(
        [
            "annotate",
            "promoters",
            str(input_gb),
            "--model-family",
            "dummy",
            "--threshold",
            "0.80",
            "--window-size",
            "8",
            "--step-size",
            "4",
            "--output",
            str(output_gb),
            "--predictions-csv",
            str(predictions_csv),
            "--manifest",
            str(manifest_json),
        ]
    )

    assert exit_code == 0
    assert output_gb.exists()
    assert predictions_csv.exists()
    assert manifest_json.exists()


def test_annotation_uses_threshold_and_window_from_benchmark_manifest(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(
        """
{
  "evaluation": {"selected_threshold": 0.9},
  "preprocessing": {"sequence_length": 8}
}
""",
        encoding="utf-8",
    )

    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_gb,
            output_file=tmp_path / "annotated.gb",
            predictions_csv=tmp_path / "predictions.csv",
            manifest=tmp_path / "manifest.json",
            model_family="dummy",
            benchmark_manifest=benchmark_manifest,
            step_size=4,
        )
    )

    assert manifest["threshold"] == 0.9
    assert manifest["threshold_source"] == "benchmark_manifest"
    assert manifest["window_size"] == 8


def test_annotation_accepts_windows_utf8_bom_benchmark_manifest(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")
    benchmark_manifest = tmp_path / "benchmark_manifest.json"
    benchmark_manifest.write_text(
        '{"evaluation": {"selected_threshold": 0.9}, "preprocessing": {"sequence_length": 8}}',
        encoding="utf-8-sig",
    )

    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_gb,
            output_file=tmp_path / "annotated.gb",
            predictions_csv=tmp_path / "predictions.csv",
            manifest=tmp_path / "manifest.json",
            model_family="dummy",
            benchmark_manifest=benchmark_manifest,
            step_size=4,
        )
    )

    assert manifest["threshold"] == 0.9
    assert manifest["window_size"] == 8


def test_annotation_resolves_model_bundle_paths(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")
    bundle = tmp_path / "model_bundle"
    (bundle / "checkpoints").mkdir(parents=True)
    checkpoint = bundle / "checkpoints" / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")
    benchmark_manifest = bundle / "manifest.json"
    benchmark_manifest.write_text(
        '{"evaluation": {"selected_threshold": 0.91}, "preprocessing": {"sequence_length": 8}}',
        encoding="utf-8",
    )

    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_gb,
            output_file=tmp_path / "annotated.gb",
            predictions_csv=tmp_path / "predictions.csv",
            manifest=tmp_path / "annotation_manifest.json",
            model_family="dummy",
            model_bundle=bundle,
            step_size=4,
        )
    )

    assert manifest["checkpoint"] == str(checkpoint)
    assert manifest["benchmark_manifest"] == str(benchmark_manifest)
    assert manifest["model_bundle"] == str(bundle)
    assert manifest["threshold"] == 0.91


def test_annotation_allows_missing_manifest_when_cli_values_are_explicit(tmp_path):
    input_gb = _write_synthetic_genbank(tmp_path / "input.gb")

    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_gb,
            output_file=tmp_path / "annotated.gb",
            predictions_csv=tmp_path / "predictions.csv",
            manifest=tmp_path / "manifest.json",
            model_family="dummy",
            benchmark_manifest=tmp_path / "missing_benchmark_manifest.json",
            threshold=0.80,
            window_size=8,
            step_size=4,
        )
    )

    assert manifest["threshold"] == 0.80
    assert manifest["threshold_source"] == "cli"
    assert manifest["window_size"] == 8
    assert "Benchmark manifest not found" in manifest["warnings"][0]
