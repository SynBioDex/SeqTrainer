from pathlib import Path

import pandas as pd

from seqtrainer.annotation import PromoterAnnotationConfig, run_promoter_annotation
from seqtrainer.annotation.collection import run_promoter_collection
from seqtrainer.annotation.coordinate_conversion import sbol_orientation, sbol_ranges_for_location
from seqtrainer.annotation.ground_truth import extract_ground_truth_promoters
from seqtrainer.annotation.windows import generate_sliding_windows


def _record():
    from Bio.Seq import Seq
    from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
    from Bio.SeqRecord import SeqRecord

    record = SeqRecord(Seq("A" * 30), id="labelled_plasmid", name="labelled_plasmid")
    record.annotations.update({"molecule_type": "DNA", "topology": "circular"})
    record.features = [
        SeqFeature(FeatureLocation(4, 9, strand=1), type="promoter", qualifiers={"label": ["deposited_p"]}),
        SeqFeature(FeatureLocation(12, 16, strand=-1), type="regulatory", qualifiers={"regulatory_class": ["promoter"]}),
        SeqFeature(FeatureLocation(26, 30, strand=1), type="misc_feature", qualifiers={"label": ["explicit promoter marker"]}),
        SeqFeature(CompoundLocation([FeatureLocation(27, 30, strand=1), FeatureLocation(0, 3, strand=1)]), type="promoter", qualifiers={"label": ["origin promoter"]}),
        SeqFeature(FeatureLocation(18, 20), type="misc_feature", qualifiers={"label": ["promoterless cassette"]}),
    ]
    return record


def test_ground_truth_evidence_tiers_and_exclusions():
    promoters = extract_ground_truth_promoters(_record(), plasmid_id="p1")
    assert [item.evidence_tier for item in promoters] == ["A", "A", "B", "A"]
    assert promoters[-1].wraps_origin is True
    strict = extract_ground_truth_promoters(_record(), label_mode="strict")
    assert len(strict) == 3
    assert all(item.evidence_tier == "A" for item in strict)


def test_so_cross_reference_is_tier_a_and_plasmid_name_is_not_evidence():
    from Bio.SeqFeature import FeatureLocation, SeqFeature

    record = _record()
    record.features.append(SeqFeature(FeatureLocation(20, 24), type="misc_feature", qualifiers={"db_xref": ["SO:0000167"]}))
    assert any(item.evidence_rule == "db_xref=SO:0000167" for item in extract_ground_truth_promoters(record))
    record.features = []
    record.id = "promoter_expected_but_unlabelled"
    assert extract_ground_truth_promoters(record) == []


def test_coordinate_conversion_is_one_based_and_bounded():
    record = _record()
    ranges = sbol_ranges_for_location(record.features[3].location, len(record.seq))
    assert [(item["start"], item["end"]) for item in ranges] == [(28, 30), (1, 3)]
    assert sbol_orientation(1).endswith("#inline")
    assert sbol_orientation(-1).endswith("#reverseComplement")


def test_labelled_evaluation_writes_ground_truth_and_metrics(tmp_path: Path):
    from Bio import SeqIO

    input_path = tmp_path / "input.gb"
    SeqIO.write(_record(), input_path, "genbank")
    evaluation_dir = tmp_path / "evaluation"
    manifest = run_promoter_annotation(
        PromoterAnnotationConfig(
            input_file=input_path,
            output_file=tmp_path / "annotated.gb",
            predictions_csv=tmp_path / "predictions.csv",
            manifest=tmp_path / "annotation_manifest.json",
            model_family="dummy",
            threshold=0.8,
            window_size=8,
            step_size=4,
            evaluation_dir=evaluation_dir,
            sbol_output=tmp_path / "annotated.nt",
            annotation_completeness="verified_complete",
        )
    )
    assert (evaluation_dir / "gold_promoters.csv").exists()
    assert (evaluation_dir / "window_predictions.csv").exists()
    assert (evaluation_dir / "merged_predictions.csv").exists()
    assert (evaluation_dir / "promoter_matches.csv").exists()
    assert (evaluation_dir / "metrics.json").exists()
    assert (evaluation_dir / "metrics.csv").exists()
    assert (tmp_path / "annotated.nt").exists()
    assert manifest["evaluation"]["gold_csv"]
    assert pd.read_csv(evaluation_dir / "gold_promoters.csv").shape[0] == 4


def test_window_centre_labels_same_strand():
    record = _record()
    gold = extract_ground_truth_promoters(record)
    windows = generate_sliding_windows(str(record.seq), window_size=6, step_size=6, circular=True, scan_both_strands=True)
    labels, ids = __import__("seqtrainer.annotation.evaluation", fromlist=["window_gold_labels"]).window_gold_labels(windows, gold, len(record.seq))
    assert len(labels) == len(windows)
    assert any(label == 1 for label in labels)
    assert any(ids)


def test_collection_skips_missing_files_and_writes_audit(tmp_path: Path):
    from Bio import SeqIO

    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    SeqIO.write(_record(), input_dir / "available.gb", "genbank")
    manifest = tmp_path / "collection.csv"
    manifest.write_text(
        "addgene_id,plasmid_name,expected_local_filename,plasmid_url\n"
        "1,available,available.gb,https://www.addgene.org/1/\n"
        "2,missing,missing.gb,https://www.addgene.org/2/\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    result = run_promoter_collection(
        manifest,
        input_dir=input_dir,
        output_dir=output_dir,
        predictor="dummy",
        continue_on_error=True,
    )
    assert result["included_count"] == 1
    assert result["excluded_count"] == 1
    assert (output_dir / "included_plasmids.csv").exists()
    excluded = pd.read_csv(output_dir / "excluded_plasmids.csv")
    assert "unavailable" in excluded.loc[0, "exclusion_reason"]
