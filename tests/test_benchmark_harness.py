from pathlib import Path

import pandas as pd
import pytest

from seqtrainer.benchmarks import (
    build_run_manifest,
    decide_imbalance_policy,
    load_benchmark_config,
    load_predefined_split_frames,
    summarize_split_frames,
    threshold_metric_from_strategy,
    write_benchmark_outputs,
)
from seqtrainer.cli.main import main
from seqtrainer.metrics import best_threshold_by_metric, binary_classification_metrics


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config-examples" / "benchmarks"


def test_binary_metrics_include_required_fields_and_confusion_matrix():
    metrics = binary_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.8, 0.4, 0.3],
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_binary_metrics_handle_one_class_split_gracefully():
    metrics = binary_classification_metrics(
        y_true=[0, 0, 0],
        y_score=[0.1, 0.2, 0.3],
        threshold=0.5,
    )

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert "warning" in metrics


def test_best_threshold_uses_score_extremes_not_only_middle_grid():
    threshold, score = best_threshold_by_metric(
        y_true=[0, 1],
        y_score=[0.01, 0.04],
        metric="mcc",
    )

    assert score == 1.0
    assert ((pd.Series([0.01, 0.04], dtype=float) >= threshold).astype(int).tolist()) == [0, 1]


def test_predefined_split_loader_and_summary(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for split, filename in {
        "train": "train_EP_DNA_BERT2_genomic_order.csv",
        "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
        "test": "test_EP_DNA_BERT2_genomic_order.csv",
    }.items():
        pd.DataFrame(
            {
                "sequence": ["ACGT", "TGCA", "AAAA"],
                "label": [0, 1, 1],
                "split_name": split,
            }
        ).to_csv(split_dir / filename, index=False)

    frames = load_predefined_split_frames(config, base_dir=tmp_path)
    summary = summarize_split_frames(config, frames)

    assert set(frames) == {"train", "validation", "test"}
    assert summary["train"]["rows"] == 3
    assert summary["train"]["class_counts"] == {"0": 1, "1": 2}
    assert summary["train"]["sequence_length"]["max"] == 4


def test_predefined_split_loader_validates_required_columns(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for filename in (
        "train_EP_DNA_BERT2_genomic_order.csv",
        "eval_EP_DNA_BERT2_genomic_order.csv",
        "test_EP_DNA_BERT2_genomic_order.csv",
    ):
        pd.DataFrame({"sequence": ["ACGT"]}).to_csv(split_dir / filename, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_predefined_split_frames(config, base_dir=tmp_path)


def test_benchmark_artifact_writers_create_json_and_csv(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    metrics = {
        "validation": binary_classification_metrics(
            y_true=[0, 1],
            y_score=[0.2, 0.8],
            threshold=0.5,
        )
    }
    manifest = build_run_manifest(
        config,
        split_summary={"validation": {"rows": 2, "class_counts": {"0": 1, "1": 1}}},
        threshold=0.5,
    )
    predictions = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "idx": [0, 1],
            "label": [0, 1],
            "probability": [0.2, 0.8],
            "threshold": [0.5, 0.5],
            "prediction": [0, 1],
        }
    )

    written = write_benchmark_outputs(
        tmp_path,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        config=config,
    )

    assert written["manifest"].exists()
    assert written["metrics_json"].exists()
    assert written["metrics_csv"].exists()
    assert written["predictions"].exists()

    metrics_csv = pd.read_csv(tmp_path / "metrics.csv")
    assert {"tn", "fp", "fn", "tp"}.issubset(metrics_csv.columns)
    assert metrics_csv.loc[0, "split"] == "validation"


def test_benchmark_manifest_cli_writes_shared_manifest(tmp_path, capsys):
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for filename in (
        "train_EP_DNA_BERT2_genomic_order.csv",
        "eval_EP_DNA_BERT2_genomic_order.csv",
        "test_EP_DNA_BERT2_genomic_order.csv",
    ):
        pd.DataFrame({"sequence": ["ACGT", "TGCA"], "label": [0, 1]}).to_csv(
            split_dir / filename,
            index=False,
        )

    output_dir = tmp_path / "manifest_run"
    exit_code = main(
        [
            "benchmark-manifest",
            "--config",
            str(CONFIG_DIR / "cnn.toml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "train: rows=2" in captured.out
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "config.json").exists()


def test_imbalance_policy_uses_training_split_only():
    split_summary = {
        "train": {"class_counts": {"0": 10, "1": 20}},
        "validation": {"class_counts": {"0": 1, "1": 100}},
    }

    policy = decide_imbalance_policy(split_summary, ratio_threshold=1.5)

    assert policy.apply_to_training is True
    assert policy.strategy == "class_weighting"
    assert policy.imbalance_ratio == 2.0

    with pytest.raises(ValueError, match="training split"):
        decide_imbalance_policy(split_summary, split="validation")


def test_imbalance_policy_does_not_weight_balanced_training_data():
    split_summary = {"train": {"class_counts": {"0": 20, "1": 22}}}

    policy = decide_imbalance_policy(split_summary, ratio_threshold=1.5)

    assert policy.apply_to_training is False
    assert policy.strategy == "none"


def test_threshold_metric_from_strategy_maps_validation_strategies():
    assert threshold_metric_from_strategy("validation_mcc") == "mcc"
    assert threshold_metric_from_strategy("validation_f1") == "f1"
    assert threshold_metric_from_strategy("validation_balanced_accuracy") == "balanced_accuracy"
    assert threshold_metric_from_strategy("fixed_0_5") is None

