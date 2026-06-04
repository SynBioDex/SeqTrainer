from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import pandas as pd

from seqtrainer.metrics import best_threshold_by_mcc, binary_classification_metrics
from seqtrainer.torch.cnn_baseline import (
    CnnBaselineConfig,
    CnnCsvSplitConfig,
    TinyDNACNN,
    run_cnn_baseline,
    run_cnn_csv_splits,
)


def test_tiny_dna_cnn_forward_shape():
    model = TinyDNACNN()
    x = torch.zeros((2, 5, 120), dtype=torch.float32)
    logits = model(x)

    assert logits.shape == (2, 2)


def test_binary_classification_metrics_include_required_fields():
    metrics = binary_classification_metrics(
        y_true=torch.tensor([0, 1, 1, 0]).numpy(),
        y_score=torch.tensor([0.1, 0.8, 0.4, 0.3]).numpy(),
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert "balanced_accuracy" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "mcc" in metrics
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_best_threshold_by_mcc_uses_validation_scores():
    threshold, score = best_threshold_by_mcc(
        y_true=torch.tensor([0, 0, 1, 1]).numpy(),
        y_score=torch.tensor([0.1, 0.2, 0.8, 0.9]).numpy(),
        thresholds=torch.tensor([0.25, 0.5, 0.75]).numpy(),
    )

    assert threshold == 0.25
    assert score == 1.0


def test_cnn_baseline_smoke_run_writes_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    result = run_cnn_baseline(
        CnnBaselineConfig(
            data_dir=repo_root / "data" / "sbol_data",
            output_dir=tmp_path,
            max_files=12,
            cycles=1,
            seed=42,
        )
    )

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "history.csv").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert set(result.metrics) == {"train", "validation", "test"}
    assert result.manifest["dataset"]["rows"] == 12


def test_cnn_csv_split_smoke_run_writes_artifacts(tmp_path):
    sequences = [
        "ACGTACGTACGTACGT",
        "TGCATGCATGCATGCA",
        "AAAAACCCCCGGGGGTT",
        "TTTTTGGGGGCCCCCAA",
    ]
    train = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1]})
    validation = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1]})
    test = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1]})
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    result = run_cnn_csv_splits(
        CnnCsvSplitConfig(
            train_csv=train_path,
            validation_csv=validation_path,
            test_csv=test_path,
            output_dir=tmp_path / "outputs",
            sequence_length=32,
            batch_size=2,
            cycles=1,
            seed=42,
        )
    )

    assert (tmp_path / "outputs" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "predictions.csv").exists()
    assert result.manifest["dataset"]["splits"]["train"]["rows"] == 4
    assert result.manifest["threshold_selection"]["strategy"] == "validation_mcc"
