from pathlib import Path

import pytest

# The torch extra is optional for SeqTrainer, so this module intentionally
# skips before importing torch-dependent benchmark objects.
# ruff: noqa: E402
torch = pytest.importorskip("torch")

import pandas as pd

import seqtrainer.torch.cnn_baseline as cnn_baseline
from seqtrainer.metrics import best_threshold_by_mcc, binary_classification_metrics
from seqtrainer.torch.cnn_baseline import (
    CnnBaselineConfig,
    CnnCsvSplitConfig,
    EnhancedDNACNN,
    TinyDNACNN,
    run_cnn_baseline,
    run_cnn_csv_splits,
)


def test_tiny_dna_cnn_forward_shape():
    model = TinyDNACNN()
    x = torch.zeros((2, 5, 120), dtype=torch.float32)
    logits = model(x)

    assert logits.shape == (2, 2)


def test_enhanced_dna_cnn_forward_shape():
    model = EnhancedDNACNN()
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
    assert "precision" in metrics
    assert "recall" in metrics
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


def test_best_threshold_by_mcc_handles_low_confidence_scores():
    scores = torch.tensor([0.01, 0.04]).numpy()
    threshold, score = best_threshold_by_mcc(
        y_true=torch.tensor([0, 1]).numpy(),
        y_score=scores,
    )

    assert ((scores.astype(float) >= threshold).astype(int) == [0, 1]).all()
    assert score == 1.0


def test_best_threshold_by_mcc_handles_high_confidence_scores():
    scores = torch.tensor([0.96, 0.99]).numpy()
    threshold, score = best_threshold_by_mcc(
        y_true=torch.tensor([0, 1]).numpy(),
        y_score=scores,
    )

    assert ((scores.astype(float) >= threshold).astype(int) == [0, 1]).all()
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
            model_variant="enhanced",
            weight_decay=1e-4,
            optimizer_name="adamw",
            scheduler_name="one_cycle",
            select_best_by_mcc=True,
            class_weighting=True,
            seed=42,
        )
    )

    assert (tmp_path / "outputs" / "metrics.json").exists()
    assert (tmp_path / "outputs" / "predictions.csv").exists()
    assert (tmp_path / "outputs" / "checkpoints" / "best_model.pt").exists()
    assert result.manifest["dataset"]["splits"]["train"]["rows"] == 4
    assert result.manifest["model"]["variant"] == "enhanced"
    assert result.manifest["threshold_selection"]["strategy"] == "validation_mcc"
    predictions = pd.read_csv(tmp_path / "outputs" / "predictions.csv")
    assert "threshold" in predictions.columns
    assert "logit_argmax_prediction" in predictions.columns
    assert (
        predictions["prediction"]
        == (predictions["probability"] >= predictions["threshold"]).astype(int)
    ).all()


def test_cnn_csv_train_predictions_preserve_csv_row_order(tmp_path):
    sequences = [
        "AAAAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCCCC",
        "GGGGGGGGGGGGGGGG",
        "TTTTTTTTTTTTTTTT",
        "ACACACACACACACAC",
        "TGTGTGTGTGTGTGTG",
    ]
    train = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1, 0, 1]})
    validation = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1, 0, 1]})
    test = pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1, 0, 1]})
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
            model_variant="tiny",
            seed=42,
        )
    )

    predictions = pd.read_csv(result.output_dir / "predictions.csv")
    train_predictions = predictions[predictions["split"] == "train"].reset_index(drop=True)

    assert train_predictions["sequence"].tolist() == sequences
    assert train_predictions["label"].tolist() == train["label"].tolist()
    assert train_predictions["idx"].tolist() == list(range(len(sequences)))


def test_cnn_csv_split_honors_fixed_threshold_strategy(tmp_path):
    sequences = [
        "AAAAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCCCC",
        "GGGGGGGGGGGGGGGG",
        "TTTTTTTTTTTTTTTT",
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
            threshold_strategy="fixed_0_5",
            seed=42,
        )
    )

    assert result.manifest["threshold_selection"]["strategy"] == "fixed_0_5"
    assert result.manifest["threshold_selection"]["threshold"] == 0.5


def test_cnn_csv_checkpoint_selection_uses_mcc_not_threshold_metric(tmp_path, monkeypatch):
    sequences = [
        "AAAAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCCCC",
        "GGGGGGGGGGGGGGGG",
        "TTTTTTTTTTTTTTTT",
    ]
    for name in ("train", "validation", "test"):
        pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1]}).to_csv(
            tmp_path / f"{name}.csv",
            index=False,
        )

    threshold_choices = iter([(0.2, 10.0), (0.8, 1.0)])
    mcc_scores = iter([(0.5, 0.1), (0.5, 0.9)])

    def fake_select_threshold(strategy, labels, probabilities):
        assert strategy == "validation_f1"
        return next(threshold_choices)

    def fake_best_threshold_by_metric(labels, probabilities, metric="mcc", thresholds=None):
        assert metric == "mcc"
        return next(mcc_scores)

    def fake_predict(model, loader, criterion, device):
        return {
            "label": torch.tensor([0, 1, 0, 1]).numpy(),
            "probability": torch.tensor([0.1, 0.9, 0.2, 0.8]).numpy(),
            "prediction": torch.tensor([0, 1, 0, 1]).numpy(),
            "logit_argmax_prediction": torch.tensor([0, 1, 0, 1]).numpy(),
            "loss": 0.1,
        }

    monkeypatch.setattr(cnn_baseline, "_run_epoch", lambda *args, **kwargs: (0.1, 1.0))
    monkeypatch.setattr(cnn_baseline, "_predict", fake_predict)
    monkeypatch.setattr(cnn_baseline, "_select_threshold", fake_select_threshold)
    monkeypatch.setattr(cnn_baseline, "best_threshold_by_metric", fake_best_threshold_by_metric)

    result = run_cnn_csv_splits(
        CnnCsvSplitConfig(
            train_csv=tmp_path / "train.csv",
            validation_csv=tmp_path / "validation.csv",
            test_csv=tmp_path / "test.csv",
            output_dir=tmp_path / "outputs",
            sequence_length=32,
            batch_size=2,
            cycles=2,
            select_best_by_mcc=True,
            threshold_strategy="validation_f1",
            seed=42,
        )
    )

    assert result.manifest["threshold_selection"]["strategy"] == "validation_f1"
    assert result.manifest["threshold_selection"]["threshold"] == 0.8
    assert result.manifest["threshold_selection"]["validation_score"] == 1.0
