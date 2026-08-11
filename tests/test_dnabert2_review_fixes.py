from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from seqtrainer.adapters.ipromp import (
    build_ipromp_mapping,
    normalize_ipromp_predictions,
)
from seqtrainer.benchmarks.config import load_benchmark_config
from seqtrainer.torch.dnabert2_benchmark import _normalize_binary_labels


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config-examples"
    / "benchmarks"
    / "ipromp_external.toml"
)


def test_dnabert2_maps_configured_string_labels():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        label=replace(
            config.label,
            negative_label="background",
            positive_label="promoter",
        ),
    )
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA", "AAAA"],
            "label": ["background", "promoter", "promoter"],
        }
    )

    labels = _normalize_binary_labels(config, frame)

    assert labels.tolist() == [0.0, 1.0, 1.0]


def test_dnabert2_respects_reversed_numeric_label_configuration():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        label=replace(config.label, negative_label=1, positive_label=0),
    )
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT", "TGCA"],
            "label": [1, 0],
        }
    )

    labels = _normalize_binary_labels(config, frame)

    assert labels.tolist() == [0.0, 1.0]


def test_dnabert2_rejects_unconfigured_labels():
    config = load_benchmark_config(CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "sequence": ["ACGT"],
            "label": ["unknown"],
        }
    )

    with pytest.raises(ValueError, match="do not match configured"):
        _normalize_binary_labels(config, frame)


def test_ipromp_mapping_rejects_duplicate_configured_ids():
    config = load_benchmark_config(CONFIG_PATH)
    config = replace(
        config,
        dataset=replace(config.dataset, id_field="id"),
    )
    frames = {
        split: pd.DataFrame(
            {
                "sequence": ["ACGT", "TGCA"],
                "label": [0, 1],
                "id": ["same-id", "same-id"],
            }
        )
        for split in ("train", "validation", "test")
    }

    with pytest.raises(ValueError, match="duplicate split/sequence_id"):
        build_ipromp_mapping(config, frames)


def test_ipromp_rejects_cached_mapping_from_different_splits(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    original_frames = {
        split: pd.DataFrame(
            {"sequence": ["ACGT", "TGCA"], "label": [0, 1]}
        )
        for split in ("train", "validation", "test")
    }
    mapping_path = tmp_path / "mapping.csv"
    build_ipromp_mapping(config, original_frames).to_csv(mapping_path, index=False)

    changed_frames = {
        split: frame.copy()
        for split, frame in original_frames.items()
    }
    changed_frames["test"].loc[0, "sequence"] = "AAAA"
    predictions = pd.DataFrame(
        {
            "split": [row.split for row in build_ipromp_mapping(config, original_frames).itertuples()],
            "sequence_id": [row.sequence_id for row in build_ipromp_mapping(config, original_frames).itertuples()],
            "probability": 0.5,
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="does not match"):
        normalize_ipromp_predictions(
            config,
            mapping_csv=mapping_path,
            predictions_csv=predictions_path,
            frames=changed_frames,
        )


def test_ipromp_rejects_reordered_idless_labels(tmp_path):
    config = load_benchmark_config(CONFIG_PATH)
    frames = {
        split: pd.DataFrame(
            {"sequence": ["ACGT", "TGCA"], "label": [0, 1]}
        )
        for split in ("train", "validation", "test")
    }
    mapping = build_ipromp_mapping(config, frames)
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    predictions = mapping[["split", "label"]].copy()
    validation_rows = predictions["split"] == "validation"
    predictions.loc[validation_rows, "label"] = predictions.loc[validation_rows, "label"].iloc[::-1].to_numpy()
    predictions["probability"] = 0.5
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    with pytest.raises(ValueError, match="labels do not match"):
        normalize_ipromp_predictions(
            config,
            mapping_csv=mapping_path,
            predictions_csv=predictions_path,
            frames=frames,
        )


def test_dnabert2_scales_final_partial_accumulation_window():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    from seqtrainer.torch.dnabert2_benchmark import _run_epoch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.0]]))

        def forward(self, input_ids, attention_mask):
            return input_ids.float() @ self.weight

    class NoOpScheduler:
        def step(self):
            return None

    model = TinyModel()
    loader = DataLoader(
        TensorDataset(
            torch.ones(3, 1),
            torch.ones(3, 1),
            torch.ones(3, 1),
        ),
        batch_size=1,
        shuffle=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _run_epoch(
        model,
        loader,
        torch.nn.MSELoss(),
        optimizer,
        NoOpScheduler(),
        torch.device("cpu"),
        torch,
        gradient_accumulation_steps=2,
        max_grad_norm=100.0,
    )

    assert model.weight.item() == pytest.approx(0.36, abs=1e-6)
