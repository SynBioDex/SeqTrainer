"""Reproducible PyTorch CNN baseline from the tutorial notebook."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extra
    raise ModuleNotFoundError(
        "The CNN baseline requires PyTorch. Install with `pip install -e '.[torch]'` "
        "or install torch in your notebook/Colab environment."
    ) from exc

from seqtrainer.data.materialized import MaterializedDataset
from seqtrainer.data.sbol import build_dataset_from_files
from seqtrainer.benchmarks.policy import decide_imbalance_policy
from seqtrainer.metrics import best_threshold_by_mcc, binary_classification_metrics
from seqtrainer.transforms.dna import one_hot_encode, pad_or_trim


@dataclass(frozen=True)
class CnnBaselineConfig:
    """Configuration for reproducing the original tutorial CNN baseline."""

    data_dir: str | Path = "data/sbol_data"
    output_dir: str | Path = "outputs/cnn_baseline_reference"
    max_files: int = 40
    sequence_length: int = 120
    train_size: float = 0.7
    validation_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    batch_size: int = 16
    cycles: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    deterministic: bool = True


@dataclass(frozen=True)
class CnnCsvSplitConfig:
    """Configuration for training the CNN on predefined CSV split files."""

    train_csv: str | Path
    validation_csv: str | Path
    test_csv: str | Path
    output_dir: str | Path = "outputs/cnn_csv_split_baseline"
    dataset_name: str = "ep_dnabert2_genomic_order"
    source_accession: str = "GSE144621"
    source_url: str = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144621"
    sequence_field: str = "sequence"
    label_field: str = "label"
    sequence_length: int = 300
    seed: int = 42
    batch_size: int = 16
    cycles: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer_name: str = "adam"
    scheduler_name: str = "none"
    select_best_by_mcc: bool = False
    early_stopping_patience: int | None = None
    model_variant: str = "tiny"
    dropout: float = 0.25
    class_weighting: bool = False
    device: str = "cpu"
    deterministic: bool = True


@dataclass(frozen=True)
class CnnBaselineResult:
    """Artifacts returned after a CNN baseline run."""

    output_dir: Path
    metrics: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    history: list[dict[str, float]]


class TinyDNACNN(nn.Module):
    """Small Conv1D classifier matching the tutorial notebook architecture."""

    def __init__(self, channels: int = 5, n_classes: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class EnhancedDNACNN(nn.Module):
    """Stronger Conv1D classifier for controlled CNN baseline improvements."""

    def __init__(self, channels: int = 5, n_classes: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=7, padding=12, dilation=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = torch.cat([self.avg_pool(features), self.max_pool(features)], dim=1)
        return self.head(pooled)


def run_cnn_baseline(config: CnnBaselineConfig | None = None) -> CnnBaselineResult:
    """Train the tutorial CNN baseline and write reference artifacts."""
    cfg = config or CnnBaselineConfig()
    _seed_everything(cfg.seed, cfg.deterministic)

    data = _build_dataset(cfg)
    device = torch.device(cfg.device)
    model = TinyDNACNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    loaders = {
        split: _loader_for_split(data["x"], data["y"], indices, cfg.batch_size, cfg.seed, split == "train")
        for split, indices in data["indices"].items()
    }
    prediction_loaders = {
        split: _loader_for_split(data["x"], data["y"], indices, cfg.batch_size, cfg.seed, False)
        for split, indices in data["indices"].items()
    }

    history: list[dict[str, float]] = []
    for cycle in range(1, cfg.cycles + 1):
        train_loss, train_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss, val_acc = _run_epoch(model, loaders["validation"], criterion, optimizer, device, train=False)
        history.append(
            {
                "cycle": float(cycle),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
            }
        )

    predictions = {
        split: _predict(model, loader, criterion, device)
        for split, loader in prediction_loaders.items()
    }
    metrics = {}
    prediction_frames = []
    for split, pred in predictions.items():
        split_metrics = binary_classification_metrics(pred["label"], pred["probability"], threshold=0.5)
        split_metrics["loss"] = pred["loss"]
        metrics[split] = split_metrics
        prediction_frames.append(_prediction_frame(split, data["frame"], data["indices"][split], pred))

    output_dir = Path(cfg.output_dir)
    checkpoint_path = output_dir / "checkpoints" / "final_model.pt"
    manifest = _manifest(cfg, data, metrics)
    manifest["checkpoint"] = {"path": str(checkpoint_path), "selection": "final_cycle"}
    _write_outputs(output_dir, cfg, metrics, manifest, history, prediction_frames, model.state_dict())
    return CnnBaselineResult(output_dir=output_dir, metrics=metrics, manifest=manifest, history=history)


def run_cnn_csv_splits(config: CnnCsvSplitConfig) -> CnnBaselineResult:
    """Train the CNN on predefined train/validation/test CSV files."""
    _seed_everything(config.seed, config.deterministic)

    frames = _load_csv_split_frames(config)
    device = torch.device(config.device)
    model = _build_csv_model(config).to(device)
    criterion = _csv_criterion(frames["train"], config, device)
    loaders = {
        split: _loader_for_frame(frame, config, shuffle=(split == "train"))
        for split, frame in frames.items()
    }
    prediction_loaders = {
        split: _loader_for_frame(frame, config, shuffle=False)
        for split, frame in frames.items()
    }
    optimizer = _csv_optimizer(model, config)
    scheduler = _csv_scheduler(optimizer, loaders["train"], config)

    history: list[dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_threshold = 0.5
    best_validation_mcc = float("-inf")
    bad_cycles = 0

    for cycle in range(1, config.cycles + 1):
        train_loss, train_acc = _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            train=True,
            scheduler=scheduler,
        )
        val_loss, val_acc = _run_epoch(model, loaders["validation"], criterion, optimizer, device, train=False)
        val_pred = _predict(model, loaders["validation"], criterion, device)
        val_threshold, val_mcc = best_threshold_by_mcc(val_pred["label"], val_pred["probability"])

        history_row = {
            "cycle": float(cycle),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_mcc": float(val_mcc),
            "validation_threshold": float(val_threshold),
        }
        history.append(history_row)

        if config.select_best_by_mcc or config.early_stopping_patience is not None:
            if val_mcc > best_validation_mcc:
                best_validation_mcc = float(val_mcc)
                best_threshold = float(val_threshold)
                best_state = deepcopy(model.state_dict())
                bad_cycles = 0
            else:
                bad_cycles += 1
                if config.early_stopping_patience is not None and bad_cycles >= config.early_stopping_patience:
                    history_row["stopped_early"] = 1.0
                    break

    if config.select_best_by_mcc or config.early_stopping_patience is not None:
        model.load_state_dict(best_state)

    predictions = {
        split: _predict(model, loader, criterion, device)
        for split, loader in prediction_loaders.items()
    }
    if config.select_best_by_mcc or config.early_stopping_patience is not None:
        threshold, validation_mcc = best_threshold, best_validation_mcc
    else:
        threshold, validation_mcc = best_threshold_by_mcc(
            predictions["validation"]["label"],
            predictions["validation"]["probability"],
        )

    metrics = {}
    prediction_frames = []
    for split, pred in predictions.items():
        split_metrics = binary_classification_metrics(pred["label"], pred["probability"], threshold=threshold)
        split_metrics["loss"] = pred["loss"]
        metrics[split] = split_metrics
        prediction_frames.append(_csv_prediction_frame(split, frames[split], config, pred, threshold))

    output_dir = Path(config.output_dir)
    uses_validation_checkpoint = config.select_best_by_mcc or config.early_stopping_patience is not None
    checkpoint_name = "best_model.pt" if uses_validation_checkpoint else "final_model.pt"
    checkpoint_path = output_dir / "checkpoints" / checkpoint_name
    manifest = _csv_manifest(config, frames, metrics, threshold, validation_mcc)
    manifest["checkpoint"] = {
        "path": str(checkpoint_path),
        "selection": "best_validation_mcc" if uses_validation_checkpoint else "final_cycle",
    }
    _write_outputs(output_dir, config, metrics, manifest, history, prediction_frames, model.state_dict())
    return CnnBaselineResult(output_dir=output_dir, metrics=metrics, manifest=manifest, history=history)


def _build_dataset(cfg: CnnBaselineConfig) -> dict[str, Any]:
    data_dir = Path(cfg.data_dir)
    files = sorted(data_dir.glob("sample_design_*.xml"))[: cfg.max_files]
    if not files:
        raise FileNotFoundError(f"No sample_design_*.xml files found in {data_dir}")

    frame = build_dataset_from_files(files)
    if frame.empty:
        raise RuntimeError("No rows were materialized from SBOL inputs.")

    label_threshold = float(frame["target"].median())
    frame = frame.copy()
    frame["label"] = (frame["target"] >= label_threshold).astype(int)

    fixed_sequences = [pad_or_trim(seq, length=cfg.sequence_length) for seq in frame["sequence"].tolist()]
    encoded = one_hot_encode(fixed_sequences)
    labels = frame["label"].to_numpy(dtype=np.int64)

    x_tensor = torch.tensor(np.transpose(encoded, (0, 2, 1)), dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    examples = [{"idx": i, "label": int(label)} for i, label in enumerate(labels)]
    materialized = MaterializedDataset(examples, metadata={"tutorial": "cnn_demo"})
    train_ds, val_ds, test_ds = materialized.train_val_test_split(
        cfg.train_size,
        cfg.validation_size,
        cfg.test_size,
        seed=cfg.seed,
    )

    return {
        "frame": frame,
        "files": files,
        "label_threshold": label_threshold,
        "x": x_tensor,
        "y": y_tensor,
        "indices": {
            "train": _indices(train_ds),
            "validation": _indices(val_ds),
            "test": _indices(test_ds),
        },
    }


def _load_csv_split_frames(cfg: CnnCsvSplitConfig) -> dict[str, pd.DataFrame]:
    paths = {
        "train": Path(cfg.train_csv),
        "validation": Path(cfg.validation_csv),
        "test": Path(cfg.test_csv),
    }
    frames = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{split} CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = {cfg.sequence_field, cfg.label_field}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{path} is empty")
        frames[split] = frame.copy()
    return frames


def _build_csv_model(cfg: CnnCsvSplitConfig) -> nn.Module:
    if cfg.model_variant == "tiny":
        return TinyDNACNN()
    if cfg.model_variant == "enhanced":
        return EnhancedDNACNN(dropout=cfg.dropout)
    raise ValueError("model_variant must be either 'tiny' or 'enhanced'")


def _csv_criterion(frame: pd.DataFrame, cfg: CnnCsvSplitConfig, device: torch.device) -> nn.Module:
    if not cfg.class_weighting:
        return nn.CrossEntropyLoss()

    labels = frame[cfg.label_field].astype(int).to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(float)
    if counts.min() == 0:
        return nn.CrossEntropyLoss()

    weights = counts.sum() / (len(counts) * counts)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def _csv_optimizer(model: nn.Module, cfg: CnnCsvSplitConfig) -> torch.optim.Optimizer:
    if cfg.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError("optimizer_name must be either 'adam' or 'adamw'")


def _csv_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    cfg: CnnCsvSplitConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.scheduler_name == "none":
        return None
    if cfg.scheduler_name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            epochs=cfg.cycles,
            steps_per_epoch=len(train_loader),
        )
    raise ValueError("scheduler_name must be either 'none' or 'one_cycle'")


def _loader_for_frame(frame: pd.DataFrame, cfg: CnnCsvSplitConfig, shuffle: bool) -> DataLoader:
    sequences = [pad_or_trim(seq, length=cfg.sequence_length) for seq in frame[cfg.sequence_field].astype(str)]
    encoded = one_hot_encode(sequences)
    labels = frame[cfg.label_field].astype(int).to_numpy(dtype=np.int64)
    x_tensor = torch.tensor(np.transpose(encoded, (0, 2, 1)), dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, generator=generator)


def _indices(dataset: MaterializedDataset) -> torch.Tensor:
    return torch.tensor([row["idx"] for row in dataset.examples], dtype=torch.long)


def _loader_for_split(
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(x_tensor[indices], y_tensor[indices])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[float, float]:
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)

    return total_loss / total, total_correct / total


def _predict(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels, probabilities, predictions = [], [], []
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            labels.extend(yb.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)

    return {
        "label": np.asarray(labels, dtype=int),
        "probability": np.asarray(probabilities, dtype=float),
        "prediction": np.asarray(predictions, dtype=int),
        "loss": float(total_loss / total),
    }


def _prediction_frame(
    split: str,
    frame: pd.DataFrame,
    indices: torch.Tensor,
    prediction: dict[str, Any],
) -> pd.DataFrame:
    rows = frame.iloc[indices.numpy()].copy()
    if "label" in rows and not np.array_equal(rows["label"].astype(int).to_numpy(), prediction["label"]):
        raise ValueError(f"Predictions for split {split!r} are not aligned with source rows")

    rows.insert(0, "split", split)
    rows.insert(1, "idx", indices.numpy())
    rows["probability"] = prediction["probability"]
    rows["prediction"] = prediction["prediction"]
    return rows[["split", "idx", "source", "target", "label", "probability", "prediction"]]


def _csv_prediction_frame(
    split: str,
    frame: pd.DataFrame,
    cfg: CnnCsvSplitConfig,
    prediction: dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    labels = frame[cfg.label_field].astype(int).to_numpy()
    if not np.array_equal(labels, prediction["label"]):
        raise ValueError(f"Predictions for split {split!r} are not aligned with source rows")

    thresholded_prediction = (prediction["probability"] >= threshold).astype(int)
    rows = pd.DataFrame(
        {
            "split": split,
            "idx": np.arange(len(frame)),
            "sequence": frame[cfg.sequence_field].astype(str),
            "label": labels,
            "probability": prediction["probability"],
            "threshold": float(threshold),
            "prediction": thresholded_prediction,
            "logit_argmax_prediction": prediction["prediction"],
        }
    )
    return rows


def _manifest(cfg: CnnBaselineConfig, data: dict[str, Any], metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    class_counts = data["frame"]["label"].value_counts().sort_index().to_dict()
    split_sizes = {split: int(len(indices)) for split, indices in data["indices"].items()}
    return {
        "task": "tutorial_cnn_baseline_reproduction",
        "dataset": {
            "data_dir": str(cfg.data_dir),
            "selected_files": [path.name for path in data["files"]],
            "rows": int(len(data["frame"])),
            "label_threshold": float(data["label_threshold"]),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "split_sizes": split_sizes,
            "split_seed": cfg.seed,
        },
        "preprocessing": {
            "sequence_length": cfg.sequence_length,
            "encoding": "one_hot",
            "channels": ["A", "C", "G", "T", "N"],
        },
        "model": {
            "name": "TinyDNACNN",
            "architecture": [
                "Conv1d(5, 32, kernel_size=7, padding=3)",
                "ReLU",
                "MaxPool1d(kernel_size=2)",
                "Conv1d(32, 64, kernel_size=5, padding=2)",
                "ReLU",
                "AdaptiveMaxPool1d(1)",
                "Flatten",
                "Linear(64, 32)",
                "ReLU",
                "Linear(32, 2)",
            ],
        },
        "training": asdict(cfg),
        "metrics": metrics,
    }


def _csv_model_metadata(config: CnnCsvSplitConfig) -> dict[str, Any]:
    if config.model_variant == "tiny":
        return {
            "name": "TinyDNACNN",
            "variant": "tiny",
            "architecture": [
                "Conv1d(5, 32, kernel_size=7, padding=3)",
                "ReLU",
                "MaxPool1d(kernel_size=2)",
                "Conv1d(32, 64, kernel_size=5, padding=2)",
                "ReLU",
                "AdaptiveMaxPool1d(1)",
                "Flatten",
                "Linear(64, 32)",
                "ReLU",
                "Linear(32, 2)",
            ],
        }

    if config.model_variant == "enhanced":
        return {
            "name": "EnhancedDNACNN",
            "variant": "enhanced",
            "dropout": float(config.dropout),
            "architecture": [
                "Conv1d(5, 64, kernel_size=15, padding=7)",
                "BatchNorm1d(64)",
                "GELU",
                "Conv1d(64, 64, kernel_size=7, padding=3)",
                "BatchNorm1d(64)",
                "GELU",
                "MaxPool1d(kernel_size=2)",
                "Dropout",
                "Conv1d(64, 128, kernel_size=7, padding=6, dilation=2)",
                "BatchNorm1d(128)",
                "GELU",
                "Conv1d(128, 128, kernel_size=7, padding=12, dilation=4)",
                "BatchNorm1d(128)",
                "GELU",
                "MaxPool1d(kernel_size=2)",
                "Dropout",
                "Conv1d(128, 256, kernel_size=3, padding=1)",
                "BatchNorm1d(256)",
                "GELU",
                "AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1)",
                "Linear(512, 128)",
                "GELU",
                "Dropout",
                "Linear(128, 2)",
            ],
        }

    raise ValueError("model_variant must be either 'tiny' or 'enhanced'")


def _csv_manifest(
    cfg: CnnCsvSplitConfig,
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    threshold: float,
    validation_mcc: float,
) -> dict[str, Any]:
    split_summary = {}
    for split, frame in frames.items():
        counts = frame[cfg.label_field].astype(int).value_counts().sort_index().to_dict()
        split_summary[split] = {
            "rows": int(len(frame)),
            "class_counts": {str(key): int(value) for key, value in counts.items()},
        }
    imbalance_policy = decide_imbalance_policy(split_summary)

    return {
        "task": "csv_split_cnn_baseline",
        "dataset": {
            "name": cfg.dataset_name,
            "source_accession": cfg.source_accession,
            "source_url": cfg.source_url,
            "split_files": {
                "train": str(cfg.train_csv),
                "validation": str(cfg.validation_csv),
                "test": str(cfg.test_csv),
            },
            "sequence_field": cfg.sequence_field,
            "label_field": cfg.label_field,
            "splits": split_summary,
        },
        "preprocessing": {
            "sequence_length": cfg.sequence_length,
            "encoding": "one_hot",
            "channels": ["A", "C", "G", "T", "N"],
        },
        "model": _csv_model_metadata(config=cfg),
        "training": asdict(cfg),
        "threshold_selection": {
            "strategy": "validation_mcc",
            "threshold": float(threshold),
            "validation_mcc": float(validation_mcc),
        },
        "imbalance_policy": {
            "apply_to_training": imbalance_policy.apply_to_training,
            "strategy": imbalance_policy.strategy,
            "class_counts": imbalance_policy.class_counts,
            "imbalance_ratio": imbalance_policy.imbalance_ratio,
            "reason": imbalance_policy.reason,
        },
        "metrics": metrics,
    }


def _write_outputs(
    output_dir: Path,
    cfg: CnnBaselineConfig,
    metrics: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    history: list[dict[str, float]],
    prediction_frames: list[pd.DataFrame],
    checkpoint_state: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(_json_ready(asdict(cfg)), indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    pd.DataFrame(_flatten_metrics(metrics)).to_csv(output_dir / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    if checkpoint_state is not None:
        uses_validation_checkpoint = isinstance(cfg, CnnCsvSplitConfig) and (
            cfg.select_best_by_mcc or cfg.early_stopping_patience is not None
        )
        checkpoint_name = "best_model.pt" if uses_validation_checkpoint else "final_model.pt"
        checkpoint_path = output_dir / "checkpoints" / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_state, checkpoint_path)


def _flatten_metrics(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for split, values in metrics.items():
        row = {"split": split}
        for key, value in values.items():
            if key == "confusion_matrix":
                row.update(value)
            else:
                row[key] = value
        rows.append(row)
    return rows


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
