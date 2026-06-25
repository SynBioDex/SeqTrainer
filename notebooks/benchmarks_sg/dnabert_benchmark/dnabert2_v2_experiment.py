"""DNABERT2-v2 frozen-embedding benchmark.

The experiment keeps the SeqTrainer CNN-v2 comparison contract fixed:

- predefined train/validation/test CSV files
- seed policy rooted at 42
- candidate selection and threshold tuning on validation only
- held-out test evaluation only after the candidate family is selected
- MCC primary and AUPRC secondary

Colab and Alpine invoke this same script with different compute profiles.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from seqtrainer.metrics import best_threshold_by_metric
from seqtrainer.torch.dnabert2_benchmark import (
    _last_hidden_state,
    _load_huggingface_dnabert2,
)


SPLIT_FILES = {
    "train": "train_EP_DNA_BERT2_genomic_order.csv",
    "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
    "test": "test_EP_DNA_BERT2_genomic_order.csv",
}
POOLINGS = ("mean", "cls", "max")


@dataclass(frozen=True)
class Profile:
    token_lengths: tuple[int, ...]
    poolings: tuple[str, ...]
    heads: tuple[str, ...]
    learning_rates: tuple[float, ...]
    seeds: tuple[int, ...]
    extraction_batch_size: int
    head_batch_size: int
    max_epochs: int
    patience: int


PROFILES = {
    "colab": Profile(
        # 70 follows the official DNABERT2 recommendation for 300 bp
        # promoter sequences. 104 is retained as a controlled ablation.
        token_lengths=(70, 104),
        poolings=POOLINGS,
        heads=("linear", "mlp"),
        learning_rates=(3e-4,),
        seeds=(42,),
        extraction_batch_size=8,
        head_batch_size=512,
        max_epochs=75,
        patience=10,
    ),
    "hpc": Profile(
        # Keep the official 70-token reference and test only modestly longer
        # contexts. A 256-token grid is unnecessarily expensive for 300 bp.
        token_lengths=(70, 104, 128),
        poolings=POOLINGS,
        heads=("linear", "mlp"),
        learning_rates=(3e-4, 1e-3),
        seeds=(42, 43, 44),
        extraction_batch_size=32,
        head_batch_size=1024,
        max_epochs=100,
        patience=12,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="colab")
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name", default="zhihan1996/DNABERT-2-117M")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-extract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    data_dir = (args.data_dir or repo_dir / "data" / "promoter_classification").resolve()
    output_dir = (
        args.output_dir
        or repo_dir / "outputs" / "benchmarks" / f"dnabert2_v2_{args.profile}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]

    _seed_everything(42)
    device = _resolve_device(args.device)
    frames = _load_frames(data_dir)
    _write_dataset_audit(frames, output_dir)

    started = time.perf_counter()
    tokenizer, encoder = _load_huggingface_dnabert2(
        args.model_name,
        device=device,
        trust_remote_code=True,
        local_files_only=False,
        disable_flash_attention=True,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()

    embedding_root = output_dir / "embeddings"
    for token_length in profile.token_lengths:
        _extract_length_embeddings(
            frames=frames,
            tokenizer=tokenizer,
            encoder=encoder,
            token_length=token_length,
            poolings=profile.poolings,
            batch_size=profile.extraction_batch_size,
            device=device,
            output_dir=embedding_root / f"tokens_{token_length}",
            resume=args.resume,
        )

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.only_extract:
        print(f"Embedding extraction complete: {embedding_root}")
        return 0

    trial_rows: list[dict[str, Any]] = []
    checkpoint_root = output_dir / "checkpoints"
    history_root = output_dir / "history"

    for token_length in profile.token_lengths:
        cache_dir = embedding_root / f"tokens_{token_length}"
        for pooling in profile.poolings:
            embeddings = _load_embedding_set(cache_dir, pooling)
            for head in profile.heads:
                for learning_rate in profile.learning_rates:
                    for seed in profile.seeds:
                        trial_name = _trial_name(
                            token_length, pooling, head, learning_rate, seed
                        )
                        result = _train_trial(
                            embeddings=embeddings,
                            head_name=head,
                            learning_rate=learning_rate,
                            seed=seed,
                            batch_size=profile.head_batch_size,
                            max_epochs=profile.max_epochs,
                            patience=profile.patience,
                            device=device,
                            checkpoint_path=checkpoint_root / f"{trial_name}.pt",
                            history_path=history_root / f"{trial_name}.csv",
                        )
                        trial_rows.append(
                            {
                                "trial": trial_name,
                                "token_length": token_length,
                                "pooling": pooling,
                                "head": head,
                                "learning_rate": learning_rate,
                                "seed": seed,
                                **result,
                            }
                        )
                        print(
                            f"{trial_name}: validation_mcc={result['validation_mcc']:.4f} "
                            f"validation_auprc={result['validation_auprc']:.4f}"
                        )

    trials = pd.DataFrame(trial_rows)
    trials.to_csv(output_dir / "validation_trials.csv", index=False)
    candidates = _aggregate_candidates(trials)
    candidates.to_csv(output_dir / "validation_candidate_summary.csv", index=False)
    selected = candidates.iloc[0].to_dict()
    (output_dir / "selected_candidate.json").write_text(
        json.dumps(_json_ready(selected), indent=2), encoding="utf-8"
    )

    selected_trials = trials[
        (trials["token_length"] == selected["token_length"])
        & (trials["pooling"] == selected["pooling"])
        & (trials["head"] == selected["head"])
        & (trials["learning_rate"] == selected["learning_rate"])
    ].sort_values("seed")

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    selected_embeddings = _load_embedding_set(
        embedding_root / f"tokens_{int(selected['token_length'])}",
        str(selected["pooling"]),
    )
    for row in selected_trials.itertuples(index=False):
        metrics, predictions = _evaluate_selected_trial(
            embeddings=selected_embeddings,
            head_name=str(selected["head"]),
            checkpoint_path=checkpoint_root / f"{row.trial}.pt",
            threshold=float(row.validation_threshold),
            device=device,
        )
        metrics.insert(0, "trial", row.trial)
        metrics.insert(1, "seed", int(row.seed))
        metric_frames.append(metrics)
        predictions.insert(0, "trial", row.trial)
        predictions.insert(1, "seed", int(row.seed))
        prediction_frames.append(predictions)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    test_metrics = all_metrics.loc[all_metrics["split"] == "test"].reset_index(drop=True)
    test_metrics.to_csv(output_dir / "selected_test_metrics.csv", index=False)
    test_predictions = pd.concat(prediction_frames, ignore_index=True)
    test_predictions.to_csv(output_dir / "selected_test_predictions.csv", index=False)
    test_summary = _summarize_test_metrics(test_metrics)
    test_summary.to_csv(output_dir / "selected_test_summary.csv", index=False)
    _write_standard_artifacts(
        output_dir=output_dir,
        selected_trials=selected_trials,
        all_metrics=all_metrics,
        test_predictions=test_predictions,
        test_summary=test_summary,
        history_root=history_root,
    )

    manifest = {
        "experiment": "dnabert2_v2_frozen_embedding_ablation",
        "profile": args.profile,
        "model": args.model_name,
        "device": str(device),
        "profile_config": asdict(profile),
        "selection_policy": {
            "candidate_ranking": ["mean_validation_mcc", "mean_validation_auprc"],
            "threshold": "validation_mcc_only",
            "test_usage": "selected_candidate_final_reporting_only",
        },
        "scientific_reference": {
            "tokenization": "DNABERT2 BPE; k-mer preprocessing is not used",
            "official_300bp_reference_max_length": 70,
            "official_finetune_learning_rate": 3e-5,
            "current_stage": "frozen encoder with trainable classifier head",
        },
        "selected_candidate": selected,
        "dataset": {
            "split_files": {key: str(data_dir / value) for key, value in SPLIT_FILES.items()},
            "split_sizes": {key: int(len(value)) for key, value in frames.items()},
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "environment": _environment_metadata(repo_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2), encoding="utf-8"
    )
    print("\nSelected candidate:")
    print(json.dumps(_json_ready(selected), indent=2))
    print("\nHeld-out test summary:")
    print(test_summary.to_string(index=False))
    return 0


def _load_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split, filename in SPLIT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split: {path}")
        frame = pd.read_csv(path)
        missing = {"sequence", "label"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["sequence"] = (
            frame["sequence"].astype(str).str.upper().str.replace("U", "T", regex=False)
        )
        frame["label"] = frame["label"].astype(int)
        if not set(frame["label"].unique()).issubset({0, 1}):
            raise ValueError(f"{path} contains labels outside 0/1")
        frames[split] = frame.reset_index(drop=True)
    return frames


def _write_dataset_audit(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []
    for split, frame in frames.items():
        counts = frame["label"].value_counts().to_dict()
        lengths = frame["sequence"].str.len()
        rows.append(
            {
                "split": split,
                "rows": len(frame),
                "negative": int(counts.get(0, 0)),
                "positive": int(counts.get(1, 0)),
                "positive_fraction": float(frame["label"].mean()),
                "duplicates": int(frame["sequence"].duplicated().sum()),
                "min_bp": int(lengths.min()),
                "median_bp": float(lengths.median()),
                "max_bp": int(lengths.max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "dataset_audit.csv", index=False)


def _write_standard_artifacts(
    *,
    output_dir: Path,
    selected_trials: pd.DataFrame,
    all_metrics: pd.DataFrame,
    test_predictions: pd.DataFrame,
    test_summary: pd.DataFrame,
    history_root: Path,
) -> None:
    """Write the same top-level artifact names used by other benchmarks."""
    all_metrics.to_csv(output_dir / "metrics.csv", index=False)
    metrics_payload = {
        "selection": "validation MCC, validation AUPRC tie-break",
        "per_split_per_seed": all_metrics.to_dict(orient="records"),
        "test_summary": test_summary.to_dict(orient="records"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_ready(metrics_payload), indent=2), encoding="utf-8"
    )
    test_predictions.to_csv(output_dir / "predictions.csv", index=False)

    histories = []
    for row in selected_trials.itertuples(index=False):
        path = history_root / f"{row.trial}.csv"
        history = pd.read_csv(path)
        history.insert(0, "trial", row.trial)
        history.insert(1, "seed", int(row.seed))
        histories.append(history)
    pd.concat(histories, ignore_index=True).to_csv(
        output_dir / "history.csv", index=False
    )


def _extract_length_embeddings(
    *,
    frames: dict[str, pd.DataFrame],
    tokenizer: Any,
    encoder: Any,
    token_length: int,
    poolings: Iterable[str],
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(poolings)
    for split, frame in frames.items():
        targets = {pooling: output_dir / f"{split}_{pooling}.pt" for pooling in requested}
        labels_path = output_dir / f"{split}_labels.pt"
        if resume and labels_path.exists() and all(path.exists() for path in targets.values()):
            print(f"Reusing cached embeddings: tokens={token_length} split={split}")
            continue

        pooled_batches = {pooling: [] for pooling in requested}
        sequences = frame["sequence"].tolist()
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch_sequences,
                padding="longest",
                truncation=True,
                max_length=token_length,
                # Hugging Face requires max_length to be divisible by the
                # padding multiple. The official 300 bp setting is 70.
                pad_to_multiple_of=8 if token_length % 8 == 0 else None,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(device)
            with torch.inference_mode():
                hidden = _last_hidden_state(
                    encoder(input_ids=input_ids, attention_mask=attention_mask)
                )
            for pooling in requested:
                pooled = _pool(hidden, attention_mask, pooling)
                pooled_batches[pooling].append(pooled.detach().cpu().to(torch.float16))
            if start == 0 or (start // batch_size) % 500 == 0:
                print(
                    f"tokens={token_length} split={split}: "
                    f"{min(start + batch_size, len(sequences))}/{len(sequences)}"
                )

        for pooling, batches in pooled_batches.items():
            torch.save(torch.cat(batches, dim=0), targets[pooling])
        torch.save(torch.tensor(frame["label"].to_numpy(), dtype=torch.float32), labels_path)


def _pool(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return hidden[:, 0, :]
    expanded = mask.unsqueeze(-1).bool()
    if pooling == "max":
        return hidden.masked_fill(~expanded, torch.finfo(hidden.dtype).min).max(dim=1).values
    weights = expanded.to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _load_embedding_set(cache_dir: Path, pooling: str) -> dict[str, torch.Tensor]:
    data: dict[str, torch.Tensor] = {}
    for split in SPLIT_FILES:
        data[f"{split}_x"] = torch.load(
            cache_dir / f"{split}_{pooling}.pt", map_location="cpu"
        ).float()
        data[f"{split}_y"] = torch.load(
            cache_dir / f"{split}_labels.pt", map_location="cpu"
        ).float()
    return data


def _build_head(name: str, hidden_size: int) -> nn.Module:
    if name == "linear":
        return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
    if name == "mlp":
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )
    raise ValueError(f"Unknown head: {name}")


def _train_trial(
    *,
    embeddings: dict[str, torch.Tensor],
    head_name: str,
    learning_rate: float,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = _build_head(head_name, int(embeddings["train_x"].shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(embeddings["train_x"], embeddings["train_y"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )

    best_state = None
    best_threshold = 0.5
    best_mcc = float("-inf")
    best_auprc = float("-inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        scheduler.step()

        validation_scores = _predict_scores(
            model, embeddings["validation_x"], batch_size, device
        )
        validation_labels = embeddings["validation_y"].numpy().astype(int)
        threshold, mcc = best_threshold_by_metric(
            validation_labels, validation_scores, metric="mcc"
        )
        auprc = float(average_precision_score(validation_labels, validation_scores))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total, 1),
                "validation_mcc": float(mcc),
                "validation_auprc": auprc,
                "validation_threshold": float(threshold),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        improved = mcc > best_mcc or (np.isclose(mcc, best_mcc) and auprc > best_auprc)
        if improved:
            best_mcc = float(mcc)
            best_auprc = auprc
            best_threshold = float(threshold)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                history[-1]["stopped_early"] = True
                break

    if best_state is None:
        raise RuntimeError("No classifier checkpoint was selected")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "validation_mcc": best_mcc,
        "validation_auprc": best_auprc,
        "validation_threshold": best_threshold,
        "epochs_ran": len(history),
    }


def _predict_scores(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for (xb,) in loader:
            logits = model(xb.to(device, non_blocking=True)).squeeze(-1)
            scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy().astype(float)


def _aggregate_candidates(trials: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["token_length", "pooling", "head", "learning_rate"]
    summary = (
        trials.groupby(group_cols, as_index=False)
        .agg(
            mean_validation_mcc=("validation_mcc", "mean"),
            std_validation_mcc=("validation_mcc", "std"),
            mean_validation_auprc=("validation_auprc", "mean"),
            std_validation_auprc=("validation_auprc", "std"),
            seeds=("seed", "count"),
        )
        .fillna(0.0)
    )
    return summary.sort_values(
        ["mean_validation_mcc", "mean_validation_auprc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _evaluate_selected_trial(
    *,
    embeddings: dict[str, torch.Tensor],
    head_name: str,
    checkpoint_path: Path,
    threshold: float,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = _build_head(head_name, int(embeddings["train_x"].shape[1])).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metric_rows = []
    prediction_frames = []
    for split in SPLIT_FILES:
        scores = _predict_scores(model, embeddings[f"{split}_x"], 1024, device)
        labels = embeddings[f"{split}_y"].numpy().astype(int)
        predictions = (scores >= threshold).astype(int)
        metric_rows.append(
            {
                "split": split,
                "threshold": threshold,
                **_metrics(labels, scores, predictions),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "idx": np.arange(len(labels)),
                    "label": labels,
                    "probability": scores,
                    "threshold": threshold,
                    "prediction": predictions,
                }
            )
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def _metrics(
    labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _summarize_test_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "sensitivity",
        "specificity",
        "auroc",
        "auprc",
    ]
    rows = []
    for metric in columns:
        rows.append(
            {
                "metric": metric,
                "mean": float(metrics[metric].mean()),
                "std": float(metrics[metric].std(ddof=0)),
                "seed_42": float(
                    metrics.loc[metrics["seed"] == 42, metric].iloc[0]
                )
                if (metrics["seed"] == 42).any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def _trial_name(
    token_length: int,
    pooling: str,
    head: str,
    learning_rate: float,
    seed: int,
) -> str:
    lr = f"{learning_rate:.0e}".replace("-", "m")
    return f"tokens{token_length}_{pooling}_{head}_lr{lr}_seed{seed}"


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment_metadata(repo_dir: Path) -> dict[str, Any]:
    def git_value(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
