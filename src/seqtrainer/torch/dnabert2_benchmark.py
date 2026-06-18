"""Dependency-gated DNABERT2 benchmark runner."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqtrainer.benchmarks.artifacts import write_benchmark_outputs
from seqtrainer.benchmarks.config import BenchmarkConfig
from seqtrainer.benchmarks.manifest import build_run_manifest
from seqtrainer.benchmarks.runner import BenchmarkRunResult, BenchmarkSkipped
from seqtrainer.benchmarks.splits import load_predefined_split_frames, summarize_split_frames
from seqtrainer.metrics import best_threshold_by_metric, binary_classification_metrics


@dataclass(frozen=True)
class _EncodedSplit:
    input_ids: Any
    attention_mask: Any
    labels: Any


def run_dnabert2_csv_splits(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> BenchmarkRunResult:
    """Run DNABERT2 frozen/fine-tuned benchmark on predefined CSV splits."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
        raise BenchmarkSkipped(
            "DNABERT2 benchmark requires optional torch dependencies."
        ) from exc

    _seed_everything(config.training.seed)
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    params = dict(config.model.params)
    train_params = dict(config.training.params)
    allow_download = bool(params.get("allow_download", False))
    local_files_only = not allow_download
    trust_remote_code = bool(params.get("trust_remote_code", True))

    device = _resolve_device(config.environment.device, torch)

    if tokenizer is None or encoder is None:
        tokenizer, encoder = _load_huggingface_dnabert2(
            config.model.name,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    freeze_encoder = str(params.get("mode", "frozen_embedding_classifier")) != "full_finetune"
    if freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    hidden_size = int(getattr(encoder.config, "hidden_size", 768))
    model = _DnaBert2Classifier(
        encoder=encoder,
        hidden_size=hidden_size,
        pooling=str(params.get("pooling", "mean")),
        dropout=float(params.get("classifier_dropout", 0.1)),
    ).to(device)

    encoded = {
        split: _encode_split(config, frame, tokenizer, torch)
        for split, frame in frames.items()
    }
    loaders = {
        split: DataLoader(
            TensorDataset(value.input_ids, value.attention_mask, value.labels),
            batch_size=config.training.batch_size or 16,
            shuffle=(split == "train"),
        )
        for split, value in encoded.items()
    }

    pos = int(frames["train"][config.dataset.label_field].astype(int).sum())
    neg = int(len(frames["train"]) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if freeze_encoder:
        return _run_frozen_embedding_classifier(
            config,
            frames=frames,
            encoded=encoded,
            encoder=encoder,
            criterion=criterion,
            device=device,
            torch=torch,
            nn=nn,
            output_dir=output_dir,
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.training.learning_rate or 1e-4,
        weight_decay=float(train_params.get("weight_decay", 0.01)),
    )
    max_epochs = config.training.max_epochs or 3
    total_steps = max(1, max_epochs * len(loaders["train"]))
    warmup_steps = int(total_steps * float(train_params.get("warmup_ratio", 0.0)))
    scheduler = _linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    history: list[dict[str, float]] = []
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_mcc = float("-inf")
    best_threshold = 0.5
    patience = int(train_params.get("early_stopping_patience", 3))
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, loaders["train"], criterion, optimizer, scheduler, device, torch)
        validation = _predict(model, loaders["validation"], criterion, device, torch)
        threshold, validation_mcc = best_threshold_by_metric(
            validation["label"],
            validation["probability"],
            metric="mcc",
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation["loss"]),
                "validation_mcc": float(validation_mcc),
                "validation_threshold": float(threshold),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_mcc > best_mcc:
            best_mcc = float(validation_mcc)
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                history[-1]["stopped_early"] = 1.0
                break

    model.load_state_dict(best_state)
    predictions = {split: _predict(model, loader, criterion, device, torch) for split, loader in loaders.items()}
    metrics: dict[str, dict[str, Any]] = {}
    prediction_frames = []
    for split, pred in predictions.items():
        split_metrics = binary_classification_metrics(pred["label"], pred["probability"], best_threshold)
        split_metrics["loss"] = pred["loss"]
        metrics[split] = split_metrics
        frame = frames[split].reset_index(drop=True)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "idx": np.arange(len(frame)),
                    "sequence": frame[config.dataset.sequence_field].astype(str),
                    "label": frame[config.dataset.label_field].astype(int),
                    "probability": pred["probability"],
                    "threshold": best_threshold,
                    "prediction": (pred["probability"] >= best_threshold).astype(int),
                }
            )
        )

    out_dir = Path(output_dir or config.outputs.output_dir)
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    manifest = build_run_manifest(
        config,
        split_summary=summarize_split_frames(config, frames),
        threshold=best_threshold,
        model_metadata={
            "mode": params.get("mode", "frozen_embedding_classifier"),
            "pooling": params.get("pooling", "mean"),
            "freeze_encoder": freeze_encoder,
            "checkpoint": str(checkpoint_path),
            "pos_weight": float(pos_weight.item()),
            "optimizer": "adamw",
            "warmup_ratio": float(train_params.get("warmup_ratio", 0.0)),
            "early_stopping_metric": "validation_mcc",
        },
        extra={"status": "completed"},
    )
    write_benchmark_outputs(
        out_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        history=pd.DataFrame(history),
        config=config,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return BenchmarkRunResult(output_dir=out_dir, status="completed", metrics=metrics, manifest=manifest)


def _run_frozen_embedding_classifier(
    config: BenchmarkConfig,
    *,
    frames: dict[str, pd.DataFrame],
    encoded: dict[str, _EncodedSplit],
    encoder: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    nn: Any,
    output_dir: str | Path | None,
) -> BenchmarkRunResult:
    params = dict(config.model.params)
    train_params = dict(config.training.params)
    pooling = str(params.get("pooling", "mean"))
    out_dir = Path(output_dir or config.outputs.output_dir)
    embedding_dir = out_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    encoder.to(device)
    encoder.eval()
    embeddings: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for split, value in encoded.items():
        emb = _extract_embeddings(encoder, value, pooling=pooling, device=device, torch=torch)
        embeddings[split] = emb
        labels[split] = value.labels
        torch.save(
            {
                "embeddings": emb.detach().cpu(),
                "labels": value.labels.detach().cpu(),
                "model_name": config.model.name,
                "pooling": pooling,
                "split": split,
            },
            embedding_dir / f"{split}_embeddings.pt",
        )

    hidden_size = int(embeddings["train"].shape[1])
    classifier = nn.Sequential(
        nn.Dropout(float(params.get("classifier_dropout", 0.1))),
        nn.Linear(hidden_size, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.training.learning_rate or 1e-3,
        weight_decay=float(train_params.get("weight_decay", 0.01)),
    )
    max_epochs = config.training.max_epochs or 20
    total_steps = max(1, max_epochs)
    warmup_steps = int(total_steps * float(train_params.get("warmup_ratio", 0.0)))
    scheduler = _linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    patience = int(train_params.get("early_stopping_patience", 4))
    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    best_mcc = float("-inf")
    best_threshold = 0.5
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        classifier.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = classifier(embeddings["train"].to(device)).squeeze(-1)
        train_labels = labels["train"].to(device)
        train_loss = criterion(train_logits, train_labels)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        validation = _predict_from_embeddings(classifier, embeddings["validation"], labels["validation"], criterion, device, torch)
        threshold, validation_mcc = best_threshold_by_metric(
            validation["label"],
            validation["probability"],
            metric="mcc",
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation["loss"]),
                "validation_mcc": float(validation_mcc),
                "validation_threshold": float(threshold),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_mcc > best_mcc:
            best_mcc = float(validation_mcc)
            best_threshold = float(threshold)
            best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                history[-1]["stopped_early"] = 1.0
                break

    classifier.load_state_dict(best_state)
    predictions = {
        split: _predict_from_embeddings(classifier, embeddings[split], labels[split], criterion, device, torch)
        for split in ("train", "validation", "test")
    }
    metrics: dict[str, dict[str, Any]] = {}
    prediction_frames = []
    for split, pred in predictions.items():
        split_metrics = binary_classification_metrics(pred["label"], pred["probability"], best_threshold)
        split_metrics["loss"] = pred["loss"]
        metrics[split] = split_metrics
        frame = frames[split].reset_index(drop=True)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "idx": np.arange(len(frame)),
                    "sequence": frame[config.dataset.sequence_field].astype(str),
                    "label": frame[config.dataset.label_field].astype(int),
                    "probability": pred["probability"],
                    "threshold": best_threshold,
                    "prediction": (pred["probability"] >= best_threshold).astype(int),
                }
            )
        )

    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    manifest = build_run_manifest(
        config,
        split_summary=summarize_split_frames(config, frames),
        threshold=best_threshold,
        model_metadata={
            "mode": "frozen_embedding_classifier",
            "pooling": pooling,
            "freeze_encoder": True,
            "embedding_cache_dir": str(embedding_dir),
            "checkpoint": str(checkpoint_path),
            "pos_weight": float(criterion.pos_weight.item()) if getattr(criterion, "pos_weight", None) is not None else None,
            "optimizer": "adamw",
            "warmup_ratio": float(train_params.get("warmup_ratio", 0.0)),
            "early_stopping_metric": "validation_mcc",
        },
        extra={"status": "completed"},
    )
    write_benchmark_outputs(
        out_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        history=pd.DataFrame(history),
        config=config,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return BenchmarkRunResult(output_dir=out_dir, status="completed", metrics=metrics, manifest=manifest)


class _DnaBert2Classifier:
    def __init__(self, encoder: Any, hidden_size: int, pooling: str, dropout: float) -> None:
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.pooling = pooling
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                pooled = _pool_hidden_states(hidden, attention_mask, self.pooling)
                return self.head(self.dropout(pooled)).squeeze(-1)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def _encode_split(config: BenchmarkConfig, frame: pd.DataFrame, tokenizer: Any, torch: Any) -> _EncodedSplit:
    preprocessing = dict(config.preprocessing.params)
    pad_to_multiple_of = preprocessing.get("pad_to_multiple_of")
    encoded = tokenizer(
        frame[config.dataset.sequence_field].astype(str).tolist(),
        padding=str(preprocessing.get("padding", "longest")),
        truncation=True,
        max_length=int(preprocessing.get("model_max_length", config.preprocessing.sequence_length or 512)),
        pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of is not None else None,
        return_tensors="pt",
    )
    labels = torch.tensor(frame[config.dataset.label_field].astype(int).to_numpy(), dtype=torch.float32)
    return _EncodedSplit(encoded["input_ids"], encoded["attention_mask"], labels)


def _load_huggingface_dnabert2(
    model_name: str,
    *,
    device: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
        raise BenchmarkSkipped(
            "DNABERT2 benchmark requires transformers. Install with `python -m pip install -e \".[torch]\"`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
            device_map=None,
            local_files_only=local_files_only,
        )
        meta_params = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta_params:
            raise RuntimeError(
                "DNABERT2 loaded with parameters on the meta device. "
                "Reload with low_cpu_mem_usage=False and device_map=None. "
                f"Example meta params: {meta_params[:5]}"
            )
        _ensure_pad_token_id(model.config, tokenizer)
        model.to(device)
        model.eval()
    except OSError as exc:
        raise BenchmarkSkipped(
            "DNABERT2 model/tokenizer files are not available locally. "
            "Set model.params.allow_download=true only when the environment may download them."
        ) from exc
    return tokenizer, model


def _ensure_pad_token_id(config: Any, tokenizer: Any) -> None:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    config.pad_token_id = pad_token_id
    setattr(config, "pad_token_id", pad_token_id)
    setattr(config.__class__, "pad_token_id", pad_token_id)
    if hasattr(config, "__dict__"):
        config.__dict__["pad_token_id"] = pad_token_id
    if hasattr(config, "update"):
        config.update({"pad_token_id": pad_token_id})


def _patch_bert_config_pad_token_id(pad_token_id: int | None) -> None:
    pad_token_id = pad_token_id if pad_token_id is not None else 0
    try:
        from transformers.models.bert.configuration_bert import BertConfig

        setattr(BertConfig, "pad_token_id", pad_token_id)
    except Exception:
        return


def _load_dnabert2_from_state_dict(
    model_name: str,
    *,
    config: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load DNABERT2 without Transformers' meta-device checkpoint path.

    Newer Colab/PyTorch/Transformers combinations can route remote-code models
    through meta tensors even when low_cpu_mem_usage is disabled. Instantiating
    from config and loading the state dict directly keeps all weights on CPU.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    encoder = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
    checkpoint_path = hf_hub_download(
        repo_id=model_name,
        filename="pytorch_model.bin",
        local_files_only=local_files_only,
    )
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    encoder.load_state_dict(state_dict, strict=False)
    return encoder


def _load_dnabert2_official_encoder(
    model_name: str,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load DNABERT2 with the simple path documented on its model card."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            dtype="auto",
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )


def _load_dnabert2_encoder_from_sequence_classifier(
    model_name: str,
    *,
    config: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    """Load DNABERT2 through the sequence-classification path used upstream."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        config=config,
        num_labels=2,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    for attr in ("bert", "base_model", "encoder"):
        encoder = getattr(model, attr, None)
        if encoder is not None:
            return encoder
    raise RuntimeError("Could not locate the DNABERT2 encoder inside the sequence-classification model.")


def _extract_embeddings(encoder: Any, encoded: _EncodedSplit, *, pooling: str, device: Any, torch: Any) -> Any:
    with torch.no_grad():
        batch = {
            "input_ids": encoded.input_ids.to(device),
            "attention_mask": encoded.attention_mask.to(device),
        }
        outputs = encoder(**batch)
        return _pool_hidden_states(outputs.last_hidden_state, batch["attention_mask"], pooling).detach().cpu()


def _pool_hidden_states(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling == "cls":
        return hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _predict_from_embeddings(classifier: Any, embeddings: Any, labels: Any, criterion: Any, device: Any, torch: Any) -> dict[str, Any]:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.to(device)).squeeze(-1)
        target = labels.to(device)
        loss = criterion(logits, target)
        probs = torch.sigmoid(logits)
    return {
        "label": labels.detach().cpu().numpy().astype(int),
        "probability": probs.detach().cpu().numpy().astype(float),
        "loss": float(loss.item()),
    }


def _linear_warmup_scheduler(optimizer: Any, warmup_steps: int, total_steps: int) -> Any:
    import torch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        remaining = max(0, total_steps - step)
        decay_steps = max(1, total_steps - warmup_steps)
        return float(remaining) / float(decay_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _run_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, scheduler: Any, device: Any, torch: Any) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += int(labels.shape[0])
    return total_loss / max(total, 1)


def _predict(model: Any, loader: Any, criterion: Any, device: Any, torch: Any) -> dict[str, Any]:
    model.eval()
    labels_out, probs_out = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            labels_out.extend(labels.detach().cpu().numpy())
            probs_out.extend(probs.detach().cpu().numpy())
            total_loss += float(loss.item()) * labels.shape[0]
            total += int(labels.shape[0])
    return {
        "label": np.asarray(labels_out, dtype=int),
        "probability": np.asarray(probs_out, dtype=float),
        "loss": total_loss / max(total, 1),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def _resolve_device(device: str, torch: Any) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
