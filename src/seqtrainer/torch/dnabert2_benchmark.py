"""Dependency-gated DNABERT2 benchmark runner."""

from __future__ import annotations

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seqtrainer.benchmarks.artifacts import write_benchmark_outputs
from seqtrainer.benchmarks.config import BenchmarkConfig
from seqtrainer.benchmarks.manifest import build_run_manifest
from seqtrainer.benchmarks.policy import decide_imbalance_policy
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
    split_summary = summarize_split_frames(config, frames)
    imbalance_policy = decide_imbalance_policy(split_summary)
    params = dict(config.model.params)
    train_params = dict(config.training.params)
    allow_download = bool(params.get("allow_download", False))
    local_files_only = not allow_download
    trust_remote_code = bool(params.get("trust_remote_code", True))

    device = _resolve_device(config.environment.device, torch)
    run_started = time.perf_counter()
    _reset_peak_memory(torch, device)

    if tokenizer is None or encoder is None:
        tokenizer, encoder = _load_huggingface_dnabert2(
            config.model.name,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            disable_flash_attention=bool(params.get("disable_flash_attention", False)),
            revision=params.get("revision"),
        )
    freeze_encoder = str(params.get("mode", "frozen_embedding_classifier")) != "full_finetune"
    if freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    gradient_checkpointing = bool(train_params.get("gradient_checkpointing", False))
    if gradient_checkpointing and not freeze_encoder:
        _enable_gradient_checkpointing(encoder)

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
    prediction_loaders = {
        split: DataLoader(
            TensorDataset(value.input_ids, value.attention_mask, value.labels),
            batch_size=config.training.batch_size or 16,
            shuffle=False,
        )
        for split, value in encoded.items()
    }

    pos_weight = None
    if imbalance_policy.apply_to_training:
        train_labels = encoded["train"].labels
        pos = int(train_labels.sum().item())
        neg = int(len(train_labels) - pos)
        pos_weight = torch.tensor(
            [neg / max(pos, 1)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
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
            run_started=run_started,
            output_dir=output_dir,
        )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.training.learning_rate or 1e-4,
        weight_decay=float(train_params.get("weight_decay", 0.01)),
    )
    max_epochs = 3 if config.training.max_epochs is None else int(config.training.max_epochs)
    gradient_accumulation_steps = max(
        1, int(train_params.get("gradient_accumulation_steps", 1))
    )
    optimizer_steps_per_epoch = ceil(
        len(loaders["train"]) / gradient_accumulation_steps
    )
    total_steps = max(1, max_epochs * optimizer_steps_per_epoch)
    warmup_steps = int(total_steps * float(train_params.get("warmup_ratio", 0.0)))
    scheduler = _linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    precision = str(config.environment.precision).lower()
    max_grad_norm = float(train_params.get("max_grad_norm", 1.0))

    history: list[dict[str, float]] = []
    best_mcc = float("-inf")
    best_threshold = 0.5
    patience = int(train_params.get("early_stopping_patience", 3))
    bad_epochs = 0
    out_dir = Path(output_dir or config.outputs.output_dir)
    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            scheduler,
            device,
            torch,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )
        validation = _predict(
            model,
            prediction_loaders["validation"],
            criterion,
            device,
            torch,
            precision=precision,
        )
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
            torch.save(model.state_dict(), checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                history[-1]["stopped_early"] = 1.0
                break

    if not checkpoint_path.exists():
        raise RuntimeError("DNABERT2 fine-tuning did not produce a checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    predictions = {
        split: _predict(
            model,
            loader,
            criterion,
            device,
            torch,
            precision=precision,
        )
        for split, loader in prediction_loaders.items()
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
                    "label": encoded[split].labels.detach().cpu().numpy().astype(int),
                    "probability": pred["probability"],
                    "threshold": best_threshold,
                    "prediction": (pred["probability"] >= best_threshold).astype(int),
                }
            )
        )

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=best_threshold,
        model_metadata={
            "mode": params.get("mode", "frozen_embedding_classifier"),
            "pooling": params.get("pooling", "mean"),
            "freeze_encoder": freeze_encoder,
            "checkpoint": str(checkpoint_path),
            "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
            "optimizer": "adamw",
            "warmup_ratio": float(train_params.get("warmup_ratio", 0.0)),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "physical_batch_size": config.training.batch_size or 16,
            "effective_batch_size": (config.training.batch_size or 16)
            * gradient_accumulation_steps,
            "precision": precision,
            "max_grad_norm": max_grad_norm,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "early_stopping_metric": "validation_mcc",
            "resolved_device": str(device),
            "runtime_seconds": float(time.perf_counter() - run_started),
            "peak_memory_mb": _peak_memory_mb(torch, device),
        },
        extra={
            "status": "completed",
            "imbalance_policy": {
                "apply_to_training": imbalance_policy.apply_to_training,
                "strategy": imbalance_policy.strategy,
                "class_counts": imbalance_policy.class_counts,
                "imbalance_ratio": imbalance_policy.imbalance_ratio,
                "reason": imbalance_policy.reason,
            },
        },
    )
    write_benchmark_outputs(
        out_dir,
        manifest=manifest,
        metrics=metrics,
        predictions=pd.concat(prediction_frames, ignore_index=True),
        history=pd.DataFrame(history),
        config=config,
    )
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
    run_started: float,
    output_dir: str | Path | None,
) -> BenchmarkRunResult:
    from torch.utils.data import DataLoader, TensorDataset

    params = dict(config.model.params)
    train_params = dict(config.training.params)
    pooling = str(params.get("pooling", "mean"))
    out_dir = Path(output_dir or config.outputs.output_dir)
    embedding_dir = out_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    split_summary = summarize_split_frames(config, frames)
    imbalance_policy = decide_imbalance_policy(split_summary)

    encoder.to(device)
    encoder.eval()
    embeddings: dict[str, Any] = {}
    labels: dict[str, Any] = {}
    for split, value in encoded.items():
        emb = _extract_embeddings(
            encoder,
            value,
            pooling=pooling,
            device=device,
            torch=torch,
            batch_size=config.training.batch_size or 16,
        )
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
    train_loader = DataLoader(
        TensorDataset(embeddings["train"], labels["train"]),
        batch_size=config.training.batch_size or 16,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.training.learning_rate or 1e-3,
        weight_decay=float(train_params.get("weight_decay", 0.01)),
    )
    max_epochs = 20 if config.training.max_epochs is None else int(config.training.max_epochs)
    total_steps = max(1, max_epochs * len(train_loader))
    warmup_steps = int(total_steps * float(train_params.get("warmup_ratio", 0.0)))
    scheduler = _linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    patience = int(train_params.get("early_stopping_patience", 4))
    best_state = {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()}
    best_mcc = float("-inf")
    best_auprc = float("-inf")
    best_threshold = 0.5
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        classifier.train()
        train_loss_total = 0.0
        train_examples = 0
        for train_embeddings, train_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            train_logits = classifier(train_embeddings.to(device)).squeeze(-1)
            train_labels = train_labels.to(device)
            train_loss = criterion(train_logits, train_labels)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            batch_size = int(train_labels.shape[0])
            train_loss_total += float(train_loss.item()) * batch_size
            train_examples += batch_size
        train_loss_value = train_loss_total / max(train_examples, 1)

        validation = _predict_from_embeddings(classifier, embeddings["validation"], labels["validation"], criterion, device, torch)
        threshold, validation_mcc = best_threshold_by_metric(
            validation["label"],
            validation["probability"],
            metric="mcc",
        )
        validation_metrics = binary_classification_metrics(
            validation["label"],
            validation["probability"],
            threshold,
        )
        validation_auprc = validation_metrics["auprc"]
        validation_auprc_score = float(validation_auprc) if validation_auprc is not None else float("-inf")
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss_value,
                "validation_loss": float(validation["loss"]),
                "validation_mcc": float(validation_mcc),
                "validation_auprc": validation_auprc_score,
                "validation_threshold": float(threshold),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if (
            validation_mcc > best_mcc
            or (validation_mcc == best_mcc and validation_auprc_score > best_auprc)
        ):
            best_mcc = float(validation_mcc)
            best_auprc = validation_auprc_score
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
                    "label": encoded[split].labels.detach().cpu().numpy().astype(int),
                    "probability": pred["probability"],
                    "threshold": best_threshold,
                    "prediction": (pred["probability"] >= best_threshold).astype(int),
                }
            )
        )

    checkpoint_path = out_dir / "checkpoints" / "best_model.pt"
    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
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
            "early_stopping_tie_breaker": "validation_auprc",
            "resolved_device": str(device),
            "runtime_seconds": float(time.perf_counter() - run_started),
            "peak_memory_mb": _peak_memory_mb(torch, device),
        },
        extra={
            "status": "completed",
            "imbalance_policy": {
                "apply_to_training": imbalance_policy.apply_to_training,
                "strategy": imbalance_policy.strategy,
                "class_counts": imbalance_policy.class_counts,
                "imbalance_ratio": imbalance_policy.imbalance_ratio,
                "reason": imbalance_policy.reason,
            },
        },
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
                hidden = _last_hidden_state(outputs)
                pooled = _pool_hidden_states(hidden, attention_mask, self.pooling)
                return self.head(self.dropout(pooled)).squeeze(-1)

        self._model = _Model()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)


def _normalize_binary_labels(config: BenchmarkConfig, frame: pd.DataFrame) -> Any:
    """Map configured negative/positive labels to model targets 0/1."""
    negative_label = config.label.negative_label
    positive_label = config.label.positive_label
    if negative_label == positive_label:
        raise ValueError("Configured negative_label and positive_label must differ.")

    raw_labels = frame[config.dataset.label_field]
    known = raw_labels.isin([negative_label, positive_label])
    if not bool(known.all()):
        unexpected = raw_labels.loc[~known].drop_duplicates().tolist()
        raise ValueError(
            f"Labels {unexpected!r} do not match configured negative/positive labels "
            f"{negative_label!r}/{positive_label!r}."
        )

    return raw_labels.map({negative_label: 0, positive_label: 1}).to_numpy(dtype=np.float32)


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
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(encoded["input_ids"])
    labels = torch.tensor(
        _normalize_binary_labels(config, frame),
        dtype=torch.float32,
    )
    return _EncodedSplit(encoded["input_ids"], attention_mask, labels)


def _load_huggingface_dnabert2(
    model_name: str,
    *,
    device: Any,
    trust_remote_code: bool,
    local_files_only: bool,
    disable_flash_attention: bool = False,
    revision: str | None = None,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional extras
        raise BenchmarkSkipped(
            "DNABERT2 benchmark requires transformers. Install with `python -m pip install -e \".[torch]\"`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=revision,
        )
        pad_token_id = _safe_pad_token_id(tokenizer)
        _patch_bert_config_pad_token_id(pad_token_id)
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            revision=revision,
        )
        _set_pad_token_id(config, pad_token_id)
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=False,
                device_map=None,
                local_files_only=local_files_only,
                config=config,
                revision=revision,
            )
        except RuntimeError as exc:
            if not _is_meta_device_error(exc):
                raise
            model = _load_dnabert2_from_state_dict(
                model_name,
                config=config,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                revision=revision,
            )
        meta_params = _meta_parameter_names(model)
        if meta_params:
            try:
                model = _load_dnabert2_from_state_dict(
                    model_name,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                    revision=revision,
                )
            except Exception as fallback_exc:
                raise BenchmarkSkipped(
                    "DNABERT2 loaded with parameters on the meta device and the CPU state-dict "
                    "fallback also failed. "
                    f"Example meta params: {meta_params[:5]}. Fallback error: {fallback_exc}"
                ) from fallback_exc
            meta_params = _meta_parameter_names(model)
            if meta_params:
                raise BenchmarkSkipped(
                    "DNABERT2 still has parameters on the meta device after state-dict fallback. "
                    f"Example meta params: {meta_params[:5]}"
                )
        _set_pad_token_id(getattr(model, "config", None), pad_token_id)
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            _set_pad_token_id(generation_config, pad_token_id)
        try:
            model.to(device)
        except RuntimeError as exc:
            if not _is_meta_device_error(exc):
                raise
            model = _load_dnabert2_from_state_dict(
                model_name,
                config=config,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                revision=revision,
            )
            _set_pad_token_id(getattr(model, "config", None), pad_token_id)
            model.to(device)
        meta_params = _meta_parameter_names(model)
        if meta_params:
            raise BenchmarkSkipped(
                "DNABERT2 parameters remained on the meta device after moving to the target device. "
                f"Example meta params: {meta_params[:5]}"
            )
        if disable_flash_attention or getattr(device, "type", None) == "cpu":
            _disable_dnabert2_flash_attention(model)
        model.eval()
    except AttributeError as exc:
        if "pad_token_id" not in str(exc):
            raise
        raise BenchmarkSkipped(
            "DNABERT2 remote code still failed while resolving pad_token_id after config patching. "
            "Retry with the updated branch in Colab or pin a compatible Transformers revision."
        ) from exc
    except OSError as exc:
        raise BenchmarkSkipped(
            "DNABERT2 model/tokenizer files are not available locally. "
            "Set model.params.allow_download=true only when the environment may download them."
        ) from exc
    return tokenizer, model


def _disable_dnabert2_flash_attention(model: Any) -> None:
    """Route DNABERT2 remote code through its standard PyTorch attention path.

    The official DNABERT2 remote module prefers a Triton FlashAttention kernel
    when available. That path is CUDA-only and can also fail on incompatible
    Triton/PyTorch combinations. The remote code already contains a plain
    PyTorch fallback, selected when its module-level flash function is None.
    """
    import importlib

    module_name = getattr(model.__class__, "__module__", "")
    if not module_name:
        return
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return
    if hasattr(module, "flash_attn_qkvpacked_func"):
        setattr(module, "flash_attn_qkvpacked_func", None)


def _enable_gradient_checkpointing(model: Any) -> None:
    """Enable activation checkpointing for resource-constrained fine-tuning."""
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise BenchmarkSkipped(
            "This DNABERT2 implementation does not expose gradient checkpointing. "
            "Disable training.params.gradient_checkpointing or use the pinned model revision."
        )
    enable()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


def _ensure_pad_token_id(config: Any, tokenizer: Any) -> None:
    _set_pad_token_id(config, _safe_pad_token_id(tokenizer))


def _safe_pad_token_id(tokenizer: Any) -> int:
    return int(tokenizer.pad_token_id) if getattr(tokenizer, "pad_token_id", None) is not None else 0


def _set_pad_token_id(target: Any, pad_token_id: int) -> None:
    if target is None:
        return
    try:
        setattr(target, "pad_token_id", pad_token_id)
    except (AttributeError, TypeError):
        pass
    try:
        setattr(target.__class__, "pad_token_id", pad_token_id)
    except (AttributeError, TypeError):
        pass
    if hasattr(target, "__dict__"):
        target.__dict__["pad_token_id"] = pad_token_id
    if hasattr(target, "update"):
        target.update({"pad_token_id": pad_token_id})


def _is_meta_device_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "meta" in message and "device" in message


def _patch_bert_config_pad_token_id(pad_token_id: int | None) -> None:
    pad_token_id = pad_token_id if pad_token_id is not None else 0
    try:
        from transformers.configuration_utils import PretrainedConfig
        from transformers.models.bert.configuration_bert import BertConfig

        setattr(PretrainedConfig, "pad_token_id", pad_token_id)
        setattr(BertConfig, "pad_token_id", pad_token_id)
    except Exception:
        return


def _meta_parameter_names(model: Any) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if getattr(parameter, "is_meta", False)
    ]


def _load_dnabert2_from_state_dict(
    model_name: str,
    *,
    config: Any,
    trust_remote_code: bool,
    local_files_only: bool,
    revision: str | None = None,
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
        revision=revision,
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


def _extract_embeddings(
    encoder: Any,
    encoded: _EncodedSplit,
    *,
    pooling: str,
    device: Any,
    torch: Any,
    batch_size: int,
) -> Any:
    pooled_batches = []
    total = int(encoded.input_ids.shape[0])
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, total, step):
            stop = min(start + step, total)
            batch = {
                "input_ids": _tensor_to_device(encoded.input_ids[start:stop], device),
                "attention_mask": _tensor_to_device(encoded.attention_mask[start:stop], device),
            }
            outputs = encoder(**batch)
            pooled = _pool_hidden_states(_last_hidden_state(outputs), batch["attention_mask"], pooling)
            pooled_batches.append(pooled.detach().cpu())
    return torch.cat(pooled_batches, dim=0)


def _last_hidden_state(outputs: Any) -> Any:
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise AttributeError("DNABERT2 output did not contain last_hidden_state.")


def _pool_hidden_states(hidden: Any, attention_mask: Any, pooling: str) -> Any:
    if pooling == "cls":
        return hidden[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _predict_from_embeddings(classifier: Any, embeddings: Any, labels: Any, criterion: Any, device: Any, torch: Any) -> dict[str, Any]:
    classifier.eval()
    with torch.no_grad():
        logits = classifier(embeddings.to(device)).squeeze(-1)
        target = _tensor_to_device(labels, device)
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


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    torch: Any,
    *,
    precision: str = "float32",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss, total = 0.0, 0
    accumulation = max(1, int(gradient_accumulation_steps))
    scaler = _grad_scaler(torch, device, precision)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, (input_ids, attention_mask, labels) in enumerate(loader, start=1):
        input_ids = _tensor_to_device(input_ids, device)
        attention_mask = _tensor_to_device(attention_mask, device)
        labels = _tensor_to_device(labels, device)
        window_start = ((batch_index - 1) // accumulation) * accumulation + 1
        window_size = min(accumulation, len(loader) - window_start + 1)
        with _autocast_context(torch, device, precision):
            logits = model(input_ids, attention_mask)
            raw_loss = criterion(logits, labels)
            loss = raw_loss / window_size
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        should_step = batch_index % accumulation == 0 or batch_index == len(loader)
        if should_step:
            if scaler is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(raw_loss.item()) * labels.shape[0]
        total += int(labels.shape[0])
    return total_loss / max(total, 1)


def _predict(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch: Any,
    *,
    precision: str = "float32",
) -> dict[str, Any]:
    model.eval()
    labels_out, probs_out = [], []
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = _tensor_to_device(input_ids, device)
            attention_mask = _tensor_to_device(attention_mask, device)
            labels = _tensor_to_device(labels, device)
            with _autocast_context(torch, device, precision):
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


def _autocast_context(torch: Any, device: Any, precision: str) -> Any:
    if getattr(device, "type", None) != "cuda":
        return nullcontext()
    normalized = str(precision).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(torch: Any, device: Any, precision: str) -> Any | None:
    if (
        getattr(device, "type", None) == "cuda"
        and str(precision).lower() in {"fp16", "float16"}
    ):
        return torch.cuda.amp.GradScaler()
    return None


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


def _tensor_to_device(tensor: Any, device: Any) -> Any:
    return tensor.to(device, non_blocking=getattr(device, "type", None) == "cuda")


def _reset_peak_memory(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) != "cuda":
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def _peak_memory_mb(torch: Any, device: Any) -> float | None:
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    except Exception:
        return None
