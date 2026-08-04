#!/usr/bin/env python3
"""Train the default ~12M Titan MAC DNA LM on staged Colab token shards."""

from __future__ import annotations

import argparse
import csv
import inspect
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seqtrainer.data.bacteria_titan import TokenShardDataset
from seqtrainer.torch.titans_mac import (
    TitansMACForCausalLM,
    TitansMACLMConfig,
    compute_lm_metrics,
    count_parameters,
    load_training_checkpoint,
    save_training_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dataset-class", default="bacteria_titan_v1_ecoli_related_15gbp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--local-root", type=Path, default=Path("/content/bacteria_titan_local"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def config_for_context(context_length: int) -> TitansMACLMConfig:
    return TitansMACLMConfig(
        vocab_size=6,
        pad_token_id=0,
        unk_token_id=1,
        d_model=384,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1536,
        max_length=context_length,
        memory_slots=64,
        memory_depth=2,
        memory_context_tokens=8,
        persistent_tokens=8,
        dropout=0.1,
        retention_gate=0.95,
        use_persistent_memory=True,
        tie_embeddings=True,
    )


def stage_shards(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        target = destination / split
        target.mkdir(parents=True, exist_ok=True)
        for shard in sorted((source / split).glob("tokens_*.npy")):
            copied = target / shard.name
            if not copied.exists() or copied.stat().st_size != shard.stat().st_size:
                shutil.copy2(shard, copied)
    tokenizer = source / "tokenizer.json"
    if tokenizer.exists():
        shutil.copy2(tokenizer, destination / tokenizer.name)


def loader(dataset: TokenShardDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    options = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers:
        options.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **options)


def optimizer_for(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    options = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        options["fused"] = True
    return torch.optim.AdamW(model.parameters(), **options)


@torch.no_grad()
def validate(model: torch.nn.Module, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    for inputs, labels in data:
        inputs = inputs.to(device, non_blocking=True).long()
        labels = labels.to(device, non_blocking=True).long()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(inputs, labels=labels, update_memory=False)
        metrics = compute_lm_metrics(output["logits"], labels, input_ids=inputs)
        batch = inputs.size(0)
        for key, value in metrics.items():
            if np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + value * batch
        examples += batch
    return {key: value / examples for key, value in totals.items()}


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_history(path: Path, history: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set(xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_perplexity"] for row in history], label="perplexity")
    axes[1].plot(epochs, [row["val_token_accuracy"] for row in history], label="accuracy")
    axes[1].set(xlabel="epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.context_length != 2048:
        print(f"Using requested context length {args.context_length}; the v1 reference config uses 2048")
    drive_tokens = (
        args.drive_root / args.dataset_class / "tokenized" / args.dataset_class / f"ctx{args.context_length}"
    )
    local_tokens = args.local_root / args.dataset_class / f"ctx{args.context_length}"
    stage_shards(drive_tokens, local_tokens)
    train_data = TokenShardDataset(local_tokens / "train")
    val_data = TokenShardDataset(local_tokens / "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    config = config_for_context(args.context_length)
    model = TitansMACForCausalLM(config).to(device)
    optimizer = optimizer_for(model, args)
    run_dir = args.drive_root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = run_dir / "latest.pt"
    start_epoch, global_step, history = 0, 0, []
    if latest.exists() and not args.no_resume:
        checkpoint = load_training_checkpoint(latest, model, optimizer, map_location=device, trusted=True)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("step", 0))
        history = checkpoint.get("history", [])
        print(f"Resumed {latest} at epoch {start_epoch}")
    print(f"Device: {device}; trainable parameters: {count_parameters(model):,}")
    training_model = torch.compile(model) if args.compile and hasattr(torch, "compile") else model
    train_loader = loader(train_data, args, shuffle=True)
    val_loader = loader(val_data, args, shuffle=False)
    best = min((row.get("overall_validation_score", row["val_loss"]) for row in history), default=float("inf"))

    for epoch in range(start_epoch, args.epochs):
        training_model.train()
        model.reset_memory()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, token_count = 0.0, 0
        started = time.perf_counter()
        for batch_index, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True).long()
            labels = labels.to(device, non_blocking=True).long()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = training_model(inputs, labels=labels)
                loss = output["loss"] / args.grad_accumulation
            loss.backward()
            if (batch_index + 1) % args.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            valid_tokens = labels.ne(config.pad_token_id).sum().item()
            loss_sum += float(output["loss"].detach()) * valid_tokens
            token_count += valid_tokens
            if (batch_index + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - started
                allocated = torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                metrics = compute_lm_metrics(output["logits"].detach(), labels, input_ids=inputs)
                print(
                    f"epoch={epoch} batch={batch_index + 1} loss={loss_sum / token_count:.4f} "
                    f"ppl={metrics['perplexity']:.3f} bpb={metrics['bits_per_base']:.3f} "
                    f"acc={metrics['token_accuracy']:.3f} top2={metrics['top_2_accuracy']:.3f} "
                    f"confidence={metrics['confidence']:.3f} entropy={metrics['entropy']:.3f} "
                    f"gc_losses=({metrics['gc_0_40_loss']:.3f},{metrics['gc_40_60_loss']:.3f},"
                    f"{metrics['gc_60_100_loss']:.3f}) tokens/s={token_count / elapsed:,.0f} "
                    f"gpu_gb={allocated:.2f}/{peak:.2f}"
                )
        elapsed = time.perf_counter() - started
        validation = validate(training_model, val_loader, device)
        overall_score = (
            validation["loss"]
            + (1.0 - validation["token_accuracy"])
            + 0.25 * (1.0 - validation["top_2_accuracy"])
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / token_count,
            "val_loss": validation["loss"],
            "val_perplexity": validation["perplexity"],
            "val_bits_per_base": validation["bits_per_base"],
            "val_token_accuracy": validation["token_accuracy"],
            "val_top_2_accuracy": validation["top_2_accuracy"],
            "val_confidence": validation["confidence"],
            "val_entropy": validation["entropy"],
            "val_gc_0_40_loss": validation.get("gc_0_40_loss", float("nan")),
            "val_gc_40_60_loss": validation.get("gc_40_60_loss", float("nan")),
            "val_gc_60_100_loss": validation.get("gc_60_100_loss", float("nan")),
            "overall_validation_score": overall_score,
            "tokens_per_second": token_count / elapsed,
        }
        history.append(row)
        write_history(run_dir / "history.csv", history)
        plot_history(run_dir / "metrics.png", history)
        save_training_checkpoint(latest, model, optimizer, epoch=epoch, step=global_step, history=history)
        if row["overall_validation_score"] < best:
            best = row["overall_validation_score"]
            save_training_checkpoint(
                run_dir / "best_overall.pt", model, optimizer, epoch=epoch, step=global_step, history=history
            )
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val={validation}")


if __name__ == "__main__":
    main()
