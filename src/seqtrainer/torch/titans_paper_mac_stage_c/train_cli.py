"""Resumable Stage C ordered-stream foundation and bounded-pilot trainer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from typing import Sequence

import torch

from seqtrainer.data.bacteria_titan import TokenStreamDataset
from seqtrainer.torch.titans_paper_mac_stage_b import (
    ActivationDType,
    AttentionBackend,
    MemoryBackend,
    StageBBackendConfig,
)

from .checkpoints import load_stage_c_checkpoint, save_stage_c_checkpoint
from .config import MemoryMode, StageCModelConfig
from .evaluation import evaluate_ordered_streams
from .model import StageCPaperMACForCausalLM
from .reporting import write_training_history
from .trainer import StageCTrainer, StreamBatchScheduler
from .study import StudyProtocol


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--memory-mode", choices=[mode.value for mode in MemoryMode], default="adaptive")
    parser.add_argument("--horizon", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--max-valid-bases", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--validation-streams", type=int, default=16)
    parser.add_argument("--validation-segments", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260741)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--activation", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--persistent-tokens", type=int, default=4)
    parser.add_argument("--memory-depth", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify-dataset", action="store_true")
    parser.add_argument("--protocol", type=Path, help="frozen Stage C study protocol")
    parser.add_argument("--run-id", help="protocol run-matrix identifier")
    args = parser.parse_args(argv)
    if args.max_valid_bases is None and args.max_optimizer_steps is None:
        parser.error("set --max-valid-bases or --max-optimizer-steps")
    if bool(args.protocol) != bool(args.run_id):
        parser.error("--protocol and --run-id must be supplied together")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        parser.error("--learning-rate and --gradient-clip-norm must be positive")
    return args


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol:
        protocol = StudyProtocol.from_path(args.protocol)
        requested_budget = args.max_valid_bases
        protocol.validate_run_config(
            args.run_id,
            {
                "memory_mode": args.memory_mode,
                **({"budget_bases": requested_budget} if requested_budget is not None else {}),
                "seed": args.seed,
            },
        )
    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=args.verify_dataset)
    tokenizer = dataset.manifest["tokenizer"]
    activation = ActivationDType(args.activation)
    backend = StageBBackendConfig(
        memory_backend=MemoryBackend.EXACT_ACCELERATED,
        attention_backend=AttentionBackend.SDPA,
        activation_dtype=activation,
    )
    config = StageCModelConfig(
        vocab_size=int(tokenizer["vocab_size"]),
        pad_token_id=int(tokenizer["pad_token_id"]),
        tokenizer_name=str(tokenizer["name"]),
        tokenizer_checksum=str(tokenizer["checksum"]),
        block_count=args.block_count,
        d_model=args.d_model,
        num_heads=args.num_heads,
        persistent_tokens=args.persistent_tokens,
        memory_depth=args.memory_depth,
        gradient_horizon=args.horizon,
        memory_mode=MemoryMode(args.memory_mode),
        backend=backend,
    )
    model = StageCPaperMACForCausalLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_streams = dataset.streams(split="train")
    validation_streams = dataset.streams(split="val")
    train_corpus_bases = sum(
        item.base_count for item in dataset.index if item.split == "train"
    )
    train_predictable_bases = sum(
        item.base_count
        - int(dataset.base_lengths[item.shard_index][item.token_offset])
        for item in dataset.index
        if item.split == "train"
    )
    scheduler = StreamBatchScheduler(
        train_streams,
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=True,
    )
    trainer = StageCTrainer(
        model,
        optimizer,
        device=device,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.run_dir / "LIVE_STATUS.json"
    latest_record: dict[str, object] | None = None

    def write_status(*, state: str, latest: dict[str, object] | None = None, error: BaseException | None = None) -> None:
        payload: dict[str, object] = {
            "format_version": 1,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_bases": trainer.processed_bases,
            "requested_max_valid_bases": args.max_valid_bases,
            "optimizer_steps": trainer.optimizer_step,
            "latest": latest,
        }
        if args.max_valid_bases:
            payload["progress_fraction"] = min(1.0, trainer.processed_bases / args.max_valid_bases)
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
            payload["traceback"] = traceback.format_exc()
        _write_json(progress_path, payload)

    write_status(state="starting")
    latest = args.run_dir / "latest.pt"
    manifest_path = args.dataset_dir / "token_stream_manifest.json"
    dataset_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    commit = _git_commit()
    if latest.exists() and not args.no_resume:
        load_stage_c_checkpoint(
            latest,
            trainer,
            scheduler,
            dataset_fingerprint=dataset_fingerprint,
            trusted=True,
        )

    def on_step(record) -> None:
        nonlocal latest_record
        latest_record = record.to_dict()
        print(json.dumps(record.to_dict(), sort_keys=True), flush=True)
        write_training_history(trainer.history, args.run_dir, write_plots=False)
        write_status(state="running", latest=latest_record)
        if trainer.optimizer_step % args.checkpoint_every == 0:
            save_stage_c_checkpoint(
                latest,
                trainer,
                scheduler,
                dataset_fingerprint=dataset_fingerprint,
                code_commit=commit,
            )

    try:
        trainer.train(
            scheduler,
            max_valid_bases=args.max_valid_bases,
            max_optimizer_steps=args.max_optimizer_steps,
            on_step=on_step,
        )
    except BaseException as error:
        write_status(state="failed", latest=latest_record, error=error)
        raise
    write_training_history(trainer.history, args.run_dir)
    save_stage_c_checkpoint(
        latest,
        trainer,
        scheduler,
        dataset_fingerprint=dataset_fingerprint,
        code_commit=commit,
    )
    max_validation = None if args.validation_streams == 0 else args.validation_streams
    max_validation_segments = None if args.validation_segments == 0 else args.validation_segments
    evaluation = evaluate_ordered_streams(
        model,
        validation_streams,
        device=device,
        memory_mode=args.memory_mode,
        max_streams=max_validation,
        max_segments=max_validation_segments,
    )
    (args.run_dir / "validation.json").write_text(
        json.dumps(evaluation.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest = {
        "format_version": 1,
        "model_config": config.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "code_commit": commit,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameter_count": model.count_parameters(),
        "processed_bases": trainer.processed_bases,
        "train_corpus_bases": train_corpus_bases,
        "train_predictable_bases": train_predictable_bases,
        "observed_corpus_passes": (
            trainer.processed_bases / train_predictable_bases
            if train_predictable_bases
            else 0.0
        ),
        "requested_max_valid_bases": args.max_valid_bases,
        "requested_max_optimizer_steps": args.max_optimizer_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "scheduler_exhausted": scheduler.exhausted,
        "stop_reason": (
            "corpus_exhausted"
            if scheduler.exhausted
            else "valid_base_budget"
            if args.max_valid_bases is not None and trainer.processed_bases >= args.max_valid_bases
            else "optimizer_step_budget"
        ),
        "optimizer_steps": trainer.optimizer_step,
        "validation": evaluation.to_dict(),
    }
    (args.run_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_status(state="completed", latest=latest_record)
    print(json.dumps(run_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
