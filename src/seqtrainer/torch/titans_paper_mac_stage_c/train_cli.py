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

from seqtrainer.data.bacteria_titan import (
    StageCPanelManifest,
    TokenStreamDataset,
    validate_panel_against_dataset,
)
from seqtrainer.torch.titans_paper_mac_stage_b import (
    ActivationDType,
    AttentionBackend,
    MemoryBackend,
    StageBBackendConfig,
)

from .checkpoints import (
    load_stage_c_checkpoint,
    save_stage_c_checkpoint,
    warm_start_stage_c_checkpoint,
)
from .config import MemoryMode, StageCModelConfig
from .evaluation import evaluate_ordered_streams
from .model import StageCPaperMACForCausalLM
from .reporting import write_training_history
from .trainer import (
    BaseCosineLRSchedule,
    StageCTrainer,
    StatefulRotationScheduler,
    StreamBatchScheduler,
)
from .study import StudyProtocol, sha256_file


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


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
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--validation-panel-manifest", type=Path)
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
    parser.add_argument(
        "--scheduler-policy",
        choices=("stream_to_completion", "stateful_rotation"),
        default="stream_to_completion",
    )
    parser.add_argument("--scheduler-burst-segments", type=int, default=96)
    parser.add_argument("--require-panel-completion", action="store_true")
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
    parser.add_argument(
        "--memory-architecture",
        choices=("legacy_mlp_v1", "paper_residual_mlp_v2"),
        default="legacy_mlp_v1",
    )
    parser.add_argument("--memory-expansion-factor", type=int, default=4)
    parser.add_argument("--memory-projection-convolution-kernel", type=int)
    parser.add_argument(
        "--memory-normalize-queries-and-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--memory-gate-granularity",
        choices=("shared_scalar", "per_layer_channel"),
        default="shared_scalar",
    )
    parser.add_argument(
        "--memory-recurrence-policy",
        choices=("legacy_configured", "paper_exact", "stabilized_rms_v1"),
        default="legacy_configured",
    )
    parser.add_argument("--memory-surprise-clip-norm", type=_optional_positive_float, default=4.0)
    parser.add_argument(
        "--memory-associative-loss-reduction",
        choices=("sum", "mean"),
        default="sum",
    )
    parser.add_argument("--memory-max-gradient-rms", type=_optional_positive_float)
    parser.add_argument("--memory-max-gradient-rms-ratio", type=_optional_positive_float)
    parser.add_argument("--memory-theta-max", type=float, default=1.0)
    parser.add_argument("--memory-theta-initial", type=float)
    parser.add_argument("--memory-alpha-initial", type=float)
    parser.add_argument("--memory-eta-initial", type=float)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--warm-start-checkpoint", type=Path)
    parser.add_argument("--lr-warmup-bases", type=int, default=0)
    parser.add_argument("--lr-decay-bases", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=3e-6)
    parser.add_argument("--verify-dataset", action="store_true")
    parser.add_argument("--protocol", type=Path, help="frozen Stage C study protocol")
    parser.add_argument(
        "--protocol-amendment",
        type=Path,
        action="append",
        default=[],
        help="source-controlled additive run-matrix amendment linked to --protocol",
    )
    parser.add_argument("--run-id", help="protocol run-matrix identifier")
    args = parser.parse_args(argv)
    if (
        args.max_valid_bases is None
        and args.max_optimizer_steps is None
        and not args.require_panel_completion
    ):
        parser.error(
            "set --max-valid-bases, --max-optimizer-steps, or --require-panel-completion"
        )
    if args.require_panel_completion and (
        args.max_valid_bases is not None or args.max_optimizer_steps is not None
    ):
        parser.error("--require-panel-completion cannot be combined with a stop cap")
    if args.require_panel_completion and args.panel_manifest is None:
        parser.error("--require-panel-completion requires --panel-manifest")
    if args.scheduler_burst_segments <= 0:
        parser.error("--scheduler-burst-segments must be positive")
    if args.warm_start_checkpoint and not args.no_resume:
        parser.error("--warm-start-checkpoint requires --no-resume")
    if args.lr_decay_bases:
        if args.lr_decay_bases <= args.lr_warmup_bases:
            parser.error("--lr-decay-bases must exceed --lr-warmup-bases")
        if not 0 < args.min_learning_rate <= args.learning_rate:
            parser.error("--min-learning-rate must be in (0, --learning-rate]")
    elif args.lr_warmup_bases:
        parser.error("--lr-warmup-bases requires --lr-decay-bases")
    if bool(args.protocol) != bool(args.run_id):
        parser.error("--protocol and --run-id must be supplied together")
    if args.protocol_amendment and not args.protocol:
        parser.error("--protocol-amendment requires --protocol and --run-id")
    if args.learning_rate <= 0 or args.gradient_clip_norm <= 0:
        parser.error("--learning-rate and --gradient-clip-norm must be positive")
    if not 0.0 < args.memory_theta_max <= 1.0:
        parser.error("--memory-theta-max must be in (0, 1]")
    if args.memory_theta_initial is not None and not (
        0.0 < args.memory_theta_initial < args.memory_theta_max
    ):
        parser.error("--memory-theta-initial must be in (0, --memory-theta-max)")
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
                **(
                    {"max_optimizer_steps": args.max_optimizer_steps}
                    if args.max_optimizer_steps is not None
                    else {}
                ),
                "seed": args.seed,
                "block_count": args.block_count,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "gradient_horizon": args.horizon,
                "memory_depth": args.memory_depth,
                "memory_architecture": args.memory_architecture,
                "memory_recurrence_policy": args.memory_recurrence_policy,
                "scheduler_policy": args.scheduler_policy,
                "scheduler_burst_segments": args.scheduler_burst_segments,
                "require_panel_completion": args.require_panel_completion,
                "lr_warmup_bases": args.lr_warmup_bases,
                "lr_decay_bases": args.lr_decay_bases,
                "minimum_learning_rate": args.min_learning_rate,
            },
            amendment_paths=args.protocol_amendment,
        )
    torch.manual_seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=args.verify_dataset)
    train_panel = (
        StageCPanelManifest.from_path(args.panel_manifest)
        if args.panel_manifest
        else None
    )
    validation_panel = (
        StageCPanelManifest.from_path(args.validation_panel_manifest)
        if args.validation_panel_manifest
        else None
    )
    if train_panel:
        validate_panel_against_dataset(train_panel, dataset)
        if train_panel.payload["split"] != "train":
            raise ValueError("training panel must select the train split")
    if validation_panel:
        validate_panel_against_dataset(validation_panel, dataset)
        if validation_panel.payload["split"] != "val":
            raise ValueError("validation panel must select the val split")
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
        memory_architecture=args.memory_architecture,
        memory_expansion_factor=args.memory_expansion_factor,
        memory_projection_convolution_kernel=args.memory_projection_convolution_kernel,
        memory_normalize_queries_and_keys=args.memory_normalize_queries_and_keys,
        memory_gate_granularity=args.memory_gate_granularity,
        memory_recurrence_policy=args.memory_recurrence_policy,
        memory_surprise_clip_norm=args.memory_surprise_clip_norm,
        memory_associative_loss_reduction=args.memory_associative_loss_reduction,
        memory_max_gradient_rms=args.memory_max_gradient_rms,
        memory_max_gradient_rms_ratio=args.memory_max_gradient_rms_ratio,
        memory_theta_max=args.memory_theta_max,
        memory_theta_initial=args.memory_theta_initial,
        memory_alpha_initial=args.memory_alpha_initial,
        memory_eta_initial=args.memory_eta_initial,
        gradient_horizon=args.horizon,
        memory_mode=MemoryMode(args.memory_mode),
        backend=backend,
        format_version=2 if args.memory_architecture == "paper_residual_mlp_v2" else 1,
    )
    model = StageCPaperMACForCausalLM(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_streams = dataset.streams(
        split="train",
        stream_ids=train_panel.stream_ids if train_panel else None,
    )
    validation_streams = dataset.streams(
        split="val",
        stream_ids=validation_panel.stream_ids if validation_panel else None,
    )
    selected_train_ids = set(train_streams)
    train_corpus_bases = sum(
        item.base_count
        for item in dataset.index
        if item.split == "train" and item.stream_id in selected_train_ids
    )
    train_predictable_bases = sum(
        item.base_count
        - int(dataset.base_lengths[item.shard_index][item.token_offset])
        for item in dataset.index
        if item.split == "train" and item.stream_id in selected_train_ids
    )
    if args.scheduler_policy == "stateful_rotation":
        scheduler = StatefulRotationScheduler(
            train_streams,
            batch_size=args.batch_size,
            burst_segments=args.scheduler_burst_segments,
            seed=args.seed,
            shuffle=True,
        )
    else:
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
    starting_bases = 0
    starting_steps = 0

    def write_status(*, state: str, latest: dict[str, object] | None = None, error: BaseException | None = None) -> None:
        payload: dict[str, object] = {
            "format_version": 1,
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_bases": trainer.processed_bases,
            "run_processed_bases": trainer.processed_bases - starting_bases,
            "requested_max_valid_bases": args.max_valid_bases,
            "optimizer_steps": trainer.optimizer_step,
            "latest": latest,
        }
        if args.require_panel_completion:
            payload["progress_fraction"] = min(
                1.0,
                (trainer.processed_bases - starting_bases)
                / max(train_predictable_bases, 1),
            )
        elif args.max_valid_bases:
            payload["progress_fraction"] = min(1.0, trainer.processed_bases / args.max_valid_bases)
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
            payload["traceback"] = traceback.format_exc()
        _write_json(progress_path, payload)

    latest = args.run_dir / "latest.pt"
    manifest_path = args.dataset_dir / "token_stream_manifest.json"
    fingerprint_payload = {
        "dataset": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_panel": train_panel.hash if train_panel else None,
        "validation_panel": validation_panel.hash if validation_panel else None,
    }
    dataset_fingerprint = (
        hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if train_panel or validation_panel
        else str(fingerprint_payload["dataset"])
    )
    commit = _git_commit()
    warm_start_payload: dict[str, object] | None = None
    if latest.exists() and not args.no_resume:
        load_stage_c_checkpoint(
            latest,
            trainer,
            scheduler,
            dataset_fingerprint=dataset_fingerprint,
            trusted=True,
        )
    elif args.warm_start_checkpoint:
        warm_start_payload = warm_start_stage_c_checkpoint(
            args.warm_start_checkpoint,
            trainer,
            trusted=True,
        )
    run_start_path = args.run_dir / "RUN_START.json"
    if run_start_path.exists() and warm_start_payload is None:
        run_start = json.loads(run_start_path.read_text(encoding="utf-8"))
        starting_bases = int(run_start["starting_processed_bases"])
        starting_steps = int(run_start["starting_optimizer_steps"])
    else:
        starting_bases = trainer.processed_bases
        starting_steps = trainer.optimizer_step
        _write_json(
            run_start_path,
            {
                "format_version": 1,
                "starting_processed_bases": starting_bases,
                "starting_optimizer_steps": starting_steps,
                "warm_start_checkpoint": (
                    str(args.warm_start_checkpoint)
                    if args.warm_start_checkpoint
                    else None
                ),
            },
        )
    learning_rate_schedule = (
        BaseCosineLRSchedule(
            peak_lr=args.learning_rate,
            minimum_lr=args.min_learning_rate,
            warmup_bases=args.lr_warmup_bases,
            decay_bases=args.lr_decay_bases,
        )
        if args.lr_decay_bases
        else None
    )
    write_status(state="starting")

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
                dataset_components=fingerprint_payload,
            )

    try:
        trainer.train(
            scheduler,
            max_valid_bases=args.max_valid_bases,
            max_optimizer_steps=args.max_optimizer_steps,
            on_step=on_step,
            learning_rate_schedule=learning_rate_schedule,
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
        dataset_components=fingerprint_payload,
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
        "dataset_components": fingerprint_payload,
        "train_panel": (
            {
                "path": str(args.panel_manifest),
                "sha256": sha256_file(args.panel_manifest),
                "panel_hash": train_panel.hash,
                "panel_id": train_panel.payload["panel_id"],
            }
            if train_panel
            else None
        ),
        "validation_panel": (
            {
                "path": str(args.validation_panel_manifest),
                "sha256": sha256_file(args.validation_panel_manifest),
                "panel_hash": validation_panel.hash,
                "panel_id": validation_panel.payload["panel_id"],
            }
            if validation_panel
            else None
        ),
        "code_commit": commit,
        "protocol": {
            "path": str(args.protocol) if args.protocol else None,
            "run_id": args.run_id,
            "amendments": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in args.protocol_amendment
            ],
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "parameter_count": model.count_parameters(),
        "processed_bases": trainer.processed_bases,
        "run_processed_bases": trainer.processed_bases - starting_bases,
        "starting_processed_bases": starting_bases,
        "starting_optimizer_steps": starting_steps,
        "train_corpus_bases": train_corpus_bases,
        "train_predictable_bases": train_predictable_bases,
        "observed_corpus_passes": (
            (trainer.processed_bases - starting_bases) / train_predictable_bases
            if train_predictable_bases
            else 0.0
        ),
        "requested_max_valid_bases": args.max_valid_bases,
        "requested_max_optimizer_steps": args.max_optimizer_steps,
        "require_panel_completion": args.require_panel_completion,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.min_learning_rate,
        "lr_warmup_bases": args.lr_warmup_bases,
        "lr_decay_bases": args.lr_decay_bases,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "memory_surprise_clip_norm": args.memory_surprise_clip_norm,
        "memory_associative_loss_reduction": args.memory_associative_loss_reduction,
        "memory_max_gradient_rms": args.memory_max_gradient_rms,
        "memory_max_gradient_rms_ratio": args.memory_max_gradient_rms_ratio,
        "memory_theta_max": args.memory_theta_max,
        "memory_theta_initial": args.memory_theta_initial,
        "scheduler_exhausted": scheduler.exhausted,
        "scheduler_policy": args.scheduler_policy,
        "scheduler_burst_segments": args.scheduler_burst_segments,
        "warm_start": (
            {
                "path": str(args.warm_start_checkpoint),
                "sha256": sha256_file(args.warm_start_checkpoint),
                "parent_dataset_fingerprint": warm_start_payload.get(
                    "dataset_fingerprint"
                ),
                "parent_code_commit": warm_start_payload.get("code_commit"),
            }
            if args.warm_start_checkpoint and warm_start_payload
            else None
        ),
        "stop_reason": (
            "panel_exhausted"
            if scheduler.exhausted and args.require_panel_completion
            else "corpus_exhausted"
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
