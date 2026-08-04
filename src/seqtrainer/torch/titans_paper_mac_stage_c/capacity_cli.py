"""T4/A100 Stage C horizon, precision, checkpoint, and capacity matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
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

from .checkpoints import load_stage_c_checkpoint, save_stage_c_checkpoint
from .config import MemoryMode, StageCModelConfig
from .evaluation import evaluate_ordered_streams
from .model import StageCPaperMACForCausalLM
from .reporting import bar_svg
from .trainer import StageCTrainer, StreamBatchScheduler


VARIANTS = {
    "reference_fp32": (MemoryMode.REFERENCE, ActivationDType.FP32),
    "exact_sdpa_fp32": (MemoryMode.ADAPTIVE, ActivationDType.FP32),
    "exact_sdpa_float16": (MemoryMode.ADAPTIVE, ActivationDType.FP16),
    "exact_sdpa_bfloat16": (MemoryMode.ADAPTIVE, ActivationDType.BF16),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--validation-panel-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require", choices=("T4", "A100"), required=True)
    parser.add_argument("--horizons", type=int, nargs="+", choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=("exact_sdpa_fp32",))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
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
    parser.add_argument(
        "--memory-recurrence-policy",
        choices=("paper_exact", "stabilized_rms_v1"),
        default="stabilized_rms_v1",
    )
    parser.add_argument("--seed", type=int, default=20260743)
    parser.add_argument("--validation-segments", type=int, default=32)
    parser.add_argument(
        "--append",
        action="store_true",
        help="merge completed probes into an existing capacity_matrix.json",
    )
    return parser.parse_args(argv)


def _result_key(result: dict[str, object]) -> tuple[int, str]:
    return int(result["horizon"]), str(result["variant"])


def _write_capacity_artifacts(
    *,
    args: argparse.Namespace,
    device_name: str,
    train_predictable_bases: int,
    results: list[dict[str, object]],
) -> dict[str, object]:
    """Persist completed probes so a later Colab interruption loses no evidence."""

    payload = {
        "format_version": 1,
        "classification": "stage_c_capacity_matrix",
        "hardware": {
            "required": args.require,
            "device_name": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "geometry": {
            "block_count": args.block_count,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "persistent_tokens": args.persistent_tokens,
            "memory_depth": args.memory_depth,
            "memory_architecture": args.memory_architecture,
            "memory_recurrence_policy": args.memory_recurrence_policy,
            "batch_size": args.batch_size,
        },
        "train_predictable_bases": train_predictable_bases,
        "results": results,
    }
    json_path = args.output_dir / "capacity_matrix.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    available_results = [result for result in results if result["available"]]
    labels = [f"{result['variant']} h{result['horizon']}" for result in available_results]
    (args.output_dir / "capacity_throughput.svg").write_text(
        bar_svg(
            labels,
            [float(result["bases_per_second"]) for result in available_results],
            title=f"Stage C throughput on {device_name}",
            x_label="Bases per second",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "capacity_memory.svg").write_text(
        bar_svg(
            labels,
            [float(result["peak_allocated_bytes"]) / 1_000_000_000 for result in available_results],
            title=f"Stage C peak allocated memory on {device_name}",
            x_label="GB",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "capacity_validation_bpb.svg").write_text(
        bar_svg(
            labels,
            [float(result["validation_bpb"]) for result in available_results],
            title="Matched bounded validation by horizon/backend",
            x_label="Bits per base (lower is better)",
        ),
        encoding="utf-8",
    )
    lines = [
        "# Stage C capacity matrix",
        "",
        f"Hardware: `{device_name}`",
        "",
        "| Horizon | Variant | Available | Validation BPB | Bases/s | Peak allocated | State/stream | Pass hours | Written-state grad |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['horizon']} | {result['variant']} | {result['available']} | "
            f"{float(result.get('validation_bpb', 0)):.4f} | "
            f"{float(result.get('bases_per_second', 0)):.2f} | "
            f"{int(result.get('peak_allocated_bytes', 0))} | "
            f"{int(result.get('functional_state_bytes_per_stream', 0))} | "
            f"{float(result.get('projected_one_train_pass_hours', 0)):.2f} | "
            f"{float(result.get('written_state_gradient_norm', 0)):.6f} |"
        )
    (args.output_dir / "capacity_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage C capacity evidence requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    if args.require.upper() not in device_name.upper():
        raise RuntimeError(f"required {args.require}, found {device_name}")
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    panel = (
        StageCPanelManifest.from_path(args.panel_manifest)
        if args.panel_manifest
        else None
    )
    validation_panel = (
        StageCPanelManifest.from_path(args.validation_panel_manifest)
        if args.validation_panel_manifest
        else None
    )
    for selected, split in ((panel, "train"), (validation_panel, "val")):
        if selected:
            validate_panel_against_dataset(selected, dataset)
            if selected.payload["split"] != split:
                raise ValueError(f"{split} panel has the wrong split")
    streams = dataset.streams(
        split="train",
        stream_ids=panel.stream_ids if panel else None,
    )
    validation_streams = dataset.streams(
        split="val",
        stream_ids=validation_panel.stream_ids if validation_panel else None,
    )
    train_predictable_bases = (
        int(panel.payload["predictable_bases"])
        if panel
        else sum(
            item.base_count
            - int(dataset.base_lengths[item.shard_index][item.token_offset])
            for item in dataset.index
            if item.split == "train"
        )
    )
    tokenizer = dataset.manifest["tokenizer"]
    dataset_fingerprint = hashlib.sha256(
        (args.dataset_dir / "token_stream_manifest.json").read_bytes()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_results: dict[tuple[int, str], dict[str, object]] = {}
    prior_path = args.output_dir / "capacity_matrix.json"
    if args.append and prior_path.is_file():
        prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        for item in prior_payload.get("results", []):
            if isinstance(item, dict) and {"horizon", "variant", "available"} <= item.keys():
                prior_results[_result_key(item)] = item
    results: list[dict[str, object]] = list(prior_results.values())
    for horizon in args.horizons:
        for variant in args.variants:
            mode, activation = VARIANTS[variant]
            print(
                json.dumps(
                    {
                        "event": "capacity_variant_started",
                        "horizon": horizon,
                        "variant": variant,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.manual_seed(args.seed)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            result: dict[str, object] = {
                "horizon": horizon,
                "variant": variant,
                "available": False,
            }
            model = None
            optimizer = None
            scheduler = None
            trainer = None
            restored_model = None
            restored_optimizer = None
            restored_scheduler = None
            restored_trainer = None
            history = ()
            validation = None
            try:
                common = dict(
                    vocab_size=int(tokenizer["vocab_size"]),
                    pad_token_id=int(tokenizer["pad_token_id"]),
                    tokenizer_name=str(tokenizer["name"]),
                    tokenizer_checksum=str(tokenizer["checksum"]),
                    block_count=args.block_count,
                    d_model=args.d_model,
                    num_heads=args.num_heads,
                    persistent_tokens=args.persistent_tokens,
                    memory_depth=args.memory_depth,
                    gradient_horizon=horizon,
                    memory_mode=mode,
                    backend=StageBBackendConfig(
                        memory_backend=MemoryBackend.EXACT_ACCELERATED,
                        attention_backend=AttentionBackend.SDPA,
                        activation_dtype=activation,
                    ),
                )
                config = (
                    StageCModelConfig.paper_deep(
                        recurrence_policy=args.memory_recurrence_policy,
                        **{**common, "memory_depth": 2},
                    )
                    if args.memory_architecture == "paper_residual_mlp_v2"
                    else StageCModelConfig(**common)
                )
                model = StageCPaperMACForCausalLM(config)
                initial_memory_state_dtypes = sorted(
                    {
                        str(value.dtype).removeprefix("torch.")
                        for state in model.initial_states("dtype-probe")
                        for value in state.fast_weights.values()
                    }
                )
                initial_state = model.initial_states("size-probe")
                functional_state_bytes_per_stream = sum(
                    value.numel() * value.element_size()
                    for state in initial_state
                    for mapping in (state.fast_weights, state.surprise)
                    for value in mapping.values()
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                scheduler = StreamBatchScheduler(
                    streams,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    shuffle=True,
                )
                trainer = StageCTrainer(model, optimizer, device=device)
                started = time.perf_counter()
                history = trainer.train(
                    scheduler,
                    max_optimizer_steps=args.steps,
                    on_step=lambda record: print(
                        json.dumps(
                            {
                                "event": "capacity_training_step",
                                "horizon": horizon,
                                "variant": variant,
                                "optimizer_step": record.optimizer_step,
                                "loss_per_token": record.loss_per_token,
                                "gradient_norm": record.gradient_norm,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    ),
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                checkpoint = args.output_dir / f"{variant}_h{horizon}.pt"
                save_started = time.perf_counter()
                save_stage_c_checkpoint(
                    checkpoint,
                    trainer,
                    scheduler,
                    dataset_fingerprint=dataset_fingerprint,
                    code_commit="capacity-matrix",
                )
                save_seconds = time.perf_counter() - save_started
                restored_model = StageCPaperMACForCausalLM(config)
                restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=3e-4)
                restored_scheduler = StreamBatchScheduler(
                    streams,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    shuffle=True,
                )
                restored_trainer = StageCTrainer(restored_model, restored_optimizer, device=device)
                load_started = time.perf_counter()
                load_stage_c_checkpoint(
                    checkpoint,
                    restored_trainer,
                    restored_scheduler,
                    dataset_fingerprint=dataset_fingerprint,
                    trusted=True,
                )
                load_seconds = time.perf_counter() - load_started
                valid_bases = sum(record.valid_bases for record in history)
                print(
                    json.dumps(
                        {
                            "event": "capacity_validation_started",
                            "horizon": horizon,
                            "variant": variant,
                            "segments": args.validation_segments,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                validation = evaluate_ordered_streams(
                    model,
                    validation_streams,
                    device=device,
                    memory_mode=mode,
                    max_segments=(
                        None if args.validation_segments == 0 else args.validation_segments
                    ),
                )
                print(
                    json.dumps(
                        {
                            "event": "capacity_validation_finished",
                            "horizon": horizon,
                            "variant": variant,
                            "bits_per_base": validation.bits_per_base,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                result.update(
                    {
                        "available": True,
                        "parameter_count": model.count_parameters(),
                        "optimizer_steps": trainer.optimizer_step,
                        "valid_bases": valid_bases,
                        "elapsed_seconds": elapsed,
                        "bases_per_second": valid_bases / elapsed,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "checkpoint_save_seconds": save_seconds,
                        "checkpoint_load_seconds": load_seconds,
                        "functional_state_bytes_per_stream": functional_state_bytes_per_stream,
                        "projected_one_train_pass_hours": (
                            train_predictable_bases / (valid_bases / elapsed) / 3600
                        ),
                        "finite": all(
                            torch.isfinite(torch.tensor(record.loss_per_token))
                            and torch.isfinite(torch.tensor(record.gradient_norm))
                            for record in history
                        ),
                        "written_state_gradient_norm": max(
                            (record.written_state_gradient_norm for record in history), default=0.0
                        ),
                        "memory_state_dtypes": initial_memory_state_dtypes,
                        "validation": validation.to_dict(),
                        "validation_bpb": validation.bits_per_base,
                    }
                )
            except (RuntimeError, ValueError) as error:
                result["reason"] = str(error)
            prior_results[_result_key(result)] = result
            results = [prior_results[key] for key in sorted(prior_results)]
            print(
                json.dumps(
                    {
                        "event": "capacity_variant_finished",
                        "horizon": horizon,
                        "variant": variant,
                        "available": result["available"],
                        "reason": result.get("reason"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            # Release each full model/checkpoint graph before the next variant.
            # This does not make the intentionally unaccelerated reference
            # recurrence T4-capacity eligible, but avoids retaining completed
            # exact-SDPA variant allocations in one Colab process.
            model = optimizer = scheduler = trainer = None
            restored_model = restored_optimizer = restored_scheduler = restored_trainer = None
            history = ()
            validation = None
            gc.collect()
            torch.cuda.empty_cache()
            _write_capacity_artifacts(
                args=args,
                device_name=device_name,
                train_predictable_bases=train_predictable_bases,
                results=results,
            )
    payload = _write_capacity_artifacts(
        args=args,
        device_name=device_name,
        train_predictable_bases=train_predictable_bases,
        results=results,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
