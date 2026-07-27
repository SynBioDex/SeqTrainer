"""CUDA functional smoke gate for the Stage C production model geometry."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from seqtrainer.data.bacteria_titan import TokenStreamDataset
from seqtrainer.data.bacteria_titan.stage_c_streams import StreamSegment
from seqtrainer.torch.titans_paper_mac_stage_b import (
    ActivationDType,
    AttentionBackend,
    MemoryBackend,
    StageBBackendConfig,
)

from .config import MemoryMode, StageCModelConfig
from .model import BlockStates, StageCPaperMACForCausalLM
from .trainer import StageCTrainer, StreamBatchScheduler


PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-3
CAUSAL_ATOL = 5e-4
CAUSAL_RTOL = 5e-3


def resolve_stage_c_dataset(stage_c_root: Path) -> Path:
    """Resolve and verify the 00b stream directory selected by Notebook 00."""

    selection_path = stage_c_root / "runs" / "c1_tokenizers_cpu" / "tokenizer_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing frozen tokenizer selection: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected_tokenizer")
    if not isinstance(selected, str) or not selected:
        raise ValueError("tokenizer selection must contain a non-empty selected_tokenizer")
    dataset = stage_c_root / "stage_c_dataset" / "ordered_streams" / selected
    manifest = dataset / "token_stream_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "selected Stage C stream dataset is missing; run Notebook 00b first: "
            f"{manifest}"
        )
    return dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require", choices=("T4", "A100"), default="T4")
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
    parser.add_argument("--gradient-horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260744)
    return parser.parse_args(argv)


def _config(
    *,
    tokenizer: Mapping[str, object],
    args: argparse.Namespace,
    activation: ActivationDType,
) -> StageCModelConfig:
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
        gradient_horizon=args.gradient_horizon,
        memory_mode=MemoryMode.ADAPTIVE,
        backend=StageBBackendConfig(
            memory_backend=MemoryBackend.EXACT_ACCELERATED,
            attention_backend=AttentionBackend.SDPA,
            activation_dtype=activation,
        ),
    )
    if args.memory_architecture == "paper_residual_mlp_v2":
        common["memory_depth"] = 2
        return StageCModelConfig.paper_deep(
            recurrence_policy=args.memory_recurrence_policy,
            **common,
        )
    return StageCModelConfig(**common)


def _batch(segment: StreamSegment, device: torch.device) -> dict[str, Tensor]:
    return {
        "input_ids": torch.tensor([segment.input_ids], dtype=torch.long, device=device),
        "labels": torch.tensor([segment.labels], dtype=torch.long, device=device),
        "valid_mask": torch.tensor([segment.valid_mask], dtype=torch.bool, device=device),
        "loss_mask": torch.tensor([segment.loss_mask], dtype=torch.bool, device=device),
        "represented_base_counts": torch.tensor(
            [segment.represented_base_counts], dtype=torch.long, device=device
        ),
    }


def _require_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"non-finite {name}")


def _state_tensors(states: Sequence[BlockStates]) -> list[Tensor]:
    tensors = [
        value
        for row in states
        for state in row
        for mapping in (state.fast_weights, state.surprise)
        for value in mapping.values()
    ]
    tensors.extend(
        history
        for row in states
        for state in row
        for history in (state.query_history, state.write_history)
        if history is not None
    )
    return tensors


def _max_difference(left: Tensor, right: Tensor) -> dict[str, float]:
    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    scale = right.detach().float().cpu().abs().clamp_min(1e-8)
    return {
        "max_absolute": float(difference.max()),
        "max_relative": float((difference / scale).max()),
    }


def _forward_backward(
    model: StageCPaperMACForCausalLM,
    segment: StreamSegment,
    device: torch.device,
) -> tuple[Tensor, Tensor, dict[str, Tensor], list[str]]:
    model.to(device)
    batch = _batch(segment, device)
    output = model.forward_segment(
        (model.initial_states("smoke-parity"),),
        **batch,
    )
    if output.loss is None:
        raise RuntimeError("smoke parity forward did not return a loss")
    _require_finite("parity logits", output.logits)
    _require_finite("parity loss", output.loss)
    output.loss.backward()
    gradients: dict[str, Tensor] = {}
    inactive_parameters: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            # A first-segment loss cannot differentiate through memory writes
            # that are consumed only by a later segment. These parameters are
            # expected to be inactive in this direct attention parity probe.
            inactive_parameters.append(name)
            continue
        _require_finite(f"parity gradient: {name}", parameter.grad)
        gradients[name] = parameter.grad.detach().cpu().clone()
    inactive_attention = [name for name in inactive_parameters if ".attention." in name]
    if inactive_attention:
        raise RuntimeError(f"missing direct attention gradients: {inactive_attention}")
    return output.logits.detach().cpu(), output.loss.detach().cpu(), gradients, inactive_parameters


def _run_training_step(
    config: StageCModelConfig,
    stream_id: str,
    segments: Sequence[StreamSegment],
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = StageCPaperMACForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = StageCTrainer(model, optimizer, device=device)
    scheduler = StreamBatchScheduler(
        {stream_id: tuple(segments)}, batch_size=1, seed=seed, shuffle=False
    )
    history = trainer.train(scheduler, max_optimizer_steps=1)
    if len(history) != 1:
        raise RuntimeError("smoke training did not complete exactly one optimizer step")
    record = history[0]
    for name, parameter in model.named_parameters():
        _require_finite(f"parameter after optimizer step: {name}", parameter)
    for state_index, optimizer_state in enumerate(optimizer.state.values()):
        for name, value in optimizer_state.items():
            if isinstance(value, Tensor):
                _require_finite(f"optimizer state[{state_index}].{name}", value)
    for index, state in enumerate(_state_tensors(tuple(trainer.stream_states.values()))):
        _require_finite(f"state after optimizer step: {index}", state)
    values = {
        "loss_per_token": record.loss_per_token,
        "bits_per_base": record.bits_per_base,
        "gradient_norm": record.gradient_norm,
        "written_state_gradient_norm": record.written_state_gradient_norm,
        "memory_update_norm": record.memory_update_norm,
        "state_drift_norm": record.state_drift_norm,
    }
    for name, value in values.items():
        _require_finite(f"training metric: {name}", torch.tensor(value))
    return {
        "optimizer_steps": trainer.optimizer_step,
        "segments": record.segments,
        "valid_tokens": record.valid_tokens,
        "valid_bases": record.valid_bases,
        "metrics": values,
    }


def _causal_check(
    config: StageCModelConfig,
    segment: StreamSegment,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = StageCPaperMACForCausalLM(config).to(device).eval()
    batch = _batch(segment, device)
    valid_positions = torch.nonzero(batch["valid_mask"][0], as_tuple=False).flatten()
    if valid_positions.numel() < 2:
        raise RuntimeError("smoke segment does not contain enough valid tokens for a causal check")
    split = int(valid_positions[valid_positions.numel() // 2])
    changed = batch["input_ids"].clone()
    replacement = (changed[0, split:].add(1)).remainder(config.vocab_size)
    replacement[replacement == config.pad_token_id] = (config.pad_token_id + 1) % config.vocab_size
    changed[0, split:] = replacement
    with torch.no_grad():
        baseline = model.forward_segment(
            (model.initial_states("smoke-causal"),),
            batch["input_ids"],
            valid_mask=batch["valid_mask"],
            memory_mode=MemoryMode.NONE,
        )
        perturbed = model.forward_segment(
            (model.initial_states("smoke-causal"),),
            changed,
            valid_mask=batch["valid_mask"],
            memory_mode=MemoryMode.NONE,
        )
    earlier = valid_positions[valid_positions < split]
    baseline_earlier = baseline.logits[0, earlier]
    perturbed_earlier = perturbed.logits[0, earlier]
    passed = torch.allclose(
        baseline_earlier, perturbed_earlier, atol=CAUSAL_ATOL, rtol=CAUSAL_RTOL
    )
    result = {
        "passed": bool(passed),
        "checked_positions": int(earlier.numel()),
        "split_token_index": split,
        **_max_difference(baseline_earlier, perturbed_earlier),
    }
    if not passed:
        raise RuntimeError(f"causal mask smoke check failed: {result}")
    return result


def _select_smoke_stream(dataset: TokenStreamDataset, horizon: int) -> tuple[str, tuple[StreamSegment, ...]]:
    for stream_id, segments in sorted(dataset.streams(split="train").items()):
        # Leave a live stream state after the horizon-sized optimizer step so
        # the finite-state assertion is substantive rather than vacuous.
        if len(segments) > horizon:
            return stream_id, tuple(segments)
    raise RuntimeError(f"dataset has no train stream with more than {horizon} segments")


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage C GPU smoke requires CUDA")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    if args.require.upper() not in device_name.upper():
        raise RuntimeError(f"required {args.require}, found {device_name}")
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    stream_id, stream = _select_smoke_stream(dataset, args.gradient_horizon)
    segment = stream[0]
    tokenizer = dataset.manifest["tokenizer"]
    if not isinstance(tokenizer, Mapping):
        raise ValueError("dataset manifest is missing tokenizer metadata")
    fp32_config = _config(tokenizer=tokenizer, args=args, activation=ActivationDType.FP32)
    fp16_config = _config(tokenizer=tokenizer, args=args, activation=ActivationDType.FP16)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    torch.manual_seed(args.seed)
    cpu_model = StageCPaperMACForCausalLM(fp32_config)
    gpu_model = copy.deepcopy(cpu_model)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        cpu_logits, cpu_loss, cpu_gradients, cpu_inactive = _forward_backward(
            cpu_model, segment, torch.device("cpu")
        )
        gpu_logits, gpu_loss, gpu_gradients, gpu_inactive = _forward_backward(
            gpu_model, segment, device
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    logits_match = torch.allclose(cpu_logits, gpu_logits, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    loss_match = torch.allclose(cpu_loss, gpu_loss, atol=PARITY_ATOL, rtol=PARITY_RTOL)
    if set(cpu_gradients) != set(gpu_gradients):
        raise RuntimeError("CPU/GPU FP32 parity has different active gradient parameter sets")
    if cpu_inactive != gpu_inactive:
        raise RuntimeError("CPU/GPU FP32 parity has different inactive gradient parameter sets")
    gradient_differences = {
        name: _max_difference(cpu_gradients[name], gpu_gradients[name])
        for name in cpu_gradients
    }
    gradients_match = all(
        torch.allclose(cpu_gradients[name], gpu_gradients[name], atol=PARITY_ATOL, rtol=PARITY_RTOL)
        for name in cpu_gradients
    )
    parity = {
        "passed": bool(logits_match and loss_match and gradients_match),
        "tf32_disabled": True,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "logits": _max_difference(cpu_logits, gpu_logits),
        "loss": _max_difference(cpu_loss, gpu_loss),
        "gradients": gradient_differences,
        "inactive_parameters": cpu_inactive,
    }
    if not parity["passed"]:
        raise RuntimeError(f"CPU/GPU FP32 parity smoke check failed: {parity}")

    fp32_training = _run_training_step(
        fp32_config, stream_id, stream, device, args.seed
    )
    fp16_training = _run_training_step(
        fp16_config, stream_id, stream, device, args.seed + 1
    )
    causal = _causal_check(fp16_config, segment, device, args.seed + 2)
    torch.cuda.synchronize(device)
    return {
        "format_version": 1,
        "classification": "stage_c_gpu_smoke",
        "passed": True,
        "hardware": {
            "required": args.require,
            "device_name": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "geometry": {
            "block_count": args.block_count,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "persistent_tokens": args.persistent_tokens,
            "memory_depth": args.memory_depth,
            "gradient_horizon": args.gradient_horizon,
            "activation_dtypes": ["fp32", "float16"],
        },
        "dataset": {
            "path": str(args.dataset_dir),
            "smoke_stream_id": stream_id,
            "tokenizer": dict(tokenizer),
        },
        "fp32_cpu_gpu_parity": parity,
        "fp32_training": fp32_training,
        "fp16_training": fp16_training,
        "fp16_causal_mask": causal,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_smoke(args)
    except Exception as error:
        payload = {
            "format_version": 1,
            "classification": "stage_c_gpu_smoke",
            "passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
