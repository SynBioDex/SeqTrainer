"""Repeated two-segment outer-training measurements for Stage B backends."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ActivationDType, AttentionBackend, MemoryBackend, StageBBackendConfig
from .long_context_benchmark import _configure_stable_stress_gates
from .stack import StageBMACStack, StageBStackOutput


@dataclass(frozen=True)
class TrainingStepScale:
    name: str
    block_count: int
    d_model: int
    num_heads: int
    persistent_tokens: int = 4
    memory_depth: int = 1


A100_TRAINING_STEP_SCALE = TrainingStepScale(
    "a100_pilot",
    block_count=8,
    d_model=384,
    num_heads=8,
)


TRAINING_VARIANTS = (
    "reference_fp32",
    "exact_fp32",
    "exact_sdpa_fp32",
    "exact_sdpa_bfloat16",
    "exact_sdpa_float16",
)


def _training_config(name: str) -> StageBBackendConfig:
    if name == "reference_fp32":
        return StageBBackendConfig()
    if name == "exact_fp32":
        return StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED)
    if name == "exact_sdpa_fp32":
        return StageBBackendConfig(
            memory_backend=MemoryBackend.EXACT_ACCELERATED,
            attention_backend=AttentionBackend.SDPA,
        )
    if name == "exact_sdpa_bfloat16":
        return StageBBackendConfig(
            memory_backend=MemoryBackend.EXACT_ACCELERATED,
            attention_backend=AttentionBackend.SDPA,
            activation_dtype=ActivationDType.BF16,
        )
    if name == "exact_sdpa_float16":
        return StageBBackendConfig(
            memory_backend=MemoryBackend.EXACT_ACCELERATED,
            attention_backend=AttentionBackend.SDPA,
            activation_dtype=ActivationDType.FP16,
        )
    raise ValueError(f"unknown training-step variant: {name}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_dtypes(output: StageBStackOutput) -> list[str]:
    return sorted(
        {
            str(value.dtype).removeprefix("torch.")
            for state in output.states
            for value in (*state.fast_weights.values(), *state.surprise.values())
        }
    )


def _finite_output(output: StageBStackOutput) -> bool:
    return bool(torch.isfinite(output.sequence).all()) and all(
        bool(torch.isfinite(value).all())
        for state in output.states
        for value in (*state.fast_weights.values(), *state.surprise.values())
    )


def _gradient_metrics(stack: StageBMACStack) -> dict[str, float | int | bool]:
    gradients = [
        parameter.grad.detach().float().reshape(-1)
        for parameter in stack.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return {"parameter_tensors": 0, "global_norm": 0.0, "finite": False}
    flattened = torch.cat(gradients)
    return {
        "parameter_tensors": len(gradients),
        "global_norm": float(flattened.norm().item()),
        "finite": bool(torch.isfinite(flattened).all()),
    }


def _step(
    stack: StageBMACStack,
    optimizer: torch.optim.Optimizer,
    segments: Sequence[Tensor],
    target: Tensor,
    config: StageBBackendConfig,
    stream_id: str,
) -> tuple[float, dict[str, float | int | bool], StageBStackOutput]:
    optimizer.zero_grad(set_to_none=True)
    states = stack.initial_states(stream_id)
    first = stack(states, segments[0], config=config)
    written_state_tensors = [
        value
        for state in first.states
        for value in state.fast_weights.values()
    ]
    for value in written_state_tensors:
        value.retain_grad()
    second = stack(first.states, segments[1], config=config)
    # The loss is on segment two, whose retrieval reads the differentiable
    # state written by segment one. Backward therefore traverses the memory
    # update rather than measuring only the same-segment causal core.
    loss = F.mse_loss(second.sequence, target)
    loss.backward()
    gradients = _gradient_metrics(stack)
    written_state_gradients = [
        value.grad.detach().float().reshape(-1)
        for value in written_state_tensors
        if value.grad is not None
    ]
    if written_state_gradients:
        flattened_state_gradient = torch.cat(written_state_gradients)
        gradients.update(
            {
                "written_state_tensors": len(written_state_gradients),
                "written_state_gradient_norm": float(
                    flattened_state_gradient.norm().item()
                ),
                "written_state_gradient_finite": bool(
                    torch.isfinite(flattened_state_gradient).all()
                ),
            }
        )
    else:
        gradients.update(
            {
                "written_state_tensors": 0,
                "written_state_gradient_norm": 0.0,
                "written_state_gradient_finite": False,
            }
        )
    optimizer.step()
    return float(loss.detach().item()), gradients, second


def _unavailable_result(
    name: str,
    config: StageBBackendConfig,
    reason: str,
) -> dict[str, object]:
    return {
        "variant": name,
        "config": config.to_dict(),
        "available": False,
        "reason": reason,
        "classification": (
            "mixed_precision_behavioral"
            if config.activation_dtype is not ActivationDType.FP32
            else "exact_fp32"
        ),
    }


def run_training_step_matrix(
    *,
    scale: TrainingStepScale = A100_TRAINING_STEP_SCALE,
    variants: Sequence[str] = TRAINING_VARIANTS,
    seed: int = 20260740,
    warmup_runs: int = 1,
    repetitions: int = 3,
    learning_rate: float = 1e-4,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, object]:
    """Measure complete two-segment forward/backward/optimizer steps."""

    if warmup_runs < 0 or repetitions <= 0:
        raise ValueError("warmup_runs must be nonnegative and repetitions positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    template = StageBMACStack(
        scale.block_count,
        scale.d_model,
        num_heads=scale.num_heads,
        persistent_tokens=scale.persistent_tokens,
        memory_depth=scale.memory_depth,
    ).to(device=selected_device, dtype=torch.float32)
    _configure_stable_stress_gates(template)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    segments = tuple(
        torch.randn(32, scale.d_model, generator=generator).to(selected_device)
        for _ in range(2)
    )
    target = torch.randn(32, scale.d_model, generator=generator).to(selected_device)
    results: list[dict[str, object]] = []
    for name in variants:
        config = _training_config(name)
        stack = copy.deepcopy(template)
        optimizer = torch.optim.AdamW(stack.parameters(), lr=learning_rate)
        try:
            for warmup in range(warmup_runs):
                warmup_stack = copy.deepcopy(template)
                warmup_optimizer = torch.optim.AdamW(
                    warmup_stack.parameters(), lr=learning_rate
                )
                _step(
                    warmup_stack,
                    warmup_optimizer,
                    segments,
                    target,
                    config,
                    f"warmup-{name}-{warmup}",
                )
                del warmup_stack, warmup_optimizer
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
            if selected_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected_device)
            samples: list[float] = []
            losses: list[float] = []
            gradient_metrics: list[dict[str, float | int | bool]] = []
            last_output: StageBStackOutput | None = None
            for repetition in range(repetitions):
                _synchronize(selected_device)
                started = time.perf_counter()
                loss, gradients, last_output = _step(
                    stack,
                    optimizer,
                    segments,
                    target,
                    config,
                    f"measured-{name}-{repetition}",
                )
                _synchronize(selected_device)
                samples.append(time.perf_counter() - started)
                losses.append(loss)
                gradient_metrics.append(gradients)
            assert last_output is not None
            median = statistics.median(samples)
            results.append(
                {
                    "variant": name,
                    "config": config.to_dict(),
                    "available": True,
                    "reason": "measured",
                    "classification": (
                        "mixed_precision_behavioral"
                        if config.activation_dtype is not ActivationDType.FP32
                        else "exact_fp32"
                    ),
                    "warmup_runs": warmup_runs,
                    "repetitions": repetitions,
                    "samples_seconds": samples,
                    "median_step_seconds": median,
                    "input_tokens_per_step": 64,
                    "tokens_per_second": 64 / median,
                    "losses": losses,
                    "gradient_metrics": gradient_metrics,
                    "all_gradients_finite": all(
                        bool(metric["finite"])
                        and bool(metric["written_state_gradient_finite"])
                        and float(metric["written_state_gradient_norm"]) > 0.0
                        for metric in gradient_metrics
                    ),
                    "output_and_state_finite": _finite_output(last_output),
                    "published_sequence_dtype": str(
                        last_output.sequence.dtype
                    ).removeprefix("torch."),
                    "memory_state_dtypes": _state_dtypes(last_output),
                    "cuda_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(selected_device))
                        if selected_device.type == "cuda"
                        else None
                    ),
                    "cuda_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(selected_device))
                        if selected_device.type == "cuda"
                        else None
                    ),
                }
            )
        except (RuntimeError, ValueError) as error:
            results.append(_unavailable_result(name, config, str(error)))
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "format_version": 1,
        "classification": "two_segment_outer_training_step",
        "seed": seed,
        "scale": asdict(scale),
        "protocol": {
            "segments": 2,
            "segment_length": 32,
            "loss": "MSE on segment-two sequence after reading segment-one memory write",
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
            "stable_gate_initialization": {
                "projection_weight_multiplier": 0.01,
                "alpha": 1e-4,
                "eta": 1e-2,
                "theta": 1e-3,
            },
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": str(selected_device),
            "device_name": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda"
                else f"CPU {platform.machine()}"
            ),
            "cuda_available": torch.cuda.is_available(),
        },
        "parameter_count": sum(parameter.numel() for parameter in template.parameters()),
        "variants": results,
    }


def write_training_step_matrix(
    result: Mapping[str, object],
    output_directory: Path | str,
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "a100_training_step_matrix.json",
        "report": output / "a100_training_step_matrix.md",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage B two-segment outer-training matrix",
        "",
        f"Device: `{result['hardware']['device_name']}`; scale: `{result['scale']['name']}`.",
        "",
        "| Variant | Available | Median step s | Tokens/s | CUDA peak allocated | State dtype |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for variant in result["variants"]:
        if not variant["available"]:
            lines.append(
                f"| {variant['variant']} | no: {variant['reason']} | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {variant['variant']} | yes | {variant['median_step_seconds']:.6f} | "
            f"{variant['tokens_per_second']:.2f} | "
            f"{variant['cuda_peak_allocated_bytes']} | "
            f"{','.join(variant['memory_state_dtypes'])} |"
        )
    lines.extend(
        (
            "",
            "Each step processes two 32-token segments. Segment two reads the differentiable memory state written by segment one before backward and AdamW update.",
            "",
        )
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths
