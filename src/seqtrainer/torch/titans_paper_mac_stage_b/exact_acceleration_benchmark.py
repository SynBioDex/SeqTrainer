"""Matched reference/exact-functional-loop measurements for B3 scales."""

from __future__ import annotations

import copy
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .backends import StageBBackendRegistry
from .config import MemoryBackend, StageBBackendConfig
from .stack import StageBMACStack, StageBStackOutput


@dataclass(frozen=True)
class StageBScale:
    name: str
    block_count: int
    d_model: int
    num_heads: int
    persistent_tokens: int = 4
    memory_depth: int = 1
    segment_count: int = 1


@dataclass(frozen=True)
class BackendScaleTiming:
    backend: str
    available: bool
    reason: str
    wall_times_seconds: tuple[float, ...]
    median_wall_time_seconds: float | None
    tokens_per_second: float | None
    state_payload_bytes: int | None
    runtime_metadata: dict[str, object] | None


@dataclass(frozen=True)
class ExactAccelerationScaleResult:
    scale: StageBScale
    device: str
    device_name: str
    dtype: str
    seed: int
    warmup_runs: int
    repetitions: int
    parameter_count: int | None
    tensor_exact: bool | None
    reference: BackendScaleTiming
    exact_accelerated: BackendScaleTiming
    speedup: float | None


@dataclass(frozen=True)
class ExactAccelerationMatrix:
    host_platform: str
    torch_version: str
    results: tuple[ExactAccelerationScaleResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SCALES = (
    StageBScale("debug", block_count=2, d_model=64, num_heads=4),
    StageBScale("nimble", block_count=4, d_model=256, num_heads=8),
    StageBScale("a100_pilot", block_count=8, d_model=384, num_heads=8),
)


def _state_bytes(output: StageBStackOutput) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for state in output.states
        for tensor in (*state.fast_weights.values(), *state.surprise.values())
    )


def _tensor_exact(left: StageBStackOutput, right: StageBStackOutput) -> bool:
    if not torch.equal(left.sequence, right.sequence):
        return False
    if any(not torch.equal(a, b) for a, b in zip(left.retrievals, right.retrievals)):
        return False
    for left_state, right_state in zip(left.states, right.states):
        for name in left_state.fast_weights:
            if not torch.equal(left_state.fast_weights[name], right_state.fast_weights[name]):
                return False
            if not torch.equal(left_state.surprise[name], right_state.surprise[name]):
                return False
    return True


def _run_segments(
    stack: StageBMACStack,
    segments: Sequence[Tensor],
    config: StageBBackendConfig,
    registry: StageBBackendRegistry,
    stream_id: str,
) -> StageBStackOutput:
    states = stack.initial_states(stream_id)
    output: StageBStackOutput | None = None
    for segment in segments:
        output = stack(states, segment, config=config, registry=registry)
        states = output.states
    assert output is not None
    return output


def _measure(
    stack: StageBMACStack,
    segments: Sequence[Tensor],
    config: StageBBackendConfig,
    *,
    warmup_runs: int,
    repetitions: int,
) -> tuple[BackendScaleTiming, StageBStackOutput]:
    registry = StageBBackendRegistry()
    device = next(stack.parameters()).device
    for index in range(warmup_runs):
        _run_segments(stack, segments, config, registry, f"warmup-{index}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_times: list[float] = []
    output: StageBStackOutput | None = None
    for index in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _run_segments(stack, segments, config, registry, f"timed-{index}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_times.append(time.perf_counter() - started)
    assert output is not None
    median = statistics.median(wall_times)
    token_count = stack.block_count * len(segments) * stack.segment_length
    return (
        BackendScaleTiming(
            backend=config.memory_backend.value,
            available=True,
            reason="measured",
            wall_times_seconds=tuple(wall_times),
            median_wall_time_seconds=median,
            tokens_per_second=token_count / median,
            state_payload_bytes=_state_bytes(output),
            runtime_metadata=registry.runtime_metadata(config),
        ),
        output,
    )


def _unavailable_timing(backend: str, reason: str) -> BackendScaleTiming:
    return BackendScaleTiming(
        backend=backend,
        available=False,
        reason=reason,
        wall_times_seconds=(),
        median_wall_time_seconds=None,
        tokens_per_second=None,
        state_payload_bytes=None,
        runtime_metadata=None,
    )


def run_exact_acceleration_matrix(
    *,
    scales: Sequence[StageBScale] = DEFAULT_SCALES,
    seed: int = 20260727,
    warmup_runs: int = 1,
    repetitions: int = 1,
    device: torch.device | str = torch.device("cpu"),
) -> ExactAccelerationMatrix:
    """Measure CPU scales and honestly mark an absent named A100 environment."""

    selected_device = torch.device(device)
    results: list[ExactAccelerationScaleResult] = []
    for scale_index, scale in enumerate(scales):
        requires_a100 = scale.name == "a100_pilot"
        is_a100 = (
            selected_device.type == "cuda"
            and torch.cuda.is_available()
            and "A100" in torch.cuda.get_device_name(selected_device).upper()
        )
        if requires_a100 and not is_a100:
            reason = "named Colab Pro A100 environment is unavailable in this execution"
            results.append(
                ExactAccelerationScaleResult(
                    scale=scale,
                    device=str(selected_device),
                    device_name=(
                        torch.cuda.get_device_name(selected_device)
                        if selected_device.type == "cuda" and torch.cuda.is_available()
                        else f"CPU {platform.machine()}"
                    ),
                    dtype="float32",
                    seed=seed,
                    warmup_runs=warmup_runs,
                    repetitions=repetitions,
                    parameter_count=None,
                    tensor_exact=None,
                    reference=_unavailable_timing("reference", reason),
                    exact_accelerated=_unavailable_timing("exact_accelerated", reason),
                    speedup=None,
                )
            )
            continue

        torch.manual_seed(seed + scale_index)
        reference_stack = StageBMACStack(
            scale.block_count,
            scale.d_model,
            num_heads=scale.num_heads,
            persistent_tokens=scale.persistent_tokens,
            memory_depth=scale.memory_depth,
        ).to(device=selected_device, dtype=torch.float32)
        exact_stack = copy.deepcopy(reference_stack)
        generator = torch.Generator(device="cpu").manual_seed(seed + scale_index)
        segments = [
            torch.randn(32, scale.d_model, generator=generator, dtype=torch.float32).to(selected_device)
            for _ in range(scale.segment_count)
        ]
        reference_config = StageBBackendConfig()
        exact_config = StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED)
        reference_timing, reference_output = _measure(
            reference_stack,
            segments,
            reference_config,
            warmup_runs=warmup_runs,
            repetitions=repetitions,
        )
        exact_timing, exact_output = _measure(
            exact_stack,
            segments,
            exact_config,
            warmup_runs=warmup_runs,
            repetitions=repetitions,
        )
        assert reference_timing.median_wall_time_seconds is not None
        assert exact_timing.median_wall_time_seconds is not None
        results.append(
            ExactAccelerationScaleResult(
                scale=scale,
                device=str(selected_device),
                device_name=(
                    torch.cuda.get_device_name(selected_device)
                    if selected_device.type == "cuda"
                    else f"CPU {platform.machine()}"
                ),
                dtype="float32",
                seed=seed + scale_index,
                warmup_runs=warmup_runs,
                repetitions=repetitions,
                parameter_count=sum(parameter.numel() for parameter in reference_stack.parameters()),
                tensor_exact=_tensor_exact(reference_output, exact_output),
                reference=reference_timing,
                exact_accelerated=exact_timing,
                speedup=(
                    reference_timing.median_wall_time_seconds
                    / exact_timing.median_wall_time_seconds
                ),
            )
        )
    return ExactAccelerationMatrix(
        host_platform=platform.platform(),
        torch_version=torch.__version__,
        results=tuple(results),
    )


def write_exact_acceleration_matrix(
    matrix: ExactAccelerationMatrix,
    output_dir: Path | str,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "b3_exact_acceleration_matrix.json"
    markdown_path = destination / "b3_exact_acceleration_matrix.md"
    json_path.write_text(json.dumps(matrix.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# B3 exact acceleration matrix",
        "",
        f"Host: `{matrix.host_platform}`; PyTorch `{matrix.torch_version}`.",
        "",
        "| Scale | Geometry | Device | Exact | Reference s | Exact s | Speedup |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in matrix.results:
        geometry = f"{result.scale.block_count}x d={result.scale.d_model}"
        if not result.reference.available:
            lines.append(
                f"| {result.scale.name} | {geometry} | {result.device_name} | unavailable | - | - | - |"
            )
            lines.append(f"\nUnavailable reason: {result.reference.reason}\n")
            continue
        lines.append(
            "| "
            f"{result.scale.name} | {geometry} | {result.device_name} | {result.tensor_exact} | "
            f"{result.reference.median_wall_time_seconds:.6f} | "
            f"{result.exact_accelerated.median_wall_time_seconds:.6f} | "
            f"{result.speedup:.3f}x |"
        )
    lines.extend(
        (
            "",
            "The functional-loop path preserves evolving gradients and token order. A speedup below 1x is reported as a regression, not an acceleration claim.",
        )
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}

