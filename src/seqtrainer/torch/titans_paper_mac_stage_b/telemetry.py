"""Deterministic MacBook/CUDA telemetry and Stage B artifact writers."""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState

from .backends import StageBBackendRegistry
from .config import StageBBackendConfig
from .parity import ParityReport


@dataclass(frozen=True)
class ModelGeometry:
    block_count: int
    d_model: int
    num_heads: int
    persistent_tokens: int
    memory_depth: int
    segment_length: int
    parameter_count: int


@dataclass(frozen=True)
class HardwareTelemetry:
    platform: str
    machine: str
    device: str
    device_name: str
    torch_version: str
    dtype: str
    cuda_available: bool
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None


@dataclass(frozen=True)
class TimingTelemetry:
    warmup_runs: int
    repetitions: int
    wall_times_seconds: tuple[float, ...]
    median_wall_time_seconds: float
    tokens_per_second: float
    token_count: int


@dataclass(frozen=True)
class StageBBenchmarkResult:
    config: dict[str, object]
    seed: int
    segment_count: int
    geometry: ModelGeometry
    hardware: HardwareTelemetry
    timing: TimingTelemetry
    state_payload_bytes: int
    parity: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _state_payload_bytes(state: PaperMACStreamState) -> int:
    tensors = (*state.fast_weights.values(), *state.surprise.values())
    return sum(value.numel() * value.element_size() for value in tensors)


def _memory_depth(block: PaperMACBlock) -> int:
    return sum(isinstance(layer, nn.Linear) for layer in block.memory.memory_mlp)


def _geometry(block: PaperMACBlock) -> ModelGeometry:
    return ModelGeometry(
        block_count=1,
        d_model=block.d_model,
        num_heads=block.attention.num_heads,
        persistent_tokens=block.persistent_token_count,
        memory_depth=_memory_depth(block),
        segment_length=block.segment_length,
        parameter_count=sum(parameter.numel() for parameter in block.parameters()),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_stage_b(
    block: PaperMACBlock,
    segments: Sequence[Tensor],
    *,
    config: StageBBackendConfig = StageBBackendConfig(),
    seed: int = 20260727,
    warmup_runs: int = 1,
    repetitions: int = 3,
    registry: StageBBackendRegistry | None = None,
    parity: ParityReport | None = None,
) -> StageBBenchmarkResult:
    """Measure a complete multi-segment transition with honest device fields."""

    if not segments:
        raise ValueError("segments must contain at least one segment")
    if warmup_runs < 0 or repetitions <= 0:
        raise ValueError("warmup_runs must be non-negative and repetitions must be positive")
    active_registry = StageBBackendRegistry() if registry is None else registry
    active_registry.validate(config)
    parameter = next(block.parameters())
    device = parameter.device
    dtype = parameter.dtype
    for segment in segments:
        if segment.shape != (block.segment_length, block.d_model):
            raise ValueError("every segment must match the block segment geometry")
        if segment.device != device or segment.dtype != dtype:
            raise ValueError("segments must match the block device and dtype")

    torch.manual_seed(seed)

    def run_once(stream_id: str) -> PaperMACStreamState:
        state = block.initial_state(stream_id)
        for segment in segments:
            state = active_registry.execute(block, state, segment, config=config).state
        return state

    for index in range(warmup_runs):
        run_once(f"warmup-{index}")
    _synchronize(device)

    cuda_measured = device.type == "cuda" and torch.cuda.is_available()
    if cuda_measured:
        torch.cuda.reset_peak_memory_stats(device)
    wall_times: list[float] = []
    final_state: PaperMACStreamState | None = None
    for index in range(repetitions):
        _synchronize(device)
        started = time.perf_counter()
        final_state = run_once(f"timed-{index}")
        _synchronize(device)
        wall_times.append(time.perf_counter() - started)
    assert final_state is not None

    median = statistics.median(wall_times)
    token_count = len(segments) * block.segment_length
    if cuda_measured:
        allocated = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
        device_name = torch.cuda.get_device_name(device)
    else:
        allocated = None
        reserved = None
        device_name = f"CPU {platform.machine()}"
    return StageBBenchmarkResult(
        config=config.to_dict(),
        seed=seed,
        segment_count=len(segments),
        geometry=_geometry(block),
        hardware=HardwareTelemetry(
            platform=platform.platform(),
            machine=platform.machine(),
            device=str(device),
            device_name=device_name,
            torch_version=torch.__version__,
            dtype=str(dtype).removeprefix("torch."),
            cuda_available=torch.cuda.is_available(),
            cuda_allocated_bytes=allocated,
            cuda_reserved_bytes=reserved,
        ),
        timing=TimingTelemetry(
            warmup_runs=warmup_runs,
            repetitions=repetitions,
            wall_times_seconds=tuple(wall_times),
            median_wall_time_seconds=median,
            tokens_per_second=token_count / median,
            token_count=token_count,
        ),
        state_payload_bytes=_state_payload_bytes(final_state),
        parity=None if parity is None else parity.to_dict(),
    )


def write_stage_b_artifacts(
    result: StageBBenchmarkResult,
    output_dir: Path | str,
    *,
    stem: str = "stage_b_backend_benchmark",
) -> dict[str, Path]:
    """Write stable JSON and human-readable Markdown from the same payload."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / f"{stem}.json"
    markdown_path = destination / f"{stem}.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parity_status = "not requested" if result.parity is None else str(result.parity["passed"])
    cuda_memory = (
        "unavailable (non-CUDA execution)"
        if result.hardware.cuda_allocated_bytes is None
        else f"{result.hardware.cuda_allocated_bytes} allocated / {result.hardware.cuda_reserved_bytes} reserved bytes"
    )
    markdown_path.write_text(
        "\n".join(
            (
                "# Titans paper-MAC Stage B backend benchmark",
                "",
                f"- Memory backend: `{result.config['memory_backend']}`",
                f"- Attention backend: `{result.config['attention_backend']}`",
                f"- Activation dtype: `{result.config['activation_dtype']}`",
                f"- Seed: `{result.seed}`",
                f"- Device: `{result.hardware.device_name}` (`{result.hardware.device}`)",
                f"- Torch: `{result.hardware.torch_version}`",
                f"- Geometry: {result.geometry}",
                f"- Segments/tokens: {result.segment_count}/{result.timing.token_count}",
                f"- Warmups/repetitions: {result.timing.warmup_runs}/{result.timing.repetitions}",
                f"- Median wall time: {result.timing.median_wall_time_seconds:.9f} s",
                f"- Throughput: {result.timing.tokens_per_second:.3f} tokens/s",
                f"- State payload: {result.state_payload_bytes} bytes",
                f"- CUDA memory: {cuda_memory}",
                f"- Parity passed: {parity_status}",
                "",
                "Raw per-repetition timings and complete parity metrics are in the JSON artifact.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": markdown_path}
