"""Command-line B1 reference parity and hardware measurement."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock

from .config import StageBBackendConfig
from .parity import compare_backends
from .telemetry import benchmark_stage_b, write_stage_b_artifacts


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return selected


def run_b1_reference_benchmark(
    *,
    d_model: int = 64,
    num_heads: int = 4,
    persistent_tokens: int = 4,
    memory_depth: int = 2,
    segment_count: int = 2,
    seed: int = 20260727,
    warmup_runs: int = 1,
    repetitions: int = 3,
    device: str = "auto",
    output_dir: Path | str = Path("artifacts/titans_stage_b"),
    stem: str = "b1_reference",
) -> dict[str, Path]:
    """Run the available B1 backend and persist parity plus telemetry."""

    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    selected_device = _device(device)
    torch.manual_seed(seed)
    block = PaperMACBlock(
        d_model=d_model,
        num_heads=num_heads,
        persistent_tokens=persistent_tokens,
        memory_depth=memory_depth,
    ).to(device=selected_device, dtype=torch.float32)
    candidate = copy.deepcopy(block)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    segments = [
        torch.randn(32, d_model, generator=generator, dtype=torch.float32).to(selected_device)
        for _ in range(segment_count)
    ]
    parity = compare_backends(block, candidate, segments[0])
    result = benchmark_stage_b(
        block,
        segments,
        config=StageBBackendConfig(),
        seed=seed,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        parity=parity,
    )
    return write_stage_b_artifacts(result, output_dir, stem=stem)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the Stage B reference backend with complete parity evidence."
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--persistent-tokens", type=int, default=4)
    parser.add_argument("--memory-depth", type=int, default=2)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/titans_stage_b"))
    parser.add_argument("--stem", default="b1_reference")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = run_b1_reference_benchmark(
        d_model=arguments.d_model,
        num_heads=arguments.num_heads,
        persistent_tokens=arguments.persistent_tokens,
        memory_depth=arguments.memory_depth,
        segment_count=arguments.segments,
        seed=arguments.seed,
        warmup_runs=arguments.warmup_runs,
        repetitions=arguments.repetitions,
        device=arguments.device,
        output_dir=arguments.output_dir,
        stem=arguments.stem,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

