"""Reproducible parity and capability evidence for B6 attention adapters."""

from __future__ import annotations

import copy
import json
import platform
import time
from pathlib import Path
from typing import Callable, Mapping

import torch
from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock

from .attention import (
    integrate_sdpa_attention,
    probe_flash_mask_support,
    sdpa_allowed_attention_mask,
)
from .backends import execute_stage_b
from .config import ActivationDType, AttentionBackend, StageBBackendConfig


def _error(reference: Tensor, candidate: Tensor) -> dict[str, float | bool]:
    difference = (candidate.detach().double() - reference.detach().double()).abs()
    denominator = reference.detach().double().abs().clamp_min(1e-12)
    return {
        "maximum_absolute_error": float(difference.max().item()),
        "maximum_relative_error": float((difference / denominator).max().item()),
        "tensor_exact": bool(torch.equal(candidate, reference)),
    }


def _maximum_mapping_error(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> dict[str, float]:
    absolute = 0.0
    relative = 0.0
    for name in reference:
        metric = _error(reference[name], candidate[name])
        absolute = max(absolute, float(metric["maximum_absolute_error"]))
        relative = max(relative, float(metric["maximum_relative_error"]))
    return {
        "maximum_absolute_error": absolute,
        "maximum_relative_error": relative,
    }


def _parity_case(seed: int, dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(seed)
    reference = PaperMACBlock(
        d_model=8, num_heads=2, persistent_tokens=3, memory_depth=1
    ).to(dtype=dtype)
    candidate = copy.deepcopy(reference)
    reference_segment = torch.randn(32, 8, dtype=dtype, requires_grad=True)
    candidate_segment = reference_segment.detach().clone().requires_grad_(True)
    reference_output = execute_stage_b(
        reference, reference.initial_state("b6-reference"), reference_segment
    )
    candidate_output = execute_stage_b(
        candidate,
        candidate.initial_state("b6-sdpa"),
        candidate_segment,
        config=StageBBackendConfig(attention_backend=AttentionBackend.SDPA),
    )
    reference_loss = reference_output.sequence.square().mean() + sum(
        value.square().mean() for value in reference_output.state.fast_weights.values()
    )
    candidate_loss = candidate_output.sequence.square().mean() + sum(
        value.square().mean() for value in candidate_output.state.fast_weights.values()
    )
    reference_loss.backward()
    candidate_loss.backward()
    reference_parameters = dict(reference.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "sequence": _error(reference_output.sequence, candidate_output.sequence),
        "retrieval": _error(reference_output.retrieval, candidate_output.retrieval),
        "fast_weights": _maximum_mapping_error(
            reference_output.state.fast_weights,
            candidate_output.state.fast_weights,
        ),
        "surprise": _maximum_mapping_error(
            reference_output.state.surprise,
            candidate_output.state.surprise,
        ),
        "input_gradient": _error(reference_segment.grad, candidate_segment.grad),
        "persistent_token_gradient": _error(
            reference.persistent_tokens.grad,
            candidate.persistent_tokens.grad,
        ),
        "attention_gradient": {
            name: _error(reference_parameters[name].grad, candidate_parameters[name].grad)
            for name in (
                "attention.in_proj_weight",
                "attention.in_proj_bias",
                "attention.out_proj.weight",
                "attention.out_proj.bias",
            )
        },
    }


def _timing(
    function: Callable[[], Tensor],
    *,
    warmup_runs: int,
    repetitions: int,
) -> dict[str, object]:
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(warmup_runs):
            function()
        for _ in range(repetitions):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
    mean_seconds = sum(samples) / len(samples)
    return {
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "samples_seconds": samples,
        "mean_seconds": mean_seconds,
        "tokens_per_second": 32 / mean_seconds,
    }


def _mixed_precision_case(
    block: PaperMACBlock,
    segment: Tensor,
    activation_dtype: ActivationDType,
) -> dict[str, object]:
    if activation_dtype is ActivationDType.FP16 and segment.device.type != "cuda":
        return {
            "available": False,
            "classification": "unavailable",
            "reason": "Stage B permits FP16 SDPA only on CUDA",
        }
    reference = copy.deepcopy(block)
    candidate = copy.deepcopy(block)
    try:
        reference_output = execute_stage_b(
            reference,
            reference.initial_state("mixed-reference"),
            segment,
        )
        candidate_output = execute_stage_b(
            candidate,
            candidate.initial_state("mixed-candidate"),
            segment,
            config=StageBBackendConfig(
                attention_backend=AttentionBackend.SDPA,
                activation_dtype=activation_dtype,
            ),
        )
        return {
            "available": True,
            "classification": "behavioral_parity_not_numerical_parity",
            "sequence": _error(reference_output.sequence, candidate_output.sequence),
            "fast_weights": _maximum_mapping_error(
                reference_output.state.fast_weights,
                candidate_output.state.fast_weights,
            ),
            "surprise": _maximum_mapping_error(
                reference_output.state.surprise,
                candidate_output.state.surprise,
            ),
            "published_sequence_dtype": str(candidate_output.sequence.dtype).removeprefix(
                "torch."
            ),
            "memory_state_dtypes": sorted(
                {str(value.dtype).removeprefix("torch.") for value in candidate_output.state.fast_weights.values()}
            ),
        }
    except (RuntimeError, ValueError) as error:
        return {
            "available": False,
            "classification": "unavailable",
            "reason": str(error),
        }


def run_attention_backend_evidence(
    *,
    seed: int = 20260735,
    warmup_runs: int = 2,
    repetitions: int = 10,
) -> dict[str, object]:
    """Run CPU oracle/parity evidence and device-dependent capability probes."""

    if warmup_runs < 0 or repetitions <= 0:
        raise ValueError("warmup_runs must be nonnegative and repetitions positive")
    fp64 = _parity_case(seed, torch.float64)
    fp32 = _parity_case(seed, torch.float32)

    torch.manual_seed(seed + 1)
    timing_block = PaperMACBlock(
        d_model=64, num_heads=4, persistent_tokens=4, memory_depth=1
    ).float().eval()
    timing_segment = torch.randn(32, 64)
    timing_state = timing_block.initial_state("timing")
    timing_retrieval = timing_block.memory.read_segment(timing_state, timing_segment)
    timing = {
        "geometry": {
            "d_model": 64,
            "num_heads": 4,
            "persistent_tokens": 4,
            "segment_length": 32,
        },
        "multihead_attention": _timing(
            lambda: timing_block.integrate(timing_retrieval, timing_segment),
            warmup_runs=warmup_runs,
            repetitions=repetitions,
        ),
        "sdpa": _timing(
            lambda: integrate_sdpa_attention(
                timing_block, timing_retrieval, timing_segment
            ),
            warmup_runs=warmup_runs,
            repetitions=repetitions,
        ),
    }

    mask = timing_block.attention_mask()
    allowed = sdpa_allowed_attention_mask(timing_block, device=mask.device)
    torch.manual_seed(seed + 2)
    causal_block = PaperMACBlock(
        d_model=8, num_heads=2, persistent_tokens=3, memory_depth=1
    ).double().eval()
    causal_changed_block = copy.deepcopy(causal_block)
    causal_segment = torch.randn(32, 8, dtype=torch.float64)
    changed = causal_segment.clone()
    changed[19] += 100.0
    sdpa_config = StageBBackendConfig(attention_backend=AttentionBackend.SDPA)
    baseline = execute_stage_b(
        causal_block,
        causal_block.initial_state("causal-base"),
        causal_segment,
        config=sdpa_config,
    )
    perturbed = execute_stage_b(
        causal_changed_block,
        causal_changed_block.initial_state("causal-changed"),
        changed,
        config=sdpa_config,
    )
    prefix_error = float(
        (perturbed.sequence[:19] - baseline.sequence[:19]).abs().max().item()
    )

    mixed_block = PaperMACBlock(
        d_model=8, num_heads=2, persistent_tokens=3, memory_depth=1
    ).float().eval()
    mixed_segment = torch.randn(32, 8)
    return {
        "format_version": 1,
        "seed": seed,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
        "mask": {
            "layout": "[persistent, retrieval, sequence]",
            "shape": list(mask.shape),
            "allowed_edges": int(allowed.sum().item()),
            "blocked_edges": int(mask.sum().item()),
            "boolean_complement_mismatches": int((allowed == mask).sum().item()),
            "persistent_queries_restricted_to_persistent_keys": bool(
                allowed[: timing_block.persistent_token_count, timing_block.persistent_token_count :]
                .logical_not()
                .all()
                .item()
            ),
        },
        "parity": {"fp64_oracle": fp64, "fp32_numerical": fp32},
        "causality": {
            "future_perturb_position": 19,
            "prefix_maximum_error": prefix_error,
            "passed": prefix_error == 0.0,
        },
        "mixed_precision_behavioral": {
            "bfloat16": _mixed_precision_case(
                mixed_block, mixed_segment, ActivationDType.BF16
            ),
            "float16": _mixed_precision_case(
                mixed_block, mixed_segment, ActivationDType.FP16
            ),
        },
        "timing": timing,
        "flash_probe": probe_flash_mask_support(mixed_block),
    }


def write_attention_backend_evidence(
    result: Mapping[str, object], output_directory: Path | str
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "b6_attention_backend_evidence.json",
        "report": output / "b6_attention_backend_evidence.md",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parity = result["parity"]
    timing = result["timing"]
    mixed = result["mixed_precision_behavioral"]
    flash = result["flash_probe"]
    causal = result["causality"]
    assert all(isinstance(item, Mapping) for item in (parity, timing, mixed, flash, causal))
    fp64 = parity["fp64_oracle"]
    fp32 = parity["fp32_numerical"]
    assert isinstance(fp64, Mapping) and isinstance(fp32, Mapping)
    reference_timing = timing["multihead_attention"]
    sdpa_timing = timing["sdpa"]
    assert isinstance(reference_timing, Mapping) and isinstance(sdpa_timing, Mapping)
    lines = [
        "# B6 attention backend evidence",
        "",
        "| Evidence | FP64 oracle | FP32 numerical |",
        "| --- | ---: | ---: |",
        f"| Sequence max abs error | {fp64['sequence']['maximum_absolute_error']:.3e} | {fp32['sequence']['maximum_absolute_error']:.3e} |",
        f"| Input-gradient max abs error | {fp64['input_gradient']['maximum_absolute_error']:.3e} | {fp32['input_gradient']['maximum_absolute_error']:.3e} |",
        f"| Persistent-gradient max abs error | {fp64['persistent_token_gradient']['maximum_absolute_error']:.3e} | {fp32['persistent_token_gradient']['maximum_absolute_error']:.3e} |",
        "",
        f"Causal future-prefix error: `{causal['prefix_maximum_error']:.3e}`.",
        "",
        "| CPU attention path | Tokens/s |",
        "| --- | ---: |",
        f"| MultiheadAttention | {reference_timing['tokens_per_second']:.2f} |",
        f"| Functional SDPA | {sdpa_timing['tokens_per_second']:.2f} |",
        "",
        f"BF16 behavioral status: `{mixed['bfloat16']['classification']}`.",
        f"FP16 behavioral status: `{mixed['float16']['classification']}`.",
        f"Flash exact-mask probe: `{flash['available']}` — {flash['reason']}",
        "",
        "The JSON stores full state/surprise/attention-gradient errors, raw timing samples, mask counts, mixed-precision state dtypes, and hardware provenance.",
        "",
    ]
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths
