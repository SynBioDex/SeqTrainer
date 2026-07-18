"""Matched evidence for the opt-in B2 causal convolutional gate path."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock

from .backends import execute_stage_b
from .config import GateBackend, StageBBackendConfig
from .convolution import CausalConvolutionalUpdateGates


def _summary(values: Tensor) -> dict[str, float]:
    detached = values.detach().double()
    return {
        "minimum": float(detached.min().item()),
        "mean": float(detached.mean().item()),
        "maximum": float(detached.max().item()),
    }


def _state_delta_norm(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> float:
    flattened = [
        (after[name] - before[name]).detach().double().reshape(-1)
        for name in before
    ]
    return float(torch.linalg.vector_norm(torch.cat(flattened)).item())


def _maximum_state_difference(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> float:
    return max(
        float((candidate[name] - reference[name]).detach().abs().max().item())
        for name in reference
    )


def _stage_a_summary(payload: Mapping[str, object]) -> dict[str, object]:
    gates = payload["gates"]
    variants = payload["variants"]
    assert isinstance(gates, Mapping)
    assert isinstance(variants, Mapping)
    summary: dict[str, object] = {"gates": dict(gates), "variants": {}}
    variant_summary: dict[str, object] = {}
    for name in ("adaptive", "frozen_memory", "no_memory"):
        metric = variants[name]
        assert isinstance(metric, Mapping)
        delayed = metric["delayed_accuracy_by_delay"]
        update_norm = metric["memory_update_norm"]
        assert isinstance(delayed, Mapping)
        assert isinstance(update_norm, Mapping)
        variant_summary[name] = {
            "delayed_accuracy_beyond_32": delayed[">32"],
            "memory_update_norm_mean": update_norm["mean"],
            "overwrite_correctness": metric["overwrite_correctness"],
            "reset_correctness": metric["reset_correctness"],
        }
    summary["variants"] = variant_summary
    return summary


def run_convolution_comparison(
    *,
    seed: int = 20260733,
    d_model: int = 8,
    num_heads: int = 2,
    persistent_tokens: int = 2,
    memory_depth: int = 1,
    kernel_size: int = 3,
    perturb_position: int = 19,
    stage_a_artifact: Path | str = Path(
        "artifacts/titans_stage_a/stage_a_benchmark.json"
    ),
) -> dict[str, object]:
    """Compare token-wise and causal-convolution gates from matched weights."""

    if not 0 < perturb_position < 32:
        raise ValueError("perturb_position must be in [1, 31]")
    torch.manual_seed(seed)
    reference = PaperMACBlock(
        d_model=d_model,
        num_heads=num_heads,
        persistent_tokens=persistent_tokens,
        memory_depth=memory_depth,
    ).double()
    convolution = copy.deepcopy(reference)
    convolutional_gates = CausalConvolutionalUpdateGates(
        d_model,
        kernel_size,
        reference_gates=convolution.memory.gates,
    ).double()
    segment = torch.randn(32, d_model, dtype=torch.float64)
    reference_state = reference.initial_state("b2-reference")
    convolution_state = convolution.initial_state("b2-convolution")
    reference_output = execute_stage_b(reference, reference_state, segment)
    convolution_config = StageBBackendConfig(
        gate_backend=GateBackend.CAUSAL_CONVOLUTION,
        convolution_kernel_size=kernel_size,
    )
    convolution_output = execute_stage_b(
        convolution,
        convolution_state,
        segment,
        config=convolution_config,
        convolutional_gates=convolutional_gates,
    )

    token_gates = reference.memory.gates(reference_output.sequence)
    convolution_gates = convolutional_gates(convolution_output.sequence)
    changed = segment.clone()
    changed[perturb_position] += 50.0
    perturbed_block = copy.deepcopy(convolution)
    perturbed_gates = copy.deepcopy(convolutional_gates)
    perturbed_output = execute_stage_b(
        perturbed_block,
        perturbed_block.initial_state("b2-perturbed"),
        changed,
        config=convolution_config,
        convolutional_gates=perturbed_gates,
    )
    baseline_gate_values = convolutional_gates(convolution_output.sequence)
    perturbed_gate_values = perturbed_gates(perturbed_output.sequence)
    gate_prefix_error = max(
        float(
            (candidate[:perturb_position] - baseline[:perturb_position])
            .detach()
            .abs()
            .max()
            .item()
        )
        for baseline, candidate in (
            (baseline_gate_values.alpha, perturbed_gate_values.alpha),
            (baseline_gate_values.eta, perturbed_gate_values.eta),
            (baseline_gate_values.theta, perturbed_gate_values.theta),
        )
    )
    output_prefix_error = float(
        (
            perturbed_output.sequence[:perturb_position]
            - convolution_output.sequence[:perturb_position]
        )
        .detach()
        .abs()
        .max()
        .item()
    )

    gradient_segment = segment.clone().requires_grad_(True)
    gradient_output = execute_stage_b(
        convolution,
        convolution.initial_state("b2-gradient"),
        gradient_segment,
        config=convolution_config,
        convolutional_gates=convolutional_gates,
    )
    gradient_loss = sum(
        value.square().mean() for value in gradient_output.state.fast_weights.values()
    )
    gradient_loss.backward()

    stage_a_path = Path(stage_a_artifact)
    stage_a_payload = json.loads(stage_a_path.read_text(encoding="utf-8"))
    comparison = {
        "format_version": 1,
        "classification": "repository_defined_opt_in_causal_gate_context",
        "paper_exact": False,
        "configuration": {
            "seed": seed,
            "dtype": "float64",
            "device": str(segment.device),
            "d_model": d_model,
            "num_heads": num_heads,
            "persistent_tokens": persistent_tokens,
            "memory_depth": memory_depth,
            "segment_length": 32,
            "kernel_size": kernel_size,
            "placement": "left-padded depthwise temporal convolution before gate projection",
            "groups": d_model,
            "padding": kernel_size - 1,
            "perturb_position": perturb_position,
        },
        "matched_comparison": {
            "token_wise_gates": {
                "alpha": _summary(token_gates.alpha),
                "eta": _summary(token_gates.eta),
                "theta": _summary(token_gates.theta),
            },
            "causal_convolution_gates": {
                "alpha": _summary(convolution_gates.alpha),
                "eta": _summary(convolution_gates.eta),
                "theta": _summary(convolution_gates.theta),
            },
            "maximum_gate_difference": {
                "alpha": float((convolution_gates.alpha - token_gates.alpha).abs().max().item()),
                "eta": float((convolution_gates.eta - token_gates.eta).abs().max().item()),
                "theta": float((convolution_gates.theta - token_gates.theta).abs().max().item()),
            },
            "memory_update_norm": {
                "token_wise": _state_delta_norm(
                    reference_state.fast_weights, reference_output.state.fast_weights
                ),
                "causal_convolution": _state_delta_norm(
                    convolution_state.fast_weights,
                    convolution_output.state.fast_weights,
                ),
            },
            "final_fast_weight_maximum_difference": _maximum_state_difference(
                reference_output.state.fast_weights,
                convolution_output.state.fast_weights,
            ),
        },
        "causality": {
            "gate_prefix_maximum_error": gate_prefix_error,
            "current_output_prefix_maximum_error": output_prefix_error,
            "passed": gate_prefix_error == 0.0 and output_prefix_error == 0.0,
        },
        "gradients": {
            "input_norm": float(gradient_segment.grad.detach().norm().item()),
            "depthwise_weight_norm": float(
                convolutional_gates.depthwise.weight.grad.detach().norm().item()
            ),
            "projection_weight_norm": float(
                convolutional_gates.projection.weight.grad.detach().norm().item()
            ),
        },
        "unchanged_stage_a_evidence": {
            "source_artifact": str(stage_a_path),
            **_stage_a_summary(stage_a_payload),
        },
    }
    return comparison


def write_convolution_comparison(
    result: Mapping[str, object],
    output_directory: Path | str,
) -> dict[str, Path]:
    """Write JSON and concise Markdown evidence for B2."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "b2_convolution_comparison.json",
        "report": output / "b2_convolution_comparison.md",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matched = result["matched_comparison"]
    causal = result["causality"]
    stage_a = result["unchanged_stage_a_evidence"]
    assert isinstance(matched, Mapping)
    assert isinstance(causal, Mapping)
    assert isinstance(stage_a, Mapping)
    update_norm = matched["memory_update_norm"]
    variants = stage_a["variants"]
    assert isinstance(update_norm, Mapping)
    assert isinstance(variants, Mapping)
    lines = [
        "# B2 causal convolution comparison",
        "",
        "> This is a repository-defined, opt-in interpretation; it is not claimed paper-exact.",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| Token-wise memory update norm | {float(update_norm['token_wise']):.8f} |",
        f"| Convolutional memory update norm | {float(update_norm['causal_convolution']):.8f} |",
        f"| Final fast-weight max difference | {float(matched['final_fast_weight_maximum_difference']):.8f} |",
        f"| Gate prefix maximum error | {float(causal['gate_prefix_maximum_error']):.3e} |",
        f"| Current-output prefix maximum error | {float(causal['current_output_prefix_maximum_error']):.3e} |",
        "",
        "## Unchanged Stage A control evidence",
        "",
        "| Variant | Delayed accuracy >32 | Update norm mean | Overwrite | Reset |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("adaptive", "frozen_memory", "no_memory"):
        metric = variants[name]
        assert isinstance(metric, Mapping)
        lines.append(
            f"| {name} | {float(metric['delayed_accuracy_beyond_32']):.3f} | "
            f"{float(metric['memory_update_norm_mean']):.6f} | "
            f"{float(metric['overwrite_correctness']):.3f} | "
            f"{float(metric['reset_correctness']):.3f} |"
        )
    lines.extend(
        (
            "",
            f"Stage A gates passed: **{bool(stage_a['gates']['passed'])}**.",
            "The JSON contains matched gate statistics, gradient norms, configuration, and provenance.",
            "",
        )
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    return paths

