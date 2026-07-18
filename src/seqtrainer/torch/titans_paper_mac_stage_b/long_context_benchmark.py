"""Long multi-segment synthetic stress matrix for Stage B backends."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import time
import tracemalloc
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from seqtrainer.torch.titans_paper_mac.benchmark import (
    _PaperBenchmarkModel,
    _key_value_embedding,
    _query_embedding,
)
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState
from seqtrainer.torch.titans_paper_mac.synthetic import DEFAULT_VOCABULARY

from .approximate_scan import update_segment_with_stale_windows
from .backends import StageBBackendRegistry
from .config import (
    APPROXIMATE_WINDOWS,
    AttentionBackend,
    GateBackend,
    MemoryBackend,
    StageBBackendConfig,
)
from .convolution import (
    CausalConvolutionalUpdateGates,
    update_segment_with_convolutional_gates,
)
from .stack import StageBMACStack, StageBStackOutput


@dataclass(frozen=True)
class LongContextScale:
    name: str
    block_count: int
    d_model: int
    num_heads: int
    segment_count: int
    persistent_tokens: int = 4
    memory_depth: int = 1
    requires_a100: bool = False


DEFAULT_LONG_CONTEXT_SCALES = (
    LongContextScale("debug_d64", 2, 64, 4, 4),
    LongContextScale("debug_d128", 2, 128, 4, 3),
    LongContextScale("nimble", 4, 256, 8, 2),
    LongContextScale("a100_pilot", 8, 384, 8, 4, requires_a100=True),
)

DEFAULT_LONG_CONTEXT_VARIANTS = (
    "reference",
    "exact_accelerated",
    *(f"approximate_w{window}" for window in APPROXIMATE_WINDOWS),
    "causal_convolution",
    "sdpa",
    "frozen_memory",
    "no_memory",
)


def _variant_config(name: str) -> StageBBackendConfig | None:
    if name == "reference":
        return StageBBackendConfig()
    if name == "exact_accelerated":
        return StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED)
    if name.startswith("approximate_w"):
        window = int(name.removeprefix("approximate_w"))
        return StageBBackendConfig(
            memory_backend=MemoryBackend.APPROXIMATE_SCAN,
            approximate_window=window,
        )
    if name == "causal_convolution":
        return StageBBackendConfig(
            gate_backend=GateBackend.CAUSAL_CONVOLUTION,
            convolution_kernel_size=3,
        )
    if name == "sdpa":
        return StageBBackendConfig(attention_backend=AttentionBackend.SDPA)
    if name in ("frozen_memory", "no_memory"):
        return None
    raise ValueError(f"unknown long-context variant: {name}")


def _state_payload_bytes(states: Sequence[PaperMACStreamState]) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in states
        for value in (*state.fast_weights.values(), *state.surprise.values())
    )


def _mapping_flat(values: Mapping[str, Tensor]) -> Tensor:
    return torch.cat([value.detach().double().reshape(-1) for value in values.values()])


def _stack_state_relative_error(
    reference: Sequence[PaperMACStreamState],
    candidate: Sequence[PaperMACStreamState],
) -> dict[str, float | None]:
    reference_weights = torch.cat(
        [_mapping_flat(state.fast_weights) for state in reference]
    )
    candidate_weights = torch.cat(
        [_mapping_flat(state.fast_weights) for state in candidate]
    )
    reference_surprise = torch.cat(
        [_mapping_flat(state.surprise) for state in reference]
    )
    candidate_surprise = torch.cat(
        [_mapping_flat(state.surprise) for state in candidate]
    )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            reference_weights,
            candidate_weights,
            reference_surprise,
            candidate_surprise,
        )
    ):
        return {
            "fast_weight_relative_l2": None,
            "surprise_relative_l2": None,
        }
    return {
        "fast_weight_relative_l2": float(
            (candidate_weights - reference_weights)
            .norm()
            .div(reference_weights.norm().clamp_min(1e-12))
            .item()
        ),
        "surprise_relative_l2": float(
            (candidate_surprise - reference_surprise)
            .norm()
            .div(reference_surprise.norm().clamp_min(1e-12))
            .item()
        ),
    }


def _states_finite(states: Sequence[PaperMACStreamState]) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for state in states
        for value in (*state.fast_weights.values(), *state.surprise.values())
    )


def _configure_stable_stress_gates(stack: StageBMACStack) -> None:
    """Use declared safe gates so random untrained memory remains interpretable."""

    target_bias = torch.logit(torch.tensor([1e-4, 1e-2, 1e-3]))
    with torch.no_grad():
        for block in stack.blocks:
            block.memory.gates.projection.weight.mul_(0.01)
            block.memory.gates.projection.bias.copy_(
                target_bias.to(block.memory.gates.projection.bias)
            )
        if stack.convolutional_gates is not None:
            for convolutional, block in zip(
                stack.convolutional_gates, stack.blocks
            ):
                convolutional.projection.load_state_dict(
                    block.memory.gates.projection.state_dict()
                )


def _control_segment(
    stack: StageBMACStack,
    states: Sequence[PaperMACStreamState],
    segment: Tensor,
    *,
    no_memory: bool,
) -> StageBStackOutput:
    sequence = segment
    block_sequences: list[Tensor] = []
    retrievals: list[Tensor] = []
    next_states: list[PaperMACStreamState] = []
    for block, state in zip(stack.blocks, states):
        retrieval = (
            torch.zeros_like(sequence)
            if no_memory
            else block.memory.read_segment(state, sequence)
        )
        sequence = block.integrate(retrieval, sequence)
        block_sequences.append(sequence)
        retrievals.append(retrieval)
        next_states.append(
            state.replace(
                fast_weights=state.fast_weights,
                surprise=state.surprise,
                segment_index=state.segment_index + 1,
            )
        )
    return StageBStackOutput(
        sequence,
        tuple(block_sequences),
        tuple(retrievals),
        tuple(next_states),
    )


def _execute_variant_segment(
    name: str,
    stack: StageBMACStack,
    states: Sequence[PaperMACStreamState],
    segment: Tensor,
    registry: StageBBackendRegistry,
) -> StageBStackOutput:
    config = _variant_config(name)
    if config is None:
        return _control_segment(
            stack,
            states,
            segment,
            no_memory=name == "no_memory",
        )
    return stack(states, segment, config=config, registry=registry)


def _serialize_resume_states(
    states: Sequence[PaperMACStreamState], device: torch.device
) -> tuple[tuple[PaperMACStreamState, ...], float, int]:
    started = time.perf_counter()
    payloads = [state.to_state_dict() for state in states]
    restored = tuple(
        PaperMACStreamState.from_state_dict(payload, device=device)
        for payload in payloads
    )
    elapsed = time.perf_counter() - started
    return restored, elapsed, _state_payload_bytes(restored)


def _gate_values(
    name: str,
    stack: StageBMACStack,
    output: StageBStackOutput,
) -> tuple[list[float], list[float], list[float]]:
    alphas: list[float] = []
    etas: list[float] = []
    thetas: list[float] = []
    for index, block in enumerate(stack.blocks):
        if name == "causal_convolution":
            assert stack.convolutional_gates is not None
            gates = stack.convolutional_gates[index](output.block_sequences[index])
        else:
            gates = block.memory.gates(output.block_sequences[index])
        alphas.extend(float(value) for value in gates.alpha.detach().flatten())
        etas.extend(float(value) for value in gates.eta.detach().flatten())
        thetas.extend(float(value) for value in gates.theta.detach().flatten())
    return alphas, etas, thetas


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "minimum": 0.0, "maximum": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _run_scale_variant(
    name: str,
    template: StageBMACStack,
    segments: Sequence[Tensor],
    reference_states_by_segment: Sequence[Sequence[PaperMACStreamState]] | None,
) -> tuple[dict[str, object], tuple[tuple[PaperMACStreamState, ...], ...]]:
    stack = copy.deepcopy(template)
    registry = StageBBackendRegistry()
    device = next(stack.parameters()).device
    states = stack.initial_states(f"{name}-long")
    state_history: list[tuple[PaperMACStreamState, ...]] = []
    segment_latencies: list[float] = []
    resume_latencies: list[float] = []
    state_payloads: list[int] = []
    state_errors: list[dict[str, float | int]] = []
    finite_states: list[bool] = []
    update_norms: list[float] = []
    alphas: list[float] = []
    etas: list[float] = []
    thetas: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tracemalloc.start()
    started_total = time.perf_counter()
    for index, segment in enumerate(segments):
        incoming = states
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        output = _execute_variant_segment(name, stack, states, segment, registry)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        segment_latencies.append(time.perf_counter() - started)
        for old_state, new_state in zip(incoming, output.states):
            before = _mapping_flat(old_state.fast_weights)
            after = _mapping_flat(new_state.fast_weights)
            update_norms.append(float((after - before).norm().item()))
        gate_values = _gate_values(name, stack, output)
        alphas.extend(gate_values[0])
        etas.extend(gate_values[1])
        thetas.extend(gate_values[2])
        states, resume_seconds, payload_bytes = _serialize_resume_states(
            output.states, device
        )
        resume_latencies.append(resume_seconds)
        state_payloads.append(payload_bytes)
        state_history.append(states)
        finite_states.append(_states_finite(states))
        if reference_states_by_segment is not None:
            state_errors.append(
                {
                    "context_tokens": (index + 1) * 32,
                    **_stack_state_relative_error(
                        reference_states_by_segment[index], states
                    ),
                }
            )
    total_seconds = time.perf_counter() - started_total
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime = (
        {"control": name}
        if _variant_config(name) is None
        else registry.runtime_metadata(_variant_config(name))
    )
    return (
        {
            "variant": name,
            "available": True,
            "reason": "measured",
            "runtime_metadata": runtime,
            "input_tokens": len(segments) * 32,
            "total_seconds": total_seconds,
            "segment_latencies_seconds": segment_latencies,
            "latency_per_segment_seconds": statistics.mean(segment_latencies),
            "tokens_per_second": len(segments) * 32 / total_seconds,
            "state_resume_seconds": resume_latencies,
            "mean_state_resume_seconds": statistics.mean(resume_latencies),
            "state_payload_bytes": max(state_payloads),
            "python_tracemalloc_peak_bytes": python_peak,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "cpu_tensor_memory_limitation": (
                "tracemalloc excludes native PyTorch tensor storage"
            ),
            "gate_statistics": {
                "alpha": _summary(alphas),
                "eta": _summary(etas),
                "theta": _summary(thetas),
            },
            "memory_update_norm": _summary(update_norms),
            "state_error_by_context": state_errors,
            "finite_state_by_segment": finite_states,
            "first_nonfinite_context_tokens": next(
                (
                    (index + 1) * 32
                    for index, finite in enumerate(finite_states)
                    if not finite
                ),
                None,
            ),
        },
        tuple(state_history),
    )


def _unavailable_variant(name: str, reason: str) -> dict[str, object]:
    return {"variant": name, "available": False, "reason": reason}


def _causality_matrix(seed: int, variants: Sequence[str]) -> dict[str, object]:
    torch.manual_seed(seed)
    template = StageBMACStack(
        1,
        8,
        num_heads=2,
        persistent_tokens=2,
        memory_depth=1,
        convolution_kernel_size=3,
    ).double().eval()
    segment = torch.randn(32, 8, dtype=torch.float64)
    changed = segment.clone()
    changed[17] += 100.0
    results: dict[str, object] = {}
    for name in variants:
        baseline_stack = copy.deepcopy(template)
        changed_stack = copy.deepcopy(template)
        baseline = _execute_variant_segment(
            name,
            baseline_stack,
            baseline_stack.initial_states(f"causal-{name}"),
            segment,
            StageBBackendRegistry(),
        )
        perturbed = _execute_variant_segment(
            name,
            changed_stack,
            changed_stack.initial_states(f"changed-{name}"),
            changed,
            StageBBackendRegistry(),
        )
        error = float(
            (perturbed.sequence[:17] - baseline.sequence[:17]).abs().max().item()
        )
        results[name] = {"prefix_maximum_error": error, "passed": error == 0.0}
    return {
        "perturb_position": 17,
        "variants": results,
        "passed": all(bool(item["passed"]) for item in results.values()),
    }


def _controlled_update(
    name: str,
    model: _PaperBenchmarkModel,
    state: PaperMACStreamState,
    embeddings: Tensor,
    valid_mask: Tensor,
    convolutional_gates: CausalConvolutionalUpdateGates | None,
) -> PaperMACStreamState:
    if name.startswith("approximate_w"):
        return update_segment_with_stale_windows(
            model.memory,
            state,
            embeddings,
            window_size=int(name.removeprefix("approximate_w")),
            valid_mask=valid_mask,
        )
    if name == "causal_convolution":
        assert convolutional_gates is not None
        return update_segment_with_convolutional_gates(
            model.memory,
            convolutional_gates,
            state,
            embeddings,
            valid_mask=valid_mask,
        )
    if name in ("frozen_memory", "no_memory"):
        return state.replace(
            fast_weights=state.fast_weights,
            surprise=state.surprise,
            segment_index=state.segment_index + 1,
        )
    return model.memory.update_segment(state, embeddings, valid_mask=valid_mask)


def _long_recall_variant(name: str, delays: Sequence[int]) -> dict[str, object]:
    vocabulary = DEFAULT_VOCABULARY
    accuracies: dict[str, float] = {}
    losses: dict[str, float] = {}
    for delay_segments in delays:
        correct: list[bool] = []
        delay_losses: list[float] = []
        for stream_index in range(4):
            model = _PaperBenchmarkModel(vocabulary.size)
            convolutional_gates = (
                CausalConvolutionalUpdateGates(
                    model.d_model,
                    3,
                    reference_gates=model.memory.gates,
                )
                if name == "causal_convolution"
                else None
            )
            key = vocabulary.keys[stream_index]
            value = vocabulary.values[stream_index]
            embeddings = torch.zeros(32, model.d_model)
            valid = torch.zeros(32, dtype=torch.bool)
            embeddings[0] = _key_value_embedding(key, value, vocabulary.size)
            valid[0] = True
            state = _controlled_update(
                name,
                model,
                model.memory.initial_state(f"{name}-{stream_index}"),
                embeddings,
                valid,
                convolutional_gates,
            )
            empty = torch.zeros_like(embeddings)
            empty_valid = torch.zeros_like(valid)
            for _ in range(delay_segments - 1):
                state = _controlled_update(
                    name,
                    model,
                    state,
                    empty,
                    empty_valid,
                    convolutional_gates,
                )
            query = torch.zeros_like(embeddings)
            query[0] = _query_embedding(key, vocabulary.size)
            retrieval = (
                torch.zeros_like(query)
                if name == "no_memory"
                else model.memory.read_segment(state, query)
            )
            logits = retrieval[0, vocabulary.size :]
            delay_losses.append(
                float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([value])))
            )
            correct.append(int(logits.argmax().item()) == value)
        context_tokens = delay_segments * 32
        accuracies[str(context_tokens)] = sum(correct) / len(correct)
        losses[str(context_tokens)] = statistics.mean(delay_losses)

    model = _PaperBenchmarkModel(vocabulary.size)
    convolutional_gates = (
        CausalConvolutionalUpdateGates(
            model.d_model, 3, reference_gates=model.memory.gates
        )
        if name == "causal_convolution"
        else None
    )
    state = model.memory.initial_state(f"overwrite-{name}")
    key = vocabulary.keys[0]
    first_value, second_value = vocabulary.values[0], vocabulary.values[1]
    for value in (first_value, second_value):
        embeddings = torch.zeros(32, model.d_model)
        valid = torch.zeros(32, dtype=torch.bool)
        embeddings[0] = _key_value_embedding(key, value, vocabulary.size)
        valid[0] = True
        state = _controlled_update(
            name, model, state, embeddings, valid, convolutional_gates
        )
    query = torch.zeros(32, model.d_model)
    query[0] = _query_embedding(key, vocabulary.size)
    retrieval = (
        torch.zeros_like(query)
        if name == "no_memory"
        else model.memory.read_segment(state, query)
    )
    overwrite_correct = int(
        retrieval[0, vocabulary.size :].argmax().item() == second_value
    )
    reset = model.memory.reset_state(state)
    initial = model.memory.initial_state("reset-reference")
    reset_correct = all(
        torch.equal(reset.fast_weights[param], initial.fast_weights[param])
        and torch.equal(reset.surprise[param], initial.surprise[param])
        for param in initial.fast_weights
    )
    aggregate_loss = statistics.mean(losses.values())
    return {
        "delay_accuracy_by_context_tokens": accuracies,
        "loss_by_context_tokens": losses,
        "mean_loss": aggregate_loss,
        "mean_bpb": aggregate_loss / math.log(2),
        "overwrite_correctness": float(overwrite_correct),
        "reset_correctness": float(reset_correct),
        "task_memory_semantics": (
            "reference (attention substitution is irrelevant to controlled memory task)"
            if name == "sdpa"
            else name
        ),
    }


def run_long_context_study(
    *,
    scales: Sequence[LongContextScale] = DEFAULT_LONG_CONTEXT_SCALES,
    variants: Sequence[str] = DEFAULT_LONG_CONTEXT_VARIANTS,
    seed: int = 20260738,
    device: torch.device | str = torch.device("cpu"),
) -> dict[str, object]:
    """Run fixed-seed multi-segment stress and long-delay controlled recall."""

    selected_device = torch.device(device)
    is_a100 = (
        selected_device.type == "cuda"
        and torch.cuda.is_available()
        and "A100" in torch.cuda.get_device_name(selected_device).upper()
    )
    scale_results: list[dict[str, object]] = []
    for scale_index, scale in enumerate(scales):
        if scale.requires_a100 and not is_a100:
            reason = "named Colab Pro A100 environment is unavailable in this execution"
            scale_results.append(
                {
                    "scale": asdict(scale),
                    "environment": "a100",
                    "available": False,
                    "reason": reason,
                    "variants": [
                        _unavailable_variant(name, reason) for name in variants
                    ],
                }
            )
            continue
        torch.manual_seed(seed + scale_index)
        template = StageBMACStack(
            scale.block_count,
            scale.d_model,
            num_heads=scale.num_heads,
            persistent_tokens=scale.persistent_tokens,
            memory_depth=scale.memory_depth,
            convolution_kernel_size=3,
        ).to(device=selected_device, dtype=torch.float32)
        _configure_stable_stress_gates(template)
        generator = torch.Generator(device="cpu").manual_seed(seed + scale_index)
        segments = [
            torch.randn(32, scale.d_model, generator=generator).to(selected_device)
            for _ in range(scale.segment_count)
        ]
        reference_result, reference_history = _run_scale_variant(
            "reference", template, segments, None
        )
        measured = [reference_result]
        for name in variants:
            if name == "reference":
                continue
            result, _ = _run_scale_variant(
                name,
                template,
                segments,
                reference_history,
            )
            measured.append(result)
        scale_results.append(
            {
                "scale": asdict(scale),
                "environment": (
                    "a100" if selected_device.type == "cuda" else "macbook_cpu"
                ),
                "available": True,
                "device": str(selected_device),
                "parameter_count": sum(
                    parameter.numel() for parameter in template.parameters()
                ),
                "variants": measured,
            }
        )

    recall = {
        name: _long_recall_variant(name, (2, 4, 8, 16)) for name in variants
    }
    return {
        "format_version": 1,
        "classification": "synthetic_system_validation_not_biology_performance",
        "protocol": {
            "seed": seed,
            "dtype": "float32",
            "segment_length": 32,
            "state_resume": "serialize/detach/restore after every segment",
            "timing_measurement": {
                "warmup_streams": 0,
                "stream_repetitions": 1,
                "per_segment_samples": "one sample for every segment in the stream",
            },
            "stable_gate_initialization": {
                "projection_weight_multiplier": 0.01,
                "bias_gate_values": {
                    "alpha": 0.0001,
                    "eta": 0.01,
                    "theta": 0.001,
                },
                "reason": (
                    "untrained default sigmoid gates near 0.5 make the random reference "
                    "recurrence non-finite and prevent interpretable backend comparison"
                ),
            },
            "variants": list(variants),
            "scales": [asdict(scale) for scale in scales],
            "rerun_command": (
                "/Users/gonzalovidal/opt/anaconda3/envs/seqtrainer/bin/"
                "seqtrainer-titans-stage-b-long-context"
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": str(selected_device),
            "device_name": (
                torch.cuda.get_device_name(selected_device)
                if selected_device.type == "cuda" and torch.cuda.is_available()
                else f"CPU {platform.machine()}"
            ),
            "cuda_available": torch.cuda.is_available(),
            "a100_available": is_a100,
        },
        "long_stream_scales": scale_results,
        "controlled_long_recall": recall,
        "causality": _causality_matrix(seed + 100, variants),
        "limitations": [
            "No genomic corpus was loaded or trained.",
            "Controlled recall isolates memory persistence and is not biology performance.",
            "CPU tracemalloc excludes native tensor storage; CUDA peak is reported only when available.",
            "A100 results are unavailable unless the named hardware is actually attached.",
        ],
    }


def _render_long_context_svg(result: Mapping[str, object]) -> str:
    recall = result["controlled_long_recall"]
    assert isinstance(recall, Mapping)
    selected = [
        name
        for name in ("reference", "approximate_w2", "approximate_w32", "frozen_memory", "no_memory")
        if name in recall
    ]
    delays = (64, 128, 256, 512)
    colors = {
        "reference": "#166534",
        "approximate_w2": "#7c3aed",
        "approximate_w32": "#c026d3",
        "frozen_memory": "#6b7280",
        "no_memory": "#9ca3af",
    }
    width, height, left, top, plot_width, plot_height = 680, 340, 75, 45, 500, 230
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="sans-serif" font-size="16">B7 controlled recall by context length</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
    ]
    for index, delay in enumerate(delays):
        x = left + index * (plot_width / (len(delays) - 1))
        lines.append(f'<text x="{x - 12:.1f}" y="{top + plot_height + 22}" font-family="sans-serif" font-size="12">{delay}</text>')
    for variant_index, name in enumerate(selected):
        metric = recall[name]
        assert isinstance(metric, Mapping)
        accuracies = metric["delay_accuracy_by_context_tokens"]
        points = []
        for index, delay in enumerate(delays):
            x = left + index * (plot_width / (len(delays) - 1))
            y = top + plot_height - float(accuracies[str(delay)]) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[name]
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{590}" y="{60 + variant_index * 22}" font-family="sans-serif" font-size="12" fill="{color}">{name}</text>')
    lines.extend(
        (
            f'<text x="{left + 170}" y="{height - 15}" font-family="sans-serif" font-size="13">context tokens</text>',
            '<text x="8" y="55" font-family="sans-serif" font-size="12">accuracy</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def write_long_context_study(
    result: Mapping[str, object], output_directory: Path | str
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "b7_long_context_study.json",
        "report": output / "b7_long_context_study.md",
        "plot": output / "b7_long_context_recall.svg",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# B7 long-context synthetic validation",
        "",
        "> Synthetic system validation only; no genomic training or biology-performance claim.",
        "",
        "## MacBook CPU",
        "",
        "| Scale | Variant | Tokens/s | Segment latency s | State/resume s | State bytes | Final state rel L2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scale in result["long_stream_scales"]:
        if not scale["available"]:
            continue
        for variant in scale["variants"]:
            errors = variant["state_error_by_context"]
            final_error = 0.0 if not errors else errors[-1]["fast_weight_relative_l2"]
            final_error_text = (
                "nonfinite" if final_error is None else f"{final_error:.3e}"
            )
            lines.append(
                f"| {scale['scale']['name']} | {variant['variant']} | "
                f"{variant['tokens_per_second']:.2f} | "
                f"{variant['latency_per_segment_seconds']:.4f} | "
                f"{variant['mean_state_resume_seconds']:.5f} | "
                f"{variant['state_payload_bytes']} | {final_error_text} |"
            )
    lines.extend(("", "## A100", ""))
    unavailable = [
        scale for scale in result["long_stream_scales"] if not scale["available"]
    ]
    if unavailable:
        for scale in unavailable:
            lines.append(f"- `{scale['scale']['name']}` unavailable: {scale['reason']}")
    else:
        lines.append("- A100 scale measured; see JSON for complete per-variant telemetry.")
    lines.extend(
        (
            "",
            "## Controlled long recall",
            "",
            "| Variant | 64 | 128 | 256 | 512 | Overwrite | Reset | Mean BPB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for name, metric in result["controlled_long_recall"].items():
        accuracy = metric["delay_accuracy_by_context_tokens"]
        lines.append(
            f"| {name} | {accuracy['64']:.3f} | {accuracy['128']:.3f} | "
            f"{accuracy['256']:.3f} | {accuracy['512']:.3f} | "
            f"{metric['overwrite_correctness']:.3f} | {metric['reset_correctness']:.3f} | "
            f"{metric['mean_bpb']:.3f} |"
        )
    lines.extend(
        (
            "",
            f"All backend future-prefix checks passed: **{result['causality']['passed']}**.",
            "",
            f"Independent rerun: `{result['protocol']['rerun_command']}`",
            "",
            "The JSON contains raw segment/resume timings, context-indexed drift, peak memory, gates/updates, hardware separation, and limitations.",
            "",
        )
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    paths["plot"].write_text(_render_long_context_svg(result), encoding="utf-8")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/titans_stage_b")
    )
    parser.add_argument("--seed", type=int, default=20260738)
    args = parser.parse_args(argv)
    result = run_long_context_study(seed=args.seed)
    paths = write_long_context_study(result, args.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
