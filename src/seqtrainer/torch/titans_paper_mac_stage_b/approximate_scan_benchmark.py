"""Speed–fidelity study for the explicitly approximate stale-window backend."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time
import tracemalloc
from statistics import mean
from typing import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from seqtrainer.torch.titans_paper_mac.benchmark import (
    BenchmarkConfig,
    ScalarSummary,
    _PaperBenchmarkModel,
    _accuracy,
    _delay_bucket,
    _encode_segment,
    _fixture_schedule,
    _global_gradient_norm,
)
from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState
from seqtrainer.torch.titans_paper_mac.synthetic import (
    DEFAULT_VOCABULARY,
    SyntheticSegment,
    build_stage_a_fixtures,
)

from .approximate_scan import update_segment_with_stale_windows
from .backends import execute_stage_b
from .config import APPROXIMATE_WINDOWS, MemoryBackend, StageBBackendConfig


def _tensor_mapping_metrics(
    reference: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> dict[str, float]:
    reference_flat = torch.cat(
        [value.detach().double().reshape(-1) for value in reference.values()]
    )
    candidate_flat = torch.cat(
        [candidate[name].detach().double().reshape(-1) for name in reference]
    )
    difference = candidate_flat - reference_flat
    return {
        "maximum_absolute_error": float(difference.abs().max().item()),
        "relative_l2_error": float(
            difference.norm().div(reference_flat.norm().clamp_min(1e-12)).item()
        ),
    }


def _mapping_delta_norm(
    before: Mapping[str, Tensor], after: Mapping[str, Tensor]
) -> float:
    difference = torch.cat(
        [
            (after[name] - before[name]).detach().double().reshape(-1)
            for name in before
        ]
    )
    return float(difference.norm().item())


def _gradient_vector(block: PaperMACBlock, segment: Tensor) -> Tensor:
    pieces = [segment.grad.detach().double().reshape(-1)]
    pieces.extend(
        parameter.grad.detach().double().reshape(-1)
        for parameter in block.parameters()
        if parameter.grad is not None
    )
    return torch.cat(pieces)


def _run_backend_with_gradients(
    template: PaperMACBlock,
    segment_values: Tensor,
    config: StageBBackendConfig,
    stream_id: str,
) -> tuple[PaperMACBlock, Tensor, object, Tensor]:
    block = copy.deepcopy(template)
    segment = segment_values.detach().clone().requires_grad_(True)
    output = execute_stage_b(
        block,
        block.initial_state(stream_id),
        segment,
        config=config,
    )
    loss = output.sequence.square().mean() + sum(
        value.square().mean() for value in output.state.fast_weights.values()
    )
    loss.backward()
    return block, segment, output, _gradient_vector(block, segment)


def _timed_backend(
    template: PaperMACBlock,
    segment: Tensor,
    config: StageBBackendConfig,
    *,
    warmup_runs: int,
    repetitions: int,
) -> dict[str, object]:
    for repetition in range(warmup_runs):
        block = copy.deepcopy(template)
        execute_stage_b(
            block,
            block.initial_state(f"warmup-{repetition}"),
            segment,
            config=config,
        )
    samples: list[float] = []
    python_peaks: list[int] = []
    cuda_peaks: list[int] = []
    for repetition in range(repetitions):
        block = copy.deepcopy(template)
        if segment.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(segment.device)
        tracemalloc.start()
        start = time.perf_counter()
        execute_stage_b(
            block,
            block.initial_state(f"timing-{repetition}"),
            segment,
            config=config,
        )
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append(elapsed)
        python_peaks.append(peak)
        if segment.device.type == "cuda":
            cuda_peaks.append(torch.cuda.max_memory_allocated(segment.device))
    mean_seconds = mean(samples)
    return {
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "samples_seconds": samples,
        "mean_seconds": mean_seconds,
        "tokens_per_second": 32 / mean_seconds,
        "python_tracemalloc_peak_bytes": max(python_peaks),
        "cuda_peak_allocated_bytes": max(cuda_peaks) if cuda_peaks else None,
        "cpu_tensor_memory_limitation": (
            "tracemalloc covers Python allocations, not native PyTorch tensor storage"
        ),
    }


def _gate_summary(block: PaperMACBlock, sequence: Tensor) -> dict[str, object]:
    gates = block.memory.gates(sequence.detach())
    return {
        name: {
            "mean": float(value.mean().item()),
            "minimum": float(value.min().item()),
            "maximum": float(value.max().item()),
        }
        for name, value in (
            ("alpha", gates.alpha),
            ("eta", gates.eta),
            ("theta", gates.theta),
        )
    }


def _run_approximate_schedule(
    *,
    model: _PaperBenchmarkModel,
    segments: Sequence[SyntheticSegment],
    window_size: int,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[
    list[float],
    list[bool],
    dict[str, list[bool]],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Stage A matched protocol with only the memory updater substituted."""

    vocabulary_size = model.vocabulary_size
    states: dict[str, PaperMACStreamState] = {}
    losses: list[float] = []
    correctness: list[bool] = []
    task_correctness: dict[str, list[bool]] = {
        "delayed_recall": [],
        "overwrite": [],
        "boundary_reset": [],
    }
    gradients: list[float] = []
    alphas: list[float] = []
    etas: list[float] = []
    thetas: list[float] = []
    update_norms: list[float] = []

    for segment in segments:
        state = states.setdefault(
            segment.stream_id, model.memory.initial_state(segment.stream_id)
        )
        if segment.reset:
            state = model.memory.reset_state(state)
        embeddings, valid_mask = _encode_segment(segment, vocabulary_size)
        retrieval = model.memory.read_segment(state, embeddings)
        candidate = update_segment_with_stale_windows(
            model.memory,
            state,
            embeddings,
            window_size=window_size,
            valid_mask=valid_mask,
        )

        if segment.target_token is not None:
            if segment.reset:
                features = torch.cat(
                    (
                        torch.zeros(vocabulary_size),
                        F.one_hot(
                            torch.tensor(DEFAULT_VOCABULARY.no_memory),
                            num_classes=vocabulary_size,
                        ).float(),
                    )
                )
            else:
                assert segment.query_position is not None
                features = retrieval[segment.query_position].detach()
            logits = model(features.unsqueeze(0))
            target = torch.tensor([segment.target_token])
            loss = F.cross_entropy(logits, target)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                gradients.append(_global_gradient_norm(model.readout.parameters()))
                optimizer.step()
            losses.append(float(loss.detach()))
            correct = int(logits.argmax(dim=-1).item()) == segment.target_token
            correctness.append(correct)
            task_correctness[segment.task].append(correct)

        if bool(valid_mask.any()):
            gates = model.memory.gates(embeddings[valid_mask])
            alphas.extend(float(value) for value in gates.alpha.detach().flatten())
            etas.extend(float(value) for value in gates.eta.detach().flatten())
            thetas.extend(float(value) for value in gates.theta.detach().flatten())
            update = torch.stack(
                [
                    (candidate.fast_weights[name] - state.fast_weights[name])
                    .detach()
                    .square()
                    .sum()
                    for name in state.fast_weights
                ]
            ).sum().sqrt()
            update_norms.append(float(update))
        states[segment.stream_id] = candidate
    return (
        losses,
        correctness,
        task_correctness,
        gradients,
        alphas,
        etas,
        thetas,
        update_norms,
    )


def _synthetic_window_metrics(
    window_size: int,
    config: BenchmarkConfig,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    train_schedules = [
        _fixture_schedule(
            build_stage_a_fixtures(seed=seed, num_streams=config.num_streams)
        )
        for seed in config.train_seeds
    ]
    evaluation_fixtures = build_stage_a_fixtures(
        seed=config.evaluation_seed, num_streams=config.num_streams
    )
    evaluation_schedule = _fixture_schedule(evaluation_fixtures)
    model = _PaperBenchmarkModel(DEFAULT_VOCABULARY.size)
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=config.learning_rate)
    train_losses: list[float] = []
    train_correct: list[bool] = []
    gradient_norms: list[float] = []
    alphas: list[float] = []
    etas: list[float] = []
    thetas: list[float] = []
    update_norms: list[float] = []
    for _ in range(config.train_epochs):
        for schedule in train_schedules:
            outcome = _run_approximate_schedule(
                model=model,
                segments=schedule,
                window_size=window_size,
                train=True,
                optimizer=optimizer,
            )
            train_losses.extend(outcome[0])
            train_correct.extend(outcome[1])
            gradient_norms.extend(outcome[3])
            alphas.extend(outcome[4])
            etas.extend(outcome[5])
            thetas.extend(outcome[6])
            update_norms.extend(outcome[7])
    evaluation = _run_approximate_schedule(
        model=model,
        segments=evaluation_schedule,
        window_size=window_size,
        train=False,
        optimizer=None,
    )
    delayed_scores: dict[str, list[bool]] = {}
    for query, correct in zip(
        evaluation_fixtures["delayed_recall"].query_segments(),
        evaluation[2]["delayed_recall"],
    ):
        delayed_scores.setdefault(_delay_bucket(query), []).append(correct)
    train_accuracy = _accuracy(train_correct)
    evaluation_accuracy = _accuracy(evaluation[1])
    return {
        "window_size": window_size,
        "staleness": "within-window gradients use incoming fast-weight snapshot",
        "train_loss": mean(train_losses),
        "train_bpb": mean(train_losses) / math.log(2),
        "evaluation_loss": mean(evaluation[0]),
        "evaluation_bpb": mean(evaluation[0]) / math.log(2),
        "train_accuracy": train_accuracy,
        "evaluation_accuracy": evaluation_accuracy,
        "train_eval_gap": train_accuracy - evaluation_accuracy,
        "delayed_accuracy_by_delay": {
            bucket: _accuracy(scores) for bucket, scores in delayed_scores.items()
        },
        "overwrite_correctness": _accuracy(evaluation[2]["overwrite"]),
        "reset_correctness": _accuracy(evaluation[2]["boundary_reset"]),
        "gradient_norm": ScalarSummary.from_values(gradient_norms).__dict__,
        "alpha": ScalarSummary.from_values(alphas).__dict__,
        "eta": ScalarSummary.from_values(etas).__dict__,
        "theta": ScalarSummary.from_values(thetas).__dict__,
        "memory_update_norm": ScalarSummary.from_values(update_norms).__dict__,
    }


def run_approximate_scan_study(
    *,
    seed: int = 20260736,
    warmup_runs: int = 1,
    repetitions: int = 3,
    synthetic_config: BenchmarkConfig = BenchmarkConfig(),
    stage_a_artifact: Path | str = Path(
        "artifacts/titans_stage_a/stage_a_benchmark.json"
    ),
) -> dict[str, object]:
    """Measure dense-update drift and matched synthetic task behavior."""

    torch.manual_seed(seed)
    template = PaperMACBlock(
        d_model=8, num_heads=2, persistent_tokens=3, memory_depth=1
    ).float()
    segment = torch.randn(32, 8)
    reference_config = StageBBackendConfig()
    exact_config = StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED)
    reference_block, _, reference_output, reference_gradient = _run_backend_with_gradients(
        template, segment, reference_config, "reference"
    )
    _, _, exact_output, exact_gradient = _run_backend_with_gradients(
        template, segment, exact_config, "exact"
    )
    reference_timing = _timed_backend(
        template,
        segment,
        reference_config,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )
    exact_timing = _timed_backend(
        template,
        segment,
        exact_config,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )
    exact_difference = exact_gradient - reference_gradient
    mechanism: dict[str, object] = {
        "reference": {
            "timing": reference_timing,
            "memory_update_norm": _mapping_delta_norm(
                template.initial_state("initial").fast_weights,
                reference_output.state.fast_weights,
            ),
        },
        "exact_accelerated": {
            "fast_weights": _tensor_mapping_metrics(
                reference_output.state.fast_weights,
                exact_output.state.fast_weights,
            ),
            "surprise": _tensor_mapping_metrics(
                reference_output.state.surprise,
                exact_output.state.surprise,
            ),
            "gradient_relative_l2_error": float(
                exact_difference.norm()
                .div(reference_gradient.norm().clamp_min(1e-12))
                .item()
            ),
            "gradient_cosine": float(
                F.cosine_similarity(exact_gradient, reference_gradient, dim=0).item()
            ),
            "timing": exact_timing,
        },
        "approximate_windows": {},
    }
    approximate_results: dict[str, object] = {}
    for window in APPROXIMATE_WINDOWS:
        config = StageBBackendConfig(
            memory_backend=MemoryBackend.APPROXIMATE_SCAN,
            approximate_window=window,
        )
        block, _, output, gradient = _run_backend_with_gradients(
            template, segment, config, f"approximate-{window}"
        )
        difference = gradient - reference_gradient
        timing = _timed_backend(
            template,
            segment,
            config,
            warmup_runs=warmup_runs,
            repetitions=repetitions,
        )
        approximate_results[str(window)] = {
            "backend": "approximate_scan",
            "window_size": window,
            "staleness": "within-window gradients use incoming fast-weight snapshot",
            "dtype": str(segment.dtype).removeprefix("torch."),
            "seed": seed,
            "sequence_maximum_absolute_error": float(
                (output.sequence - reference_output.sequence).abs().max().item()
            ),
            "fast_weights": _tensor_mapping_metrics(
                reference_output.state.fast_weights, output.state.fast_weights
            ),
            "surprise": _tensor_mapping_metrics(
                reference_output.state.surprise, output.state.surprise
            ),
            "gradient_relative_l2_error": float(
                difference.norm().div(reference_gradient.norm().clamp_min(1e-12)).item()
            ),
            "gradient_cosine": float(
                F.cosine_similarity(gradient, reference_gradient, dim=0).item()
            ),
            "gate_statistics": _gate_summary(block, output.sequence),
            "memory_update_norm": _mapping_delta_norm(
                template.initial_state("initial").fast_weights,
                output.state.fast_weights,
            ),
            "timing": timing,
            "speedup_over_reference": (
                float(timing["tokens_per_second"])
                / float(reference_timing["tokens_per_second"])
            ),
        }
    mechanism["approximate_windows"] = approximate_results

    stage_a_path = Path(stage_a_artifact)
    stage_a_payload = json.loads(stage_a_path.read_text(encoding="utf-8"))
    synthetic_windows = {
        str(window): _synthetic_window_metrics(window, synthetic_config)
        for window in APPROXIMATE_WINDOWS
    }
    fastest_window = max(
        APPROXIMATE_WINDOWS,
        key=lambda window: float(
            approximate_results[str(window)]["speedup_over_reference"]
        ),
    )
    return {
        "format_version": 1,
        "classification": "experimental_approximation_not_parity_equivalent",
        "protocol": {
            "seed": seed,
            "dtype": "float32",
            "device": str(segment.device),
            "dense_mechanism_geometry": {
                "d_model": 8,
                "num_heads": 2,
                "persistent_tokens": 3,
                "memory_depth": 1,
                "segment_length": 32,
            },
            "windows": list(APPROXIMATE_WINDOWS),
            "warmup_runs": warmup_runs,
            "repetitions": repetitions,
            "synthetic_config": {
                "seed": synthetic_config.seed,
                "train_seeds": list(synthetic_config.train_seeds),
                "evaluation_seed": synthetic_config.evaluation_seed,
                "num_streams": synthetic_config.num_streams,
                "train_epochs": synthetic_config.train_epochs,
                "learning_rate": synthetic_config.learning_rate,
            },
        },
        "dense_mechanism": mechanism,
        "synthetic_task_comparison": {
            "reference": stage_a_payload["variants"]["adaptive"],
            "exact_accelerated": {
                **stage_a_payload["variants"]["adaptive"],
                "provenance": "B3 tensor-exact backend; same task semantics as reference",
            },
            "frozen_memory": stage_a_payload["variants"]["frozen_memory"],
            "no_memory": stage_a_payload["variants"]["no_memory"],
            "approximate_windows": synthetic_windows,
            "sparse_fixture_limitation": (
                "the controlled Stage A fixture has at most one valid memory write in each segment, "
                "so within-window staleness may be inactive; dense mechanism drift is decisive"
            ),
        },
        "decision": {
            "recommended_default": "reference",
            "approximate_scan_status": "experimental_only",
            "fastest_observed_window": fastest_window,
            "fastest_observed_speedup": approximate_results[str(fastest_window)][
                "speedup_over_reference"
            ],
            "promotion_allowed": False,
            "reason": (
                "every supported window changes dense-update state/gradients and no approximation "
                "may be described as parity-equivalent; task controls do not exercise dense writes"
            ),
        },
    }


def _render_speed_fidelity_svg(result: Mapping[str, object]) -> str:
    windows = result["dense_mechanism"]["approximate_windows"]
    assert isinstance(windows, Mapping)
    width, height = 620, 320
    left, top, plot_width, plot_height = 80, 45, 480, 220
    points = []
    for name, raw in windows.items():
        assert isinstance(raw, Mapping)
        speedup = float(raw["speedup_over_reference"])
        error = float(raw["fast_weights"]["relative_l2_error"])
        points.append((name, speedup, error))
    max_speed = max(1.05, max(point[1] for point in points) * 1.1)
    max_error = max(1e-6, max(point[2] for point in points) * 1.1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="sans-serif" font-size="16">B5 stale-window speed–fidelity (lower error, higher speedup)</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<text x="{left + 160}" y="{height - 15}" font-family="sans-serif" font-size="13">speedup over reference</text>',
        f'<text x="8" y="{top + 10}" font-family="sans-serif" font-size="12">relative</text>',
        f'<text x="8" y="{top + 25}" font-family="sans-serif" font-size="12">state error</text>',
    ]
    for name, speedup, error in points:
        x = left + (speedup / max_speed) * plot_width
        y = top + plot_height - (error / max_error) * plot_height
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#7c3aed"/>')
        lines.append(
            f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" font-family="sans-serif" font-size="12">w={name}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_approximate_scan_study(
    result: Mapping[str, object], output_directory: Path | str
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output / "b5_approximate_scan_study.json",
        "report": output / "b5_approximate_scan_study.md",
        "plot": output / "b5_approximate_scan_speed_fidelity.svg",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mechanism = result["dense_mechanism"]
    tasks = result["synthetic_task_comparison"]
    decision = result["decision"]
    assert isinstance(mechanism, Mapping)
    assert isinstance(tasks, Mapping)
    assert isinstance(decision, Mapping)
    windows = mechanism["approximate_windows"]
    task_windows = tasks["approximate_windows"]
    assert isinstance(windows, Mapping) and isinstance(task_windows, Mapping)
    lines = [
        "# B5 approximate-scan speed–fidelity study",
        "",
        "> Classification: experimental approximation; never parity-equivalent.",
        "",
        "| Window | Speedup | State rel L2 | Surprise rel L2 | Gradient rel L2 | Gradient cosine | Delayed >32 | Eval BPB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in APPROXIMATE_WINDOWS:
        dense = windows[str(window)]
        task = task_windows[str(window)]
        assert isinstance(dense, Mapping) and isinstance(task, Mapping)
        lines.append(
            f"| {window} | {dense['speedup_over_reference']:.3f} | "
            f"{dense['fast_weights']['relative_l2_error']:.3e} | "
            f"{dense['surprise']['relative_l2_error']:.3e} | "
            f"{dense['gradient_relative_l2_error']:.3e} | "
            f"{dense['gradient_cosine']:.6f} | "
            f"{task['delayed_accuracy_by_delay'].get('>32', 0.0):.3f} | "
            f"{task['evaluation_bpb']:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            f"- Default: `{decision['recommended_default']}`.",
            f"- Approximate scan: `{decision['approximate_scan_status']}`.",
            f"- Promotion allowed: `{decision['promotion_allowed']}`.",
            f"- Reason: {decision['reason']}",
            "",
            f"Fixture limitation: {tasks['sparse_fixture_limitation']}",
            "",
            "The JSON includes raw timing samples, peak-memory telemetry, gates/update statistics, reference/exact/control metrics, and complete reproducibility metadata.",
            "",
        )
    )
    paths["report"].write_text("\n".join(lines), encoding="utf-8")
    paths["plot"].write_text(_render_speed_fidelity_svg(result), encoding="utf-8")
    return paths
