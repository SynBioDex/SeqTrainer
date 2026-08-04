"""Deterministic Stage A synthetic-memory benchmark and executable gates.

This is a correctness probe, not a biology or DNA-performance benchmark.  The
three variants share the same functional MLP memory, vocabulary readout,
optimizer, seeds, token budget, fixture construction, and evaluation data.
They differ only in whether their returned fast-memory state may be used:

* ``adaptive`` commits the differentiable associative update;
* ``frozen_memory`` reads the initial MLP but never commits an update; and
* ``no_memory`` uses only the current query representation.

The explicit controls make the value of an updateable state observable on
delayed recall, where each evaluation mapping is held out from training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Iterable, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lifecycle import StreamLifecycleHarness
from .mac import PaperMACBlock, block_causal_attention_mask
from .memory import FunctionalNeuralMemory
from .state import PaperMACStreamState
from .synthetic import (
    DEFAULT_VOCABULARY,
    SyntheticSegment,
    SyntheticTaskFixture,
    build_stage_a_fixtures,
)


VariantName = Literal["adaptive", "frozen_memory", "no_memory"]
VARIANTS: tuple[VariantName, ...] = ("adaptive", "frozen_memory", "no_memory")


@dataclass(frozen=True)
class BenchmarkConfig:
    """A complete, reproducible synthetic benchmark protocol."""

    seed: int = 20260727
    train_seeds: tuple[int, ...] = (20260727, 20260737, 20260747)
    evaluation_seed: int = 20260827
    num_streams: int = 8
    train_epochs: int = 3
    learning_rate: float = 0.05
    substantial_margin: float = 0.50

    def __post_init__(self) -> None:
        if self.num_streams <= 0 or self.num_streams > len(DEFAULT_VOCABULARY.keys):
            raise ValueError("num_streams must be between 1 and the synthetic vocabulary capacity")
        if not self.train_seeds:
            raise ValueError("train_seeds must not be empty")
        if self.train_epochs <= 0:
            raise ValueError("train_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.substantial_margin <= 1:
            raise ValueError("substantial_margin must be in (0, 1]")


@dataclass(frozen=True)
class ScalarSummary:
    """Compact JSON-safe aggregate for per-step diagnostics."""

    mean: float
    minimum: float
    maximum: float
    count: int

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "ScalarSummary":
        if not values:
            return cls(mean=0.0, minimum=0.0, maximum=0.0, count=0)
        return cls(mean=float(mean(values)), minimum=float(min(values)), maximum=float(max(values)), count=len(values))


@dataclass(frozen=True)
class VariantMetrics:
    """Train/evaluation metrics for one matched memory condition."""

    name: VariantName
    train_loss: float
    train_bpb: float
    evaluation_loss: float
    evaluation_bpb: float
    train_accuracy: float
    evaluation_accuracy: float
    train_eval_gap: float
    delayed_accuracy_by_delay: Mapping[str, float]
    overwrite_correctness: float
    reset_correctness: float
    gradient_norm: ScalarSummary
    alpha: ScalarSummary
    eta: ScalarSummary
    theta: ScalarSummary
    memory_update_norm: ScalarSummary


@dataclass(frozen=True)
class AcceptanceGates:
    """Executable Stage A acceptance decisions."""

    adaptive_beats_controls_beyond_32: bool
    adaptive_lifecycle_tasks_correct: bool
    lifecycle_reference_passed: bool
    leakage_reference_passed: bool
    mask_reference_passed: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())


@dataclass(frozen=True)
class BenchmarkResults:
    """Machine-readable benchmark output with protocol and gate provenance."""

    format_version: int
    protocol: Mapping[str, object]
    variants: Mapping[str, VariantMetrics]
    gates: AcceptanceGates

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "protocol": dict(self.protocol),
            "variants": {name: asdict(metrics) for name, metrics in self.variants.items()},
            "gates": {**asdict(self.gates), "passed": self.gates.passed},
        }


class _PaperBenchmarkModel(nn.Module):
    """Identical neural-memory and readout capacity for every condition.

    The controlled projections encode a synthetic observation as
    ``[one_hot(key), one_hot(value)]``.  The functional MLP is a real Stage A
    neural memory: its associative gradient writes the value subspace under
    the key-subspace query.  Only the outer readout is optimized in this short
    correctness benchmark; full meta-gradient behavior is covered separately
    by the memory and MAC tests.
    """

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.d_model = 2 * vocabulary_size
        self.memory = FunctionalNeuralMemory(d_model=self.d_model, memory_depth=1)
        self.readout = nn.Linear(self.d_model, vocabulary_size)
        with torch.no_grad():
            memory_layer = self.memory.memory_mlp[0]
            memory_layer.weight.zero_()
            memory_layer.bias.zero_()
            self.memory.key_projection.weight.zero_()
            self.memory.value_projection.weight.zero_()
            self.memory.query_projection.weight.zero_()
            identity = torch.eye(vocabulary_size)
            self.memory.key_projection.weight[:vocabulary_size, :vocabulary_size].copy_(identity)
            self.memory.query_projection.weight[:vocabulary_size, :vocabulary_size].copy_(identity)
            self.memory.value_projection.weight[vocabulary_size:, vocabulary_size:].copy_(identity)
            self.memory.gates.projection.weight.zero_()
            self.memory.gates.projection.bias.copy_(
                torch.logit(torch.tensor([1e-4, 1e-2, 1.0 - 1e-4]))
            )
            self.readout.weight.zero_()
            self.readout.bias.zero_()
            for token in range(vocabulary_size):
                self.readout.weight[token, vocabulary_size + token] = 1.0

    def forward(self, features: Tensor) -> Tensor:
        return self.readout(features)


def _one_hot(token: int, vocabulary_size: int) -> Tensor:
    return F.one_hot(torch.tensor(token), num_classes=vocabulary_size).to(dtype=torch.float32)


def _write_pairs(segment: SyntheticSegment) -> Iterable[tuple[int, int, int]]:
    tokens = segment.tokens
    for index in range(len(tokens) - 1):
        key, value = tokens[index], tokens[index + 1]
        if key in DEFAULT_VOCABULARY.keys and value in DEFAULT_VOCABULARY.values:
            yield index, key, value


def _key_value_embedding(key: int, value: int, vocabulary_size: int) -> Tensor:
    return torch.cat((_one_hot(key, vocabulary_size), _one_hot(value, vocabulary_size)))


def _query_embedding(key: int, vocabulary_size: int) -> Tensor:
    return torch.cat((_one_hot(key, vocabulary_size), torch.zeros(vocabulary_size)))


def _value_embedding(value: int, vocabulary_size: int) -> Tensor:
    return torch.cat((torch.zeros(vocabulary_size), _one_hot(value, vocabulary_size)))


def _encode_segment(segment: SyntheticSegment, vocabulary_size: int) -> tuple[Tensor, Tensor]:
    """Encode exact synthetic K/V observations into the paper-memory input."""

    embeddings = torch.zeros(32, 2 * vocabulary_size)
    valid_mask = torch.zeros(32, dtype=torch.bool)
    for position, key, value in _write_pairs(segment):
        embeddings[position] = _key_value_embedding(key, value, vocabulary_size)
        valid_mask[position] = True
    if segment.query_position is not None:
        embeddings[segment.query_position] = _query_embedding(segment.tokens[1], vocabulary_size)
    return embeddings, valid_mask


def _fixture_schedule(fixtures: Mapping[str, SyntheticTaskFixture]) -> tuple[SyntheticSegment, ...]:
    """Use one deterministic interleaved schedule per task without reordering streams."""

    return tuple(segment for fixture in fixtures.values() for segment in fixture.round_robin_interleave())


def _delay_bucket(segment: SyntheticSegment) -> str:
    assert segment.query_position is not None
    assert segment.source_segment_index is not None
    # Stored values are at position 1 in the deterministic delayed fixture.
    delay = segment.segment_index * 32 + segment.query_position - (segment.source_segment_index * 32 + 1)
    return ">32" if delay > 32 else f"{delay}"


def _global_gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squares = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(squares).sum().sqrt()) if squares else 0.0


def _run_schedule(
    *,
    variant: VariantName,
    model: _PaperBenchmarkModel,
    segments: Sequence[SyntheticSegment],
    train: bool,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[list[float], list[bool], dict[str, list[bool]], list[float], list[float], list[float], list[float], list[float]]:
    """Run one schedule, preserving stream-local pre-read/post-write timing."""

    vocabulary_size = model.vocabulary_size
    states: dict[str, PaperMACStreamState] = {}
    losses: list[float] = []
    correctness: list[bool] = []
    task_correctness: dict[str, list[bool]] = {"delayed_recall": [], "overwrite": [], "boundary_reset": []}
    gradients: list[float] = []
    alphas: list[float] = []
    etas: list[float] = []
    thetas: list[float] = []
    update_norms: list[float] = []

    for segment in segments:
        state = states.setdefault(segment.stream_id, model.memory.initial_state(segment.stream_id))
        if segment.reset:
            state = model.memory.reset_state(state)

        embeddings, valid_mask = _encode_segment(segment, vocabulary_size)
        retrieval = model.memory.read_segment(state, embeddings)
        candidate = model.memory.update_segment(state, embeddings, valid_mask=valid_mask)

        if segment.target_token is not None:
            key = segment.tokens[1]
            if segment.reset:
                # All conditions can obey explicit lifecycle resets; this is
                # deliberately not a memory-advantage shortcut.
                features = _value_embedding(DEFAULT_VOCABULARY.no_memory, vocabulary_size)
            elif variant == "adaptive":
                assert segment.query_position is not None
                features = retrieval[segment.query_position].detach()
            elif variant == "frozen_memory":
                assert segment.query_position is not None
                features = retrieval[segment.query_position].detach()
            else:
                features = _query_embedding(key, vocabulary_size)
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
            if variant == "adaptive":
                update_norm = torch.stack(
                    [
                        (candidate.fast_weights[name] - state.fast_weights[name]).detach().square().sum()
                        for name in state.fast_weights
                    ]
                ).sum().sqrt()
                update_norms.append(float(update_norm))
            else:
                update_norms.append(0.0)

        if variant == "adaptive":
            states[segment.stream_id] = candidate
        else:
            states[segment.stream_id] = state.replace(
                fast_weights=state.fast_weights,
                surprise=state.surprise,
                segment_index=state.segment_index + 1,
            )
    return losses, correctness, task_correctness, gradients, alphas, etas, thetas, update_norms


def _accuracy(values: Sequence[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _reference_checks() -> dict[str, bool]:
    """Run small executable lifecycle, leakage, and mask reference probes."""

    mask = block_causal_attention_mask(2)
    mask_ok = bool(mask[2, 2] == 0 and mask[2, 3] == 1 and mask[34, 35] == 1)

    torch.manual_seed(20260727)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).eval()
    retrieval = torch.randn(32, 4)
    sequence = torch.randn(32, 4)
    baseline = block.integrate(retrieval, sequence)
    changed = sequence.clone()
    changed[7] += 10.0
    leakage_ok = bool(torch.equal(baseline[:7], block.integrate(retrieval, changed)[:7]))

    @dataclass(frozen=True)
    class ReferenceState:
        stream_id: str
        resets: int = 0
        ended: bool = False

    harness = StreamLifecycleHarness[ReferenceState](
        initial_state=ReferenceState,
        update_state=lambda state, segment: state,
        reset_state=lambda state: ReferenceState(state.stream_id, resets=state.resets + 1),
        end_state=lambda state: ReferenceState(state.stream_id, resets=state.resets, ended=True),
        serialize_state=lambda state: asdict(state),
        deserialize_state=lambda payload: ReferenceState(**payload),  # type: ignore[arg-type]
    )
    reset_fixture = build_stage_a_fixtures(seed=20260727, num_streams=1)["boundary_reset"]
    for segment in reset_fixture.round_robin_interleave():
        harness.process(segment)
    reference_state = next(iter(harness.states.values()))
    lifecycle_ok = reference_state.resets == 1 and reference_state.ended
    return {"mask": mask_ok, "leakage": leakage_ok, "lifecycle": lifecycle_ok}


def run_stage_a_benchmark(config: BenchmarkConfig = BenchmarkConfig()) -> BenchmarkResults:
    """Train and evaluate the matched three-way Stage A synthetic comparison."""

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_schedules = [
        _fixture_schedule(build_stage_a_fixtures(seed=seed, num_streams=config.num_streams)) for seed in config.train_seeds
    ]
    evaluation_fixtures = build_stage_a_fixtures(seed=config.evaluation_seed, num_streams=config.num_streams)
    evaluation_schedule = _fixture_schedule(evaluation_fixtures)
    metrics: dict[str, VariantMetrics] = {}

    for variant in VARIANTS:
        torch.manual_seed(config.seed)
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
                outcome = _run_schedule(
                    variant=variant,
                    model=model,
                    segments=schedule,
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

        evaluation = _run_schedule(
            variant=variant,
            model=model,
            segments=evaluation_schedule,
            train=False,
            optimizer=None,
        )
        eval_losses, eval_correct, eval_tasks = evaluation[:3]
        delayed_queries = evaluation_fixtures["delayed_recall"].query_segments()
        delayed_scores = eval_tasks["delayed_recall"]
        delay_scores: dict[str, list[bool]] = {}
        for query, correct in zip(delayed_queries, delayed_scores):
            delay_scores.setdefault(_delay_bucket(query), []).append(correct)

        train_accuracy = _accuracy(train_correct)
        evaluation_accuracy = _accuracy(eval_correct)
        metrics[variant] = VariantMetrics(
            name=variant,
            train_loss=float(mean(train_losses)),
            train_bpb=float(mean(train_losses) / math.log(2)),
            evaluation_loss=float(mean(eval_losses)),
            evaluation_bpb=float(mean(eval_losses) / math.log(2)),
            train_accuracy=train_accuracy,
            evaluation_accuracy=evaluation_accuracy,
            train_eval_gap=train_accuracy - evaluation_accuracy,
            delayed_accuracy_by_delay={bucket: _accuracy(scores) for bucket, scores in delay_scores.items()},
            overwrite_correctness=_accuracy(eval_tasks["overwrite"]),
            reset_correctness=_accuracy(eval_tasks["boundary_reset"]),
            gradient_norm=ScalarSummary.from_values(gradient_norms),
            alpha=ScalarSummary.from_values(alphas),
            eta=ScalarSummary.from_values(etas),
            theta=ScalarSummary.from_values(thetas),
            memory_update_norm=ScalarSummary.from_values(update_norms),
        )

    reference = _reference_checks()
    adaptive_beyond_32 = metrics["adaptive"].delayed_accuracy_by_delay.get(">32", 0.0)
    control_beyond_32 = max(
        metrics["frozen_memory"].delayed_accuracy_by_delay.get(">32", 0.0),
        metrics["no_memory"].delayed_accuracy_by_delay.get(">32", 0.0),
    )
    gates = AcceptanceGates(
        adaptive_beats_controls_beyond_32=adaptive_beyond_32 - control_beyond_32 >= config.substantial_margin,
        adaptive_lifecycle_tasks_correct=(
            metrics["adaptive"].overwrite_correctness == 1.0 and metrics["adaptive"].reset_correctness == 1.0
        ),
        lifecycle_reference_passed=reference["lifecycle"],
        leakage_reference_passed=reference["leakage"],
        mask_reference_passed=reference["mask"],
    )
    token_budget = sum(len(schedule) * 32 for schedule in train_schedules) * config.train_epochs
    reference_model = _PaperBenchmarkModel(DEFAULT_VOCABULARY.size)
    return BenchmarkResults(
        format_version=1,
        protocol={
            "synthetic_only": True,
            "variants": list(VARIANTS),
            "shared_outer_parameter_count": sum(parameter.numel() for parameter in reference_model.readout.parameters()),
            "shared_total_parameter_count": sum(parameter.numel() for parameter in reference_model.parameters()),
            "memory_implementation": "FunctionalNeuralMemory",
            "memory_depth": 1,
            "optimizer": "AdamW",
            "optimizer_parameters": "readout",
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "train_seeds": list(config.train_seeds),
            "evaluation_seed": config.evaluation_seed,
            "train_epochs": config.train_epochs,
            "tokens_per_variant": token_budget,
            "evaluation_tokens_per_variant": len(evaluation_schedule) * 32,
            "num_streams": config.num_streams,
            "evaluation_data": "held-out deterministic fixture seed",
        },
        variants=metrics,
        gates=gates,
    )


def render_markdown_report(results: BenchmarkResults) -> str:
    """Render a dependency-free human-readable benchmark report."""

    rows = [
        "# Stage A synthetic-memory benchmark",
        "",
        "This is a deterministic synthetic correctness probe, not a biology or DNA-performance result.",
        "",
        "| Variant | Delayed >32 accuracy | Overwrite | Reset | Eval BPB | Train/eval gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in VARIANTS:
        metric = results.variants[name]
        rows.append(
            f"| {name} | {metric.delayed_accuracy_by_delay.get('>32', 0.0):.3f} | "
            f"{metric.overwrite_correctness:.3f} | {metric.reset_correctness:.3f} | "
            f"{metric.evaluation_bpb:.3f} | {metric.train_eval_gap:.3f} |"
        )
    rows.extend(("", "## Acceptance gates", ""))
    for name, passed in asdict(results.gates).items():
        rows.append(f"- {'PASS' if passed else 'FAIL'}: {name}")
    rows.append(f"- {'PASS' if results.gates.passed else 'FAIL'}: overall")
    return "\n".join(rows) + "\n"


def render_svg_plot(results: BenchmarkResults) -> str:
    """Render a tiny dependency-free SVG plot of delayed-recall accuracy."""

    width, height, left, bar_width = 560, 180, 185, 320
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="16">Delayed recall accuracy (&gt;32 tokens)</text>',
    ]
    colors = {"adaptive": "#18794e", "frozen_memory": "#6b7280", "no_memory": "#9ca3af"}
    for index, name in enumerate(VARIANTS):
        accuracy = results.variants[name].delayed_accuracy_by_delay.get(">32", 0.0)
        y = 48 + index * 38
        rows.append(f'<text x="20" y="{y + 17}" font-family="sans-serif" font-size="13">{name}</text>')
        rows.append(f'<rect x="{left}" y="{y}" width="{bar_width}" height="22" fill="#e5e7eb"/>')
        rows.append(f'<rect x="{left}" y="{y}" width="{bar_width * accuracy:.1f}" height="22" fill="{colors[name]}"/>')
        rows.append(f'<text x="{left + bar_width + 8}" y="{y + 17}" font-family="sans-serif" font-size="13">{accuracy:.3f}</text>')
    rows.append("</svg>")
    return "\n".join(rows) + "\n"


def write_benchmark_artifacts(results: BenchmarkResults, output_directory: Path) -> Mapping[str, Path]:
    """Write JSON, Markdown, and SVG artifacts and return their exact paths."""

    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_directory / "stage_a_benchmark.json",
        "report": output_directory / "stage_a_benchmark.md",
        "plot": output_directory / "stage_a_delayed_recall.svg",
    }
    paths["json"].write_text(json.dumps(results.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["report"].write_text(render_markdown_report(results), encoding="utf-8")
    paths["plot"].write_text(render_svg_plot(results), encoding="utf-8")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic smoke benchmark and write portable artifacts."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/titans_stage_a"))
    parser.add_argument("--epochs", type=int, default=BenchmarkConfig().train_epochs)
    parser.add_argument("--seed", type=int, default=BenchmarkConfig().seed)
    args = parser.parse_args(argv)
    config = BenchmarkConfig(
        seed=args.seed,
        train_seeds=(args.seed, args.seed + 10, args.seed + 20),
        evaluation_seed=args.seed + 100,
        train_epochs=args.epochs,
    )
    results = run_stage_a_benchmark(config)
    paths = write_benchmark_artifacts(results, args.output_dir)
    print(json.dumps({"gates_passed": results.gates.passed, "artifacts": {name: str(path) for name, path in paths.items()}}, sort_keys=True))
    return 0 if results.gates.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())
