"""Reproducible exact-scan feasibility harness for the Stage A recurrence."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from seqtrainer.torch.titans_paper_mac.memory import FunctionalNeuralMemory


@dataclass(frozen=True)
class AffineStateMap:
    """A map ``z -> transition @ z + bias`` for ``z = [M, S]``."""

    transition: Tensor
    bias: Tensor


@dataclass(frozen=True)
class ScanFeasibilityResult:
    seed: int
    dtype: str
    classification: str
    exact_scan_available: bool
    linear_output_max_abs_error: float
    linear_state_max_abs_error: float
    linear_surprise_max_abs_error: float
    linear_gradient_max_abs_error: float
    affine_associativity_max_abs_error: float
    nonlinear_stale_output_max_abs_error: float
    nonlinear_stale_state_max_abs_error: float
    nonlinear_stale_surprise_max_abs_error: float
    nonlinear_stale_gradient_max_abs_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compose_affine(later: AffineStateMap, earlier: AffineStateMap) -> AffineStateMap:
    """Compose two maps in temporal order using an associative operator."""

    return AffineStateMap(
        transition=later.transition @ earlier.transition,
        bias=later.transition @ earlier.bias + later.bias,
    )


def linear_token_map(
    key: Tensor,
    value: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
) -> AffineStateMap:
    """Return the exact affine map for scalar linear memory and squared loss."""

    gradient_weight = key.square()
    gradient_bias = -(value * key)
    transition = torch.stack(
        (
            torch.stack(((1.0 - alpha) - theta * gradient_weight, eta)),
            torch.stack((-theta * gradient_weight, eta)),
        )
    )
    bias_value = -theta * gradient_bias
    return AffineStateMap(transition=transition, bias=torch.stack((bias_value, bias_value)))


def sequential_linear_recurrence(
    keys: Tensor,
    values: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
    initial_weight: Tensor,
    initial_surprise: Tensor,
) -> tuple[Tensor, Tensor]:
    weight = initial_weight
    surprise = initial_surprise
    for index in range(keys.numel()):
        gradient = (weight * keys[index] - values[index]) * keys[index]
        surprise = eta[index] * surprise - theta[index] * gradient
        weight = (1.0 - alpha[index]) * weight + surprise
    return weight, surprise


def affine_scan_linear_recurrence(
    keys: Tensor,
    values: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
    initial_weight: Tensor,
    initial_surprise: Tensor,
) -> tuple[Tensor, Tensor]:
    identity = torch.eye(2, dtype=keys.dtype, device=keys.device)
    combined = AffineStateMap(
        transition=identity,
        bias=torch.zeros(2, dtype=keys.dtype, device=keys.device),
    )
    for index in range(keys.numel()):
        token_map = linear_token_map(
            keys[index], values[index], alpha[index], eta[index], theta[index]
        )
        combined = compose_affine(token_map, combined)
    state = combined.transition @ torch.stack((initial_weight, initial_surprise)) + combined.bias
    return state[0], state[1]


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _max_collection_error(left: Sequence[Tensor], right: Sequence[Tensor]) -> float:
    return max((_max_abs(a, b) for a, b in zip(left, right)), default=0.0)


def _linear_trial(seed: int) -> tuple[float, float, float, float, float]:
    generator = torch.Generator().manual_seed(seed)
    base = {
        "keys": torch.randn(4, generator=generator, dtype=torch.float64),
        "values": torch.randn(4, generator=generator, dtype=torch.float64),
        "alpha": torch.sigmoid(torch.randn(4, generator=generator, dtype=torch.float64)),
        "eta": torch.sigmoid(torch.randn(4, generator=generator, dtype=torch.float64)),
        "theta": torch.sigmoid(torch.randn(4, generator=generator, dtype=torch.float64)),
        "weight": torch.randn((), generator=generator, dtype=torch.float64),
        "surprise": torch.randn((), generator=generator, dtype=torch.float64),
    }

    def leaves() -> dict[str, Tensor]:
        return {name: value.detach().clone().requires_grad_(True) for name, value in base.items()}

    sequential_inputs = leaves()
    scan_inputs = leaves()
    sequential_weight, sequential_surprise = sequential_linear_recurrence(
        sequential_inputs["keys"],
        sequential_inputs["values"],
        sequential_inputs["alpha"],
        sequential_inputs["eta"],
        sequential_inputs["theta"],
        sequential_inputs["weight"],
        sequential_inputs["surprise"],
    )
    scan_weight, scan_surprise = affine_scan_linear_recurrence(
        scan_inputs["keys"],
        scan_inputs["values"],
        scan_inputs["alpha"],
        scan_inputs["eta"],
        scan_inputs["theta"],
        scan_inputs["weight"],
        scan_inputs["surprise"],
    )
    query = torch.tensor(0.37, dtype=torch.float64)
    sequential_output = sequential_weight * query
    scan_output = scan_weight * query
    sequential_objective = sequential_output.square() + sequential_surprise.square()
    scan_objective = scan_output.square() + scan_surprise.square()
    sequential_gradients = torch.autograd.grad(
        sequential_objective, tuple(sequential_inputs.values()), create_graph=False
    )
    scan_gradients = torch.autograd.grad(
        scan_objective, tuple(scan_inputs.values()), create_graph=False
    )

    maps = [
        linear_token_map(
            sequential_inputs["keys"][i],
            sequential_inputs["values"][i],
            sequential_inputs["alpha"][i],
            sequential_inputs["eta"][i],
            sequential_inputs["theta"][i],
        )
        for i in range(3)
    ]
    left_grouped = compose_affine(maps[2], compose_affine(maps[1], maps[0]))
    right_grouped = compose_affine(compose_affine(maps[2], maps[1]), maps[0])
    associativity_error = max(
        _max_abs(left_grouped.transition, right_grouped.transition),
        _max_abs(left_grouped.bias, right_grouped.bias),
    )
    return (
        _max_abs(sequential_output, scan_output),
        _max_abs(sequential_weight, scan_weight),
        _max_abs(sequential_surprise, scan_surprise),
        _max_collection_error(sequential_gradients, scan_gradients),
        associativity_error,
    )


def _run_nonlinear(
    memory: FunctionalNeuralMemory,
    keys: Tensor,
    values: Tensor,
    queries: Tensor,
    alpha: Tensor,
    eta: Tensor,
    theta: Tensor,
    *,
    stale_gradients: bool,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    state = memory.initial_state("scan-feasibility")
    frozen = state.fast_weights
    gradients = (
        [memory.surprise_gradient(frozen, keys[i], values[i]) for i in range(keys.shape[0])]
        if stale_gradients
        else None
    )
    for index in range(keys.shape[0]):
        gradient = (
            gradients[index]
            if gradients is not None
            else memory.surprise_gradient(state.fast_weights, keys[index], values[index])
        )
        surprise = memory.momentum_update(
            state.surprise, gradient, eta[index], theta[index]
        )
        fast_weights = memory.forgetting_update(state.fast_weights, surprise, alpha[index])
        state = state.replace(fast_weights=fast_weights, surprise=surprise)
    output = memory.retrieve(state, queries)
    objective_terms = [output.square().sum()]
    objective_terms.extend(value.square().mean() for value in state.fast_weights.values())
    objective_terms.extend(value.square().mean() for value in state.surprise.values())
    objective = torch.stack(objective_terms).sum()
    gradient_targets = (*memory.memory_mlp.parameters(), keys, values, queries)
    outer_gradients = torch.autograd.grad(objective, gradient_targets, allow_unused=False)
    return (
        output,
        tuple(state.fast_weights.values()),
        tuple(state.surprise.values()),
        outer_gradients,
    )


def _nonlinear_trial(seed: int) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    exact_memory = FunctionalNeuralMemory(d_model=2, memory_depth=2).double()
    stale_memory = copy.deepcopy(exact_memory)
    generator = torch.Generator().manual_seed(seed + 1)
    base_keys = torch.randn(3, 2, generator=generator, dtype=torch.float64)
    base_values = torch.randn(3, 2, generator=generator, dtype=torch.float64)
    base_queries = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    alpha = torch.tensor([0.1, 0.2, 0.15], dtype=torch.float64)
    eta = torch.tensor([0.7, 0.6, 0.8], dtype=torch.float64)
    theta = torch.tensor([0.3, 0.4, 0.2], dtype=torch.float64)

    exact = _run_nonlinear(
        exact_memory,
        base_keys.detach().clone().requires_grad_(True),
        base_values.detach().clone().requires_grad_(True),
        base_queries.detach().clone().requires_grad_(True),
        alpha,
        eta,
        theta,
        stale_gradients=False,
    )
    stale = _run_nonlinear(
        stale_memory,
        base_keys.detach().clone().requires_grad_(True),
        base_values.detach().clone().requires_grad_(True),
        base_queries.detach().clone().requires_grad_(True),
        alpha,
        eta,
        theta,
        stale_gradients=True,
    )
    return (
        _max_abs(exact[0], stale[0]),
        _max_collection_error(exact[1], stale[1]),
        _max_collection_error(exact[2], stale[2]),
        _max_collection_error(exact[3], stale[3]),
    )


def run_scan_feasibility_harness(seed: int = 20260727) -> ScanFeasibilityResult:
    """Prove the restricted affine case and exhibit nonlinear stale divergence."""

    linear = _linear_trial(seed)
    nonlinear = _nonlinear_trial(seed)
    return ScanFeasibilityResult(
        seed=seed,
        dtype="float64",
        classification="restricted_only",
        exact_scan_available=False,
        linear_output_max_abs_error=linear[0],
        linear_state_max_abs_error=linear[1],
        linear_surprise_max_abs_error=linear[2],
        linear_gradient_max_abs_error=linear[3],
        affine_associativity_max_abs_error=linear[4],
        nonlinear_stale_output_max_abs_error=nonlinear[0],
        nonlinear_stale_state_max_abs_error=nonlinear[1],
        nonlinear_stale_surprise_max_abs_error=nonlinear[2],
        nonlinear_stale_gradient_max_abs_error=nonlinear[3],
    )


def write_scan_feasibility_artifact(
    result: ScanFeasibilityResult,
    path: Path | str,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
