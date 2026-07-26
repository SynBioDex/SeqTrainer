from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import (  # noqa: E402
    AdaptiveUpdateGates,
    FunctionalNeuralMemory,
    PaperMACStreamState,
)


def _set_tiny_parameters(memory: FunctionalNeuralMemory) -> None:
    """Set a deterministic linear memory and identity key/value projections."""

    with torch.no_grad():
        layer = memory.memory_mlp[0]
        layer.weight.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4]], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        memory.key_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        memory.value_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        memory.query_projection.weight.copy_(torch.eye(2, dtype=torch.float64))
        memory.gates.projection.weight.zero_()
        memory.gates.projection.bias.copy_(torch.logit(torch.tensor([0.1, 0.6, 0.2], dtype=torch.float64)))


def test_fp64_one_step_matches_hand_calculated_equations() -> None:
    torch.manual_seed(7)
    memory = FunctionalNeuralMemory(d_model=2, memory_depth=1).double()
    _set_tiny_parameters(memory)
    state = memory.initial_state("stream-a")
    key = torch.tensor([0.5, -1.0], dtype=torch.float64)
    value = torch.tensor([-0.25, 0.75], dtype=torch.float64)
    alpha = torch.tensor([0.1], dtype=torch.float64)
    eta = torch.tensor([0.6], dtype=torch.float64)
    theta = torch.tensor([0.2], dtype=torch.float64)

    updated = memory.update_one(state, key, value, alpha=alpha, eta=eta, theta=theta)

    weight = state.fast_weights["0.weight"]
    bias = state.fast_weights["0.bias"]
    prediction = weight @ key + bias
    residual = prediction - value
    hand_gradient = {
        "0.weight": torch.outer(residual, key),
        "0.bias": residual,
    }
    hand_surprise = {name: -theta * gradient for name, gradient in hand_gradient.items()}
    hand_weights = {name: (1.0 - alpha) * state.fast_weights[name] + hand_surprise[name] for name in hand_gradient}

    for name in hand_gradient:
        assert torch.allclose(updated.surprise[name], hand_surprise[name], atol=1e-12, rtol=0)
        assert torch.allclose(updated.fast_weights[name], hand_weights[name], atol=1e-12, rtol=0)
    assert updated.segment_index == 0
    assert state.segment_index == 0


def test_optional_surprise_trust_region_bounds_pathological_local_writes() -> None:
    """A rare associative-gradient spike must not become an unbounded state."""

    memory = FunctionalNeuralMemory(
        d_model=2,
        memory_depth=1,
        max_surprise_norm=0.25,
    ).double()
    _set_tiny_parameters(memory)
    state = memory.initial_state("bounded-stream")
    updated = memory.update_one(
        state,
        torch.tensor([50.0, -50.0], dtype=torch.float64),
        torch.tensor([-75.0, 75.0], dtype=torch.float64),
        alpha=torch.tensor([0.1], dtype=torch.float64),
        eta=torch.tensor([0.6], dtype=torch.float64),
        theta=torch.tensor([0.2], dtype=torch.float64),
    )

    surprise_norm = torch.stack(
        [value.square().sum() for value in updated.surprise.values()]
    ).sum().sqrt()
    assert float(surprise_norm.detach()) == pytest.approx(0.25)
    assert all(torch.isfinite(value).all() for value in updated.fast_weights.values())


def test_mean_associative_loss_is_exact_dimension_normalization() -> None:
    summed = FunctionalNeuralMemory(d_model=2, memory_depth=1).double()
    averaged = FunctionalNeuralMemory(
        d_model=2,
        memory_depth=1,
        associative_loss_reduction="mean",
    ).double()
    averaged.load_state_dict(summed.state_dict())
    weights = summed.initial_fast_weights()
    keys = torch.tensor([[0.5, -1.0]], dtype=torch.float64)
    values = torch.tensor([[-0.25, 0.75]], dtype=torch.float64)

    sum_loss = summed.associative_loss(weights, keys, values)
    mean_loss = averaged.associative_loss(weights, keys, values)

    assert torch.equal(mean_loss * summed.d_model, sum_loss)


def test_relative_gradient_rms_conditioning_matches_declared_equation() -> None:
    memory = FunctionalNeuralMemory(
        d_model=2,
        memory_depth=1,
        max_gradient_rms_ratio=0.25,
    ).double()
    weights = memory.initial_fast_weights()
    gradient = type(weights)(
        (name, torch.full_like(value, 100.0)) for name, value in weights.items()
    )
    memory.begin_update_telemetry(weights)

    conditioned = memory.condition_gradient(weights, gradient)

    raw_rms = memory._mapping_rms(gradient)
    admissible = 0.25 * memory._mapping_rms(weights)
    expected_scale = admissible / raw_rms
    for name in gradient:
        assert torch.allclose(conditioned[name], gradient[name] * expected_scale)
    telemetry = memory.update_telemetry()
    assert float(telemetry["gradient_scale_min"]) == pytest.approx(
        float(expected_scale.detach())
    )
    assert float(telemetry["gradient_interventions"]) == 1.0


def test_theta_gate_has_declared_maximum_and_initial_value() -> None:
    gates = AdaptiveUpdateGates(
        3,
        theta_max=0.5,
        theta_initial=0.25,
    ).double()
    with torch.no_grad():
        gates.projection.weight.zero_()

    values = gates(torch.zeros(4, 3, dtype=torch.float64))

    assert torch.allclose(values.theta, torch.full((4, 1), 0.25, dtype=torch.float64))
    assert bool(values.theta.max() < 0.5)


def test_segment_shapes_read_snapshot_and_stream_isolation() -> None:
    torch.manual_seed(11)
    memory = FunctionalNeuralMemory(d_model=4, memory_depth=2).double()
    first = memory.initial_state("first")
    second = memory.initial_state("second")
    segment = torch.randn(32, 4, dtype=torch.float64)

    expected_read = memory.retrieve(first, memory.query_projection(segment))
    read, updated_first = memory.read_then_update(first, segment)

    assert read.shape == (32, 4)
    assert torch.equal(read, expected_read)
    assert updated_first.segment_index == 1
    assert second.segment_index == 0
    for name in first.fast_weights:
        assert torch.equal(first.fast_weights[name], second.fast_weights[name])
        assert torch.equal(first.fast_weights[name], memory.initial_fast_weights()[name])
        assert not torch.equal(updated_first.fast_weights[name], second.fast_weights[name])


def test_gradients_reach_gate_parameters_through_all_32_updates() -> None:
    torch.manual_seed(13)
    memory = FunctionalNeuralMemory(d_model=3, memory_depth=1).double()
    state = memory.initial_state("gradient-stream")
    segment = torch.randn(32, 3, dtype=torch.float64, requires_grad=True)

    updated = memory.update_segment(state, segment)
    meta_loss = sum(value.square().sum() for value in updated.fast_weights.values())
    meta_loss.backward()

    gate_gradient = memory.gates.projection.weight.grad
    assert gate_gradient is not None
    assert torch.isfinite(gate_gradient).all()
    assert gate_gradient.abs().sum() > 0
    assert segment.grad is not None
    assert torch.isfinite(segment.grad).all()
    assert torch.all(segment.grad.abs().sum(dim=-1) > 0)
    assert updated.segment_index == 1


def test_state_serialization_is_lossless_and_reset_is_stream_local() -> None:
    torch.manual_seed(17)
    memory = FunctionalNeuralMemory(d_model=3, memory_depth=1).double()
    first = memory.update_segment(memory.initial_state("first"), torch.randn(32, 3, dtype=torch.float64))
    second = memory.update_segment(memory.initial_state("second"), torch.randn(32, 3, dtype=torch.float64))
    second_before_reset = {name: value.clone() for name, value in second.fast_weights.items()}

    payload = first.to_state_dict()
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    try:
        loaded = torch.load(buffer, weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        buffer.seek(0)
        loaded = torch.load(buffer)
    restored = PaperMACStreamState.from_state_dict(loaded)

    assert restored.stream_id == first.stream_id
    assert restored.segment_index == first.segment_index
    assert restored.reset_count == first.reset_count
    for name in first.fast_weights:
        assert restored.fast_weights[name].dtype == torch.float64
        assert torch.equal(restored.fast_weights[name], first.fast_weights[name])
        assert torch.equal(restored.surprise[name], first.surprise[name])

    reset_first = memory.reset_state(first)
    assert reset_first.stream_id == "first"
    assert reset_first.segment_index == 0
    assert reset_first.reset_count == first.reset_count + 1
    for name in second.fast_weights:
        assert torch.equal(second.fast_weights[name], second_before_reset[name])
