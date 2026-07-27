from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import (  # noqa: E402
    AdaptiveUpdateGates,
    FunctionalNeuralMemory,
    PaperResidualMemory,
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


def test_legacy_gate_honors_explicit_matched_initial_conditions() -> None:
    gates = AdaptiveUpdateGates(
        3,
        theta_max=1.0,
        alpha_initial=1e-3,
        eta_initial=0.9,
        theta_initial=1e-3,
    ).double()
    with torch.no_grad():
        gates.projection.weight.zero_()

    values = gates(torch.zeros(4, 3, dtype=torch.float64))

    assert torch.allclose(values.alpha, torch.full((4, 1), 1e-3, dtype=torch.float64))
    assert torch.allclose(values.eta, torch.full((4, 1), 0.9, dtype=torch.float64))
    assert torch.allclose(values.theta, torch.full((4, 1), 1e-3, dtype=torch.float64))


def test_relative_rms_conditioning_has_finite_higher_order_gradient_at_zero_state() -> None:
    memory = FunctionalNeuralMemory(
        d_model=2,
        memory_depth=1,
        max_gradient_rms_ratio=10.0,
    ).double()
    weights = type(memory.initial_fast_weights())(
        (name, torch.zeros_like(value, requires_grad=True))
        for name, value in memory.initial_fast_weights().items()
    )
    gradient = type(weights)(
        (name, torch.ones_like(value, requires_grad=True)) for name, value in weights.items()
    )

    conditioned = memory.condition_gradient(weights, gradient)
    torch.stack([value.sum() for value in conditioned.values()]).sum().backward()

    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in weights.values()
    )


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


def test_paper_residual_memory_matches_declared_two_layer_equation() -> None:
    """The v2 memory is x + LN(Wout GELU(Win x)), not a linear-depth alias."""

    torch.manual_seed(23)
    module = PaperResidualMemory(d_model=3, expansion_factor=4).double()
    inputs = torch.randn(5, 3, dtype=torch.float64)

    expected = inputs + module.normalization(
        module.out_projection(torch.nn.functional.gelu(module.in_projection(inputs)))
    )

    assert torch.equal(module(inputs), expected)
    assert module.in_projection.out_features == 12
    assert module.out_projection.in_features == 12


def test_paper_channel_gates_and_dual_recurrence_match_fp64_equations() -> None:
    torch.manual_seed(29)
    memory = FunctionalNeuralMemory(
        d_model=2,
        memory_depth=2,
        architecture="paper_residual_mlp_v2",
        expansion_factor=4,
        projection_convolution_kernel=4,
        normalize_queries_and_keys=True,
        max_surprise_norm=None,
        associative_loss_reduction="sum",
        theta_max=1.0,
        theta_initial=1e-3,
        alpha_initial=1e-3,
        eta_initial=0.9,
    ).double()
    state = memory.initial_state("paper-equations")
    key = torch.tensor([0.25, -0.5], dtype=torch.float64)
    value = torch.tensor([-0.75, 0.125], dtype=torch.float64)
    raw = memory.surprise_gradient(state.fast_weights, key, value)
    tokens = torch.zeros(1, 2, dtype=torch.float64)
    gates = memory.gates(tokens)
    alpha = type(state.fast_weights)((name, gate[0]) for name, gate in gates.alpha.items())
    eta = type(state.fast_weights)((name, gate[0]) for name, gate in gates.eta.items())
    theta = type(state.fast_weights)((name, gate[0]) for name, gate in gates.theta.items())

    updated = memory.update_one(
        state,
        key,
        value,
        alpha=alpha,
        eta=eta,
        theta=theta,
    )

    for name in state.fast_weights:
        expected_surprise = eta[name] * state.surprise[name] - theta[name] * raw[name]
        expected_weight = (1.0 - alpha[name]) * state.fast_weights[name] + expected_surprise
        assert torch.allclose(updated.surprise[name], expected_surprise, atol=1e-12, rtol=0)
        assert torch.allclose(updated.fast_weights[name], expected_weight, atol=1e-12, rtol=0)
        assert alpha[name].shape in {
            state.fast_weights[name].shape[:1],
            (*state.fast_weights[name].shape[:1], 1),
        }


def test_causal_kernel_four_projection_carries_exact_stream_history() -> None:
    torch.manual_seed(31)
    memory = FunctionalNeuralMemory(
        d_model=3,
        memory_depth=2,
        architecture="paper_residual_mlp_v2",
        projection_convolution_kernel=4,
        normalize_queries_and_keys=True,
        alpha_initial=1e-3,
        eta_initial=0.9,
        theta_initial=1e-3,
    ).double()
    with torch.no_grad():
        memory.query_convolution.weight.copy_(
            torch.randn_like(memory.query_convolution.weight)
        )
    first = torch.randn(32, 3, dtype=torch.float64)
    second = torch.randn(32, 3, dtype=torch.float64)
    state = memory.initial_state("causal-history")

    first_queries, first_history = memory.project_queries(state, first)
    advanced = memory.advance_query_history(state, first_history)
    second_queries, _ = memory.project_queries(advanced, second)
    combined = torch.cat((first, second))
    projected = memory.query_projection(combined).transpose(0, 1).unsqueeze(0)
    expected = memory.query_convolution(
        torch.nn.functional.pad(projected, (3, 0))
    ).squeeze(0).transpose(0, 1)
    expected = memory._l2_normalize(expected)

    assert torch.allclose(first_queries, expected[:32], atol=1e-12, rtol=0)
    assert torch.allclose(second_queries, expected[32:], atol=1e-12, rtol=0)


def test_dual_surprise_telemetry_separates_past_and_momentary_terms() -> None:
    memory = FunctionalNeuralMemory(d_model=2, memory_depth=1).double()
    _set_tiny_parameters(memory)
    state = memory.initial_state("telemetry")
    previous = type(state.surprise)(
        (name, torch.full_like(value, 0.25)) for name, value in state.surprise.items()
    )
    state = state.replace(fast_weights=state.fast_weights, surprise=previous)
    memory.begin_update_telemetry(state.fast_weights)

    memory.update_one(
        state,
        torch.tensor([0.5, -1.0], dtype=torch.float64),
        torch.tensor([-0.25, 0.75], dtype=torch.float64),
        alpha=torch.tensor([0.1], dtype=torch.float64),
        eta=torch.tensor([0.6], dtype=torch.float64),
        theta=torch.tensor([0.2], dtype=torch.float64),
    )

    telemetry = memory.update_telemetry()
    assert float(telemetry["past_surprise_rms_max"]) > 0
    assert float(telemetry["momentary_surprise_rms_max"]) > 0
    assert float(telemetry["combined_surprise_rms_max"]) > 0
    assert float(telemetry["forgotten_weight_rms_max"]) > 0
    assert -1.0 <= float(telemetry["past_momentary_cosine_sum"]) <= 1.0
