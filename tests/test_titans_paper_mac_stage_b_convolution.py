from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    BackendUnavailableError,
    CausalConvolutionalUpdateGates,
    GateBackend,
    MemoryBackend,
    StageBBackendConfig,
    StageBMACStack,
    execute_stage_b,
    run_convolution_comparison,
    write_convolution_comparison,
)


def test_disabled_convolution_flag_recovers_direct_reference_exactly() -> None:
    torch.manual_seed(401)
    reference = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    candidate = copy.deepcopy(reference)
    segment = torch.randn(32, 4, dtype=torch.float64)

    expected = reference(reference.initial_state("disabled"), segment)
    actual = execute_stage_b(candidate, candidate.initial_state("disabled"), segment)

    assert torch.equal(actual.sequence, expected.sequence)
    assert torch.equal(actual.retrieval, expected.retrieval)
    for name in expected.state.fast_weights:
        assert torch.equal(actual.state.fast_weights[name], expected.state.fast_weights[name])
        assert torch.equal(actual.state.surprise[name], expected.state.surprise[name])


def test_causal_convolution_has_no_future_influence_on_earlier_gates() -> None:
    torch.manual_seed(409)
    gates = CausalConvolutionalUpdateGates(d_model=6, kernel_size=3).double()
    segment = torch.randn(32, 6, dtype=torch.float64)
    changed = segment.clone()
    changed[17] += 100.0

    baseline = gates(segment)
    perturbed = gates(changed)

    for baseline_gate, perturbed_gate in (
        (baseline.alpha, perturbed.alpha),
        (baseline.eta, perturbed.eta),
        (baseline.theta, perturbed.theta),
    ):
        torch.testing.assert_close(perturbed_gate[:17], baseline_gate[:17], rtol=0, atol=0)
        assert not torch.equal(perturbed_gate[17], baseline_gate[17])


def test_convolutional_gate_path_preserves_shapes_state_timing_and_gradients() -> None:
    torch.manual_seed(419)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    gates = CausalConvolutionalUpdateGates(
        d_model=4,
        kernel_size=3,
        reference_gates=block.memory.gates,
    ).double()
    state = block.initial_state("convolution")
    segment = torch.randn(32, 4, dtype=torch.float64, requires_grad=True)
    config = StageBBackendConfig(
        gate_backend=GateBackend.CAUSAL_CONVOLUTION,
        convolution_kernel_size=3,
    )

    output = execute_stage_b(
        block,
        state,
        segment,
        config=config,
        convolutional_gates=gates,
    )
    loss = output.sequence.square().mean()
    loss = loss + sum(value.square().mean() for value in output.state.fast_weights.values())
    loss.backward()

    assert output.sequence.shape == (32, 4)
    assert output.retrieval.shape == (32, 4)
    assert state.segment_index == 0
    assert output.state.segment_index == 1
    assert segment.grad is not None and torch.isfinite(segment.grad).all()
    assert gates.depthwise.weight.grad is not None
    assert gates.depthwise.weight.grad.abs().sum() > 0
    assert gates.projection.weight.grad is not None
    assert gates.projection.weight.grad.abs().sum() > 0


def test_future_token_still_cannot_change_earlier_current_segment_output() -> None:
    torch.manual_seed(421)
    baseline_block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double().eval()
    perturbed_block = copy.deepcopy(baseline_block)
    baseline_gates = CausalConvolutionalUpdateGates(4, 3, reference_gates=baseline_block.memory.gates).double()
    perturbed_gates = copy.deepcopy(baseline_gates)
    segment = torch.randn(32, 4, dtype=torch.float64)
    changed = segment.clone()
    changed[19] += 50.0
    config = StageBBackendConfig(
        gate_backend=GateBackend.CAUSAL_CONVOLUTION,
        convolution_kernel_size=3,
    )

    baseline = execute_stage_b(
        baseline_block,
        baseline_block.initial_state("baseline"),
        segment,
        config=config,
        convolutional_gates=baseline_gates,
    )
    perturbed = execute_stage_b(
        perturbed_block,
        perturbed_block.initial_state("perturbed"),
        changed,
        config=config,
        convolutional_gates=perturbed_gates,
    )

    torch.testing.assert_close(perturbed.sequence[:19], baseline.sequence[:19], rtol=0, atol=0)
    assert not torch.equal(perturbed.sequence[19], baseline.sequence[19])


def test_convolution_comparison_records_causality_gradients_and_stage_a_controls(
    tmp_path,
) -> None:
    stage_a = {
        "gates": {"passed": True},
        "variants": {
            name: {
                "delayed_accuracy_by_delay": {">32": delayed},
                "memory_update_norm": {"mean": update},
                "overwrite_correctness": overwrite,
                "reset_correctness": 1.0,
            }
            for name, delayed, update, overwrite in (
                ("adaptive", 1.0, 1.5, 1.0),
                ("frozen_memory", 0.125, 0.0, 0.125),
                ("no_memory", 0.25, 0.0, 0.25),
            )
        },
    }
    stage_a_path = tmp_path / "stage_a.json"
    stage_a_path.write_text(json.dumps(stage_a), encoding="utf-8")

    result = run_convolution_comparison(
        seed=431,
        d_model=4,
        num_heads=2,
        stage_a_artifact=stage_a_path,
    )
    paths = write_convolution_comparison(result, tmp_path / "out")

    assert result["paper_exact"] is False
    assert result["causality"]["passed"] is True
    assert result["gradients"]["depthwise_weight_norm"] > 0
    assert result["unchanged_stage_a_evidence"]["gates"]["passed"] is True
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["causality"]["passed"]
    assert "not claimed paper-exact" in paths["report"].read_text(encoding="utf-8")


def test_stack_owns_trainable_convolution_modules_and_rejects_unreviewed_pairing() -> None:
    torch.manual_seed(439)
    stack = StageBMACStack(
        block_count=2,
        d_model=4,
        num_heads=2,
        persistent_tokens=2,
        memory_depth=1,
        convolution_kernel_size=3,
    ).double()
    segment = torch.randn(32, 4, dtype=torch.float64)
    convolution_config = StageBBackendConfig(
        gate_backend=GateBackend.CAUSAL_CONVOLUTION,
        convolution_kernel_size=3,
    )

    output = stack(stack.initial_states("stack"), segment, config=convolution_config)
    sum(
        value.square().mean()
        for state in output.states
        for value in state.fast_weights.values()
    ).backward()

    assert stack.convolutional_gates is not None
    assert len(stack.convolutional_gates) == 2
    assert all(gate.depthwise.weight.grad is not None for gate in stack.convolutional_gates)

    unreviewed = StageBBackendConfig(
        memory_backend=MemoryBackend.EXACT_ACCELERATED,
        gate_backend=GateBackend.CAUSAL_CONVOLUTION,
        convolution_kernel_size=3,
    )
    with pytest.raises(BackendUnavailableError, match="reviewed only with the reference"):
        stack(stack.initial_states("unreviewed"), segment, config=unreviewed)
