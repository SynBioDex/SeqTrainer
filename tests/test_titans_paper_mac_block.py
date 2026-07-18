from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock, block_causal_attention_mask  # noqa: E402


def test_block_causal_mask_has_exact_persistent_and_prefix_visibility() -> None:
    persistent_count = 3
    mask = block_causal_attention_mask(persistent_count)
    retrieval_start = persistent_count
    sequence_start = persistent_count + 32

    assert mask.shape == (67, 67)
    for row in range(persistent_count):
        assert not mask[row, :persistent_count].any()
        assert mask[row, persistent_count:].all()

    for position in range(32):
        expected_keys = set(range(persistent_count))
        expected_keys.update(range(retrieval_start, retrieval_start + position + 1))
        expected_keys.update(range(sequence_start, sequence_start + position + 1))
        for row in (retrieval_start + position, sequence_start + position):
            actual_keys = set(torch.where(~mask[row])[0].tolist())
            assert actual_keys == expected_keys


def test_block_reads_a_full_prewrite_segment_then_commits_one_state() -> None:
    torch.manual_seed(101)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    state = block.initial_state("stream")
    segment = torch.randn(32, 4, dtype=torch.float64)

    expected_retrieval = block.memory.retrieve(state, block.memory.query_projection(segment))
    output = block(state, segment)
    expected_state = block.memory.update_segment(state, output.sequence)
    raw_input_state = block.memory.update_segment(state, segment)

    assert output.retrieval.shape == (32, 4)
    assert output.sequence.shape == (32, 4)
    assert torch.equal(output.retrieval, expected_retrieval)
    assert state.segment_index == 0
    assert output.state.segment_index == 1
    for name in state.fast_weights:
        assert torch.equal(output.state.fast_weights[name], expected_state.fast_weights[name])
    assert any(
        not torch.equal(output.state.fast_weights[name], raw_input_state.fast_weights[name])
        for name in state.fast_weights
    )


def test_future_retrieval_or_sequence_perturbation_cannot_change_earlier_outputs() -> None:
    torch.manual_seed(103)
    block = PaperMACBlock(d_model=8, num_heads=2, persistent_tokens=2).eval()
    retrieval = torch.randn(32, 8)
    sequence = torch.randn(32, 8)
    baseline = block.integrate(retrieval, sequence)
    future_position = 17

    changed_retrieval = retrieval.clone()
    changed_retrieval[future_position] += 100.0
    changed_sequence = sequence.clone()
    changed_sequence[future_position] -= 100.0

    retrieval_perturbed = block.integrate(changed_retrieval, sequence)
    sequence_perturbed = block.integrate(retrieval, changed_sequence)

    torch.testing.assert_close(retrieval_perturbed[:future_position], baseline[:future_position], rtol=0, atol=0)
    torch.testing.assert_close(sequence_perturbed[:future_position], baseline[:future_position], rtol=0, atol=0)
    assert not torch.equal(retrieval_perturbed[future_position], baseline[future_position])
    assert not torch.equal(sequence_perturbed[future_position], baseline[future_position])


def test_future_token_cannot_change_an_earlier_output_through_memory_or_attention() -> None:
    torch.manual_seed(107)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double().eval()
    segment = torch.randn(32, 4, dtype=torch.float64)
    future_position = 19
    changed = segment.clone()
    changed[future_position] += 50.0

    baseline = block(block.initial_state("baseline"), segment)
    perturbed = block(block.initial_state("perturbed"), changed)

    torch.testing.assert_close(perturbed.sequence[:future_position], baseline.sequence[:future_position], rtol=0, atol=0)
    assert not torch.equal(perturbed.sequence[future_position], baseline.sequence[future_position])


def test_block_preserves_autograd_through_attention_and_one_write() -> None:
    torch.manual_seed(109)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    segment = torch.randn(32, 4, dtype=torch.float64, requires_grad=True)

    output = block(block.initial_state("gradient-stream"), segment)
    loss = output.sequence.square().sum() + sum(value.square().sum() for value in output.state.fast_weights.values())
    loss.backward()

    assert segment.grad is not None and torch.isfinite(segment.grad).all()
    assert block.persistent_tokens.grad is not None and block.persistent_tokens.grad.abs().sum() > 0
    assert block.attention.in_proj_weight.grad is not None and block.attention.in_proj_weight.grad.abs().sum() > 0
    assert block.memory.gates.projection.weight.grad is not None
    assert block.memory.gates.projection.weight.grad.abs().sum() > 0
