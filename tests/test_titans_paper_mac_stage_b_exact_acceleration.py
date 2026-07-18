from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    ExactAcceleratedMemoryBackend,
    MemoryBackend,
    StageBBackendConfig,
    StageBBackendRegistry,
    compare_backends,
    execute_stage_b,
)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_exact_accelerated_backend_matches_outputs_state_surprise_and_gradients(dtype) -> None:
    torch.manual_seed(307)
    reference = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=2).to(dtype=dtype)
    candidate = copy.deepcopy(reference)
    segment = torch.randn(32, 4, dtype=dtype)

    report = compare_backends(
        reference,
        candidate,
        segment,
        candidate_config=StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED),
    )

    assert report.passed
    assert report.sequence.exact
    assert report.retrieval.exact
    assert report.input_gradient.exact
    assert all(item.exact for item in report.fast_weights.values())
    assert all(item.exact for item in report.surprise.values())
    assert all(item.exact for item in report.trainable_gradients.values())


def test_exact_acceleration_falls_back_to_reference_when_support_probe_fails() -> None:
    torch.manual_seed(311)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    direct = copy.deepcopy(block)
    segment = torch.randn(32, 4, dtype=torch.float64)
    backend = ExactAcceleratedMemoryBackend(support_predicate=lambda _: False)
    registry = StageBBackendRegistry()
    registry.register_memory(
        MemoryBackend.EXACT_ACCELERATED,
        backend,
        exactness="reference_fallback",
        reason="forced unsupported test",
    )

    expected = direct(direct.initial_state("fallback"), segment)
    actual = execute_stage_b(
        block,
        block.initial_state("fallback"),
        segment,
        config=StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED),
        registry=registry,
    )

    assert torch.equal(actual.sequence, expected.sequence)
    assert torch.equal(actual.retrieval, expected.retrieval)
    assert backend.last_execution == "reference_fallback"
    assert backend.fallback_calls == 1
    for name in expected.state.fast_weights:
        assert torch.equal(actual.state.fast_weights[name], expected.state.fast_weights[name])
        assert torch.equal(actual.state.surprise[name], expected.state.surprise[name])


def test_exact_acceleration_preserves_tail_mask_and_one_published_state() -> None:
    torch.manual_seed(313)
    reference = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    candidate = copy.deepcopy(reference)
    segment = torch.randn(32, 4, dtype=torch.float64)
    valid_mask = torch.zeros(32, dtype=torch.bool)
    valid_mask[:19] = True

    expected = reference(
        reference.initial_state("tail"),
        segment,
        valid_mask=valid_mask,
    )
    actual = execute_stage_b(
        candidate,
        candidate.initial_state("tail"),
        segment,
        valid_mask=valid_mask,
        config=StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED),
    )

    assert actual.state.segment_index == 1
    for name in expected.state.fast_weights:
        assert torch.equal(actual.state.fast_weights[name], expected.state.fast_weights[name])
        assert torch.equal(actual.state.surprise[name], expected.state.surprise[name])
