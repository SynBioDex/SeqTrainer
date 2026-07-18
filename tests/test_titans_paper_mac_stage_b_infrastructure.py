from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    ActivationDType,
    AttentionBackend,
    BackendUnavailableError,
    MemoryBackend,
    StageBBackendConfig,
    StageBBackendRegistry,
    benchmark_stage_b,
    compare_backends,
    execute_stage_b,
    write_stage_b_artifacts,
)


def test_reference_is_default_and_unreviewed_modes_are_rejected() -> None:
    config = StageBBackendConfig()
    registry = StageBBackendRegistry()

    assert config.memory_backend is MemoryBackend.REFERENCE
    assert config.attention_backend is AttentionBackend.MULTIHEAD_ATTENTION
    assert config.activation_dtype is ActivationDType.FP32
    assert registry.memory_capabilities()["reference"]["available"] is True
    assert registry.memory_capabilities()["exact_accelerated"]["available"] is True
    assert registry.memory_capabilities()["exact_scan"]["available"] is False
    assert registry.memory_capabilities()["approximate_scan"]["available"] is False

    registry.validate(StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED))
    for backend in (MemoryBackend.EXACT_SCAN, MemoryBackend.APPROXIMATE_SCAN):
        kwargs = {"memory_backend": backend}
        if backend is MemoryBackend.APPROXIMATE_SCAN:
            kwargs["approximate_window"] = 4
        with pytest.raises(BackendUnavailableError):
            registry.validate(StageBBackendConfig(**kwargs))
    with pytest.raises(BackendUnavailableError):
        registry.validate(StageBBackendConfig(attention_backend=AttentionBackend.SDPA))
    with pytest.raises(BackendUnavailableError):
        registry.validate(StageBBackendConfig(activation_dtype=ActivationDType.BF16))


def test_default_dispatch_is_tensor_exact_with_direct_stage_a_forward() -> None:
    torch.manual_seed(211)
    direct = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    dispatched = copy.deepcopy(direct)
    segment = torch.randn(32, 4, dtype=torch.float64)

    expected = direct(direct.initial_state("stream"), segment)
    actual = execute_stage_b(dispatched, dispatched.initial_state("stream"), segment)

    assert torch.equal(actual.sequence, expected.sequence)
    assert torch.equal(actual.retrieval, expected.retrieval)
    for name in expected.state.fast_weights:
        assert torch.equal(actual.state.fast_weights[name], expected.state.fast_weights[name])
        assert torch.equal(actual.state.surprise[name], expected.state.surprise[name])


def test_parity_report_includes_output_state_surprise_and_trainable_gradients() -> None:
    torch.manual_seed(223)
    reference = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    candidate = copy.deepcopy(reference)
    segment = torch.randn(32, 4, dtype=torch.float64)

    report = compare_backends(reference, candidate, segment)

    assert report.passed
    assert report.sequence.exact and report.retrieval.exact
    assert report.input_gradient.exact
    assert report.fast_weights and all(item.exact for item in report.fast_weights.values())
    assert report.surprise and all(item.exact for item in report.surprise.values())
    assert report.trainable_gradients
    assert all(item.exact for item in report.trainable_gradients.values())
    assert "persistent_tokens" in report.trainable_gradients
    assert "attention.in_proj_weight" in report.trainable_gradients
    assert "memory.gates.projection.weight" in report.trainable_gradients


def test_cpu_telemetry_degrades_safely_and_writes_json_markdown(tmp_path) -> None:
    torch.manual_seed(227)
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    segments = [torch.randn(32, 4, dtype=torch.float64)]

    result = benchmark_stage_b(block, segments, warmup_runs=0, repetitions=1, seed=227)
    paths = write_stage_b_artifacts(result, tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert result.config["memory_backend"] == "reference"
    assert result.segment_count == 1
    assert result.timing.token_count == 32
    assert result.timing.tokens_per_second > 0
    assert result.state_payload_bytes > 0
    assert result.hardware.device == "cpu"
    assert result.hardware.cuda_allocated_bytes is None
    assert result.hardware.cuda_reserved_bytes is None
    assert payload["hardware"]["cuda_allocated_bytes"] is None
    assert "unavailable (non-CUDA execution)" in paths["markdown"].read_text(encoding="utf-8")
