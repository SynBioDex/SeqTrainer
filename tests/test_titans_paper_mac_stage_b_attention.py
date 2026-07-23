from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    ActivationDType,
    AttentionBackend,
    BackendUnavailableError,
    StageBBackendConfig,
    StageBBackendRegistry,
    execute_stage_b,
    probe_flash_mask_support,
    run_attention_backend_evidence,
    sdpa_allowed_attention_mask,
    write_attention_backend_evidence,
)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sdpa_matches_reference_forward_backward_and_persistent_gradient(dtype) -> None:
    torch.manual_seed(503)
    reference = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).to(dtype=dtype)
    candidate = copy.deepcopy(reference)
    reference_segment = torch.randn(32, 4, dtype=dtype, requires_grad=True)
    candidate_segment = reference_segment.detach().clone().requires_grad_(True)

    reference_output = execute_stage_b(
        reference,
        reference.initial_state("reference"),
        reference_segment,
    )
    candidate_output = execute_stage_b(
        candidate,
        candidate.initial_state("candidate"),
        candidate_segment,
        config=StageBBackendConfig(attention_backend=AttentionBackend.SDPA),
    )
    reference_loss = reference_output.sequence.square().mean() + sum(
        value.square().mean() for value in reference_output.state.fast_weights.values()
    )
    candidate_loss = candidate_output.sequence.square().mean() + sum(
        value.square().mean() for value in candidate_output.state.fast_weights.values()
    )
    reference_loss.backward()
    candidate_loss.backward()

    tolerance = 2e-12 if dtype is torch.float64 else 2e-5
    torch.testing.assert_close(
        candidate_output.sequence,
        reference_output.sequence,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        candidate_segment.grad,
        reference_segment.grad,
        rtol=tolerance * 5,
        atol=tolerance * 5,
    )
    torch.testing.assert_close(
        candidate.persistent_tokens.grad,
        reference.persistent_tokens.grad,
        rtol=tolerance * 5,
        atol=tolerance * 5,
    )
    for name in ("in_proj_weight", "out_proj.weight"):
        reference_parameter = dict(reference.attention.named_parameters())[name]
        candidate_parameter = dict(candidate.attention.named_parameters())[name]
        torch.testing.assert_close(
            candidate_parameter.grad,
            reference_parameter.grad,
            rtol=tolerance * 5,
            atol=tolerance * 5,
        )
    for name in reference_output.state.fast_weights:
        torch.testing.assert_close(
            candidate_output.state.fast_weights[name],
            reference_output.state.fast_weights[name],
            rtol=tolerance * 5,
            atol=tolerance * 5,
        )
        torch.testing.assert_close(
            candidate_output.state.surprise[name],
            reference_output.state.surprise[name],
            rtol=tolerance * 5,
            atol=tolerance * 5,
        )


def test_sdpa_mask_is_exact_boolean_complement_for_every_edge() -> None:
    block = PaperMACBlock(d_model=4, num_heads=2, persistent_tokens=3, memory_depth=1)
    blocked = block.attention_mask()
    allowed = sdpa_allowed_attention_mask(block, device=torch.device("cpu"))

    assert allowed.dtype is torch.bool
    assert allowed.shape == blocked.shape == (67, 67)
    for query in range(67):
        for key in range(67):
            assert bool(allowed[query, key]) is (not bool(blocked[query, key]))


def test_cpu_sdpa_math_path_has_a_differentiable_backward_path() -> None:
    """Guard against PyTorch selecting a CPU Flash kernel without backward."""

    block = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).float()
    retrieval = torch.randn(32, 4)
    segment = torch.randn(32, 4, requires_grad=True)

    output = integrate_sdpa_attention(block, retrieval, segment)
    output.square().mean().backward()

    assert segment.grad is not None and torch.isfinite(segment.grad).all()
    assert block.attention.in_proj_weight.grad is not None
    assert torch.isfinite(block.attention.in_proj_weight.grad).all()


@pytest.mark.parametrize("changed_position", [1, 7, 16, 31])
def test_sdpa_future_perturbation_cannot_change_earlier_outputs(changed_position) -> None:
    torch.manual_seed(509 + changed_position)
    block = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).double().eval()
    changed_block = copy.deepcopy(block)
    segment = torch.randn(32, 4, dtype=torch.float64)
    changed = segment.clone()
    changed[changed_position] += 100.0
    config = StageBBackendConfig(attention_backend=AttentionBackend.SDPA)

    baseline = execute_stage_b(
        block, block.initial_state("base"), segment, config=config
    )
    perturbed = execute_stage_b(
        changed_block,
        changed_block.initial_state("changed"),
        changed,
        config=config,
    )

    torch.testing.assert_close(
        perturbed.sequence[:changed_position],
        baseline.sequence[:changed_position],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        perturbed.sequence[changed_position], baseline.sequence[changed_position]
    )


def test_mixed_precision_boundary_and_cpu_flash_fallback_are_explicit() -> None:
    registry = StageBBackendRegistry()
    block = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).float()
    segment = torch.randn(32, 4)
    bf16 = StageBBackendConfig(
        attention_backend=AttentionBackend.SDPA,
        activation_dtype=ActivationDType.BF16,
    )

    try:
        output = execute_stage_b(block, block.initial_state("bf16"), segment, config=bf16)
    except RuntimeError as error:
        assert "BFloat16" in str(error) or "bfloat16" in str(error).lower()
    else:
        assert output.sequence.dtype is torch.float32
        assert all(value.dtype is torch.float32 for value in output.state.fast_weights.values())
        assert all(value.dtype is torch.float32 for value in output.state.surprise.values())

    with pytest.raises(BackendUnavailableError):
        registry.validate(
            StageBBackendConfig(
                attention_backend=AttentionBackend.FLASH,
                activation_dtype=ActivationDType.FP16,
            )
        )
    probe = probe_flash_mask_support(block)
    assert probe["available"] is False
    assert "no CUDA" in probe["reason"]
    registered_probe = registry.probe_and_enable_flash(block)
    assert registered_probe["available"] is False
    assert registry.attention_capabilities()["flash"]["available"] is False
    assert "no CUDA" in registry.attention_capabilities()["flash"]["reason"]


def test_attention_evidence_separates_oracle_numerical_and_mixed_precision(tmp_path) -> None:
    result = run_attention_backend_evidence(seed=557, warmup_runs=0, repetitions=1)
    paths = write_attention_backend_evidence(result, tmp_path)

    assert result["mask"]["boolean_complement_mismatches"] == 0
    assert result["causality"]["passed"] is True
    assert result["parity"]["fp64_oracle"]["sequence"]["maximum_absolute_error"] < 1e-10
    assert result["parity"]["fp32_numerical"]["sequence"]["maximum_absolute_error"] < 1e-4
    assert "behavioral" in result["mixed_precision_behavioral"]["bfloat16"]["classification"]
    assert result["mixed_precision_behavioral"]["float16"]["available"] is False
    assert paths["json"].exists() and paths["report"].exists()
