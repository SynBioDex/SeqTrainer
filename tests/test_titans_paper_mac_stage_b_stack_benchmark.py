from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    MemoryBackend,
    StageBBackendConfig,
    StageBMACStack,
    StageBScale,
    run_exact_acceleration_matrix,
    write_exact_acceleration_matrix,
)


def test_multiblock_stack_is_tensor_exact_for_exact_functional_loop() -> None:
    torch.manual_seed(317)
    reference = StageBMACStack(2, 4, num_heads=2, persistent_tokens=2, memory_depth=1).double()
    exact = copy.deepcopy(reference)
    segment = torch.randn(32, 4, dtype=torch.float64)

    reference_output = reference(reference.initial_states("stack"), segment)
    exact_output = exact(
        exact.initial_states("stack"),
        segment,
        config=StageBBackendConfig(memory_backend=MemoryBackend.EXACT_ACCELERATED),
    )

    assert torch.equal(reference_output.sequence, exact_output.sequence)
    assert all(torch.equal(a, b) for a, b in zip(reference_output.retrievals, exact_output.retrievals))
    for expected_state, actual_state in zip(reference_output.states, exact_output.states):
        for name in expected_state.fast_weights:
            assert torch.equal(expected_state.fast_weights[name], actual_state.fast_weights[name])
            assert torch.equal(expected_state.surprise[name], actual_state.surprise[name])


def test_exact_acceleration_matrix_records_geometry_parity_and_artifacts(tmp_path) -> None:
    matrix = run_exact_acceleration_matrix(
        scales=(StageBScale("test", block_count=2, d_model=4, num_heads=2),),
        seed=319,
        warmup_runs=0,
        repetitions=1,
    )
    result = matrix.results[0]

    assert result.tensor_exact is True
    assert result.parameter_count is not None and result.parameter_count > 0
    assert result.reference.median_wall_time_seconds is not None
    assert result.exact_accelerated.median_wall_time_seconds is not None
    assert result.speedup is not None and result.speedup > 0
    assert result.exact_accelerated.runtime_metadata["memory"]["stale_gradients"] is False

    paths = write_exact_acceleration_matrix(matrix, tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["results"][0]["scale"]["block_count"] == 2
    assert "Speedup" in paths["markdown"].read_text(encoding="utf-8")


def test_a100_pilot_is_explicitly_unavailable_without_named_hardware() -> None:
    matrix = run_exact_acceleration_matrix(
        scales=(StageBScale("a100_pilot", block_count=8, d_model=384, num_heads=8),),
        warmup_runs=0,
        repetitions=1,
        device="cpu",
    )
    result = matrix.results[0]
    assert result.tensor_exact is None
    assert result.reference.available is False
    assert result.exact_accelerated.available is False
    assert "A100" in result.reference.reason
