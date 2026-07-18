from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    BackendUnavailableError,
    MemoryBackend,
    StageBBackendConfig,
    StageBBackendRegistry,
    run_scan_feasibility_harness,
    write_scan_feasibility_artifact,
)


def test_exact_scan_is_restricted_to_affine_gradient_form_and_stays_unavailable(tmp_path) -> None:
    result = run_scan_feasibility_harness(seed=20260727)

    assert result.classification == "restricted_only"
    assert result.exact_scan_available is False
    assert result.linear_output_max_abs_error < 1e-12
    assert result.linear_state_max_abs_error < 1e-12
    assert result.linear_surprise_max_abs_error < 1e-12
    assert result.linear_gradient_max_abs_error < 1e-12
    assert result.affine_associativity_max_abs_error < 1e-12
    assert result.nonlinear_stale_output_max_abs_error > 1e-10
    assert result.nonlinear_stale_state_max_abs_error > 1e-10
    assert result.nonlinear_stale_surprise_max_abs_error > 1e-10
    assert result.nonlinear_stale_gradient_max_abs_error > 1e-10

    artifact = write_scan_feasibility_artifact(result, tmp_path / "scan.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["classification"] == "restricted_only"
    assert payload["exact_scan_available"] is False

    with pytest.raises(BackendUnavailableError):
        StageBBackendRegistry().validate(
            StageBBackendConfig(memory_backend=MemoryBackend.EXACT_SCAN)
        )


def test_scan_feasibility_harness_is_deterministic() -> None:
    assert run_scan_feasibility_harness(17).to_dict() == run_scan_feasibility_harness(17).to_dict()

