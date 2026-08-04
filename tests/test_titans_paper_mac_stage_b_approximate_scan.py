from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import PaperMACBlock  # noqa: E402
from seqtrainer.torch.titans_paper_mac.benchmark import BenchmarkConfig  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    APPROXIMATE_WINDOWS,
    ApproximateScanMemoryBackend,
    MemoryBackend,
    StageBBackendConfig,
    StageBBackendRegistry,
    execute_stage_b,
    run_approximate_scan_study,
    write_approximate_scan_study,
)


def _config(window: int) -> StageBBackendConfig:
    return StageBBackendConfig(
        memory_backend=MemoryBackend.APPROXIMATE_SCAN,
        approximate_window=window,
    )


def test_approximate_scan_is_explicit_and_default_reference_is_unchanged() -> None:
    torch.manual_seed(601)
    direct = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).double()
    dispatched = copy.deepcopy(direct)
    segment = torch.randn(32, 4, dtype=torch.float64)

    expected = direct(direct.initial_state("default"), segment)
    actual = execute_stage_b(dispatched, dispatched.initial_state("default"), segment)

    assert torch.equal(actual.sequence, expected.sequence)
    assert torch.equal(actual.retrieval, expected.retrieval)
    assert StageBBackendConfig().memory_backend is MemoryBackend.REFERENCE
    for name in expected.state.fast_weights:
        assert torch.equal(actual.state.fast_weights[name], expected.state.fast_weights[name])
        assert torch.equal(actual.state.surprise[name], expected.state.surprise[name])


@pytest.mark.parametrize("window", APPROXIMATE_WINDOWS)
def test_stale_windows_are_deterministic_trainable_and_publish_one_state(window) -> None:
    torch.manual_seed(607)
    first = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).double()
    second = copy.deepcopy(first)
    first_segment = torch.randn(32, 4, dtype=torch.float64, requires_grad=True)
    second_segment = first_segment.detach().clone().requires_grad_(True)

    first_output = execute_stage_b(
        first, first.initial_state("same"), first_segment, config=_config(window)
    )
    second_output = execute_stage_b(
        second, second.initial_state("same"), second_segment, config=_config(window)
    )
    first_loss = sum(value.square().mean() for value in first_output.state.fast_weights.values())
    second_loss = sum(value.square().mean() for value in second_output.state.fast_weights.values())
    first_loss.backward()
    second_loss.backward()

    assert first_output.state.segment_index == 1
    assert first_output.state.stream_id == "same"
    assert first_segment.grad is not None and torch.isfinite(first_segment.grad).all()
    assert torch.equal(first_segment.grad, second_segment.grad)
    for name in first_output.state.fast_weights:
        assert torch.equal(
            first_output.state.fast_weights[name], second_output.state.fast_weights[name]
        )
        assert torch.equal(first_output.state.surprise[name], second_output.state.surprise[name])


@pytest.mark.parametrize("window", APPROXIMATE_WINDOWS)
def test_every_supported_stale_window_differs_from_exact_on_dense_updates(window) -> None:
    torch.manual_seed(613)
    exact = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).double()
    approximate = copy.deepcopy(exact)
    segment = torch.randn(32, 4, dtype=torch.float64)
    exact_output = execute_stage_b(exact, exact.initial_state("exact"), segment)
    approximate_output = execute_stage_b(
        approximate,
        approximate.initial_state("approximate"),
        segment,
        config=_config(window),
    )

    assert torch.equal(approximate_output.sequence, exact_output.sequence)
    assert any(
        not torch.equal(
            approximate_output.state.fast_weights[name],
            exact_output.state.fast_weights[name],
        )
        for name in exact_output.state.fast_weights
    )


def test_runtime_metadata_names_window_staleness_and_single_publish() -> None:
    registry = StageBBackendRegistry()
    metadata = registry.runtime_metadata(_config(8))["memory"]

    assert metadata["implementation"] == "approximate_scan"
    assert metadata["exactness"] == "approximate_stale_within_window"
    assert metadata["window_size"] == 8
    assert "incoming fast-weight snapshot" in metadata["staleness"]
    assert metadata["published_states_per_segment"] == 1


def test_approximation_preserves_mask_safety_and_rejects_ended_stream() -> None:
    torch.manual_seed(617)
    block = PaperMACBlock(
        d_model=4, num_heads=2, persistent_tokens=2, memory_depth=1
    ).double().eval()
    changed_block = copy.deepcopy(block)
    segment = torch.randn(32, 4, dtype=torch.float64)
    changed = segment.clone()
    changed[17] += 100.0
    config = _config(4)

    baseline = execute_stage_b(
        block, block.initial_state("base"), segment, config=config
    )
    perturbed = execute_stage_b(
        changed_block,
        changed_block.initial_state("changed"),
        changed,
        config=config,
    )
    torch.testing.assert_close(perturbed.sequence[:17], baseline.sequence[:17], rtol=0, atol=0)
    ended = block.initial_state("ended").mark_ended()
    with pytest.raises(RuntimeError, match="ended"):
        ApproximateScanMemoryBackend(4)(block, ended, baseline.sequence, None)


def test_speed_fidelity_artifact_names_every_window_and_rejects_promotion(tmp_path) -> None:
    stage_a = {
        "variants": {
            name: {
                "delayed_accuracy_by_delay": {">32": delayed},
                "evaluation_bpb": bpb,
                "overwrite_correctness": overwrite,
                "reset_correctness": 1.0,
            }
            for name, delayed, bpb, overwrite in (
                ("adaptive", 1.0, 0.05, 1.0),
                ("frozen_memory", 0.125, 2.1, 0.125),
                ("no_memory", 0.25, 2.3, 0.25),
            )
        }
    }
    stage_a_path = tmp_path / "stage_a.json"
    stage_a_path.write_text(__import__("json").dumps(stage_a), encoding="utf-8")
    study = run_approximate_scan_study(
        seed=631,
        warmup_runs=0,
        repetitions=1,
        synthetic_config=BenchmarkConfig(
            seed=641,
            train_seeds=(641,),
            evaluation_seed=651,
            num_streams=2,
            train_epochs=1,
        ),
        stage_a_artifact=stage_a_path,
    )
    paths = write_approximate_scan_study(study, tmp_path / "out")

    assert set(study["dense_mechanism"]["approximate_windows"]) == {
        str(window) for window in APPROXIMATE_WINDOWS
    }
    assert study["classification"] == "experimental_approximation_not_parity_equivalent"
    assert study["decision"]["promotion_allowed"] is False
    assert all(path.exists() for path in paths.values())
