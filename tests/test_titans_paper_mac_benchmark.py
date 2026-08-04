from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac import (  # noqa: E402
    BenchmarkConfig,
    run_stage_a_benchmark,
    write_benchmark_artifacts,
)


def test_benchmark_is_deterministic_matched_and_passes_its_acceptance_gates(tmp_path) -> None:
    config = BenchmarkConfig(
        seed=20260727,
        train_seeds=(20260727, 20260737),
        evaluation_seed=20260827,
        num_streams=4,
        train_epochs=2,
    )
    first = run_stage_a_benchmark(config)
    second = run_stage_a_benchmark(config)

    assert first.to_dict() == second.to_dict()
    assert first.gates.passed
    assert set(first.variants) == {"adaptive", "frozen_memory", "no_memory"}
    assert first.protocol["memory_implementation"] == "FunctionalNeuralMemory"
    assert first.protocol["shared_total_parameter_count"] > first.protocol["shared_outer_parameter_count"]
    assert first.protocol["tokens_per_variant"] > 0
    assert first.variants["adaptive"].delayed_accuracy_by_delay[">32"] == 1.0
    assert first.variants["adaptive"].delayed_accuracy_by_delay[">32"] > max(
        first.variants["frozen_memory"].delayed_accuracy_by_delay[">32"],
        first.variants["no_memory"].delayed_accuracy_by_delay[">32"],
    )
    assert first.variants["adaptive"].overwrite_correctness == 1.0
    assert first.variants["adaptive"].reset_correctness == 1.0
    assert first.variants["adaptive"].gradient_norm.count > 0
    assert first.variants["adaptive"].memory_update_norm.mean > 0
    assert first.variants["frozen_memory"].memory_update_norm.mean == 0.0
    assert first.variants["no_memory"].memory_update_norm.mean == 0.0
    for variant in first.variants.values():
        assert variant.alpha.count > 0
        assert variant.eta.count > 0
        assert variant.theta.count > 0
        assert variant.theta.mean > variant.eta.mean > variant.alpha.mean

    paths = write_benchmark_artifacts(first, tmp_path)
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["gates"]["passed"] is True
    assert "Delayed >32 accuracy" in paths["report"].read_text(encoding="utf-8")
    assert "<svg" in paths["plot"].read_text(encoding="utf-8")
