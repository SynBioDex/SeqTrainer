from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    TrainingStepScale,
    run_training_step_matrix,
    write_training_step_matrix,
)


def test_two_segment_training_step_exercises_exact_cpu_paths_and_writes_artifact(
    tmp_path,
) -> None:
    result = run_training_step_matrix(
        scale=TrainingStepScale(
            "test",
            block_count=1,
            d_model=4,
            num_heads=2,
            persistent_tokens=2,
        ),
        variants=("reference_fp32", "exact_fp32", "exact_sdpa_fp32"),
        seed=733,
        warmup_runs=0,
        repetitions=1,
        device="cpu",
    )
    paths = write_training_step_matrix(result, tmp_path)

    assert result["classification"] == "two_segment_outer_training_step"
    assert result["protocol"]["segments"] == 2
    assert result["parameter_count"] > 0
    assert {variant["variant"] for variant in result["variants"]} == {
        "reference_fp32",
        "exact_fp32",
        "exact_sdpa_fp32",
    }
    for variant in result["variants"]:
        assert variant["available"] is True
        assert variant["output_and_state_finite"] is True
        assert variant["all_gradients_finite"] is True
        assert variant["gradient_metrics"][0]["written_state_gradient_norm"] > 0
        assert variant["memory_state_dtypes"] == ["float32"]
        assert variant["tokens_per_second"] > 0
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["protocol"]["loss"].startswith("MSE on segment-two")
    assert paths["report"].exists()
