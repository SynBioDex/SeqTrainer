from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    APPROXIMATE_WINDOWS,
    DEFAULT_LONG_CONTEXT_VARIANTS,
    LongContextScale,
    run_long_context_study,
    write_long_context_study,
)


def test_long_context_study_separates_hardware_backends_controls_and_recall(tmp_path) -> None:
    scales = (
        LongContextScale("test_cpu", 1, 4, 2, 2, persistent_tokens=2),
        LongContextScale(
            "a100_pilot", 8, 384, 8, 4, requires_a100=True
        ),
    )
    result = run_long_context_study(scales=scales, seed=701)
    paths = write_long_context_study(result, tmp_path)

    cpu, a100 = result["long_stream_scales"]
    assert cpu["available"] is True and cpu["environment"] == "macbook_cpu"
    assert {item["variant"] for item in cpu["variants"]} == set(
        DEFAULT_LONG_CONTEXT_VARIANTS
    )
    assert a100["available"] is False and a100["environment"] == "a100"
    assert "unavailable" in a100["reason"]
    assert result["causality"]["passed"] is True
    assert all(
        all(item["finite_state_by_segment"])
        for item in cpu["variants"]
    )
    assert set(result["controlled_long_recall"]) == set(DEFAULT_LONG_CONTEXT_VARIANTS)
    assert all(
        f"approximate_w{window}" in result["controlled_long_recall"]
        for window in APPROXIMATE_WINDOWS
    )
    assert result["controlled_long_recall"]["reference"][
        "delay_accuracy_by_context_tokens"
    ]["512"] == 1.0
    assert result["controlled_long_recall"]["no_memory"][
        "delay_accuracy_by_context_tokens"
    ]["512"] == 0.0
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["classification"] == "synthetic_system_validation_not_biology_performance"
    assert all(path.exists() for path in paths.values())
