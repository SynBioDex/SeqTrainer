from __future__ import annotations

import json
from pathlib import Path

import pytest

from seqtrainer.torch.titans_paper_mac_stage_c.study import (
    STUDY_ID,
    StudyProtocol,
    amend,
    canonical_json,
    initialize,
    record,
    report,
    sha256_file,
    validate_protocol,
    verify,
)


PROTOCOL = Path(__file__).parents[1] / "studies" / STUDY_ID / "protocol.json"
PAPER_DEEP_PROTOCOL = (
    Path(__file__).parents[1]
    / "studies"
    / "stage_c_ecoli_escherichia_paper_deep_memory_v2"
    / "protocol.json"
)
PAPER_DEEP_ADAPTIVE_5M_AMENDMENT = (
    PAPER_DEEP_PROTOCOL.parent / "amendments" / "adaptive_exploration_5m_v1.json"
)


def test_frozen_protocol_is_valid_and_canonical_hash_is_deterministic() -> None:
    protocol = validate_protocol(PROTOCOL)
    assert protocol.payload["study_id"] == STUDY_ID
    assert protocol.hash == StudyProtocol(json.loads(canonical_json(protocol.payload))).hash
    assert protocol.hash == protocol.hash
    with pytest.raises(ValueError, match="conflicts"):
        protocol.validate_run_config("adaptive_discovery_25m", {"memory_mode": "no_memory"})


def test_ledger_chain_amendment_and_artifact_verification(tmp_path: Path) -> None:
    root = tmp_path / "drive" / STUDY_ID
    artifact = tmp_path / "run_manifest.json"
    artifact.write_text(json.dumps({"format_version": 1, "optimizer_steps": 1}) + "\n", encoding="utf-8")
    initialize(PROTOCOL, root)
    event = record(PROTOCOL, root, "calibration_50k", [artifact], status="completed", evidence_tier="engineering")
    assert event["artifacts"][0]["sha256"] == sha256_file(artifact)
    amendment = amend(PROTOCOL, root, "amendment_001", "repair logging", "engineering", "no scientific impact", {"logging": "capture CUDA details"})
    assert amendment.exists()
    checked = verify(PROTOCOL, root)
    assert checked["valid"] is True
    assert checked["events"] == 3
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify(PROTOCOL, root)


def test_linked_amendment_can_add_but_not_replace_an_exploratory_run(tmp_path: Path) -> None:
    protocol = validate_protocol(PROTOCOL)
    amendment = tmp_path / "adaptive_5m.json"
    amendment.write_text(
        canonical_json(
            {
                "format_version": 1,
                "amendment_id": "adaptive_5m",
                "preceding_protocol_hash": protocol.hash,
                "classification": "exploratory",
                "changes": {
                    "run_matrix_additions": {
                        "adaptive_exploration_5m": {
                            "phase": "exploratory",
                            "budget_bases": 5_000_000,
                            "memory_mode": "adaptive",
                            "memory_architecture": "paper_residual_mlp_v2",
                            "memory_recurrence_policy": "paper_exact",
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    protocol.validate_run_config(
        "adaptive_exploration_5m",
        {
            "phase": "exploratory",
            "budget_bases": 5_000_000,
            "memory_mode": "adaptive",
            "memory_architecture": "paper_residual_mlp_v2",
            "memory_recurrence_policy": "paper_exact",
        },
        amendment_paths=[amendment],
    )
    with pytest.raises(ValueError, match="conflicts"):
        protocol.validate_run_config(
            "adaptive_exploration_5m",
            {"memory_mode": "no_memory"},
            amendment_paths=[amendment],
        )


def test_record_accepts_a_source_controlled_exploratory_addition(tmp_path: Path) -> None:
    root = tmp_path / "study"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("evidence\n", encoding="utf-8")

    event = record(
        PAPER_DEEP_PROTOCOL,
        root,
        "adaptive_exploration_5m",
        [artifact],
        evidence_tier="exploratory",
        amendment_paths=[PAPER_DEEP_ADAPTIVE_5M_AMENDMENT],
    )

    assert event["run_id"] == "adaptive_exploration_5m"
    assert event["evidence_tier"] == "exploratory"


def test_failed_runs_remain_visible_and_block_final_report(tmp_path: Path) -> None:
    root = tmp_path / "study"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("evidence\n", encoding="utf-8")
    initialize(PROTOCOL, root)
    record(PROTOCOL, root, "calibration_50k", [artifact], status="failed", evidence_tier="engineering", deviation_reason="OOM")
    assert len((root / "ledger.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="missing confirmatory evidence"):
        report(PROTOCOL, root)


def test_report_requires_all_confirmatory_run_ids(tmp_path: Path) -> None:
    root = tmp_path / "study"
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("evidence\n", encoding="utf-8")
    initialize(PROTOCOL, root)
    for run_id in [
        "adaptive_discovery_25m", "control_reference_equal_budget", "control_frozen_equal_budget",
        "control_none_equal_budget", "replication_seed2", "anomaly_evaluation",
    ]:
        record(PROTOCOL, root, run_id, [artifact], evidence_tier="confirmatory")
    outputs = report(PROTOCOL, root, tmp_path / "report")
    assert outputs["report"].is_file()
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["study_id"] == STUDY_ID
