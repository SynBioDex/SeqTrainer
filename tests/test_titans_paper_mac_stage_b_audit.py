from __future__ import annotations

import json
from pathlib import Path
import shutil

from seqtrainer.torch.titans_paper_mac_stage_b import (
    build_stage_b_audit,
    write_stage_b_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _isolated_artifacts(tmp_path: Path, *, include_a100: bool) -> Path:
    root = tmp_path / "repository"
    ignore = None if include_a100 else shutil.ignore_patterns("a100")
    shutil.copytree(REPOSITORY_ROOT / "artifacts", root / "artifacts", ignore=ignore)
    return root


def test_final_audit_is_evidence_backed_separates_approximation_and_is_no_go(tmp_path) -> None:
    root = _isolated_artifacts(tmp_path, include_a100=False)
    audit = build_stage_b_audit(
        root,
        final_test_count=94,
        final_warning_count=7,
    )
    paths = write_stage_b_audit(
        audit,
        artifact_path=tmp_path / "audit.json",
        document_path=tmp_path / "audit.md",
    )

    assert audit["selected_default"] == "reference"
    assert audit["stage_c_gate"]["decision"] == "NO_GO"
    assert audit["stage_c_gate"]["ready"] is False
    assert any(
        item["criterion"] == "MacBook and named A100 measurements are reproducible"
        and item["passed"] is False
        for item in audit["requirements"]
    )
    backends = {backend["name"]: backend for backend in audit["backends"]}
    assert "restricted_only" in backends["exact_scan"]["exactness_class"]
    assert "experimental" in backends["approximate_scan"]["exactness_class"]
    assert backends["approximate_scan"]["default"] is False
    assert json.loads(paths["json"].read_text(encoding="utf-8"))[
        "stage_c_gate"
    ]["decision"] == "NO_GO"
    document = paths["document"].read_text(encoding="utf-8")
    assert "**Stage C gate: NO_GO**" in document
    assert "Stage C may not begin" in document
    assert "Exact and approximate paths are intentionally separate" in document


def test_final_audit_closes_stage_b_with_verified_a100_bundle(tmp_path) -> None:
    root = _isolated_artifacts(tmp_path, include_a100=True)
    audit = build_stage_b_audit(
        root,
        final_test_count=94,
        final_warning_count=7,
    )
    paths = write_stage_b_audit(
        audit,
        artifact_path=tmp_path / "ready-audit.json",
        document_path=tmp_path / "ready-audit.md",
    )

    assert audit["selected_default"] == "exact_accelerated+sdpa"
    assert audit["stage_c_gate"] == {
        "decision": "READY",
        "ready": True,
        "definition": "Genome/clade-separated 15 Gbp bacterial next-base foundation training.",
        "blockers": [],
        "rule": audit["stage_c_gate"]["rule"],
    }
    assert audit["hardware_summary"]["a100"]["available"] is True
    assert audit["hardware_summary"]["a100"]["verification"][
        "manifest_checksums_passed"
    ] is True
    backends = {backend["name"]: backend for backend in audit["backends"]}
    assert backends["exact_accelerated+sdpa"]["default"] is True
    assert backends["reference"]["default"] is False
    assert "A100" in backends["exact_accelerated+sdpa"]["hardware_precision"]
    document = paths["document"].read_text(encoding="utf-8")
    assert "**Stage C gate: READY**" in document
    assert "Stage C may begin" in document
    assert "Stage B is formally closed" in document
