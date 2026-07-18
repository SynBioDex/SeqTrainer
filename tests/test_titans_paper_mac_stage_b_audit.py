from __future__ import annotations

import json
from pathlib import Path

from seqtrainer.torch.titans_paper_mac_stage_b import (
    build_stage_b_audit,
    write_stage_b_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_final_audit_is_evidence_backed_separates_approximation_and_is_no_go(tmp_path) -> None:
    audit = build_stage_b_audit(
        REPOSITORY_ROOT,
        final_test_count=89,
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

