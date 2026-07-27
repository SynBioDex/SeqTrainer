from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs" / "titans_stage_a" / "fidelity_audit.md"
PAPER_PACKAGE = ROOT / "src" / "seqtrainer" / "torch" / "titans_paper_mac"


def test_fidelity_audit_records_every_required_mechanism_and_binary_decision() -> None:
    text = AUDIT.read_text(encoding="utf-8")
    required_evidence = (
        "Associative KV objective",
        "MLP neural-memory fast weights",
        "Momentary surprise",
        "Momentum / past surprise",
        "Adaptive `alpha`, `eta`, `theta`",
        "Persistent memory",
        "Full `h_t` retrieval",
        "Read/write timing",
        "Block-causal MAC mask",
        "Stream isolation and lifecycle",
        "Full differentiability",
        "Reproduction contract",
        "Binary exit checklist",
    )
    assert all(item in text for item in required_evidence)
    assert "STAGE B DECISION: READY" in text
    assert "**Binary result: READY for Stage B.**" in text
    assert "- [x] Equation-24 memory writes consume the causal-core output" in text
    assert "- [x] The matched adaptive/frozen/no-memory benchmark exercises" in text
    assert "- [x] Stage A is captured in a reproducible commit" in text


def test_paper_path_remains_isolated_and_only_declared_optimizations_are_present() -> None:
    sources = "\n".join(path.read_text(encoding="utf-8") for path in PAPER_PACKAGE.glob("*.py"))
    assert "from ..titans_mac" not in sources
    assert "from seqtrainer.torch.titans_mac" not in sources
    # V2 intentionally adds the paper's causal kernel-4 Q/K/V projection
    # convolution. Scan and fused-attention substitutions remain excluded.
    assert "projection_convolution_kernel" in sources
    assert "groups=d_model" in sources
    assert "associative_scan" not in sources
    assert "scaled_dot_product_attention" not in sources
