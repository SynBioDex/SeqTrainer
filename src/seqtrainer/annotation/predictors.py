"""Promoter predictor interfaces used by annotation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PromoterPredictor(Protocol):
    """Model-agnostic promoter probability interface."""

    def predict_proba(self, sequences: list[str]) -> list[float]:
        """Return one promoter score/probability per input sequence."""

    def metadata(self) -> dict:
        """Return model metadata for annotation manifests."""


class DummyPromoterPredictor:
    """Deterministic smoke-test predictor.

    This is intentionally simple and should not be used for biological claims.
    It assigns high scores to windows containing a TATA-like motif so tests can
    exercise annotation writing deterministically.
    """

    def predict_proba(self, sequences: list[str]) -> list[float]:
        scores = []
        for sequence in sequences:
            seq = sequence.upper()
            if "TATA" in seq or "TTGACA" in seq:
                scores.append(0.95)
            else:
                scores.append(0.10)
        return scores

    def metadata(self) -> dict:
        return {
            "model_family": "dummy",
            "mode": "deterministic_smoke_test",
            "biological_claims": False,
        }


class DNABERT2PromoterPredictor:
    """Dependency-gated DNABERT2 predictor interface."""

    def __init__(self, checkpoint: str | Path | None = None, benchmark_manifest: str | Path | None = None):
        if checkpoint is None:
            raise ValueError("DNABERT2 annotation requires --checkpoint from a completed benchmark run.")
        self.checkpoint = Path(checkpoint)
        self.benchmark_manifest = Path(benchmark_manifest) if benchmark_manifest else None
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "DNABERT2 annotation requires torch/transformers. Install with "
                "`pip install -e \".[annotation,torch]\"`."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"DNABERT2 checkpoint not found: {self.checkpoint}")
        raise NotImplementedError(
            "DNABERT2 checkpoint inference is wired as an interface but the checkpoint-specific "
            "loader is not implemented in this MVP. Use dummy mode for smoke tests or add a "
            "compatible benchmark checkpoint loader before reporting DNABERT2 annotations."
        )

    def predict_proba(self, sequences: list[str]) -> list[float]:  # pragma: no cover - constructor raises
        raise NotImplementedError

    def metadata(self) -> dict:  # pragma: no cover - constructor raises
        return {"model_family": "dnabert2", "checkpoint": str(self.checkpoint)}


class CnnV2PromoterPredictor:
    """Dependency-gated CNN-v2 predictor interface."""

    def __init__(self, checkpoint: str | Path | None = None, benchmark_manifest: str | Path | None = None):
        if checkpoint is None:
            raise ValueError("CNN-v2 annotation requires --checkpoint from a completed benchmark run.")
        self.checkpoint = Path(checkpoint)
        self.benchmark_manifest = Path(benchmark_manifest) if benchmark_manifest else None
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "CNN-v2 annotation requires torch. Install with `pip install -e \".[annotation,torch]\"`."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"CNN-v2 checkpoint not found: {self.checkpoint}")
        raise NotImplementedError(
            "CNN-v2 checkpoint inference is wired as an interface but the checkpoint-specific "
            "loader is not implemented in this MVP. Use dummy mode for smoke tests or add a "
            "compatible benchmark checkpoint loader before reporting CNN-v2 annotations."
        )

    def predict_proba(self, sequences: list[str]) -> list[float]:  # pragma: no cover - constructor raises
        raise NotImplementedError

    def metadata(self) -> dict:  # pragma: no cover - constructor raises
        return {"model_family": "cnn_v2", "checkpoint": str(self.checkpoint)}


def build_predictor(
    model_family: str,
    *,
    checkpoint: str | Path | None = None,
    benchmark_manifest: str | Path | None = None,
) -> PromoterPredictor:
    """Construct a predictor for the requested model family."""
    if model_family == "dummy":
        return DummyPromoterPredictor()
    if model_family == "dnabert2":
        return DNABERT2PromoterPredictor(checkpoint=checkpoint, benchmark_manifest=benchmark_manifest)
    if model_family == "cnn_v2":
        return CnnV2PromoterPredictor(checkpoint=checkpoint, benchmark_manifest=benchmark_manifest)
    raise ValueError(f"Unsupported model family: {model_family}")

