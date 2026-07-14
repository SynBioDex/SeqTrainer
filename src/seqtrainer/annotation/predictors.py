"""Promoter predictor interfaces used by annotation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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
    """Dependency-gated DNABERT2 predictor for benchmark checkpoints."""

    def __init__(self, checkpoint: str | Path | None = None, benchmark_manifest: str | Path | None = None):
        if checkpoint is None:
            raise ValueError("DNABERT2 annotation requires --checkpoint from a completed benchmark run.")
        if benchmark_manifest is None:
            raise ValueError(
                "DNABERT2 annotation requires --benchmark-manifest so the tokenizer, "
                "pooling, dropout, revision, and window settings match the benchmark run."
            )
        self.checkpoint = Path(checkpoint)
        self.benchmark_manifest = Path(benchmark_manifest)
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "DNABERT2 annotation requires torch/transformers. Install with "
                "`pip install -e \".[annotation,torch]\"`."
            ) from exc
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"DNABERT2 checkpoint not found: {self.checkpoint}")
        if not self.benchmark_manifest.exists():
            raise FileNotFoundError(f"DNABERT2 benchmark manifest not found: {self.benchmark_manifest}")

        try:
            from seqtrainer.torch.dnabert2_benchmark import (
                _DnaBert2Classifier,
                _load_huggingface_dnabert2,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "DNABERT2 annotation requires torch/transformers. Install with "
                "`pip install -e \".[annotation,torch]\"`."
            ) from exc

        self._torch = torch
        self._manifest = json.loads(self.benchmark_manifest.read_text(encoding="utf-8"))
        self._model_name = str(_manifest_get(self._manifest, ("model", "name"), "zhihan1996/DNABERT-2-117M"))
        self._model_params = dict(_manifest_get(self._manifest, ("model", "params"), {}))
        self._preprocessing = dict(_manifest_get(self._manifest, ("preprocessing", "params"), {}))
        self._pooling = str(self._model_params.get("pooling", "mean"))
        self._batch_size = int(_manifest_get(self._manifest, ("training", "batch_size"), 8) or 8)
        self._device = _resolve_torch_device(torch)

        allow_download = bool(self._model_params.get("allow_download", False))
        tokenizer, encoder = _load_huggingface_dnabert2(
            self._model_name,
            device=self._device,
            trust_remote_code=bool(self._model_params.get("trust_remote_code", True)),
            local_files_only=not allow_download,
            disable_flash_attention=bool(self._model_params.get("disable_flash_attention", True)),
            revision=self._model_params.get("revision"),
        )
        self._tokenizer = tokenizer
        hidden_size = int(getattr(encoder.config, "hidden_size", 768))
        self._model = _DnaBert2Classifier(
            encoder=encoder,
            hidden_size=hidden_size,
            pooling=self._pooling,
            dropout=float(self._model_params.get("classifier_dropout", 0.1)),
        ).to(self._device)
        state_dict = _load_torch_state_dict(torch, self.checkpoint)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    def predict_proba(self, sequences: list[str]) -> list[float]:
        if not sequences:
            return []
        torch = self._torch
        probabilities: list[float] = []
        max_length = int(
            self._preprocessing.get(
                "model_max_length",
                _manifest_get(self._manifest, ("preprocessing", "sequence_length"), 300),
            )
        )
        padding = str(self._preprocessing.get("padding", "longest"))
        pad_to_multiple_of = self._preprocessing.get("pad_to_multiple_of")
        with torch.inference_mode():
            for start in range(0, len(sequences), self._batch_size):
                batch_sequences = sequences[start : start + self._batch_size]
                encoded = self._tokenizer(
                    batch_sequences,
                    padding=padding,
                    truncation=True,
                    max_length=max_length,
                    pad_to_multiple_of=int(pad_to_multiple_of) if pad_to_multiple_of is not None else None,
                    return_tensors="pt",
                )
                batch = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                    if hasattr(value, "to")
                }
                if "attention_mask" not in batch:
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
                logits = self._model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
        return probabilities

    def metadata(self) -> dict:
        return {
            "model_family": "dnabert2",
            "checkpoint": str(self.checkpoint),
            "benchmark_manifest": str(self.benchmark_manifest),
            "model_name": self._model_name,
            "pooling": self._pooling,
            "device": str(self._device),
            "batch_size": self._batch_size,
            "mode": self._model_params.get("mode"),
        }


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


def _manifest_get(manifest: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    current: Any = manifest
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_torch_device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_torch_state_dict(torch: Any, checkpoint: Path) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch compatibility
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"DNABERT2 checkpoint did not contain a state dict: {checkpoint}")
    return state

