"""Validated configuration contracts for Stage C genomic paper-MAC training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping

from seqtrainer.torch.titans_paper_mac_stage_b import (
    ActivationDType,
    AttentionBackend,
    GateBackend,
    MemoryBackend,
    StageBBackendConfig,
)


class MemoryMode(str, Enum):
    """Scientific memory conditions used by the Stage C ablations."""

    ADAPTIVE = "adaptive"
    REFERENCE = "reference"
    FROZEN = "frozen_memory"
    NONE = "no_memory"


@dataclass(frozen=True)
class StageCModelConfig:
    """Serializable architecture and execution contract for the genomic LM."""

    vocab_size: int
    pad_token_id: int
    tokenizer_name: str
    tokenizer_checksum: str
    block_count: int = 8
    d_model: int = 384
    num_heads: int = 8
    persistent_tokens: int = 4
    memory_depth: int = 1
    segment_length: int = 32
    tie_embeddings: bool = True
    gradient_horizon: int = 2
    memory_mode: MemoryMode = MemoryMode.ADAPTIVE
    backend: StageBBackendConfig = field(
        default_factory=lambda: StageBBackendConfig(
            memory_backend=MemoryBackend.EXACT_ACCELERATED,
            attention_backend=AttentionBackend.SDPA,
            activation_dtype=ActivationDType.FP32,
            gate_backend=GateBackend.TOKEN_WISE,
        )
    )
    format_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_mode", MemoryMode(self.memory_mode))
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be in the vocabulary")
        if not self.tokenizer_name or not self.tokenizer_checksum:
            raise ValueError("tokenizer name and checksum are required")
        if self.block_count <= 0 or self.d_model <= 0:
            raise ValueError("block_count and d_model must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.segment_length != 32:
            raise ValueError("Stage C preserves the paper-MAC 32-token segment")
        if self.gradient_horizon not in (1, 2, 3, 4):
            raise ValueError("gradient_horizon must be one of 1, 2, 3, or 4")
        if self.backend.memory_backend in (
            MemoryBackend.APPROXIMATE_SCAN,
            MemoryBackend.EXACT_SCAN,
        ):
            raise ValueError("approximate and unproven scan backends are ineligible for Stage C")
        if self.backend.attention_backend is AttentionBackend.FLASH:
            raise ValueError("Flash attention is disabled for Stage C")

    @classmethod
    def cpu_basal(
        cls,
        *,
        vocab_size: int,
        pad_token_id: int,
        tokenizer_name: str,
        tokenizer_checksum: str,
        gradient_horizon: int = 2,
    ) -> "StageCModelConfig":
        """Return the locked small model used before accelerator experiments."""

        return cls(
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            tokenizer_name=tokenizer_name,
            tokenizer_checksum=tokenizer_checksum,
            block_count=1,
            d_model=32,
            num_heads=4,
            persistent_tokens=4,
            memory_depth=1,
            gradient_horizon=gradient_horizon,
            backend=StageBBackendConfig(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["memory_mode"] = self.memory_mode.value
        payload["backend"] = self.backend.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StageCModelConfig":
        raw_backend = payload.get("backend")
        if not isinstance(raw_backend, Mapping):
            raise ValueError("Stage C config is missing backend configuration")
        backend = StageBBackendConfig(**dict(raw_backend))
        values = dict(payload)
        values["backend"] = backend
        return cls(**values)

