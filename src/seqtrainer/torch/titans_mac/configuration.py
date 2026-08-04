"""Configuration for Titan Memory-as-Context DNA models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TitansMACLMConfig:
    """Hyperparameters for a Titan MAC causal language model.

    The vocabulary contract is fixed to ``PAD=0, N/UNK=1, A=2, C=3,
    G=4, T=5``. ``vocab_size`` remains configurable for experimentation, but
    values below six are rejected.
    """

    vocab_size: int = 6
    pad_token_id: int = 0
    unk_token_id: int = 1
    d_model: int = 384
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1536
    max_length: int = 2048
    memory_slots: int = 64
    memory_depth: int = 2
    memory_context_tokens: int = 8
    persistent_tokens: int = 8
    dropout: float = 0.1
    retention_gate: float = 0.95
    use_persistent_memory: bool = True
    tie_embeddings: bool = True
    num_labels: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size < 6:
            raise ValueError("vocab_size must include the six fixed DNA tokens")
        if self.pad_token_id != 0 or self.unk_token_id != 1:
            raise ValueError("Titan MAC requires pad_token_id=0 and unk_token_id=1")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        for name in (
            "d_model",
            "num_heads",
            "num_layers",
            "dim_feedforward",
            "max_length",
            "memory_slots",
            "memory_depth",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.memory_context_tokens <= self.memory_slots:
            raise ValueError("memory_context_tokens must be between 0 and memory_slots")
        if self.persistent_tokens < 0:
            raise ValueError("persistent_tokens cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.retention_gate <= 1.0:
            raise ValueError("retention_gate must be in [0, 1]")

    @property
    def context_tokens(self) -> int:
        persistent = self.persistent_tokens if self.use_persistent_memory else 0
        return persistent + self.memory_context_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TitansMACLMConfig":
        return cls(**values)
