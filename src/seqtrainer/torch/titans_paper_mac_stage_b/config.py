"""Typed Stage B backend selection with conservative availability checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MemoryBackend(str, Enum):
    """Neural-memory execution classes tracked by the Stage B audit."""

    REFERENCE = "reference"
    EXACT_ACCELERATED = "exact_accelerated"
    EXACT_SCAN = "exact_scan"
    APPROXIMATE_SCAN = "approximate_scan"


class AttentionBackend(str, Enum):
    """Attention implementations that must preserve the Stage A edge set."""

    MULTIHEAD_ATTENTION = "multihead_attention"
    SDPA = "sdpa"
    FLASH = "flash"


class ActivationDType(str, Enum):
    """Activation precision; the neural-memory state remains FP32 in Stage B."""

    FP32 = "float32"
    BF16 = "bfloat16"
    FP16 = "float16"


APPROXIMATE_WINDOWS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class StageBBackendConfig:
    """Feature flags for one Stage B execution.

    At B1 only the reference memory, reference attention, and FP32 activations
    are registered.  Naming future modes here makes artifacts stable without
    accidentally making unfinished research paths selectable.
    """

    memory_backend: MemoryBackend = MemoryBackend.REFERENCE
    attention_backend: AttentionBackend = AttentionBackend.MULTIHEAD_ATTENTION
    activation_dtype: ActivationDType = ActivationDType.FP32
    approximate_window: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_backend", MemoryBackend(self.memory_backend))
        object.__setattr__(self, "attention_backend", AttentionBackend(self.attention_backend))
        object.__setattr__(self, "activation_dtype", ActivationDType(self.activation_dtype))
        if self.memory_backend is MemoryBackend.APPROXIMATE_SCAN:
            if self.approximate_window not in APPROXIMATE_WINDOWS:
                raise ValueError(
                    "approximate_scan requires approximate_window in "
                    f"{APPROXIMATE_WINDOWS}"
                )
        elif self.approximate_window is not None:
            raise ValueError("approximate_window is valid only for approximate_scan")

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_backend": self.memory_backend.value,
            "attention_backend": self.attention_backend.value,
            "activation_dtype": self.activation_dtype.value,
            "approximate_window": self.approximate_window,
        }

