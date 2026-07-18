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


class GateBackend(str, Enum):
    """Adaptive-gate feature path, independent of memory execution backend."""

    TOKEN_WISE = "token_wise"
    CAUSAL_CONVOLUTION = "causal_convolution"


APPROXIMATE_WINDOWS = (2, 4, 8, 16, 32)


@dataclass(frozen=True)
class StageBBackendConfig:
    """Feature flags for one Stage B execution.

    The conservative defaults remain the reference memory, reference attention,
    token-wise gates, and FP32 activations. Naming future modes here makes
    artifacts stable without accidentally making unfinished research paths
    selectable.
    """

    memory_backend: MemoryBackend = MemoryBackend.REFERENCE
    attention_backend: AttentionBackend = AttentionBackend.MULTIHEAD_ATTENTION
    activation_dtype: ActivationDType = ActivationDType.FP32
    gate_backend: GateBackend = GateBackend.TOKEN_WISE
    convolution_kernel_size: int = 3
    approximate_window: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_backend", MemoryBackend(self.memory_backend))
        object.__setattr__(self, "attention_backend", AttentionBackend(self.attention_backend))
        object.__setattr__(self, "activation_dtype", ActivationDType(self.activation_dtype))
        object.__setattr__(self, "gate_backend", GateBackend(self.gate_backend))
        if self.convolution_kernel_size <= 0:
            raise ValueError("convolution_kernel_size must be positive")
        if self.gate_backend is GateBackend.CAUSAL_CONVOLUTION and self.convolution_kernel_size < 2:
            raise ValueError("causal_convolution requires convolution_kernel_size >= 2")
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
            "gate_backend": self.gate_backend.value,
            "convolution_kernel_size": self.convolution_kernel_size,
            "approximate_window": self.approximate_window,
        }
