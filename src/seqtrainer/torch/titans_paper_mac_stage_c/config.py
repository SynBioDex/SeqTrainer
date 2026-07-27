"""Validated configuration contracts for Stage C genomic paper-MAC training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Mapping

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
    memory_architecture: Literal[
        "legacy_mlp_v1", "paper_residual_mlp_v2"
    ] = "legacy_mlp_v1"
    memory_expansion_factor: int = 4
    memory_projection_convolution_kernel: int | None = None
    memory_normalize_queries_and_keys: bool = False
    memory_gate_granularity: Literal[
        "shared_scalar", "per_layer_channel"
    ] = "shared_scalar"
    memory_recurrence_policy: Literal[
        "legacy_configured", "paper_exact", "stabilized_rms_v1"
    ] = "legacy_configured"
    memory_surprise_clip_norm: float | None = 4.0
    memory_associative_loss_reduction: Literal["sum", "mean"] = "sum"
    memory_max_gradient_rms: float | None = None
    memory_max_gradient_rms_ratio: float | None = None
    memory_theta_max: float = 1.0
    memory_theta_initial: float | None = None
    memory_alpha_initial: float | None = None
    memory_eta_initial: float | None = None
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
        if self.memory_architecture not in {
            "legacy_mlp_v1",
            "paper_residual_mlp_v2",
        }:
            raise ValueError("unknown memory_architecture")
        if self.memory_gate_granularity not in {
            "shared_scalar",
            "per_layer_channel",
        }:
            raise ValueError("unknown memory_gate_granularity")
        if self.memory_recurrence_policy not in {
            "legacy_configured",
            "paper_exact",
            "stabilized_rms_v1",
        }:
            raise ValueError("unknown memory_recurrence_policy")
        if self.memory_expansion_factor <= 0:
            raise ValueError("memory_expansion_factor must be positive")
        if (
            self.memory_projection_convolution_kernel is not None
            and self.memory_projection_convolution_kernel < 2
        ):
            raise ValueError("memory projection convolution kernel must be at least 2")
        if self.memory_architecture == "paper_residual_mlp_v2":
            if self.format_version < 2:
                raise ValueError("paper_residual_mlp_v2 requires config format_version >= 2")
            if self.memory_depth != 2 or self.memory_expansion_factor != 4:
                raise ValueError("paper_residual_mlp_v2 requires depth 2 and expansion factor 4")
            if self.memory_projection_convolution_kernel != 4:
                raise ValueError("paper_residual_mlp_v2 requires causal Q/K/V kernel size 4")
            if not self.memory_normalize_queries_and_keys:
                raise ValueError("paper_residual_mlp_v2 requires q/k L2 normalization")
            if self.memory_gate_granularity != "per_layer_channel":
                raise ValueError("paper_residual_mlp_v2 requires per-layer channel gates")
            if self.memory_recurrence_policy == "legacy_configured":
                raise ValueError("paper_residual_mlp_v2 requires an explicit recurrence policy")
        elif self.memory_gate_granularity != "shared_scalar":
            raise ValueError("legacy_mlp_v1 supports only shared_scalar gates")
        if self.memory_recurrence_policy == "paper_exact":
            if self.memory_associative_loss_reduction != "sum":
                raise ValueError("paper_exact requires the summed associative loss")
            if self.memory_max_gradient_rms is not None or self.memory_max_gradient_rms_ratio is not None:
                raise ValueError("paper_exact does not permit inner-gradient conditioning")
            if self.memory_surprise_clip_norm is not None:
                raise ValueError("paper_exact does not permit a surprise cap")
            if self.memory_theta_max != 1.0:
                raise ValueError("paper_exact requires theta_max=1")
        if self.memory_recurrence_policy == "stabilized_rms_v1":
            if self.memory_associative_loss_reduction != "mean":
                raise ValueError("stabilized_rms_v1 requires dimension-normalized loss")
            if self.memory_max_gradient_rms_ratio is None:
                raise ValueError("stabilized_rms_v1 requires a relative gradient-RMS limit")
        if self.memory_surprise_clip_norm is not None and self.memory_surprise_clip_norm <= 0:
            raise ValueError("memory_surprise_clip_norm must be positive when supplied")
        if self.memory_associative_loss_reduction not in {"sum", "mean"}:
            raise ValueError("memory_associative_loss_reduction must be 'sum' or 'mean'")
        if self.memory_max_gradient_rms is not None and self.memory_max_gradient_rms <= 0:
            raise ValueError("memory_max_gradient_rms must be positive when supplied")
        if (
            self.memory_max_gradient_rms_ratio is not None
            and self.memory_max_gradient_rms_ratio <= 0
        ):
            raise ValueError("memory_max_gradient_rms_ratio must be positive when supplied")
        if not 0.0 < self.memory_theta_max <= 1.0:
            raise ValueError("memory_theta_max must be in (0, 1]")
        if self.memory_theta_initial is not None and not (
            0.0 < self.memory_theta_initial < self.memory_theta_max
        ):
            raise ValueError("memory_theta_initial must be in (0, memory_theta_max)")
        for name, value in (
            ("memory_alpha_initial", self.memory_alpha_initial),
            ("memory_eta_initial", self.memory_eta_initial),
        ):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
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
            # CPU MultiheadAttention may dispatch to a Flash SDPA backward
            # kernel that Colab does not implement. The SDPA adapter routes
            # CPU through the exact differentiable math implementation.
            backend=StageBBackendConfig(attention_backend=AttentionBackend.SDPA),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["memory_mode"] = self.memory_mode.value
        payload["backend"] = self.backend.to_dict()
        return payload

    @classmethod
    def paper_deep(
        cls,
        *,
        vocab_size: int,
        pad_token_id: int,
        tokenizer_name: str,
        tokenizer_checksum: str,
        recurrence_policy: Literal["paper_exact", "stabilized_rms_v1"],
        **overrides: object,
    ) -> "StageCModelConfig":
        """Construct the frozen v2 paper-default two-layer memory contract."""

        exact = recurrence_policy == "paper_exact"
        values: dict[str, object] = {
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "tokenizer_name": tokenizer_name,
            "tokenizer_checksum": tokenizer_checksum,
            "memory_depth": 2,
            "memory_architecture": "paper_residual_mlp_v2",
            "memory_expansion_factor": 4,
            "memory_projection_convolution_kernel": 4,
            "memory_normalize_queries_and_keys": True,
            "memory_gate_granularity": "per_layer_channel",
            "memory_recurrence_policy": recurrence_policy,
            "memory_surprise_clip_norm": None,
            "memory_associative_loss_reduction": "sum" if exact else "mean",
            "memory_max_gradient_rms": None,
            "memory_max_gradient_rms_ratio": None if exact else 10.0,
            "memory_theta_max": 1.0 if exact else 0.5,
            "memory_theta_initial": 1e-3,
            "memory_alpha_initial": 1e-3,
            "memory_eta_initial": 0.9,
            "format_version": 2,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StageCModelConfig":
        raw_backend = payload.get("backend")
        if not isinstance(raw_backend, Mapping):
            raise ValueError("Stage C config is missing backend configuration")
        backend = StageBBackendConfig(**dict(raw_backend))
        values = dict(payload)
        values["backend"] = backend
        return cls(**values)
