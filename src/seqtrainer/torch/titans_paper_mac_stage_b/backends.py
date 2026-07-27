"""Feature-flagged dispatch around the unchanged Stage A MAC reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock, PaperMACBlockOutput
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState

from .config import ActivationDType, AttentionBackend, GateBackend, MemoryBackend, StageBBackendConfig


class BackendUnavailableError(ValueError):
    """Raised when configuration selects a backend without proven support."""


MemoryUpdate = Callable[
    [PaperMACBlock, PaperMACStreamState, Tensor, Optional[Tensor]],
    PaperMACStreamState,
]
AttentionIntegration = Callable[[PaperMACBlock, Tensor, Tensor, ActivationDType], Tensor]


@dataclass(frozen=True)
class BackendCapability:
    """Audit-facing declaration for one registered implementation."""

    name: str
    exactness: str
    available: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "exactness": self.exactness,
            "available": self.available,
            "reason": self.reason,
        }


def _reference_memory_update(
    block: PaperMACBlock,
    state: PaperMACStreamState,
    sequence: Tensor,
    valid_mask: Optional[Tensor],
) -> PaperMACStreamState:
    return block.memory.update_segment(state, sequence, valid_mask=valid_mask)


def _reference_attention(
    block: PaperMACBlock,
    retrieval: Tensor,
    segment: Tensor,
    activation_dtype: ActivationDType,
) -> Tensor:
    if activation_dtype is not ActivationDType.FP32:
        raise BackendUnavailableError(
            "reduced precision is available only through the reviewed SDPA adapter"
        )
    return block.integrate(retrieval, segment)


class StageBBackendRegistry:
    """Registry whose initial state exposes only reviewed B1 reference paths."""

    def __init__(self) -> None:
        self._memory_updates: dict[MemoryBackend, MemoryUpdate] = {
            MemoryBackend.REFERENCE: _reference_memory_update,
        }
        self._attention_integrations: dict[AttentionBackend, AttentionIntegration] = {
            AttentionBackend.MULTIHEAD_ATTENTION: _reference_attention,
        }
        self._memory_capabilities: dict[MemoryBackend, BackendCapability] = {
            MemoryBackend.REFERENCE: BackendCapability(
                name=MemoryBackend.REFERENCE.value,
                exactness="fp64_oracle/fp32_reference",
                available=True,
                reason="unchanged Stage A FunctionalNeuralMemory.update_segment",
            ),
            MemoryBackend.EXACT_ACCELERATED: BackendCapability(
                name=MemoryBackend.EXACT_ACCELERATED.value,
                exactness="tensor_exact_functional_loop",
                available=True,
                reason="B3 functional-loop refactor with evolving gradients and reference fallback",
            ),
            MemoryBackend.EXACT_SCAN: BackendCapability(
                name=MemoryBackend.EXACT_SCAN.value,
                exactness="unproven",
                available=False,
                reason="unavailable until B4 proves an associative nonlinear update",
            ),
            MemoryBackend.APPROXIMATE_SCAN: BackendCapability(
                name=MemoryBackend.APPROXIMATE_SCAN.value,
                exactness="approximate",
                available=True,
                reason="B5 explicit stale-within-window gradient ablation",
            ),
        }
        self._attention_capabilities: dict[AttentionBackend, BackendCapability] = {
            AttentionBackend.MULTIHEAD_ATTENTION: BackendCapability(
                name=AttentionBackend.MULTIHEAD_ATTENTION.value,
                exactness="reference",
                available=True,
                reason="unchanged Stage A torch.nn.MultiheadAttention path",
            ),
            AttentionBackend.SDPA: BackendCapability(
                name=AttentionBackend.SDPA.value,
                exactness="fp64_oracle/fp32_numerical_parity",
                available=True,
                reason="B6 functional SDPA with the exact additive [P,H,S] mask",
            ),
            AttentionBackend.FLASH: BackendCapability(
                name=AttentionBackend.FLASH.value,
                exactness="pending_hardware_probe",
                available=False,
                reason="unavailable until B6 validates mask support on CUDA/A100",
            ),
        }
        self._activation_dtypes = {
            ActivationDType.FP32,
            ActivationDType.BF16,
            ActivationDType.FP16,
        }
        from .exact_acceleration import ExactAcceleratedMemoryBackend
        from .attention import integrate_sdpa_attention

        self._memory_updates[MemoryBackend.EXACT_ACCELERATED] = ExactAcceleratedMemoryBackend()
        self._attention_integrations[AttentionBackend.SDPA] = integrate_sdpa_attention

    def register_memory(
        self,
        backend: MemoryBackend,
        implementation: MemoryUpdate,
        *,
        exactness: str,
        reason: str,
    ) -> None:
        if backend is MemoryBackend.REFERENCE:
            raise ValueError("the reference memory backend cannot be replaced")
        self._memory_updates[backend] = implementation
        self._memory_capabilities[backend] = BackendCapability(
            name=backend.value,
            exactness=exactness,
            available=True,
            reason=reason,
        )

    def register_attention(
        self,
        backend: AttentionBackend,
        implementation: AttentionIntegration,
        *,
        exactness: str,
        reason: str,
    ) -> None:
        if backend is AttentionBackend.MULTIHEAD_ATTENTION:
            raise ValueError("the reference attention backend cannot be replaced")
        self._attention_integrations[backend] = implementation
        self._attention_capabilities[backend] = BackendCapability(
            name=backend.value,
            exactness=exactness,
            available=True,
            reason=reason,
        )

    def enable_activation_dtype(self, dtype: ActivationDType) -> None:
        self._activation_dtypes.add(ActivationDType(dtype))

    def probe_and_enable_flash(
        self,
        block: PaperMACBlock,
        *,
        activation_dtype: ActivationDType = ActivationDType.FP16,
    ) -> dict[str, object]:
        """Register Flash only after a forced exact-mask A100 probe succeeds."""

        from .attention import integrate_flash_attention, probe_flash_mask_support

        result = probe_flash_mask_support(
            block,
            activation_dtype=activation_dtype,
        )
        if bool(result["available"]):
            self.register_attention(
                AttentionBackend.FLASH,
                integrate_flash_attention,
                exactness="exact_mask_mixed_precision_behavioral",
                reason=str(result["reason"]),
            )
        else:
            self._attention_capabilities[AttentionBackend.FLASH] = BackendCapability(
                name=AttentionBackend.FLASH.value,
                exactness="unavailable_hardware_probe",
                available=False,
                reason=str(result["reason"]),
            )
        return result

    def validate(self, config: StageBBackendConfig) -> None:
        memory = self._memory_capabilities[config.memory_backend]
        has_implementation = (
            config.memory_backend in self._memory_updates
            or config.memory_backend is MemoryBackend.APPROXIMATE_SCAN
        )
        if not memory.available or not has_implementation:
            raise BackendUnavailableError(f"memory backend {memory.name!r} is unavailable: {memory.reason}")
        attention = self._attention_capabilities[config.attention_backend]
        if not attention.available or config.attention_backend not in self._attention_integrations:
            raise BackendUnavailableError(
                f"attention backend {attention.name!r} is unavailable: {attention.reason}"
            )
        if config.activation_dtype not in self._activation_dtypes:
            raise BackendUnavailableError(
                f"activation dtype {config.activation_dtype.value!r} is unavailable until B6"
            )
        if (
            config.activation_dtype is not ActivationDType.FP32
            and config.attention_backend
            not in (AttentionBackend.SDPA, AttentionBackend.FLASH)
        ):
            raise BackendUnavailableError(
                "reduced precision is available only for reviewed SDPA/Flash attention"
            )

    def memory_capabilities(self) -> dict[str, dict[str, object]]:
        return {backend.value: capability.to_dict() for backend, capability in self._memory_capabilities.items()}

    def attention_capabilities(self) -> dict[str, dict[str, object]]:
        return {
            backend.value: capability.to_dict()
            for backend, capability in self._attention_capabilities.items()
        }

    def runtime_metadata(self, config: StageBBackendConfig) -> dict[str, object]:
        if config.memory_backend is MemoryBackend.APPROXIMATE_SCAN:
            from .approximate_scan import ApproximateScanMemoryBackend

            assert config.approximate_window is not None
            memory_implementation = ApproximateScanMemoryBackend(
                config.approximate_window
            )
        else:
            memory_implementation = self._memory_updates[config.memory_backend]
        metadata = (
            memory_implementation.runtime_metadata()
            if hasattr(memory_implementation, "runtime_metadata")
            else {"implementation": config.memory_backend.value}
        )
        return {
            "memory": metadata,
            "attention": {
                "implementation": config.attention_backend.value,
                "activation_dtype": config.activation_dtype.value,
                "memory_state_dtype": "unchanged_fp32_island_for_mixed_precision",
            },
            "gates": {
                "implementation": config.gate_backend.value,
                "convolution_kernel_size": (
                    config.convolution_kernel_size
                    if config.gate_backend is GateBackend.CAUSAL_CONVOLUTION
                    else None
                ),
            },
        }

    def execute(
        self,
        block: PaperMACBlock,
        state: PaperMACStreamState,
        segment_embeddings: Tensor,
        *,
        config: StageBBackendConfig = StageBBackendConfig(),
        valid_mask: Optional[Tensor] = None,
        convolutional_gates: object | None = None,
    ) -> PaperMACBlockOutput:
        """Run the explicit read/integrate/write transition through dispatch."""

        self.validate(config)
        retrieval, query_history = block.memory.read_segment_with_history(
            state, segment_embeddings
        )
        sequence = self._attention_integrations[config.attention_backend](
            block,
            retrieval,
            segment_embeddings,
            config.activation_dtype,
        )
        if config.gate_backend is GateBackend.CAUSAL_CONVOLUTION:
            if config.memory_backend is not MemoryBackend.REFERENCE:
                raise BackendUnavailableError(
                    "B2 causal_convolution is reviewed only with the reference recurrence"
                )
            from .convolution import (
                CausalConvolutionalUpdateGates,
                update_segment_with_convolutional_gates,
            )

            if not isinstance(convolutional_gates, CausalConvolutionalUpdateGates):
                raise BackendUnavailableError(
                    "causal_convolution requires a CausalConvolutionalUpdateGates module"
                )
            if convolutional_gates.kernel_size != config.convolution_kernel_size:
                raise ValueError("configured convolution kernel does not match the supplied module")
            updated_state = update_segment_with_convolutional_gates(
                block.memory,
                convolutional_gates,
                state,
                sequence,
                valid_mask=valid_mask,
            )
        elif config.memory_backend is MemoryBackend.APPROXIMATE_SCAN:
            from .approximate_scan import ApproximateScanMemoryBackend

            assert config.approximate_window is not None
            updated_state = ApproximateScanMemoryBackend(config.approximate_window)(
                block,
                state,
                sequence,
                valid_mask,
            )
        else:
            updated_state = self._memory_updates[config.memory_backend](
                block, state, sequence, valid_mask
            )
        updated_state = block.memory.advance_query_history(
            updated_state, query_history
        )
        return PaperMACBlockOutput(
            sequence=sequence,
            retrieval=retrieval,
            state=updated_state,
        )


def execute_stage_b(
    block: PaperMACBlock,
    state: PaperMACStreamState,
    segment_embeddings: Tensor,
    *,
    config: StageBBackendConfig = StageBBackendConfig(),
    valid_mask: Optional[Tensor] = None,
    registry: Optional[StageBBackendRegistry] = None,
    convolutional_gates: object | None = None,
) -> PaperMACBlockOutput:
    """Convenience entrypoint using a fresh conservative registry."""

    active_registry = StageBBackendRegistry() if registry is None else registry
    return active_registry.execute(
        block,
        state,
        segment_embeddings,
        config=config,
        valid_mask=valid_mask,
        convolutional_gates=convolutional_gates,
    )
