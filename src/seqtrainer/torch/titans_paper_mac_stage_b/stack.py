"""Minimal multi-block Stage B stack for scale and long-stream experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from torch import Tensor, nn

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState

from .backends import StageBBackendRegistry
from .config import StageBBackendConfig
from .convolution import CausalConvolutionalUpdateGates


@dataclass(frozen=True)
class StageBStackOutput:
    sequence: Tensor
    block_sequences: tuple[Tensor, ...]
    retrievals: tuple[Tensor, ...]
    states: tuple[PaperMACStreamState, ...]


class StageBMACStack(nn.Module):
    """Apply the paper-MAC transition at each layer with independent memory."""

    def __init__(
        self,
        block_count: int,
        d_model: int,
        *,
        num_heads: int,
        persistent_tokens: int = 4,
        memory_depth: int = 2,
        segment_length: int = 32,
        convolution_kernel_size: int | None = None,
        max_surprise_norm: float | None = None,
        associative_loss_reduction: str = "sum",
        max_gradient_rms: float | None = None,
        max_gradient_rms_ratio: float | None = None,
        theta_max: float = 1.0,
        theta_initial: float | None = None,
    ) -> None:
        super().__init__()
        if block_count <= 0:
            raise ValueError("block_count must be positive")
        self.block_count = block_count
        self.d_model = d_model
        self.segment_length = segment_length
        self.blocks = nn.ModuleList(
            PaperMACBlock(
                d_model=d_model,
                num_heads=num_heads,
                persistent_tokens=persistent_tokens,
                memory_depth=memory_depth,
                segment_length=segment_length,
                max_surprise_norm=max_surprise_norm,
                associative_loss_reduction=associative_loss_reduction,
                max_gradient_rms=max_gradient_rms,
                max_gradient_rms_ratio=max_gradient_rms_ratio,
                theta_max=theta_max,
                theta_initial=theta_initial,
            )
            for _ in range(block_count)
        )
        self.convolutional_gates = (
            nn.ModuleList(
                CausalConvolutionalUpdateGates(
                    d_model,
                    convolution_kernel_size,
                    reference_gates=block.memory.gates,
                )
                for block in self.blocks
            )
            if convolution_kernel_size is not None
            else None
        )

    def initial_states(self, stream_id: str) -> tuple[PaperMACStreamState, ...]:
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        return tuple(
            block.initial_state(f"{stream_id}:block-{index}")
            for index, block in enumerate(self.blocks)
        )

    def forward(
        self,
        states: Sequence[PaperMACStreamState],
        segment_embeddings: Tensor,
        *,
        config: StageBBackendConfig = StageBBackendConfig(),
        valid_mask: Optional[Tensor] = None,
        registry: Optional[StageBBackendRegistry] = None,
    ) -> StageBStackOutput:
        if len(states) != self.block_count:
            raise ValueError(f"states must contain {self.block_count} entries")
        active_registry = StageBBackendRegistry() if registry is None else registry
        sequence = segment_embeddings
        retrievals: list[Tensor] = []
        block_sequences: list[Tensor] = []
        next_states: list[PaperMACStreamState] = []
        for index, (block, state) in enumerate(zip(self.blocks, states)):
            convolutional_gates = (
                None if self.convolutional_gates is None else self.convolutional_gates[index]
            )
            output = active_registry.execute(
                block,
                state,
                sequence,
                config=config,
                valid_mask=valid_mask,
                convolutional_gates=convolutional_gates,
            )
            sequence = output.sequence
            block_sequences.append(output.sequence)
            retrievals.append(output.retrieval)
            next_states.append(output.state)
        return StageBStackOutput(
            sequence=sequence,
            block_sequences=tuple(block_sequences),
            retrievals=tuple(retrievals),
            states=tuple(next_states),
        )
