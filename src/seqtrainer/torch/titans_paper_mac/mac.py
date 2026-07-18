"""Minimal paper-traceable MAC block with an explicit block-causal layout.

For a 32-token segment ``S`` and the fixed pre-write memory read ``H``, the
attention layout is ``[P, H_1..H_32, S_1..S_32]``.  ``P`` denotes learned
persistent tokens.  A retrieval or sequence query at position ``i`` sees only
``P`` and the paired retrieval/sequence prefix through ``i``.  Consequently,
neither a future retrieval nor a future segment token can alter an earlier
sequence output.

The block first calls :meth:`FunctionalNeuralMemory.read_segment`, causally
integrates ``[P, H, S]``, and only then calls ``update_segment`` with that core
output.  Thus all 32 values of ``H`` come from ``M_(t-1)`` and the only
published write is the post-core ``M_t`` state.  No scan, convolution,
flash-attention, or performance substitution is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .memory import FunctionalNeuralMemory
from .state import PaperMACStreamState


def block_causal_attention_mask(
    persistent_tokens: int,
    segment_length: int = 32,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return a boolean ``[P, H, S]`` attention mask for one MAC block.

    The returned convention matches :class:`torch.nn.MultiheadAttention`:
    ``True`` blocks an attention edge.  Persistent queries attend only to the
    persistent-token bank.  For either retrieval or sequence query ``i``, the
    allowed keys are every persistent token plus ``H_1:i`` and ``S_1:i``.
    """

    if persistent_tokens <= 0:
        raise ValueError("persistent_tokens must be positive")
    if segment_length != 32:
        raise ValueError("paper-MAC Stage A requires segment_length=32")

    layout_length = persistent_tokens + 2 * segment_length
    mask = torch.ones((layout_length, layout_length), dtype=torch.bool, device=device)
    mask[:persistent_tokens, :persistent_tokens] = False

    retrieval_start = persistent_tokens
    sequence_start = persistent_tokens + segment_length
    for position in range(segment_length):
        sequence_allowed = torch.cat(
            (
                torch.arange(persistent_tokens, device=device),
                torch.arange(retrieval_start, retrieval_start + position + 1, device=device),
                torch.arange(sequence_start, sequence_start + position + 1, device=device),
            )
        )
        mask[retrieval_start + position, sequence_allowed] = False
        mask[sequence_start + position, sequence_allowed] = False
    return mask


@dataclass(frozen=True)
class PaperMACBlockOutput:
    """Outputs from one 32-position pre-read / post-write MAC transition."""

    sequence: Tensor
    retrieval: Tensor
    state: PaperMACStreamState


class PaperMACBlock(nn.Module):
    """One minimal causal MAC block over a single 32-token stream segment.

    ``forward`` has two explicit phases: it obtains every retrieval from the
    incoming state ``M_(t-1)``, then uses those retrievals in causal attention
    and returns the once-updated ``M_t`` state.  The input state is never
    mutated, so a caller owns stream isolation by passing its matching state.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 1,
        persistent_tokens: int = 4,
        memory_depth: int = 2,
        segment_length: int = 32,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by a positive num_heads")
        if persistent_tokens <= 0:
            raise ValueError("persistent_tokens must be positive")
        if segment_length != 32:
            raise ValueError("paper-MAC Stage A requires segment_length=32")

        self.d_model = d_model
        self.segment_length = segment_length
        self.persistent_token_count = persistent_tokens
        self.memory = FunctionalNeuralMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            segment_length=segment_length,
        )
        self.persistent_tokens = nn.Parameter(torch.empty(persistent_tokens, d_model))
        nn.init.normal_(self.persistent_tokens, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def initial_state(self, stream_id: str) -> PaperMACStreamState:
        """Create the functional-memory state for one logical stream."""

        return self.memory.initial_state(stream_id)

    def attention_mask(self, *, device: Optional[torch.device] = None) -> Tensor:
        """Return this block's inspectable, no-future-leakage attention mask."""

        return block_causal_attention_mask(
            self.persistent_token_count,
            self.segment_length,
            device=device,
        )

    def _validate_layout_inputs(self, retrieval: Tensor, segment_embeddings: Tensor) -> None:
        expected = (self.segment_length, self.d_model)
        if retrieval.shape != expected:
            raise ValueError(f"retrieval must have shape {expected}")
        if segment_embeddings.shape != expected:
            raise ValueError(f"segment_embeddings must have shape {expected}")
        if retrieval.device != segment_embeddings.device:
            raise ValueError("retrieval and segment_embeddings must share a device")
        if retrieval.dtype != segment_embeddings.dtype:
            raise ValueError("retrieval and segment_embeddings must share a dtype")

    def integrate(self, retrieval: Tensor, segment_embeddings: Tensor) -> Tensor:
        """Causally integrate an already-read ``H`` and current segment ``S``.

        This separate primitive makes the mask independently testable: callers
        can perturb a future ``H_j`` or ``S_j`` and verify that earlier returned
        sequence positions are unchanged.
        """

        self._validate_layout_inputs(retrieval, segment_embeddings)
        persistent = self.persistent_tokens.to(
            device=segment_embeddings.device,
            dtype=segment_embeddings.dtype,
        )
        layout = torch.cat((persistent, retrieval, segment_embeddings), dim=0).unsqueeze(0)
        attended, _ = self.attention(
            layout,
            layout,
            layout,
            attn_mask=self.attention_mask(device=layout.device),
            need_weights=False,
        )
        sequence_start = self.persistent_token_count + self.segment_length
        return self.output_norm(segment_embeddings + attended[0, sequence_start:])

    def forward(
        self,
        state: PaperMACStreamState,
        segment_embeddings: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
    ) -> PaperMACBlockOutput:
        """Read all 32 values from ``M_(t-1)``, integrate, then write once."""

        retrieval = self.memory.read_segment(state, segment_embeddings)
        sequence = self.integrate(retrieval, segment_embeddings)
        updated_state = self.memory.update_segment(state, sequence, valid_mask=valid_mask)
        return PaperMACBlockOutput(sequence=sequence, retrieval=retrieval, state=updated_state)
