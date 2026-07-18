"""Explicit stale-window approximation for the nonlinear memory recurrence."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock
from seqtrainer.torch.titans_paper_mac.memory import FunctionalNeuralMemory
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState

from .config import APPROXIMATE_WINDOWS


class ApproximateScanMemoryBackend:
    """Compute gradients from one stale fast-weight snapshot per window.

    For window start ``w``, every associative gradient in
    ``[w, w + window_size)`` is evaluated at ``M_w``. The gradients are then
    consumed in token order while surprise, momentum, forgetting, and gates
    continue to evolve sequentially. The next window snapshots the resulting
    fast weights. This is an approximation for every supported size because
    the exact nonlinear gradient normally depends on each immediately
    preceding update.
    """

    exactness = "approximate_stale_within_window"

    def __init__(self, window_size: int) -> None:
        if window_size not in APPROXIMATE_WINDOWS:
            raise ValueError(f"window_size must be in {APPROXIMATE_WINDOWS}")
        self.window_size = window_size

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "implementation": "approximate_scan",
            "exactness": self.exactness,
            "window_size": self.window_size,
            "staleness": (
                "all surprise gradients in a window use its incoming fast-weight snapshot; "
                "gates/momentum/forgetting remain sequential"
            ),
            "published_states_per_segment": 1,
        }

    def __call__(
        self,
        block: PaperMACBlock,
        state: PaperMACStreamState,
        sequence: Tensor,
        valid_mask: Optional[Tensor],
    ) -> PaperMACStreamState:
        return update_segment_with_stale_windows(
            block.memory,
            state,
            sequence,
            window_size=self.window_size,
            valid_mask=valid_mask,
        )


def update_segment_with_stale_windows(
    memory: FunctionalNeuralMemory,
    state: PaperMACStreamState,
    segment_embeddings: Tensor,
    *,
    window_size: int,
    valid_mask: Optional[Tensor] = None,
) -> PaperMACStreamState:
    """Apply the declared B5 approximation and publish one post-segment state."""

    backend = ApproximateScanMemoryBackend(window_size)
    memory._validate_segment(segment_embeddings, valid_mask)
    if state.ended:
        raise RuntimeError("cannot update an ended stream")
    keys = memory.key_projection(segment_embeddings)
    values = memory.value_projection(segment_embeddings)
    gate_values = memory.gates(segment_embeddings)
    fast_weights = OrderedDict(state.fast_weights.items())
    surprise = OrderedDict(state.surprise.items())

    for window_start in range(0, memory.segment_length, backend.window_size):
        window_end = min(window_start + backend.window_size, memory.segment_length)
        stale_snapshot = OrderedDict(fast_weights.items())
        gradients: dict[int, OrderedDict[str, Tensor]] = {}
        for position in range(window_start, window_end):
            if valid_mask is not None and not bool(valid_mask[position].item()):
                continue
            gradients[position] = memory.surprise_gradient(
                stale_snapshot,
                keys[position],
                values[position],
            )
        for position, gradient in gradients.items():
            surprise = memory.momentum_update(
                surprise,
                gradient,
                gate_values.eta[position],
                gate_values.theta[position],
            )
            fast_weights = memory.forgetting_update(
                fast_weights,
                surprise,
                gate_values.alpha[position],
            )
    return state.replace(
        fast_weights=fast_weights,
        surprise=surprise,
        segment_index=state.segment_index + 1,
    )

