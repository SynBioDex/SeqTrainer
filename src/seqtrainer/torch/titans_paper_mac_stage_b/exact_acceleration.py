"""Exact functional-loop refactor for the evolving fast-weight recurrence."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock
from seqtrainer.torch.titans_paper_mac.memory import FunctionalNeuralMemory
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState


class ExactAcceleratedMemoryBackend:
    """Remove state-object churn without changing any tensor operation.

    The Stage A reference constructs a replacement ``PaperMACStreamState`` at
    every position.  This implementation carries the same ordered fast-weight
    and surprise pytrees through the loop and constructs the public replacement
    state once.  Gradients are still evaluated at the evolving weights in token
    order.  A conservative support predicate provides an explicit reference
    fallback rather than attempting a different algorithm.
    """

    def __init__(
        self,
        support_predicate: Optional[Callable[[PaperMACBlock], bool]] = None,
    ) -> None:
        self._support_predicate = support_predicate or self._default_support
        self.accelerated_calls = 0
        self.fallback_calls = 0
        self.last_execution = "not_run"

    @staticmethod
    def _default_support(block: PaperMACBlock) -> bool:
        return (
            type(block.memory) is FunctionalNeuralMemory
            and block.segment_length == 32
            and block.memory.segment_length == 32
        )

    def __call__(
        self,
        block: PaperMACBlock,
        state: PaperMACStreamState,
        sequence: Tensor,
        valid_mask: Optional[Tensor],
    ) -> PaperMACStreamState:
        if not self._support_predicate(block):
            self.fallback_calls += 1
            self.last_execution = "reference_fallback"
            return block.memory.update_segment(state, sequence, valid_mask=valid_mask)

        memory = block.memory
        if sequence.shape != (memory.segment_length, memory.d_model):
            raise ValueError(
                f"segment_embeddings must have shape ({memory.segment_length}, {memory.d_model})"
            )
        if valid_mask is not None and valid_mask.shape != (memory.segment_length,):
            raise ValueError(f"valid_mask must have shape ({memory.segment_length},)")
        if state.ended:
            raise RuntimeError("cannot update an ended stream")

        keys = memory.key_projection(sequence)
        values = memory.value_projection(sequence)
        gate_values = memory.gates(sequence)
        fast_weights = OrderedDict(state.fast_weights.items())
        surprise = OrderedDict(state.surprise.items())
        memory.begin_update_telemetry(fast_weights)
        for position in range(memory.segment_length):
            if valid_mask is not None and not bool(valid_mask[position].item()):
                continue
            fast_weights, surprise = memory.update_tensors(
                fast_weights,
                surprise,
                keys[position],
                values[position],
                alpha=gate_values.alpha[position],
                eta=gate_values.eta[position],
                theta=gate_values.theta[position],
            )
        self.accelerated_calls += 1
        self.last_execution = "exact_functional_loop"
        return state.replace(
            fast_weights=fast_weights,
            surprise=surprise,
            segment_index=state.segment_index + 1,
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "implementation": "exact_functional_loop",
            "last_execution": self.last_execution,
            "accelerated_calls": self.accelerated_calls,
            "fallback_calls": self.fallback_calls,
            "stale_gradients": False,
            "token_order_preserved": True,
        }
