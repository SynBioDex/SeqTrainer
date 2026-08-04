"""Opt-in causal convolution for adaptive neural-memory gates."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from seqtrainer.torch.titans_paper_mac.memory import (
    AdaptiveUpdateGates,
    FunctionalNeuralMemory,
    GateValues,
)
from seqtrainer.torch.titans_paper_mac.state import PaperMACStreamState


class CausalConvolutionalUpdateGates(nn.Module):
    """Left-padded depthwise temporal context followed by three gate logits.

    The paper does not specify kernel width, grouping, or padding for its
    convolution ablation.  Stage B chooses the smallest inspectable sequence
    mixer: one depthwise causal moving-average kernel, then a projection copied
    from the Stage A token-wise gate module.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        *,
        reference_gates: Optional[AdaptiveUpdateGates] = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            groups=d_model,
            bias=False,
        )
        with torch.no_grad():
            self.depthwise.weight.fill_(1.0 / kernel_size)
        self.projection = nn.Linear(d_model, 3)
        if reference_gates is not None:
            if reference_gates.projection.in_features != d_model:
                raise ValueError("reference gate dimension does not match d_model")
            self.projection.load_state_dict(copy.deepcopy(reference_gates.projection.state_dict()))

    def contextualize(self, token_embeddings: Tensor) -> Tensor:
        if token_embeddings.ndim != 2 or token_embeddings.shape[-1] != self.d_model:
            raise ValueError("token_embeddings must have shape (sequence, d_model)")
        channels_first = token_embeddings.transpose(0, 1).unsqueeze(0)
        padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        return self.depthwise(padded).squeeze(0).transpose(0, 1)

    def forward(self, token_embeddings: Tensor) -> GateValues:
        contextual = self.contextualize(token_embeddings)
        values = torch.sigmoid(self.projection(contextual))
        alpha, eta, theta = values.unbind(dim=-1)
        return GateValues(
            alpha=alpha.unsqueeze(-1),
            eta=eta.unsqueeze(-1),
            theta=theta.unsqueeze(-1),
        )


def update_segment_with_convolutional_gates(
    memory: FunctionalNeuralMemory,
    gates: CausalConvolutionalUpdateGates,
    state: PaperMACStreamState,
    segment_embeddings: Tensor,
    *,
    valid_mask: Optional[Tensor] = None,
) -> PaperMACStreamState:
    """Apply Stage A recurrence with only its gate input path changed."""

    if segment_embeddings.shape != (memory.segment_length, memory.d_model):
        raise ValueError(
            f"segment_embeddings must have shape ({memory.segment_length}, {memory.d_model})"
        )
    if gates.d_model != memory.d_model:
        raise ValueError("convolutional gate dimension must match the memory")
    if valid_mask is not None and valid_mask.shape != (memory.segment_length,):
        raise ValueError(f"valid_mask must have shape ({memory.segment_length},)")
    if state.ended:
        raise RuntimeError("cannot update an ended stream")
    keys = memory.key_projection(segment_embeddings)
    values = memory.value_projection(segment_embeddings)
    gate_values = gates(segment_embeddings)
    fast_weights = OrderedDict(state.fast_weights.items())
    surprise = OrderedDict(state.surprise.items())
    for position in range(memory.segment_length):
        if valid_mask is not None and not bool(valid_mask[position].item()):
            continue
        gradient = memory.surprise_gradient(
            fast_weights,
            keys[position],
            values[position],
        )
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

