"""Functional neural long-term memory for the paper-traceable Titans MAC path.

The memory follows the paper's associative objective and update equations:

``L_t = 1/2 ||M_(t-1)(k_t) - v_t||²``
``S_t = eta_t S_(t-1) - theta_t grad(L_t)``
``M_t = (1 - alpha_t) M_(t-1) + S_t``

``read_then_update`` reads the same ``M_(t-1)`` across the complete 32-token
segment and commits only the final state after all local update steps.  It
never detaches or mutates a fast-weight tensor in place.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .state import FastWeights, PaperMACStreamState


@dataclass(frozen=True)
class GateValues:
    """Data-dependent gates with shape ``(..., 1)`` for parameter broadcast."""

    alpha: Tensor
    eta: Tensor
    theta: Tensor

    def __post_init__(self) -> None:
        if self.alpha.shape != self.eta.shape or self.alpha.shape != self.theta.shape:
            raise ValueError("alpha, eta, and theta must have identical shapes")
        if not self.alpha.shape or self.alpha.shape[-1] != 1:
            raise ValueError("gate tensors must end in a singleton dimension")


class AdaptiveUpdateGates(nn.Module):
    """Project each token to paper update gates ``alpha``, ``eta``, and ``theta``.

    Input has shape ``(..., d_model)`` and every returned gate has shape
    ``(..., 1)``.  Sigmoid constrains all three gates to ``(0, 1)``; ``theta``
    is therefore a data-dependent learning-rate multiplier.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.projection = nn.Linear(d_model, 3)
        # The local update is applied 32 times before a state is committed.
        # Starting every gate near 0.5 makes that recurrence explode before
        # outer-loop gradient clipping can intervene. Keep the gates learnable,
        # but begin with rapid forgetting and conservative momentum/updates.
        with torch.no_grad():
            self.projection.bias.copy_(
                torch.tensor((4.0, -4.0, -6.0), dtype=self.projection.bias.dtype)
            )

    def forward(self, token_embeddings: Tensor) -> GateValues:
        if token_embeddings.ndim < 1 or token_embeddings.size(-1) != self.projection.in_features:
            raise ValueError("token_embeddings must have shape (..., d_model)")
        values = torch.sigmoid(self.projection(token_embeddings))
        alpha, eta, theta = values.unbind(dim=-1)
        return GateValues(alpha=alpha.unsqueeze(-1), eta=eta.unsqueeze(-1), theta=theta.unsqueeze(-1))


class FunctionalNeuralMemory(nn.Module):
    """MLP long-term memory with differentiable, stream-local fast weights.

    A state represents exactly one stream.  Segment inputs have shape
    ``(32, d_model)`` and gates have shape ``(32, 1)``.  Multiple streams are
    kept isolated by calling :meth:`initial_state` once per stream and passing
    the matching state to :meth:`read_then_update` or :meth:`update_segment`.
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        segment_length: int = 32,
        *,
        max_surprise_norm: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if memory_depth <= 0:
            raise ValueError("memory_depth must be positive")
        if segment_length != 32:
            raise ValueError("paper-MAC Stage A requires segment_length=32")
        if max_surprise_norm is not None and max_surprise_norm <= 0:
            raise ValueError("max_surprise_norm must be positive when supplied")
        self.d_model = d_model
        self.segment_length = segment_length
        self.max_surprise_norm = max_surprise_norm

        layers: list[nn.Module] = []
        for _ in range(memory_depth - 1):
            layers.extend((nn.Linear(d_model, d_model), nn.GELU()))
        layers.append(nn.Linear(d_model, d_model))
        self.memory_mlp = nn.Sequential(*layers)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.gates = AdaptiveUpdateGates(d_model)

    def initial_fast_weights(self) -> FastWeights:
        """Return the MLP's meta-parameters as the initial functional memory."""

        return OrderedDict(self.memory_mlp.named_parameters())

    def initial_state(self, stream_id: str) -> PaperMACStreamState:
        """Return an independent logical stream state with shared meta-initialization."""

        return PaperMACStreamState.initial(stream_id, self.initial_fast_weights())

    def reset_state(self, state: PaperMACStreamState) -> PaperMACStreamState:
        """Reset one stream only; this method never changes a module buffer."""

        return state.reset(self.initial_fast_weights())

    def _validate_vector(self, value: Tensor, name: str) -> None:
        if value.shape != (self.d_model,):
            raise ValueError(f"{name} must have shape ({self.d_model},)")

    def _validate_segment(self, segment_embeddings: Tensor, valid_mask: Optional[Tensor]) -> None:
        if segment_embeddings.shape != (self.segment_length, self.d_model):
            raise ValueError(
                f"segment_embeddings must have shape ({self.segment_length}, {self.d_model})"
            )
        if valid_mask is not None and valid_mask.shape != (self.segment_length,):
            raise ValueError(f"valid_mask must have shape ({self.segment_length},)")

    def _functional_memory(self, fast_weights: Mapping[str, Tensor], inputs: Tensor) -> Tensor:
        return functional_call(self.memory_mlp, OrderedDict(fast_weights.items()), (inputs,), strict=True)

    def retrieve(self, state: PaperMACStreamState, queries: Tensor) -> Tensor:
        """Read ``M_(t-1)`` without changing it; output shape matches ``queries``."""

        if queries.shape[-1:] != (self.d_model,):
            raise ValueError("queries must have shape (..., d_model)")
        return self._functional_memory(state.fast_weights, queries)

    def associative_loss(self, fast_weights: Mapping[str, Tensor], keys: Tensor, values: Tensor) -> Tensor:
        """Return unreduced per-token associative reconstruction loss (equations 11–12)."""

        if keys.shape != values.shape or keys.shape[-1:] != (self.d_model,):
            raise ValueError("keys and values must have equal shape (..., d_model)")
        reconstruction = self._functional_memory(fast_weights, keys)
        return 0.5 * (reconstruction - values).square().sum(dim=-1)

    def surprise_gradient(self, fast_weights: Mapping[str, Tensor], key: Tensor, value: Tensor) -> FastWeights:
        """Compute the exact higher-order gradient of one associative loss."""

        self._validate_vector(key, "key")
        self._validate_vector(value, "value")
        loss = self.associative_loss(fast_weights, key, value)
        gradients = torch.autograd.grad(
            loss,
            tuple(fast_weights.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        return OrderedDict(zip(fast_weights, gradients))

    @staticmethod
    def momentum_update(
        previous_surprise: Mapping[str, Tensor], gradient: Mapping[str, Tensor], eta: Tensor, theta: Tensor
    ) -> FastWeights:
        """Apply equation 14: ``eta * S_(t-1) - theta * gradient``."""

        return OrderedDict(
            (name, eta * previous_surprise[name] - theta * gradient[name]) for name in previous_surprise
        )

    def _bound_surprise(self, surprise: Mapping[str, Tensor]) -> FastWeights:
        """Project the local surprise vector into an optional trust region.

        The paper recurrence is unbounded.  During long outer-loop training a
        rare associative-gradient spike can otherwise become the next
        fast-weight state and make the following read non-finite.  A single
        scale preserves its direction and is differentiable almost everywhere;
        disabled guards retain the exact original recurrence.
        """

        bounded = OrderedDict(surprise.items())
        if self.max_surprise_norm is None:
            return bounded
        squared = torch.stack([value.square().sum() for value in bounded.values()]).sum()
        norm = squared.sqrt()
        scale = (self.max_surprise_norm / norm.clamp_min(self.max_surprise_norm)).detach()
        return OrderedDict((name, value * scale) for name, value in bounded.items())

    @staticmethod
    def forgetting_update(
        fast_weights: Mapping[str, Tensor], surprise: Mapping[str, Tensor], alpha: Tensor
    ) -> FastWeights:
        """Apply equation 13: ``(1 - alpha) * M_(t-1) + S_t``."""

        return OrderedDict((name, (1.0 - alpha) * fast_weights[name] + surprise[name]) for name in fast_weights)

    def update_one(
        self,
        state: PaperMACStreamState,
        key: Tensor,
        value: Tensor,
        *,
        alpha: Tensor,
        eta: Tensor,
        theta: Tensor,
    ) -> PaperMACStreamState:
        """Apply one exact local inner-loop update without committing a segment.

        This public primitive makes the equations independently testable.  It
        leaves ``segment_index`` unchanged; :meth:`update_segment` is the sole
        operation that commits a completed 32-token segment.
        """

        self._validate_vector(key, "key")
        self._validate_vector(value, "value")
        for name, gate in (("alpha", alpha), ("eta", eta), ("theta", theta)):
            if gate.numel() != 1:
                raise ValueError(f"{name} must be a scalar or a single-element tensor")
        gradient = self.surprise_gradient(state.fast_weights, key, value)
        surprise = self._bound_surprise(
            self.momentum_update(state.surprise, gradient, eta, theta)
        )
        fast_weights = self.forgetting_update(state.fast_weights, surprise, alpha)
        return state.replace(fast_weights=fast_weights, surprise=surprise)

    def update_segment(
        self,
        state: PaperMACStreamState,
        segment_embeddings: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
    ) -> PaperMACStreamState:
        """Run all 32 differentiable local updates and commit one replacement state.

        ``segment_embeddings`` has shape ``(32, d_model)``.  A false entry in
        ``valid_mask`` represents tail padding and is excluded from the update.
        The input state is never mutated and remains the snapshot that was read
        for every retrieval in the segment.
        """

        self._validate_segment(segment_embeddings, valid_mask)
        if state.ended:
            raise RuntimeError("cannot update an ended stream")
        keys = self.key_projection(segment_embeddings)
        values = self.value_projection(segment_embeddings)
        gate_values = self.gates(segment_embeddings)
        candidate = state
        for position in range(self.segment_length):
            if valid_mask is not None and not bool(valid_mask[position].item()):
                continue
            candidate = self.update_one(
                candidate,
                keys[position],
                values[position],
                alpha=gate_values.alpha[position],
                eta=gate_values.eta[position],
                theta=gate_values.theta[position],
            )
        return candidate.replace(
            fast_weights=candidate.fast_weights,
            surprise=candidate.surprise,
            segment_index=state.segment_index + 1,
        )

    def read_then_update(
        self,
        state: PaperMACStreamState,
        segment_embeddings: Tensor,
        *,
        valid_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, PaperMACStreamState]:
        """Read one fixed ``M_(t-1)`` across the segment, then return one new state.

        This is the boundary used by the future MAC core.  It intentionally
        exposes no intermediate writes while preserving autograd through all 32
        update steps in the returned state.
        """

        self._validate_segment(segment_embeddings, valid_mask)
        queries = self.query_projection(segment_embeddings)
        retrieval = self.retrieve(state, queries)
        return retrieval, self.update_segment(state, segment_embeddings, valid_mask=valid_mask)

    def read_segment(self, state: PaperMACStreamState, segment_embeddings: Tensor) -> Tensor:
        """Retrieve all 32 positions from one immutable incoming state.

        This is the read-only boundary used by :class:`PaperMACBlock`.  Keeping
        it separate from :meth:`update_segment` lets the MAC core decide which
        causally integrated representations are useful enough to write.
        """

        self._validate_segment(segment_embeddings, valid_mask=None)
        return self.retrieve(state, self.query_projection(segment_embeddings))
