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
from typing import Literal, Mapping, Optional, TypeAlias

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


@dataclass(frozen=True)
class ParameterGateValues:
    """Per-fast-parameter channel gates for the paper-default deep memory."""

    alpha: Mapping[str, Tensor]
    eta: Mapping[str, Tensor]
    theta: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        keys = tuple(self.alpha)
        if keys != tuple(self.eta) or keys != tuple(self.theta):
            raise ValueError("parameter alpha, eta, and theta gates must share ordered keys")
        if not keys:
            raise ValueError("parameter gates must not be empty")
        for name in keys:
            if (
                self.alpha[name].shape != self.eta[name].shape
                or self.alpha[name].shape != self.theta[name].shape
            ):
                raise ValueError(f"parameter gates for {name!r} must have identical shapes")


GateLike: TypeAlias = Tensor | Mapping[str, Tensor]


class PaperResidualMemory(nn.Module):
    """Paper-default two-layer, 4x-expanded residual neural memory."""

    def __init__(self, d_model: int, expansion_factor: int = 4) -> None:
        super().__init__()
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        hidden = d_model * expansion_factor
        self.in_projection = nn.Linear(d_model, hidden)
        self.out_projection = nn.Linear(hidden, d_model)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        transformed = self.out_projection(torch.nn.functional.gelu(self.in_projection(inputs)))
        return inputs + self.normalization(transformed)


class PerLayerChannelUpdateGates(nn.Module):
    """Generate independent channel-wise update gates for each deep-memory layer."""

    PARAMETER_LAYERS = {
        "in_projection.weight": "in_projection",
        "in_projection.bias": "in_projection",
        "out_projection.weight": "out_projection",
        "out_projection.bias": "out_projection",
        "normalization.weight": "out_projection",
        "normalization.bias": "out_projection",
    }

    def __init__(
        self,
        d_model: int,
        expansion_factor: int,
        *,
        alpha_initial: float,
        eta_initial: float,
        theta_max: float,
        theta_initial: float,
    ) -> None:
        super().__init__()
        for name, value in (
            ("alpha_initial", alpha_initial),
            ("eta_initial", eta_initial),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if not 0.0 < theta_initial < theta_max <= 1.0:
            raise ValueError("theta_initial must be in (0, theta_max] and theta_max <= 1")
        self.theta_max = float(theta_max)
        widths = {
            "in_projection": d_model * expansion_factor,
            "out_projection": d_model,
        }
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, 3 * width) for name, width in widths.items()}
        )
        with torch.no_grad():
            for head in self.heads.values():
                width = head.out_features // 3
                head.weight.zero_()
                head.bias[:width].fill_(torch.logit(torch.tensor(alpha_initial)).item())
                head.bias[width : 2 * width].fill_(
                    torch.logit(torch.tensor(eta_initial)).item()
                )
                head.bias[2 * width :].fill_(
                    torch.logit(torch.tensor(theta_initial / theta_max)).item()
                )

    @staticmethod
    def _broadcast_channels(values: Tensor, *, weight: bool) -> Tensor:
        return values.unsqueeze(-1) if weight else values

    def forward(self, token_embeddings: Tensor) -> ParameterGateValues:
        per_layer: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for layer, head in self.heads.items():
            alpha, eta, theta_unit = torch.sigmoid(head(token_embeddings)).chunk(3, dim=-1)
            per_layer[layer] = (alpha, eta, self.theta_max * theta_unit)
        mappings: dict[str, OrderedDict[str, Tensor]] = {
            name: OrderedDict() for name in ("alpha", "eta", "theta")
        }
        for parameter, layer in self.PARAMETER_LAYERS.items():
            weight = parameter.endswith(".weight") and not parameter.startswith("normalization")
            values = per_layer[layer]
            for index, name in enumerate(("alpha", "eta", "theta")):
                mappings[name][parameter] = self._broadcast_channels(
                    values[index],
                    weight=weight,
                )
        return ParameterGateValues(**mappings)


class AdaptiveUpdateGates(nn.Module):
    """Project each token to paper update gates ``alpha``, ``eta``, and ``theta``.

    Input has shape ``(..., d_model)`` and every returned gate has shape
    ``(..., 1)``.  Sigmoid constrains all three gates to ``(0, 1)``; ``theta``
    is therefore a data-dependent learning-rate multiplier.
    """

    def __init__(
        self,
        d_model: int,
        *,
        theta_max: float = 1.0,
        theta_initial: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if not 0.0 < theta_max <= 1.0:
            raise ValueError("theta_max must be in (0, 1]")
        if theta_initial is not None and not 0.0 < theta_initial < theta_max:
            raise ValueError("theta_initial must be in (0, theta_max)")
        self.theta_max = float(theta_max)
        self.projection = nn.Linear(d_model, 3)
        # The local update is applied 32 times before a state is committed.
        # Starting every gate near 0.5 makes that recurrence explode before
        # outer-loop gradient clipping can intervene. Keep the gates learnable,
        # but begin with rapid forgetting and conservative momentum/updates.
        with torch.no_grad():
            theta_fraction = (
                torch.sigmoid(torch.tensor(-6.0)).item()
                if theta_initial is None
                else theta_initial / theta_max
            )
            theta_bias = torch.logit(torch.tensor(theta_fraction)).item()
            self.projection.bias.copy_(torch.tensor(
                (4.0, -4.0, theta_bias), dtype=self.projection.bias.dtype
            ))

    def forward(self, token_embeddings: Tensor) -> GateValues:
        if token_embeddings.ndim < 1 or token_embeddings.size(-1) != self.projection.in_features:
            raise ValueError("token_embeddings must have shape (..., d_model)")
        values = torch.sigmoid(self.projection(token_embeddings))
        alpha, eta, theta_unit = values.unbind(dim=-1)
        theta = self.theta_max * theta_unit
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
        architecture: Literal["legacy_mlp_v1", "paper_residual_mlp_v2"] = "legacy_mlp_v1",
        expansion_factor: int = 4,
        projection_convolution_kernel: int | None = None,
        normalize_queries_and_keys: bool = False,
        max_surprise_norm: float | None = None,
        associative_loss_reduction: Literal["sum", "mean"] = "sum",
        max_gradient_rms: float | None = None,
        max_gradient_rms_ratio: float | None = None,
        theta_max: float = 1.0,
        theta_initial: float | None = None,
        alpha_initial: float | None = None,
        eta_initial: float | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if memory_depth <= 0:
            raise ValueError("memory_depth must be positive")
        if architecture not in {"legacy_mlp_v1", "paper_residual_mlp_v2"}:
            raise ValueError("unknown neural-memory architecture")
        if architecture == "paper_residual_mlp_v2" and memory_depth != 2:
            raise ValueError("paper_residual_mlp_v2 requires memory_depth=2")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        if projection_convolution_kernel is not None and projection_convolution_kernel < 2:
            raise ValueError("projection_convolution_kernel must be at least 2")
        if segment_length != 32:
            raise ValueError("paper-MAC Stage A requires segment_length=32")
        if max_surprise_norm is not None and max_surprise_norm <= 0:
            raise ValueError("max_surprise_norm must be positive when supplied")
        if associative_loss_reduction not in {"sum", "mean"}:
            raise ValueError("associative_loss_reduction must be 'sum' or 'mean'")
        if max_gradient_rms is not None and max_gradient_rms <= 0:
            raise ValueError("max_gradient_rms must be positive when supplied")
        if max_gradient_rms_ratio is not None and max_gradient_rms_ratio <= 0:
            raise ValueError("max_gradient_rms_ratio must be positive when supplied")
        self.d_model = d_model
        self.segment_length = segment_length
        self.architecture = architecture
        self.expansion_factor = expansion_factor
        self.projection_convolution_kernel = projection_convolution_kernel
        self.normalize_queries_and_keys = normalize_queries_and_keys
        self.max_surprise_norm = max_surprise_norm
        self.associative_loss_reduction = associative_loss_reduction
        self.max_gradient_rms = max_gradient_rms
        self.max_gradient_rms_ratio = max_gradient_rms_ratio
        self._update_telemetry: dict[str, Tensor] = {}

        if architecture == "paper_residual_mlp_v2":
            self.memory_mlp = PaperResidualMemory(d_model, expansion_factor)
        else:
            layers: list[nn.Module] = []
            for _ in range(memory_depth - 1):
                layers.extend((nn.Linear(d_model, d_model), nn.GELU()))
            layers.append(nn.Linear(d_model, d_model))
            self.memory_mlp = nn.Sequential(*layers)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        if projection_convolution_kernel is None:
            self.key_convolution = None
            self.value_convolution = None
            self.query_convolution = None
        else:
            def convolution() -> nn.Conv1d:
                module = nn.Conv1d(
                    d_model,
                    d_model,
                    projection_convolution_kernel,
                    groups=d_model,
                    bias=False,
                )
                with torch.no_grad():
                    module.weight.zero_()
                    module.weight[:, 0, -1] = 1.0
                return module

            self.key_convolution = convolution()
            self.value_convolution = convolution()
            self.query_convolution = convolution()
        if architecture == "paper_residual_mlp_v2":
            self.gates = PerLayerChannelUpdateGates(
                d_model,
                expansion_factor,
                alpha_initial=1e-3 if alpha_initial is None else alpha_initial,
                eta_initial=0.9 if eta_initial is None else eta_initial,
                theta_max=theta_max,
                theta_initial=1e-3 if theta_initial is None else theta_initial,
            )
        else:
            self.gates = AdaptiveUpdateGates(
                d_model,
                theta_max=theta_max,
                theta_initial=theta_initial,
            )

    def initial_fast_weights(self) -> FastWeights:
        """Return the MLP's meta-parameters as the initial functional memory."""

        return OrderedDict(self.memory_mlp.named_parameters())

    def initial_state(self, stream_id: str) -> PaperMACStreamState:
        """Return an independent logical stream state with shared meta-initialization."""

        history_length = (
            0
            if self.projection_convolution_kernel is None
            else self.projection_convolution_kernel - 1
        )
        return PaperMACStreamState.initial(
            stream_id,
            self.initial_fast_weights(),
            projection_history_length=history_length,
            d_model=self.d_model,
        )

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

    def _project_with_history(
        self,
        projection: nn.Linear,
        convolution: nn.Conv1d | None,
        inputs: Tensor,
        history: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if convolution is None:
            return projection(inputs), None
        expected_history = convolution.kernel_size[0] - 1
        if history is None or history.shape != (expected_history, self.d_model):
            raise ValueError("stream projection history is missing or has the wrong shape")
        combined = torch.cat((history.to(inputs), inputs), dim=0)
        projected = projection(combined).transpose(0, 1).unsqueeze(0)
        convolved = convolution(projected).squeeze(0).transpose(0, 1)
        return convolved, combined[-expected_history:]

    @staticmethod
    def _l2_normalize(values: Tensor) -> Tensor:
        epsilon = torch.finfo(values.dtype).eps
        return values / values.norm(dim=-1, keepdim=True).clamp_min(epsilon)

    def project_queries(
        self,
        state: PaperMACStreamState,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        queries, history = self._project_with_history(
            self.query_projection,
            self.query_convolution,
            inputs,
            state.query_history,
        )
        if self.normalize_queries_and_keys:
            queries = self._l2_normalize(queries)
        return queries, history

    def project_writes(
        self,
        state: PaperMACStreamState,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        keys, history = self._project_with_history(
            self.key_projection,
            self.key_convolution,
            inputs,
            state.write_history,
        )
        values, value_history = self._project_with_history(
            self.value_projection,
            self.value_convolution,
            inputs,
            state.write_history,
        )
        if history is not None and value_history is not None and not torch.equal(history, value_history):
            raise RuntimeError("key/value projection histories diverged")
        if self.normalize_queries_and_keys:
            keys = self._l2_normalize(keys)
        return keys, values, history

    def advance_query_history(
        self,
        state: PaperMACStreamState,
        history: Tensor | None,
    ) -> PaperMACStreamState:
        if history is None:
            return state
        return state.replace(
            fast_weights=state.fast_weights,
            surprise=state.surprise,
            query_history=history,
        )

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
        squared_error = (reconstruction - values).square()
        if self.associative_loss_reduction == "mean":
            return 0.5 * squared_error.mean(dim=-1)
        return 0.5 * squared_error.sum(dim=-1)

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
    def _mapping_rms(values: Mapping[str, Tensor]) -> Tensor:
        squared = torch.stack([value.square().sum() for value in values.values()]).sum()
        count = sum(value.numel() for value in values.values())
        return (squared / count).sqrt()

    def begin_update_telemetry(self, reference: Mapping[str, Tensor]) -> None:
        """Reset detached diagnostics for one segment update."""

        zero = next(iter(reference.values())).new_zeros(())
        self._update_telemetry = {
            "raw_gradient_rms_max": zero,
            "conditioned_gradient_rms_max": zero,
            "gradient_scale_min": zero.new_ones(()),
            "gradient_interventions": zero,
            "legacy_surprise_interventions": zero,
            "update_count": zero,
            "past_surprise_rms_max": zero,
            "momentary_surprise_rms_max": zero,
            "combined_surprise_rms_max": zero,
            "forgotten_weight_rms_max": zero,
            "past_momentary_cosine_sum": zero,
        }

    def update_telemetry(self) -> Mapping[str, Tensor]:
        """Return detached diagnostics from the most recent segment update."""

        return self._update_telemetry

    def _record_gradient_telemetry(
        self,
        raw_rms: Tensor,
        conditioned_rms: Tensor,
        scale: Tensor,
    ) -> None:
        if not self._update_telemetry:
            return
        telemetry = self._update_telemetry
        telemetry["raw_gradient_rms_max"] = torch.maximum(
            telemetry["raw_gradient_rms_max"], raw_rms.detach()
        )
        telemetry["conditioned_gradient_rms_max"] = torch.maximum(
            telemetry["conditioned_gradient_rms_max"], conditioned_rms.detach()
        )
        telemetry["gradient_scale_min"] = torch.minimum(
            telemetry["gradient_scale_min"], scale.detach()
        )
        telemetry["gradient_interventions"] = (
            telemetry["gradient_interventions"] + scale.detach().lt(1.0).to(scale.dtype)
        )
        telemetry["update_count"] = telemetry["update_count"] + 1.0

    def condition_gradient(
        self,
        fast_weights: Mapping[str, Tensor],
        gradient: Mapping[str, Tensor],
    ) -> FastWeights:
        """Apply the declared scale-aware inner-gradient trust region.

        For the concatenated gradient vector ``g`` and fast-weight vector ``M``,
        ``rms(g) = ||g||_2 / sqrt(N)``.  If configured, the admissible RMS is
        ``min(c, rho * rms(M))`` and the returned gradient is
        ``g * min(1, admissible / rms(g))``.  The operation is differentiable
        almost everywhere and remains in the higher-order training graph.
        """

        raw_rms = self._mapping_rms(gradient)
        limits: list[Tensor] = []
        if self.max_gradient_rms is not None:
            limits.append(raw_rms.new_tensor(self.max_gradient_rms))
        if self.max_gradient_rms_ratio is not None:
            limits.append(
                self._mapping_rms(fast_weights) * self.max_gradient_rms_ratio
            )
        if limits:
            admissible = torch.stack(limits).min()
            scale = torch.minimum(
                torch.ones_like(raw_rms),
                admissible / raw_rms.clamp_min(torch.finfo(raw_rms.dtype).eps),
            )
        else:
            scale = torch.ones_like(raw_rms)
        conditioned = OrderedDict((name, value * scale) for name, value in gradient.items())
        self._record_gradient_telemetry(raw_rms, self._mapping_rms(conditioned), scale)
        return conditioned

    @staticmethod
    def _parameter_gate(gate: GateLike, name: str) -> Tensor:
        if isinstance(gate, Tensor):
            return gate
        if name not in gate:
            raise ValueError(f"missing channel gate for fast parameter {name!r}")
        return gate[name]

    @classmethod
    def momentum_update(
        cls,
        previous_surprise: Mapping[str, Tensor],
        gradient: Mapping[str, Tensor],
        eta: GateLike,
        theta: GateLike,
    ) -> FastWeights:
        """Apply equation 14: ``eta * S_(t-1) - theta * gradient``."""

        return OrderedDict(
            (
                name,
                cls._parameter_gate(eta, name) * previous_surprise[name]
                - cls._parameter_gate(theta, name) * gradient[name],
            )
            for name in previous_surprise
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
        if self._update_telemetry:
            self._update_telemetry["legacy_surprise_interventions"] = (
                self._update_telemetry["legacy_surprise_interventions"]
                + scale.lt(1.0).to(scale.dtype)
            )
        return OrderedDict((name, value * scale) for name, value in bounded.items())

    @classmethod
    def forgetting_update(
        cls,
        fast_weights: Mapping[str, Tensor],
        surprise: Mapping[str, Tensor],
        alpha: GateLike,
    ) -> FastWeights:
        """Apply equation 13: ``(1 - alpha) * M_(t-1) + S_t``."""

        return OrderedDict(
            (
                name,
                (1.0 - cls._parameter_gate(alpha, name)) * fast_weights[name]
                + surprise[name],
            )
            for name in fast_weights
        )

    def update_one(
        self,
        state: PaperMACStreamState,
        key: Tensor,
        value: Tensor,
        *,
        alpha: GateLike,
        eta: GateLike,
        theta: GateLike,
    ) -> PaperMACStreamState:
        """Apply one exact local inner-loop update without committing a segment.

        This public primitive makes the equations independently testable.  It
        leaves ``segment_index`` unchanged; :meth:`update_segment` is the sole
        operation that commits a completed 32-token segment.
        """

        self._validate_vector(key, "key")
        self._validate_vector(value, "value")
        for gate_name, gate in (("alpha", alpha), ("eta", eta), ("theta", theta)):
            if isinstance(gate, Tensor):
                if gate.numel() != 1:
                    raise ValueError(
                        f"{gate_name} must be scalar for legacy memory or a parameter mapping"
                    )
            elif tuple(gate) != tuple(state.fast_weights):
                raise ValueError(f"{gate_name} parameter gates do not match fast weights")
        fast_weights, surprise = self.update_tensors(
            state.fast_weights,
            state.surprise,
            key,
            value,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )
        return state.replace(fast_weights=fast_weights, surprise=surprise)

    def update_tensors(
        self,
        fast_weights: Mapping[str, Tensor],
        previous_surprise: Mapping[str, Tensor],
        key: Tensor,
        value: Tensor,
        *,
        alpha: GateLike,
        eta: GateLike,
        theta: GateLike,
    ) -> tuple[FastWeights, FastWeights]:
        """Apply one centralized update shared by reference and accelerated paths."""

        raw_gradient = self.surprise_gradient(fast_weights, key, value)
        gradient = self.condition_gradient(fast_weights, raw_gradient)
        past_surprise = OrderedDict(
            (
                name,
                self._parameter_gate(eta, name) * previous_surprise[name],
            )
            for name in previous_surprise
        )
        momentary_surprise = OrderedDict(
            (
                name,
                -self._parameter_gate(theta, name) * gradient[name],
            )
            for name in gradient
        )
        unbounded_surprise = OrderedDict(
            (
                name,
                past_surprise[name] + momentary_surprise[name],
            )
            for name in previous_surprise
        )
        if self._update_telemetry:
            telemetry = self._update_telemetry
            past_rms = self._mapping_rms(past_surprise).detach()
            momentary_rms = self._mapping_rms(momentary_surprise).detach()
            combined_rms = self._mapping_rms(unbounded_surprise).detach()
            forgotten = OrderedDict(
                (
                    name,
                    self._parameter_gate(alpha, name) * fast_weights[name],
                )
                for name in fast_weights
            )
            dot = torch.stack(
                [
                    (past_surprise[name] * momentary_surprise[name]).sum()
                    for name in past_surprise
                ]
            ).sum().detach()
            epsilon = torch.finfo(dot.dtype).eps
            cosine = dot / (
                past_rms
                * momentary_rms
                * sum(value.numel() for value in past_surprise.values())
            ).clamp_min(epsilon)
            for key, value in (
                ("past_surprise_rms_max", past_rms),
                ("momentary_surprise_rms_max", momentary_rms),
                ("combined_surprise_rms_max", combined_rms),
                ("forgotten_weight_rms_max", self._mapping_rms(forgotten).detach()),
            ):
                telemetry[key] = torch.maximum(telemetry[key], value)
            telemetry["past_momentary_cosine_sum"] = (
                telemetry["past_momentary_cosine_sum"] + cosine
            )
        surprise = self._bound_surprise(unbounded_surprise)
        next_weights = self.forgetting_update(fast_weights, surprise, alpha)
        return next_weights, surprise

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
        keys, values, write_history = self.project_writes(state, segment_embeddings)
        gate_values = self.gates(segment_embeddings)
        candidate = state
        self.begin_update_telemetry(state.fast_weights)
        for position in range(self.segment_length):
            if valid_mask is not None and not bool(valid_mask[position].item()):
                continue
            if isinstance(gate_values, ParameterGateValues):
                alpha: GateLike = OrderedDict(
                    (name, value[position]) for name, value in gate_values.alpha.items()
                )
                eta: GateLike = OrderedDict(
                    (name, value[position]) for name, value in gate_values.eta.items()
                )
                theta: GateLike = OrderedDict(
                    (name, value[position]) for name, value in gate_values.theta.items()
                )
            else:
                alpha = gate_values.alpha[position]
                eta = gate_values.eta[position]
                theta = gate_values.theta[position]
            candidate = self.update_one(
                candidate,
                keys[position],
                values[position],
                alpha=alpha,
                eta=eta,
                theta=theta,
            )
        return candidate.replace(
            fast_weights=candidate.fast_weights,
            surprise=candidate.surprise,
            segment_index=state.segment_index + 1,
            write_history=write_history,
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
        queries, query_history = self.project_queries(state, segment_embeddings)
        retrieval = self.retrieve(state, queries)
        updated = self.update_segment(state, segment_embeddings, valid_mask=valid_mask)
        return retrieval, self.advance_query_history(updated, query_history)

    def read_segment(self, state: PaperMACStreamState, segment_embeddings: Tensor) -> Tensor:
        """Retrieve all 32 positions from one immutable incoming state.

        This is the read-only boundary used by :class:`PaperMACBlock`.  Keeping
        it separate from :meth:`update_segment` lets the MAC core decide which
        causally integrated representations are useful enough to write.
        """

        self._validate_segment(segment_embeddings, valid_mask=None)
        queries, _ = self.project_queries(state, segment_embeddings)
        return self.retrieve(state, queries)

    def read_segment_with_history(
        self,
        state: PaperMACStreamState,
        segment_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """Read a segment and return the uncommitted causal query history."""

        self._validate_segment(segment_embeddings, valid_mask=None)
        queries, history = self.project_queries(state, segment_embeddings)
        return self.retrieve(state, queries), history

    def gate_tensors(self, token_embeddings: Tensor) -> Mapping[str, Tensor]:
        """Flatten gate outputs for telemetry without losing channel variation."""

        values = self.gates(token_embeddings)
        if isinstance(values, ParameterGateValues):
            return {
                name: torch.cat(
                    [mapping.reshape(token_embeddings.shape[0], -1) for mapping in getattr(values, name).values()],
                    dim=-1,
                )
                for name in ("alpha", "eta", "theta")
            }
        return {
            "alpha": values.alpha,
            "eta": values.eta,
            "theta": values.theta,
        }
