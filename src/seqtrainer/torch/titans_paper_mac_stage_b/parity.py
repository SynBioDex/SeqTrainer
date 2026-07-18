"""Output, state, surprise, and gradient parity evidence for Stage B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock, PaperMACBlockOutput

from .backends import StageBBackendRegistry
from .config import StageBBackendConfig


@dataclass(frozen=True)
class TensorParity:
    exact: bool
    close: bool
    max_abs_error: float
    max_relative_error: float
    cosine_similarity: float


@dataclass(frozen=True)
class ParityReport:
    reference_config: Mapping[str, object]
    candidate_config: Mapping[str, object]
    atol: float
    rtol: float
    sequence: TensorParity
    retrieval: TensorParity
    input_gradient: TensorParity
    fast_weights: Mapping[str, TensorParity]
    surprise: Mapping[str, TensorParity]
    trainable_gradients: Mapping[str, TensorParity]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _compare(reference: Tensor, candidate: Tensor, *, atol: float, rtol: float) -> TensorParity:
    if reference.shape != candidate.shape:
        raise ValueError(f"parity shape mismatch: {reference.shape} != {candidate.shape}")
    left = reference.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    right = candidate.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    difference = (left - right).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = left.abs().clamp_min(torch.finfo(torch.float64).eps)
    max_relative = float((difference / denominator).max().item()) if difference.numel() else 0.0
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 and float(right_norm.item()) == 0.0:
        cosine = 1.0
    elif float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(left, right).div(left_norm * right_norm).item())
    return TensorParity(
        exact=torch.equal(left, right),
        close=torch.allclose(left, right, atol=atol, rtol=rtol),
        max_abs_error=max_abs,
        max_relative_error=max_relative,
        cosine_similarity=cosine,
    )


def _training_loss(output: PaperMACBlockOutput) -> Tensor:
    terms = [output.sequence.square().mean(), output.retrieval.square().mean()]
    terms.extend(value.square().mean() for value in output.state.fast_weights.values())
    terms.extend(value.square().mean() for value in output.state.surprise.values())
    return torch.stack(terms).sum()


def _capture(
    block: PaperMACBlock,
    segment: Tensor,
    config: StageBBackendConfig,
    registry: StageBBackendRegistry,
    stream_id: str,
) -> tuple[PaperMACBlockOutput, dict[str, Tensor]]:
    block.zero_grad(set_to_none=True)
    candidate_segment = segment.detach().clone().requires_grad_(True)
    output = registry.execute(
        block,
        block.initial_state(stream_id),
        candidate_segment,
        config=config,
    )
    _training_loss(output).backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    if candidate_segment.grad is None:
        raise RuntimeError("parity loss did not reach the segment input")
    gradients["__input__"] = candidate_segment.grad.detach().clone()
    return output, gradients


def compare_backends(
    reference_block: PaperMACBlock,
    candidate_block: PaperMACBlock,
    segment: Tensor,
    *,
    reference_config: StageBBackendConfig = StageBBackendConfig(),
    candidate_config: StageBBackendConfig = StageBBackendConfig(),
    reference_registry: StageBBackendRegistry | None = None,
    candidate_registry: StageBBackendRegistry | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ParityReport:
    """Compare complete differentiable transitions from matched parameters."""

    left_state = reference_block.state_dict()
    right_state = candidate_block.state_dict()
    if tuple(left_state) != tuple(right_state) or any(
        not torch.equal(left_state[name], right_state[name]) for name in left_state
    ):
        raise ValueError("parity blocks must start from identical state_dict values")
    left_registry = StageBBackendRegistry() if reference_registry is None else reference_registry
    right_registry = StageBBackendRegistry() if candidate_registry is None else candidate_registry
    reference, reference_gradients = _capture(
        reference_block, segment, reference_config, left_registry, "parity"
    )
    candidate, candidate_gradients = _capture(
        candidate_block, segment, candidate_config, right_registry, "parity"
    )
    if tuple(reference.state.fast_weights) != tuple(candidate.state.fast_weights):
        raise ValueError("fast-weight key mismatch")
    if tuple(reference.state.surprise) != tuple(candidate.state.surprise):
        raise ValueError("surprise key mismatch")
    if tuple(reference_gradients) != tuple(candidate_gradients):
        raise ValueError("trainable-gradient key mismatch")

    fast_weights = {
        name: _compare(reference.state.fast_weights[name], candidate.state.fast_weights[name], atol=atol, rtol=rtol)
        for name in reference.state.fast_weights
    }
    surprise = {
        name: _compare(reference.state.surprise[name], candidate.state.surprise[name], atol=atol, rtol=rtol)
        for name in reference.state.surprise
    }
    gradients = {
        name: _compare(reference_gradients[name], candidate_gradients[name], atol=atol, rtol=rtol)
        for name in reference_gradients
    }
    input_gradient = gradients.pop("__input__")
    sequence = _compare(reference.sequence, candidate.sequence, atol=atol, rtol=rtol)
    retrieval = _compare(reference.retrieval, candidate.retrieval, atol=atol, rtol=rtol)
    comparisons = [
        sequence,
        retrieval,
        input_gradient,
        *fast_weights.values(),
        *surprise.values(),
        *gradients.values(),
    ]
    return ParityReport(
        reference_config=reference_config.to_dict(),
        candidate_config=candidate_config.to_dict(),
        atol=atol,
        rtol=rtol,
        sequence=sequence,
        retrieval=retrieval,
        input_gradient=input_gradient,
        fast_weights=fast_weights,
        surprise=surprise,
        trainable_gradients=gradients,
        passed=all(item.close for item in comparisons),
    )
