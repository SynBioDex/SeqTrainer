"""Base-normalized Stage C language-model and biological metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class StageCMetrics:
    loss_per_token: float
    bits_per_base: float
    perplexity_per_token: float
    token_accuracy: float
    top_2_accuracy: float
    confidence: float
    entropy: float
    valid_tokens: int
    valid_bases: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_stage_c_metrics(
    logits: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    represented_base_counts: Tensor,
) -> StageCMetrics:
    """Compute token metrics plus the common tokenizer-independent BPB."""

    if logits.shape[:-1] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits, labels, and loss_mask shapes are inconsistent")
    if represented_base_counts.shape != labels.shape:
        raise ValueError("represented_base_counts must match labels")
    active = loss_mask.bool()
    valid_tokens = int(active.sum().item())
    valid_bases = int((represented_base_counts * active).sum().item())
    if not valid_tokens or not valid_bases:
        raise ValueError("metrics require at least one valid token and base")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.long().reshape(-1),
        reduction="none",
    ).reshape(labels.shape)
    loss_sum = losses[active].sum()
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    top2 = probabilities.topk(min(2, probabilities.size(-1)), dim=-1).indices
    confidence = probabilities.max(dim=-1).values[active].mean()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)[active].mean()
    token_loss = float(loss_sum / valid_tokens)
    return StageCMetrics(
        loss_per_token=token_loss,
        bits_per_base=float(loss_sum / (valid_bases * math.log(2.0))),
        perplexity_per_token=float(math.exp(min(token_loss, 50.0))),
        token_accuracy=float(predictions[active].eq(labels[active]).float().mean()),
        top_2_accuracy=float(top2[active].eq(labels[active].unsqueeze(-1)).any(dim=-1).float().mean()),
        confidence=float(confidence),
        entropy=float(entropy),
        valid_tokens=valid_tokens,
        valid_bases=valid_bases,
    )
