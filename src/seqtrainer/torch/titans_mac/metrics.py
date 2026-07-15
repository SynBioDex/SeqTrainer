"""Metrics for DNA next-token language modeling."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F


@torch.no_grad()
def compute_lm_metrics(
    logits: Tensor,
    labels: Tensor,
    pad_token_id: int = 0,
    input_ids: Optional[Tensor] = None,
) -> dict[str, float]:
    """Compute loss, calibration, ranking, and GC-stratified LM metrics."""

    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels have incompatible shapes")
    valid = labels.ne(pad_token_id)
    if not valid.any():
        raise ValueError("labels contain no non-padding tokens")
    flat_logits = logits[valid].float()
    flat_labels = labels[valid]
    token_losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    probabilities = flat_logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    top_k = min(2, probabilities.size(-1))
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    loss = token_losses.mean().item()
    metrics = {
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
        "bits_per_base": loss / math.log(2.0),
        "token_accuracy": predictions.eq(flat_labels).float().mean().item(),
        "top_2_accuracy": probabilities.topk(top_k, dim=-1).indices.eq(flat_labels.unsqueeze(-1)).any(dim=-1).float().mean().item(),
        "confidence": probabilities.max(dim=-1).values.mean().item(),
        "entropy": entropy.mean().item(),
    }
    sequences = input_ids if input_ids is not None else labels
    valid_bases = sequences.ge(2)
    gc = (sequences.eq(3) | sequences.eq(4)).sum(dim=1) / valid_bases.sum(dim=1).clamp_min(1)
    bins = ((0.0, 0.4, "gc_0_40_loss"), (0.4, 0.6, "gc_40_60_loss"), (0.6, 1.01, "gc_60_100_loss"))
    loss_grid = torch.zeros_like(labels, dtype=token_losses.dtype)
    loss_grid[valid] = token_losses
    for lower, upper, name in bins:
        selected_sequences = (gc >= lower) & (gc < upper)
        selected_tokens = valid & selected_sequences.unsqueeze(1)
        metrics[name] = loss_grid[selected_tokens].mean().item() if selected_tokens.any() else float("nan")
    return metrics
