"""Autoregressive DNA generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .model import TitansMACForCausalLM
from .tokenizer import DNABaseTokenizer


@dataclass(frozen=True)
class GeneratedSequence:
    sequence: str
    token_ids: list[int]
    prefix: str


@torch.no_grad()
def generate_dna(
    model: TitansMACForCausalLM,
    tokenizer: Optional[DNABaseTokenizer] = None,
    prefix: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device | str] = None,
) -> GeneratedSequence:
    """Sample a DNA sequence from a causal Titan MAC model."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    tokenizer = tokenizer or DNABaseTokenizer()
    prefix = prefix or "A"
    ids = tokenizer.encode(prefix, max_length=model.config.max_length, padding=False)
    if not ids:
        ids = [DNABaseTokenizer.A_TOKEN_ID]
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        context = ids[-model.config.max_length :]
        input_ids = torch.tensor([context], dtype=torch.long, device=target_device)
        logits = model(input_ids, update_memory=False)["logits"]
        next_logits = logits[0, -1].float() / temperature
        next_logits[: DNABaseTokenizer.A_TOKEN_ID] = -torch.inf
        if top_k is not None and 0 < top_k < next_logits.numel():
            threshold = torch.topk(next_logits, top_k).values[-1]
            next_logits[next_logits < threshold] = -torch.inf
        next_id = int(torch.multinomial(next_logits.softmax(dim=-1), num_samples=1).item())
        ids.append(next_id)
    model.train(was_training)
    return GeneratedSequence(sequence=tokenizer.decode(ids), token_ids=ids, prefix=prefix)
