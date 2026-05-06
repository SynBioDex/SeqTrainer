"""Educational Titans/MIRAS-inspired Memory-as-Context modules for DNA classification.

This module intentionally implements a compact, notebook-friendly approximation of
Titans/MIRAS ideas:
- short-term token encoding via TransformerEncoder
- long-term neural memory via an MLP memory bank
- optional persistent task memory tokens
- memory-as-context (MAC) by prepending memory vectors to token embeddings

It is not an official or full research reproduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TitansMIRASConfig:
    """Configuration for the educational Titans+MIRAS classifier prototype."""

    vocab_size: int = 6
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    max_length: int = 512
    memory_slots: int = 8
    memory_depth: int = 2
    memory_context_tokens: int = 4
    dropout: float = 0.1
    num_classes: int = 2
    memory_architecture: str = "mlp"
    attentional_bias: str = "mse_associative_surprise"
    retention_gate: float = 0.9
    memory_algorithm: str = "adamw_plus_ema_memory_update"
    use_persistent_memory: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class DNATokenizer:
    """Simple dependency-free DNA tokenizer with PAD and UNK/N handling."""

    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1
    VOCAB = {
        "A": 2,
        "C": 3,
        "G": 4,
        "T": 5,
        "N": UNK_TOKEN_ID,
    }

    def __init__(self, max_length: int = 512) -> None:
        self.max_length = max_length

    def encode(self, sequence: str, max_length: Optional[int] = None) -> tuple[list[int], list[int]]:
        max_len = max_length or self.max_length
        seq = (sequence or "").upper()
        token_ids = [self.VOCAB.get(ch, self.UNK_TOKEN_ID) for ch in seq[:max_len]]
        attn = [1] * len(token_ids)
        if len(token_ids) < max_len:
            pad_len = max_len - len(token_ids)
            token_ids.extend([self.PAD_TOKEN_ID] * pad_len)
            attn.extend([0] * pad_len)
        return token_ids, attn

    def batch_encode(self, sequences: list[str], max_length: Optional[int] = None) -> tuple[Tensor, Tensor]:
        encoded = [self.encode(seq, max_length=max_length) for seq in sequences]
        input_ids = torch.tensor([x[0] for x in encoded], dtype=torch.long)
        attention_mask = torch.tensor([x[1] for x in encoded], dtype=torch.long)
        return input_ids, attention_mask


class NeuralLongTermMemory(nn.Module):
    """MLP memory bank with EMA-style retention update and MAC retrieval."""

    def __init__(self, d_model: int, slots: int, depth: int, retention_gate: float = 0.9, dropout: float = 0.1) -> None:
        super().__init__()
        self.slots = slots
        self.retention_gate = float(retention_gate)

        layers: list[nn.Module] = []
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(d_model, d_model))
        self.memory_mlp = nn.Sequential(*layers)

        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.register_buffer("memory_state", torch.zeros(slots, d_model))

    def reset_memory(self) -> None:
        self.memory_state.zero_()

    def update(self, chunk_embeddings: Tensor) -> Tensor:
        batch_summary = chunk_embeddings.mean(dim=1)
        new_content = self.memory_mlp(batch_summary).mean(dim=0)
        expanded = new_content.unsqueeze(0).expand(self.slots, -1)
        with torch.no_grad():
            self.memory_state.mul_(self.retention_gate).add_((1.0 - self.retention_gate) * expanded.detach())
        return self.memory_state

    def retrieve_context(self, token_embeddings: Tensor, context_tokens: int) -> Tensor:
        query = self.query_proj(token_embeddings.mean(dim=1))
        keys = self.key_proj(self.memory_state)
        scores = torch.matmul(query, keys.t()) / (token_embeddings.shape[-1] ** 0.5)
        top_k = min(context_tokens, self.slots)
        top_idx = scores.topk(k=top_k, dim=-1).indices
        selected = self.memory_state[top_idx]
        return selected

    def forward(self, token_embeddings: Tensor, context_tokens: int, update_memory: bool = True) -> Tensor:
        if update_memory:
            self.update(token_embeddings)
        return self.retrieve_context(token_embeddings, context_tokens)


class TitansMemoryAsContextClassifier(nn.Module):
    """Educational Titans-like classifier using long-term memory as prepended context."""

    def __init__(self, config: TitansMIRASConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=DNATokenizer.PAD_TOKEN_ID)
        self.persistent_tokens = (
            nn.Parameter(torch.randn(1, config.memory_context_tokens, config.d_model) * 0.02)
            if config.use_persistent_memory
            else None
        )
        self.long_term_memory = NeuralLongTermMemory(
            d_model=config.d_model,
            slots=config.memory_slots,
            depth=config.memory_depth,
            retention_gate=config.retention_gate,
            dropout=config.dropout,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

    def reset_memory(self) -> None:
        self.long_term_memory.reset_memory()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        update_memory: bool = True,
    ) -> dict[str, Tensor] | Tensor:
        x = self.embedding(input_ids)
        memory_context = self.long_term_memory(
            x,
            context_tokens=self.config.memory_context_tokens,
            update_memory=update_memory,
        )
        if self.persistent_tokens is not None:
            persistent = self.persistent_tokens.expand(input_ids.size(0), -1, -1)
            context = torch.cat([persistent, memory_context], dim=1)
        else:
            context = memory_context

        x = torch.cat([context, x], dim=1)
        if attention_mask is not None:
            context_mask = torch.ones(
                attention_mask.size(0),
                context.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_mask = torch.cat([context_mask, attention_mask], dim=1)
            key_padding_mask = full_mask == 0
        else:
            key_padding_mask = None

        enc = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = self.norm(enc.mean(dim=1))
        logits = self.classifier(pooled)

        if labels is None:
            return logits

        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "memory_context": memory_context}
