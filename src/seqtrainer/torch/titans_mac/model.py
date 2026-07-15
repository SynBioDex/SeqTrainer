"""Scalable Titan-style Memory-as-Context models for DNA."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .configuration import TitansMACLMConfig


class _LongTermMemory(nn.Module):
    def __init__(self, config: TitansMACLMConfig) -> None:
        super().__init__()
        self.retention_gate = config.retention_gate
        self.memory_slots = nn.Parameter(torch.empty(config.memory_slots, config.d_model))
        nn.init.normal_(self.memory_slots, std=0.02)
        layers: list[nn.Module] = []
        for _ in range(config.memory_depth - 1):
            layers.extend((nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Dropout(config.dropout)))
        layers.append(nn.Linear(config.d_model, config.d_model))
        self.memory_network = nn.Sequential(*layers)
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.register_buffer("memory_state", torch.zeros(config.memory_slots, config.d_model), persistent=True)

    def reset_memory(self) -> None:
        self.memory_state.zero_()

    def _bank(self) -> Tensor:
        return self.memory_network(self.memory_slots + self.memory_state.detach())

    def retrieve(self, token_embeddings: Tensor, count: int) -> tuple[Tensor, Tensor, Tensor]:
        if count == 0:
            shape = (token_embeddings.size(0), 0)
            empty_idx = torch.empty(shape, dtype=torch.long, device=token_embeddings.device)
            empty_scores = torch.empty(shape, dtype=token_embeddings.dtype, device=token_embeddings.device)
            return token_embeddings[:, :0], empty_idx, empty_scores
        # A window-global query would expose future input bases to early logits.
        # The first input token is causal-safe for every prediction in the window.
        query = token_embeddings[:, 0]
        bank = self._bank()
        scores = self.query(query) @ self.key(bank).t()
        scores = scores / math.sqrt(token_embeddings.size(-1))
        values, indices = scores.topk(k=count, dim=-1)
        context = bank[indices]
        return context, indices, values.softmax(dim=-1)

    @torch.no_grad()
    def update(self, token_embeddings: Tensor, indices: Tensor, weights: Tensor) -> None:
        if indices.numel() == 0:
            return
        content = token_embeddings.mean(dim=1).detach()
        updates = torch.zeros_like(self.memory_state)
        counts = torch.zeros(self.memory_state.size(0), device=updates.device, dtype=updates.dtype)
        for rank in range(indices.size(1)):
            slot_ids = indices[:, rank]
            slot_weights = weights[:, rank].to(updates.dtype)
            updates.index_add_(0, slot_ids, content * slot_weights.unsqueeze(-1))
            counts.index_add_(0, slot_ids, slot_weights)
        used = counts > 0
        averaged = updates[used] / counts[used].unsqueeze(-1)
        self.memory_state[used].mul_(self.retention_gate).add_(averaged, alpha=1.0 - self.retention_gate)


class TitansMACForCausalLM(nn.Module):
    """Causal DNA LM with persistent and retrieved long-term context tokens."""

    def __init__(self, config: TitansMACLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.persistent_memory = (
            nn.Parameter(torch.empty(config.persistent_tokens, config.d_model))
            if config.use_persistent_memory and config.persistent_tokens
            else None
        )
        if self.persistent_memory is not None:
            nn.init.normal_(self.persistent_memory, std=0.02)
        self.long_term_memory = _LongTermMemory(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def reset_memory(self) -> None:
        self.long_term_memory.reset_memory()

    def _causal_mask(self, dna_length: int, context_length: int, device: torch.device) -> Tensor:
        total = dna_length + context_length
        mask = torch.ones(total, total, dtype=torch.bool, device=device)
        if context_length:
            mask[:context_length, :context_length] = False
        dna_causal = torch.triu(
            torch.ones(dna_length, dna_length, dtype=torch.bool, device=device), diagonal=1
        )
        mask[context_length:, :context_length] = False
        mask[context_length:, context_length:] = dna_causal
        return mask

    def _encode(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        update_memory: bool = True,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        if input_ids.size(1) > self.config.max_length:
            raise ValueError(f"sequence length exceeds max_length={self.config.max_length}")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        dna = self.token_embeddings(input_ids) + self.position_embeddings(positions).unsqueeze(0)
        dna = self.embedding_dropout(dna)
        memory_context, slot_indices, slot_weights = self.long_term_memory.retrieve(
            dna, self.config.memory_context_tokens
        )
        contexts: list[Tensor] = []
        if self.persistent_memory is not None:
            contexts.append(self.persistent_memory.unsqueeze(0).expand(input_ids.size(0), -1, -1))
        contexts.append(memory_context)
        context = torch.cat(contexts, dim=1) if contexts else dna[:, :0]
        hidden = torch.cat((context, dna), dim=1)
        causal_mask = self._causal_mask(input_ids.size(1), context.size(1), input_ids.device)
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            context_mask = torch.ones(
                input_ids.size(0), context.size(1), dtype=attention_mask.dtype, device=input_ids.device
            )
            key_padding_mask = torch.cat((context_mask, attention_mask), dim=1).eq(0)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        hidden = self.final_norm(hidden[:, context.size(1) :])
        if update_memory:
            self.long_term_memory.update(dna, slot_indices, slot_weights)
        diagnostics = {
            "slot_indices": slot_indices,
            "slot_weights": slot_weights,
            "slot_usage": F.one_hot(slot_indices, num_classes=self.config.memory_slots).sum(dim=(0, 1))
            if slot_indices.numel()
            else torch.zeros(self.config.memory_slots, device=input_ids.device, dtype=torch.long),
        }
        return hidden, memory_context, diagnostics

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        output_memory_context: bool = False,
        output_memory_diagnostics: bool = False,
        update_memory: Optional[bool] = None,
    ) -> dict[str, Tensor | dict[str, Tensor] | None]:
        hidden, memory_context, diagnostics = self._encode(
            input_ids,
            attention_mask=attention_mask,
            update_memory=self.training if update_memory is None else update_memory,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=self.config.pad_token_id
            )
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden if output_hidden_states else None,
            "memory_context": memory_context if output_memory_context else None,
            "memory_diagnostics": diagnostics if output_memory_diagnostics else None,
        }

    @torch.no_grad()
    def extract_sequence_embeddings(self, input_ids: Tensor, pooling: str = "mean") -> Tensor:
        attention_mask = input_ids.ne(self.config.pad_token_id)
        hidden, _, _ = self._encode(input_ids, attention_mask=attention_mask, update_memory=False)
        if pooling == "mean":
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        if pooling == "max":
            return hidden.masked_fill(~attention_mask.unsqueeze(-1), torch.finfo(hidden.dtype).min).max(dim=1).values
        if pooling == "last":
            indices = attention_mask.sum(dim=1).clamp_min(1) - 1
            return hidden[torch.arange(hidden.size(0), device=hidden.device), indices]
        raise ValueError("pooling must be one of: mean, max, last")

    @torch.no_grad()
    def get_memory_diagnostics(self, input_ids: Tensor) -> dict[str, Tensor]:
        _, _, diagnostics = self._encode(
            input_ids, attention_mask=input_ids.ne(self.config.pad_token_id), update_memory=False
        )
        bank = self.long_term_memory._bank()
        diagnostics["slot_cosine_similarity"] = F.normalize(bank, dim=-1) @ F.normalize(bank, dim=-1).t()
        diagnostics["memory_slots"] = bank
        return diagnostics


class TitansMACForSequenceClassification(TitansMACForCausalLM):
    """Titan MAC encoder with a sequence classification head."""

    def __init__(self, config: TitansMACLMConfig) -> None:
        super().__init__(config)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        update_memory: Optional[bool] = None,
        **_: object,
    ) -> dict[str, Tensor | None]:
        mask = attention_mask if attention_mask is not None else input_ids.ne(self.config.pad_token_id)
        hidden, _, _ = self._encode(
            input_ids, attention_mask=mask, update_memory=self.training if update_memory is None else update_memory
        )
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits, "hidden_states": hidden}


def count_parameters(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
