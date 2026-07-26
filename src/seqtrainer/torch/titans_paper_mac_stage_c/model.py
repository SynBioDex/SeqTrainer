"""Token-level causal LM wrapper around the reviewed Stage B paper-MAC stack."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from seqtrainer.torch.titans_paper_mac import PaperMACStreamState
from seqtrainer.torch.titans_paper_mac_stage_b import (
    ActivationDType,
    AttentionBackend,
    StageBBackendConfig,
    StageBMACStack,
)
from seqtrainer.torch.titans_paper_mac_stage_b.attention import (
    differentiable_sdpa_context,
    integrate_sdpa_attention,
)

from .config import MemoryMode, StageCModelConfig


BlockStates = tuple[PaperMACStreamState, ...]


@dataclass(frozen=True)
class StageCLMOutput:
    """One batch of logits, loss accounting, and replacement stream states."""

    logits: Tensor
    states: tuple[BlockStates, ...]
    loss: Tensor | None
    loss_sum: Tensor | None
    valid_tokens: int
    valid_bases: int
    retrieval_norm: float
    memory_update_norm: float
    surprise_norm: float
    state_drift_norm: float
    gate_statistics: dict[str, float]
    memory_gradient_statistics: dict[str, float]


def detach_stream_states(states: BlockStates) -> BlockStates:
    """Truncate autograd while retaining exact numerical functional state."""

    detached: list[PaperMACStreamState] = []
    for state in states:
        fast_weights = OrderedDict(
            (name, value.detach().requires_grad_(True)) for name, value in state.fast_weights.items()
        )
        surprise = OrderedDict(
            (name, value.detach().requires_grad_(value.requires_grad))
            for name, value in state.surprise.items()
        )
        detached.append(
            PaperMACStreamState(
                stream_id=state.stream_id,
                fast_weights=fast_weights,
                surprise=surprise,
                segment_index=state.segment_index,
                reset_count=state.reset_count,
                ended=state.ended,
            )
        )
    return tuple(detached)


class StageCPaperMACForCausalLM(nn.Module):
    """Separate Stage C language model; the legacy slot/EMA LM is untouched."""

    def __init__(self, config: StageCModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(config.segment_length, config.d_model)
        self.stack = StageBMACStack(
            config.block_count,
            config.d_model,
            num_heads=config.num_heads,
            persistent_tokens=config.persistent_tokens,
            memory_depth=config.memory_depth,
            segment_length=config.segment_length,
            max_surprise_norm=config.memory_surprise_clip_norm,
            associative_loss_reduction=config.memory_associative_loss_reduction,
            max_gradient_rms=config.memory_max_gradient_rms,
            max_gradient_rms_ratio=config.memory_max_gradient_rms_ratio,
            theta_max=config.memory_theta_max,
            theta_initial=config.memory_theta_initial,
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embeddings.weight[self.config.pad_token_id].zero_()
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def initial_states(self, stream_id: str) -> BlockStates:
        return self.stack.initial_states(stream_id)

    def _effective_backend(
        self,
        mode: MemoryMode,
        *,
        device: torch.device,
    ) -> StageBBackendConfig:
        backend = StageBBackendConfig() if mode is MemoryMode.REFERENCE else self.config.backend
        if backend.attention_backend is AttentionBackend.MULTIHEAD_ATTENTION:
            # Stage C's functional memory writes retain a higher-order graph.
            # MultiheadAttention is free to dispatch to a kernel without double
            # backward support, so use the reviewed exact SDPA adapter instead.
            # CPU evaluates it directly as tensor math; CUDA is scoped to math
            # SDPA in _forward_row below.
            return replace(backend, attention_backend=AttentionBackend.SDPA)
        return backend

    @staticmethod
    def _state_update_norm(before: BlockStates, after: BlockStates) -> Tensor:
        terms = [
            (after_state.fast_weights[name] - before_state.fast_weights[name]).square().sum()
            for before_state, after_state in zip(before, after)
            for name in before_state.fast_weights
        ]
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).sum().sqrt()

    def _state_drift_norm(self, states: BlockStates) -> Tensor:
        terms = [
            (state.fast_weights[name] - initial[name]).square().sum()
            for block, state in zip(self.stack.blocks, states)
            for initial in (block.memory.initial_fast_weights(),)
            for name in state.fast_weights
        ]
        return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.0)

    @staticmethod
    def _surprise_norm(states: BlockStates) -> Tensor:
        terms = [
            value.square().sum()
            for state in states
            for value in state.surprise.values()
        ]
        return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.0)

    def _gate_statistics(
        self,
        block_inputs: Sequence[Tensor],
        valid_mask: Tensor,
    ) -> dict[str, float]:
        values: dict[str, list[Tensor]] = {"alpha": [], "eta": [], "theta": []}
        with torch.no_grad():
            for block, sequence in zip(self.stack.blocks, block_inputs):
                gates = block.memory.gates(sequence.detach())
                for name in values:
                    values[name].append(getattr(gates, name)[valid_mask].float().flatten())
        statistics: dict[str, float] = {}
        for name, parts in values.items():
            combined = torch.cat(parts) if parts else torch.empty(0)
            statistics[f"{name}_mean"] = float(combined.mean().cpu()) if combined.numel() else 0.0
            statistics[f"{name}_std"] = (
                float(combined.std(unbiased=False).cpu()) if combined.numel() else 0.0
            )
            statistics[f"{name}_min"] = float(combined.min().cpu()) if combined.numel() else 0.0
            statistics[f"{name}_max"] = float(combined.max().cpu()) if combined.numel() else 0.0
        return statistics

    def _memory_gradient_statistics(self) -> dict[str, float]:
        telemetry = [block.memory.update_telemetry() for block in self.stack.blocks]
        populated = [item for item in telemetry if item]
        if not populated:
            return {
                "raw_gradient_rms_max": 0.0,
                "conditioned_gradient_rms_max": 0.0,
                "gradient_scale_min": 1.0,
                "gradient_intervention_fraction": 0.0,
                "legacy_surprise_intervention_fraction": 0.0,
            }
        update_count = torch.stack([item["update_count"] for item in populated]).sum()
        denominator = update_count.clamp_min(1.0)
        return {
            "raw_gradient_rms_max": float(torch.stack(
                [item["raw_gradient_rms_max"] for item in populated]
            ).max().cpu()),
            "conditioned_gradient_rms_max": float(torch.stack(
                [item["conditioned_gradient_rms_max"] for item in populated]
            ).max().cpu()),
            "gradient_scale_min": float(torch.stack(
                [item["gradient_scale_min"] for item in populated]
            ).min().cpu()),
            "gradient_intervention_fraction": float((torch.stack(
                [item["gradient_interventions"] for item in populated]
            ).sum() / denominator).cpu()),
            "legacy_surprise_intervention_fraction": float((torch.stack(
                [item["legacy_surprise_interventions"] for item in populated]
            ).sum() / denominator).cpu()),
        }

    def _no_memory_forward(
        self,
        states: BlockStates,
        embeddings: Tensor,
        backend: StageBBackendConfig,
    ) -> tuple[Tensor, BlockStates, tuple[Tensor, ...], tuple[Tensor, ...]]:
        sequence = embeddings
        retrievals: list[Tensor] = []
        block_sequences: list[Tensor] = []
        for block in self.stack.blocks:
            retrieval = torch.zeros_like(sequence)
            if backend.attention_backend is AttentionBackend.MULTIHEAD_ATTENTION:
                if backend.activation_dtype is not ActivationDType.FP32:
                    raise ValueError("reduced precision no-memory execution requires SDPA")
                sequence = block.integrate(retrieval, sequence)
            elif backend.attention_backend is AttentionBackend.SDPA:
                sequence = integrate_sdpa_attention(
                    block,
                    retrieval,
                    sequence,
                    backend.activation_dtype,
                )
            else:
                raise ValueError("Flash attention remains disabled in Stage C")
            retrievals.append(retrieval)
            block_sequences.append(sequence)
        return sequence, states, tuple(retrievals), tuple(block_sequences)

    def _forward_row(
        self,
        states: BlockStates,
        input_ids: Tensor,
        valid_mask: Tensor,
        mode: MemoryMode,
    ) -> tuple[
        Tensor,
        BlockStates,
        tuple[Tensor, ...],
        Tensor,
        Tensor,
        Tensor,
        dict[str, float],
        dict[str, float],
    ]:
        positions = torch.arange(self.config.segment_length, device=input_ids.device)
        embeddings = (
            self.token_embeddings(input_ids) * math.sqrt(self.config.d_model)
            + self.position_embeddings(positions)
        )
        backend = self._effective_backend(mode, device=embeddings.device)
        # Functional neural-memory updates call autograd.grad(create_graph=True).
        # Constrain CUDA SDPA to its exact math kernel so the subsequent outer
        # backward has a supported backward-of-backward path on T4/A100 builds.
        with differentiable_sdpa_context(embeddings.device):
            if mode is MemoryMode.NONE:
                sequence, next_states, retrievals, block_sequences = self._no_memory_forward(
                    states, embeddings, backend
                )
            else:
                output = self.stack(
                    states,
                    embeddings,
                    config=backend,
                    valid_mask=valid_mask,
                )
                sequence = output.sequence
                retrievals = output.retrievals
                block_sequences = output.block_sequences
                next_states = states if mode is MemoryMode.FROZEN else output.states
        logits = self.lm_head(self.final_norm(sequence))
        update_norm = self._state_update_norm(states, next_states).to(device=logits.device)
        surprise_norm = self._surprise_norm(next_states).to(device=logits.device)
        drift_norm = self._state_drift_norm(next_states).to(device=logits.device)
        block_inputs = (embeddings, *block_sequences[:-1])
        gate_statistics = self._gate_statistics(block_inputs, valid_mask)
        memory_gradient_statistics = (
            self._memory_gradient_statistics()
            if mode is not MemoryMode.NONE
            else {
                "raw_gradient_rms_max": 0.0,
                "conditioned_gradient_rms_max": 0.0,
                "gradient_scale_min": 1.0,
                "gradient_intervention_fraction": 0.0,
                "legacy_surprise_intervention_fraction": 0.0,
            }
        )
        return (
            logits,
            next_states,
            retrievals,
            update_norm,
            surprise_norm,
            drift_norm,
            gate_statistics,
            memory_gradient_statistics,
        )

    def forward_segment(
        self,
        states: Sequence[BlockStates],
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        valid_mask: Tensor | None = None,
        loss_mask: Tensor | None = None,
        represented_base_counts: Tensor | None = None,
        memory_mode: MemoryMode | str | None = None,
    ) -> StageCLMOutput:
        """Run one ordered segment for each independently named batch row."""

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        expected = (len(states), self.config.segment_length)
        if tuple(input_ids.shape) != expected:
            raise ValueError(f"input_ids must have shape {expected}")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        if valid_mask is None:
            valid_mask = input_ids.ne(self.config.pad_token_id)
        if tuple(valid_mask.shape) != expected:
            raise ValueError(f"valid_mask must have shape {expected}")
        mode = self.config.memory_mode if memory_mode is None else MemoryMode(memory_mode)
        logits: list[Tensor] = []
        next_states: list[BlockStates] = []
        retrieval_squares: list[Tensor] = []
        update_norms: list[Tensor] = []
        surprise_norms: list[Tensor] = []
        drift_norms: list[Tensor] = []
        gate_rows: list[dict[str, float]] = []
        memory_gradient_rows: list[dict[str, float]] = []
        for row, row_states in enumerate(states):
            (
                row_logits,
                row_next,
                retrievals,
                update_norm,
                surprise_norm,
                drift_norm,
                gate_statistics,
                memory_gradient_statistics,
            ) = self._forward_row(
                row_states,
                input_ids[row],
                valid_mask[row].bool(),
                mode,
            )
            logits.append(row_logits)
            next_states.append(row_next)
            retrieval_squares.extend(retrieval.detach().square().sum() for retrieval in retrievals)
            update_norms.append(update_norm.detach())
            surprise_norms.append(surprise_norm.detach())
            drift_norms.append(drift_norm.detach())
            gate_rows.append(gate_statistics)
            memory_gradient_rows.append(memory_gradient_statistics)
        stacked = torch.stack(logits)
        loss = None
        loss_sum = None
        valid_tokens = 0
        valid_bases = 0
        if labels is not None:
            if tuple(labels.shape) != expected:
                raise ValueError(f"labels must have shape {expected}")
            active_loss = valid_mask.bool() if loss_mask is None else loss_mask.bool()
            if tuple(active_loss.shape) != expected:
                raise ValueError(f"loss_mask must have shape {expected}")
            per_token = F.cross_entropy(
                stacked.reshape(-1, self.config.vocab_size),
                labels.long().reshape(-1),
                reduction="none",
            ).reshape(expected)
            loss_sum = (per_token * active_loss).sum()
            valid_tokens = int(active_loss.sum().item())
            loss = loss_sum / max(valid_tokens, 1)
            if represented_base_counts is None:
                valid_bases = valid_tokens
            else:
                if tuple(represented_base_counts.shape) != expected:
                    raise ValueError(f"represented_base_counts must have shape {expected}")
                valid_bases = int((represented_base_counts * active_loss).sum().item())
        retrieval_norm = (
            float(torch.stack(retrieval_squares).sum().sqrt().cpu()) if retrieval_squares else 0.0
        )
        memory_update_norm = (
            float(torch.stack(update_norms).square().sum().sqrt().cpu()) if update_norms else 0.0
        )
        surprise_norm = (
            float(torch.stack(surprise_norms).square().sum().sqrt().cpu())
            if surprise_norms
            else 0.0
        )
        state_drift_norm = (
            float(torch.stack(drift_norms).square().sum().sqrt().cpu())
            if drift_norms
            else 0.0
        )
        gate_statistics = {
            key: sum(row[key] for row in gate_rows) / len(gate_rows)
            for key in gate_rows[0]
        } if gate_rows else {}
        memory_gradient_statistics = {
            "raw_gradient_rms_max": max(
                row["raw_gradient_rms_max"] for row in memory_gradient_rows
            ),
            "conditioned_gradient_rms_max": max(
                row["conditioned_gradient_rms_max"] for row in memory_gradient_rows
            ),
            "gradient_scale_min": min(
                row["gradient_scale_min"] for row in memory_gradient_rows
            ),
            "gradient_intervention_fraction": sum(
                row["gradient_intervention_fraction"] for row in memory_gradient_rows
            ) / len(memory_gradient_rows),
            "legacy_surprise_intervention_fraction": sum(
                row["legacy_surprise_intervention_fraction"] for row in memory_gradient_rows
            ) / len(memory_gradient_rows),
        } if memory_gradient_rows else {}
        return StageCLMOutput(
            logits=stacked,
            states=tuple(next_states),
            loss=loss,
            loss_sum=loss_sum,
            valid_tokens=valid_tokens,
            valid_bases=valid_bases,
            retrieval_norm=retrieval_norm,
            memory_update_norm=memory_update_norm,
            surprise_norm=surprise_norm,
            state_drift_norm=state_drift_norm,
            gate_statistics=gate_statistics,
            memory_gradient_statistics=memory_gradient_statistics,
        )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
