"""Held-out ordered-stream evaluation with biological group aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import torch

from seqtrainer.data.bacteria_titan import StreamSegment

from .config import MemoryMode
from .model import StageCPaperMACForCausalLM, detach_stream_states
from .trainer import StageCTrainer


@dataclass(frozen=True)
class EvaluationResult:
    memory_mode: str
    bits_per_base: float
    loss_per_token: float
    perplexity: float
    token_accuracy: float
    top_2_accuracy: float
    valid_tokens: int
    valid_bases: int
    streams: int
    segments: int
    per_clade_bpb: Mapping[str, float]
    per_accession_bpb: Mapping[str, float]
    per_gc_bin_bpb: Mapping[str, float]
    retrieval_norm_mean: float
    memory_update_norm_mean: float
    surprise_norm_mean: float
    state_drift_norm_mean: float
    gate_statistics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_ordered_streams(
    model: StageCPaperMACForCausalLM,
    streams: Mapping[str, Sequence[StreamSegment]],
    *,
    device: torch.device | str,
    memory_mode: MemoryMode | str | None = None,
    max_streams: int | None = None,
    max_segments: int | None = None,
) -> EvaluationResult:
    """Evaluate from reset state without retaining graphs between segments."""

    selected_device = torch.device(device)
    if max_streams is not None and max_streams <= 0:
        raise ValueError("max_streams must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")
    mode = model.config.memory_mode if memory_mode is None else MemoryMode(memory_mode)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bases = 0
    correct = 0
    top2_correct = 0
    group_nll: dict[str, float] = defaultdict(float)
    group_bases: dict[str, int] = defaultdict(int)
    accession_nll: dict[str, float] = defaultdict(float)
    accession_bases: dict[str, int] = defaultdict(int)
    gc_nll: dict[str, float] = defaultdict(float)
    gc_bases: dict[str, int] = defaultdict(int)
    diagnostic_sums = {
        "retrieval_norm": 0.0,
        "memory_update_norm": 0.0,
        "surprise_norm": 0.0,
        "state_drift_norm": 0.0,
    }
    gate_sums: dict[str, float] = defaultdict(float)
    stream_count = 0
    segment_count = 0
    for stream_id in sorted(streams):
        if max_streams is not None and stream_count >= max_streams:
            break
        if max_segments is not None and segment_count >= max_segments:
            break
        states = model.initial_states(stream_id)
        stream_count += 1
        for segment in streams[stream_id]:
            if max_segments is not None and segment_count >= max_segments:
                break
            tensors = StageCTrainer._batch_tensors((segment,), selected_device)
            with torch.enable_grad():
                output = model.forward_segment((states,), memory_mode=mode, **tensors)
            assert output.loss_sum is not None
            nll = float(output.loss_sum.detach())
            mask = tensors["loss_mask"]
            predictions = output.logits.detach().argmax(dim=-1)
            top2 = output.logits.detach().topk(min(2, output.logits.size(-1)), dim=-1).indices
            labels = tensors["labels"]
            correct += int(predictions[mask].eq(labels[mask]).sum().item())
            top2_correct += int(
                top2[mask].eq(labels[mask].unsqueeze(-1)).any(dim=-1).sum().item()
            )
            total_nll += nll
            total_tokens += output.valid_tokens
            total_bases += output.valid_bases
            group_nll[segment.clade_group] += nll
            group_bases[segment.clade_group] += output.valid_bases
            accession_nll[segment.accession] += nll
            accession_bases[segment.accession] += output.valid_bases
            gc_bin = (
                "gc_0_40"
                if segment.gc_fraction < 0.4
                else "gc_60_100"
                if segment.gc_fraction >= 0.6
                else "gc_40_60"
            )
            gc_nll[gc_bin] += nll
            gc_bases[gc_bin] += output.valid_bases
            diagnostic_sums["retrieval_norm"] += output.retrieval_norm
            diagnostic_sums["memory_update_norm"] += output.memory_update_norm
            diagnostic_sums["surprise_norm"] += output.surprise_norm
            diagnostic_sums["state_drift_norm"] += output.state_drift_norm
            for key, value in output.gate_statistics.items():
                gate_sums[key] += value
            states = detach_stream_states(output.states[0])
            segment_count += 1
    if not total_tokens or not total_bases:
        raise ValueError("evaluation requires valid targets")
    divisor = math.log(2.0)
    loss_per_token = total_nll / total_tokens
    return EvaluationResult(
        memory_mode=mode.value,
        bits_per_base=total_nll / (total_bases * divisor),
        loss_per_token=loss_per_token,
        perplexity=math.exp(min(loss_per_token, 700.0)),
        token_accuracy=correct / total_tokens,
        top_2_accuracy=top2_correct / total_tokens,
        valid_tokens=total_tokens,
        valid_bases=total_bases,
        streams=stream_count,
        segments=segment_count,
        per_clade_bpb={key: group_nll[key] / (bases * divisor) for key, bases in group_bases.items()},
        per_accession_bpb={
            key: accession_nll[key] / (bases * divisor) for key, bases in accession_bases.items()
        },
        per_gc_bin_bpb={
            key: gc_nll[key] / (bases * divisor) for key, bases in gc_bases.items()
        },
        retrieval_norm_mean=diagnostic_sums["retrieval_norm"] / segment_count,
        memory_update_norm_mean=diagnostic_sums["memory_update_norm"] / segment_count,
        surprise_norm_mean=diagnostic_sums["surprise_norm"] / segment_count,
        state_drift_norm_mean=diagnostic_sums["state_drift_norm"] / segment_count,
        gate_statistics={key: value / segment_count for key, value in gate_sums.items()},
    )
