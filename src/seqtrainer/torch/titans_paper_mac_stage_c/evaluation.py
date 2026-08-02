"""Held-out ordered-stream evaluation with biological group aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from seqtrainer.data.bacteria_titan import StreamSegment
from seqtrainer.torch.titans_paper_mac import PaperMACStreamState

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
    per_accession_valid_bases: Mapping[str, int]
    per_accession_segments: Mapping[str, int]
    per_gc_bin_bpb: Mapping[str, float]
    retrieval_norm_mean: float
    memory_update_norm_mean: float
    surprise_norm_mean: float
    state_drift_norm_mean: float
    gate_statistics: Mapping[str, float]
    memory_gradient_statistics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EVALUATION_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class _EvaluationPlanEntry:
    stream_id: str
    accession: str
    segments: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _evaluation_plan(
    streams: Mapping[str, Sequence[StreamSegment]],
    *,
    max_streams: int | None,
    max_segments: int | None,
    max_segments_per_accession: int | None,
) -> tuple[_EvaluationPlanEntry, ...]:
    """Freeze the deterministic stream/segment work list before evaluation."""

    if max_streams is not None and max_streams <= 0:
        raise ValueError("max_streams must be positive when provided")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive when provided")
    if max_segments_per_accession is not None and max_segments_per_accession <= 0:
        raise ValueError("max_segments_per_accession must be positive when provided")
    remaining = max_segments
    accession_counts: dict[str, int] = defaultdict(int)
    planned: list[_EvaluationPlanEntry] = []
    selected_ids = sorted(streams)
    if max_streams is not None:
        selected_ids = selected_ids[:max_streams]
    for stream_id in selected_ids:
        stream = streams[stream_id]
        if not len(stream):
            continue
        accession = str(stream[0].accession)
        allowed = len(stream)
        if remaining is not None:
            allowed = min(allowed, remaining)
        if max_segments_per_accession is not None:
            allowed = min(
                allowed,
                max_segments_per_accession - accession_counts[accession],
            )
        if allowed <= 0:
            continue
        planned.append(_EvaluationPlanEntry(stream_id, accession, allowed))
        accession_counts[accession] += allowed
        if remaining is not None:
            remaining -= allowed
            if remaining == 0:
                break
    if not planned:
        raise ValueError("evaluation limits selected no stream segments")
    return tuple(planned)


def _plan_hash(plan: Sequence[_EvaluationPlanEntry]) -> str:
    encoded = json.dumps(
        [entry.to_dict() for entry in plan],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _initial_accumulator() -> dict[str, object]:
    return {
        "total_nll": 0.0,
        "total_tokens": 0,
        "total_bases": 0,
        "correct": 0,
        "top2_correct": 0,
        "group_nll": {},
        "group_bases": {},
        "accession_nll": {},
        "accession_bases": {},
        "accession_segments": {},
        "gc_nll": {},
        "gc_bases": {},
        "diagnostic_sums": {
            "retrieval_norm": 0.0,
            "memory_update_norm": 0.0,
            "surprise_norm": 0.0,
            "state_drift_norm": 0.0,
        },
        "gate_sums": {},
        "memory_gradient_sums": {},
        "memory_gradient_maxima": {},
        "memory_gradient_scale_min": 1.0,
        "processed_stream_ids": [],
        "completed_stream_ids": [],
        "segment_count": 0,
    }


def _number_map(accumulator: Mapping[str, object], key: str) -> dict[str, float]:
    value = accumulator.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation checkpoint field {key!r} is invalid")
    return {str(name): float(number) for name, number in value.items()}


def _integer_map(accumulator: Mapping[str, object], key: str) -> dict[str, int]:
    value = accumulator.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation checkpoint field {key!r} is invalid")
    return {str(name): int(number) for name, number in value.items()}


def _serialize_block_states(states: Sequence[PaperMACStreamState] | None) -> object:
    return None if states is None else [state.to_state_dict() for state in states]


def _restore_block_states(
    payload: object,
    *,
    device: torch.device,
) -> tuple[PaperMACStreamState, ...] | None:
    if payload is None:
        return None
    if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
        raise ValueError("evaluation checkpoint fast state is invalid")
    return tuple(PaperMACStreamState.from_state_dict(item, device=device) for item in payload)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_evaluation_checkpoint(
    path: Path,
    *,
    contract: Mapping[str, object],
    plan: Sequence[_EvaluationPlanEntry],
    device: torch.device,
) -> tuple[dict[str, object], int, int, tuple[PaperMACStreamState, ...] | None, float]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("format_version") != EVALUATION_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported evaluation checkpoint format")
    if payload.get("contract") != dict(contract):
        raise ValueError("evaluation checkpoint contract changed; refuse unsafe resume")
    raw_plan = payload.get("plan")
    expected_plan = [entry.to_dict() for entry in plan]
    if raw_plan != expected_plan:
        raise ValueError("evaluation stream plan changed; refuse unsafe resume")
    accumulator = payload.get("accumulator")
    cursor = payload.get("cursor")
    if not isinstance(accumulator, Mapping) or not isinstance(cursor, Mapping):
        raise ValueError("evaluation checkpoint is missing accumulator or cursor")
    plan_index = int(cursor.get("plan_index", 0))
    next_segment = int(cursor.get("next_segment", 0))
    if not 0 <= plan_index <= len(plan):
        raise ValueError("evaluation checkpoint plan cursor is invalid")
    if next_segment < 0:
        raise ValueError("evaluation checkpoint segment cursor is invalid")
    states = _restore_block_states(payload.get("active_states"), device=device)
    if next_segment and states is None:
        raise ValueError("evaluation checkpoint is missing active stream state")
    return dict(accumulator), plan_index, next_segment, states, float(payload.get("elapsed_seconds", 0.0))


def _progress_payload(
    *,
    state: str,
    run_label: str,
    plan: Sequence[_EvaluationPlanEntry],
    accumulator: Mapping[str, object],
    elapsed_seconds: float,
    plan_index: int,
    next_segment: int,
    resumed: bool,
    checkpoint_path: Path | None,
    error: str | None = None,
) -> dict[str, object]:
    completed = int(accumulator["segment_count"])
    planned = sum(entry.segments for entry in plan)
    rate = completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = max(planned - completed, 0)
    current = plan[plan_index] if plan_index < len(plan) else None
    payload: dict[str, object] = {
        "format_version": 1,
        "state": state,
        "run_label": run_label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "resumed": resumed,
        "completed_segments": completed,
        "planned_segments": planned,
        "progress_fraction": completed / planned,
        "elapsed_seconds": elapsed_seconds,
        "segments_per_second": rate,
        "eta_seconds": remaining / rate if rate > 0 else None,
        "completed_bases": int(accumulator["total_bases"]),
        "started_streams": len(accumulator["processed_stream_ids"]),
        "completed_streams": len(accumulator["completed_stream_ids"]),
        "current_stream": None if current is None else current.stream_id,
        "current_accession": None if current is None else current.accession,
        "next_segment_in_stream": next_segment,
        "resume_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    }
    if error:
        payload["error"] = error
    return payload


def _save_evaluation_checkpoint(
    path: Path,
    *,
    state: str,
    contract: Mapping[str, object],
    plan: Sequence[_EvaluationPlanEntry],
    accumulator: Mapping[str, object],
    plan_index: int,
    next_segment: int,
    active_states: Sequence[PaperMACStreamState] | None,
    elapsed_seconds: float,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format_version": EVALUATION_CHECKPOINT_FORMAT_VERSION,
            "state": state,
            "contract": dict(contract),
            "plan": [entry.to_dict() for entry in plan],
            "cursor": {"plan_index": plan_index, "next_segment": next_segment},
            "accumulator": dict(accumulator),
            "active_states": _serialize_block_states(active_states),
            "elapsed_seconds": elapsed_seconds,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def evaluate_ordered_streams(
    model: StageCPaperMACForCausalLM,
    streams: Mapping[str, Sequence[StreamSegment]],
    *,
    device: torch.device | str,
    memory_mode: MemoryMode | str | None = None,
    max_streams: int | None = None,
    max_segments: int | None = None,
    max_segments_per_accession: int | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_contract: Mapping[str, object] | None = None,
    checkpoint_every_segments: int = 128,
    progress_every_segments: int = 32,
    progress_path: str | Path | None = None,
    run_label: str = "evaluation",
) -> EvaluationResult:
    """Evaluate ordered streams with optional exact, atomic segment resume.

    The resume artifact stores the next deterministic cursor, all aggregate
    metrics, and the detached fast state for the one stream in flight.  The
    model checkpoint, dataset/panel identity, memory policy, and frozen work
    plan belong in ``resume_contract``; any change is rejected before work is
    resumed.  Numerical fast state may therefore continue on a different
    compatible device without attempting to serialize an autograd graph.
    """

    selected_device = torch.device(device)
    if checkpoint_every_segments <= 0:
        raise ValueError("checkpoint_every_segments must be positive")
    if progress_every_segments <= 0:
        raise ValueError("progress_every_segments must be positive")
    mode = model.config.memory_mode if memory_mode is None else MemoryMode(memory_mode)
    plan = _evaluation_plan(
        streams,
        max_streams=max_streams,
        max_segments=max_segments,
        max_segments_per_accession=max_segments_per_accession,
    )
    contract = {
        **dict(resume_contract or {}),
        "model_config": model.config.to_dict(),
        "memory_mode": mode.value,
        "plan_sha256": _plan_hash(plan),
    }
    checkpoint_path = None if resume_checkpoint is None else Path(resume_checkpoint)
    live_path = None if progress_path is None else Path(progress_path)
    accumulator = _initial_accumulator()
    plan_index = 0
    next_segment = 0
    active_states: tuple[PaperMACStreamState, ...] | None = None
    prior_elapsed = 0.0
    resumed = False
    if checkpoint_path is not None and checkpoint_path.exists():
        accumulator, plan_index, next_segment, active_states, prior_elapsed = (
            _load_evaluation_checkpoint(
                checkpoint_path,
                contract=contract,
                plan=plan,
                device=selected_device,
            )
        )
        resumed = True

    model.eval()
    total_nll = float(accumulator.get("total_nll", 0.0))
    total_tokens = int(accumulator.get("total_tokens", 0))
    total_bases = int(accumulator.get("total_bases", 0))
    correct = int(accumulator.get("correct", 0))
    top2_correct = int(accumulator.get("top2_correct", 0))
    group_nll = defaultdict(float, _number_map(accumulator, "group_nll"))
    group_bases = defaultdict(int, _integer_map(accumulator, "group_bases"))
    accession_nll = defaultdict(float, _number_map(accumulator, "accession_nll"))
    accession_bases = defaultdict(int, _integer_map(accumulator, "accession_bases"))
    accession_segments = defaultdict(
        int, _integer_map(accumulator, "accession_segments")
    )
    gc_nll = defaultdict(float, _number_map(accumulator, "gc_nll"))
    gc_bases = defaultdict(int, _integer_map(accumulator, "gc_bases"))
    diagnostic_sums = defaultdict(float, _number_map(accumulator, "diagnostic_sums"))
    gate_sums = defaultdict(float, _number_map(accumulator, "gate_sums"))
    memory_gradient_sums = defaultdict(
        float, _number_map(accumulator, "memory_gradient_sums")
    )
    memory_gradient_maxima = defaultdict(
        float, _number_map(accumulator, "memory_gradient_maxima")
    )
    memory_gradient_scale_min = float(
        accumulator.get("memory_gradient_scale_min", 1.0)
    )
    raw_processed = accumulator.get("processed_stream_ids", [])
    if not isinstance(raw_processed, list):
        raise ValueError("evaluation checkpoint processed stream IDs are invalid")
    processed_stream_ids = list(map(str, raw_processed))
    raw_completed = accumulator.get("completed_stream_ids", [])
    if not isinstance(raw_completed, list):
        raise ValueError("evaluation checkpoint completed stream IDs are invalid")
    completed_stream_ids = list(map(str, raw_completed))
    segment_count = int(accumulator.get("segment_count", 0))
    started = time.perf_counter()
    segment_in_progress = False

    def sync_accumulator() -> None:
        accumulator.update(
            {
                "total_nll": total_nll,
                "total_tokens": total_tokens,
                "total_bases": total_bases,
                "correct": correct,
                "top2_correct": top2_correct,
                "group_nll": dict(group_nll),
                "group_bases": dict(group_bases),
                "accession_nll": dict(accession_nll),
                "accession_bases": dict(accession_bases),
                "accession_segments": dict(accession_segments),
                "gc_nll": dict(gc_nll),
                "gc_bases": dict(gc_bases),
                "diagnostic_sums": dict(diagnostic_sums),
                "gate_sums": dict(gate_sums),
                "memory_gradient_sums": dict(memory_gradient_sums),
                "memory_gradient_maxima": dict(memory_gradient_maxima),
                "memory_gradient_scale_min": memory_gradient_scale_min,
                "processed_stream_ids": processed_stream_ids,
                "completed_stream_ids": completed_stream_ids,
                "segment_count": segment_count,
            }
        )

    def elapsed() -> float:
        return prior_elapsed + (time.perf_counter() - started)

    def publish(state: str, *, error: str | None = None) -> None:
        sync_accumulator()
        status = _progress_payload(
            state=state,
            run_label=run_label,
            plan=plan,
            accumulator=accumulator,
            elapsed_seconds=elapsed(),
            plan_index=plan_index,
            next_segment=next_segment,
            resumed=resumed,
            checkpoint_path=checkpoint_path,
            error=error,
        )
        if live_path is not None:
            _atomic_json(live_path, status)
        print(json.dumps(status, sort_keys=True), flush=True)

    publish("running")
    try:
        while plan_index < len(plan):
            entry = plan[plan_index]
            stream = streams[entry.stream_id]
            if next_segment:
                if active_states is None:
                    raise ValueError("resume cursor requires active stream state")
                states = active_states
            else:
                states = model.initial_states(entry.stream_id)
                if entry.stream_id not in processed_stream_ids:
                    processed_stream_ids.append(entry.stream_id)
            while next_segment < entry.segments:
                segment_in_progress = True
                segment = stream[next_segment]
                tensors = StageCTrainer._batch_tensors((segment,), selected_device)
                with torch.enable_grad():
                    output = model.forward_segment((states,), memory_mode=mode, **tensors)
                assert output.loss_sum is not None
                nll = float(output.loss_sum.detach())
                mask = tensors["loss_mask"]
                predictions = output.logits.detach().argmax(dim=-1)
                top2 = output.logits.detach().topk(
                    min(2, output.logits.size(-1)), dim=-1
                ).indices
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
                accession_segments[segment.accession] += 1
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
                for key in (
                    "gradient_intervention_fraction",
                    "legacy_surprise_intervention_fraction",
                ):
                    memory_gradient_sums[key] += output.memory_gradient_statistics[key]
                for key in ("raw_gradient_rms_max", "conditioned_gradient_rms_max"):
                    memory_gradient_maxima[key] = max(
                        memory_gradient_maxima[key],
                        output.memory_gradient_statistics[key],
                    )
                memory_gradient_scale_min = min(
                    memory_gradient_scale_min,
                    output.memory_gradient_statistics["gradient_scale_min"],
                )
                states = detach_stream_states(output.states[0])
                active_states = states
                next_segment += 1
                segment_count += 1
                if next_segment == entry.segments:
                    if entry.stream_id not in completed_stream_ids:
                        completed_stream_ids.append(entry.stream_id)
                    plan_index += 1
                    next_segment = 0
                    active_states = None
                segment_in_progress = False
                if checkpoint_path is not None and segment_count % checkpoint_every_segments == 0:
                    sync_accumulator()
                    _save_evaluation_checkpoint(
                        checkpoint_path,
                        state="running",
                        contract=contract,
                        plan=plan,
                        accumulator=accumulator,
                        plan_index=plan_index,
                        next_segment=next_segment,
                        active_states=active_states,
                        elapsed_seconds=elapsed(),
                    )
                if segment_count % progress_every_segments == 0:
                    publish("running")
                if plan_index >= len(plan) or next_segment == 0:
                    break
        sync_accumulator()
        if checkpoint_path is not None:
            _save_evaluation_checkpoint(
                checkpoint_path,
                state="completed",
                contract=contract,
                plan=plan,
                accumulator=accumulator,
                plan_index=len(plan),
                next_segment=0,
                active_states=None,
                elapsed_seconds=elapsed(),
            )
        publish("completed")
    except BaseException as error:
        sync_accumulator()
        # Never overwrite a known-good cursor with partially accumulated
        # statistics from a segment that raised midway through postprocessing.
        if checkpoint_path is not None and not segment_in_progress:
            _save_evaluation_checkpoint(
                checkpoint_path,
                state="interrupted",
                contract=contract,
                plan=plan,
                accumulator=accumulator,
                plan_index=plan_index,
                next_segment=next_segment,
                active_states=active_states,
                elapsed_seconds=elapsed(),
            )
        publish("interrupted", error=f"{type(error).__name__}: {error}")
        raise

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
        streams=len(processed_stream_ids),
        segments=segment_count,
        per_clade_bpb={key: group_nll[key] / (bases * divisor) for key, bases in group_bases.items()},
        per_accession_bpb={
            key: accession_nll[key] / (bases * divisor) for key, bases in accession_bases.items()
        },
        per_accession_valid_bases=dict(accession_bases),
        per_accession_segments=dict(accession_segments),
        per_gc_bin_bpb={
            key: gc_nll[key] / (bases * divisor) for key, bases in gc_bases.items()
        },
        retrieval_norm_mean=diagnostic_sums["retrieval_norm"] / segment_count,
        memory_update_norm_mean=diagnostic_sums["memory_update_norm"] / segment_count,
        surprise_norm_mean=diagnostic_sums["surprise_norm"] / segment_count,
        state_drift_norm_mean=diagnostic_sums["state_drift_norm"] / segment_count,
        gate_statistics={key: value / segment_count for key, value in gate_sums.items()},
        memory_gradient_statistics={
            **memory_gradient_maxima,
            "gradient_scale_min": memory_gradient_scale_min,
            **{
                key: value / segment_count
                for key, value in memory_gradient_sums.items()
            },
        },
    )
