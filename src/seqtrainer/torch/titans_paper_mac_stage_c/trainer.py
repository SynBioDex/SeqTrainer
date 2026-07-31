"""Ordered-stream scheduling and truncated-gradient Stage C training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from seqtrainer.data.bacteria_titan.stage_c_streams import StreamSegment

from .config import MemoryMode
from .metrics import compute_stage_c_metrics
from .model import BlockStates, StageCPaperMACForCausalLM, detach_stream_states


@dataclass(frozen=True)
class TrainingStepRecord:
    optimizer_step: int
    segment_steps: int
    segments: int
    valid_tokens: int
    valid_bases: int
    loss_per_token: float
    bits_per_base: float
    token_accuracy: float
    top_2_accuracy: float
    retrieval_norm: float
    memory_update_norm: float
    surprise_norm: float
    state_drift_norm: float
    alpha_mean: float
    alpha_std: float
    eta_mean: float
    eta_std: float
    theta_mean: float
    theta_std: float
    raw_memory_gradient_rms_max: float
    conditioned_memory_gradient_rms_max: float
    memory_gradient_scale_min: float
    memory_gradient_intervention_fraction: float
    legacy_surprise_intervention_fraction: float
    gradient_norm: float
    written_state_gradient_norm: float
    elapsed_seconds: float
    bases_per_second: float
    learning_rate: float = 0.0
    past_surprise_rms_max: float = 0.0
    momentary_surprise_rms_max: float = 0.0
    combined_surprise_rms_max: float = 0.0
    forgotten_weight_rms_max: float = 0.0
    past_momentary_cosine_mean: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class NonFiniteTrainingError(RuntimeError):
    """Stop a run before non-finite model state can contaminate its evidence."""


class StageCScheduler(Protocol):
    """Checkpointable ordered-stream scheduler contract."""

    @property
    def exhausted(self) -> bool: ...

    def next_batch(self) -> tuple[StreamSegment, ...]: ...

    def to_state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, payload: Mapping[str, object]) -> None: ...


@dataclass(frozen=True)
class BaseCosineLRSchedule:
    """Warm up and cosine-decay by cumulative predictable DNA bases."""

    peak_lr: float
    minimum_lr: float
    warmup_bases: int
    decay_bases: int

    def __post_init__(self) -> None:
        if not 0 < self.minimum_lr <= self.peak_lr:
            raise ValueError("learning-rate bounds must satisfy 0 < minimum <= peak")
        if self.warmup_bases < 0 or self.decay_bases <= self.warmup_bases:
            raise ValueError("decay_bases must exceed non-negative warmup_bases")

    def __call__(self, processed_bases: int) -> float:
        if processed_bases < 0:
            raise ValueError("processed_bases cannot be negative")
        if self.warmup_bases and processed_bases < self.warmup_bases:
            return self.peak_lr * max(processed_bases, 1) / self.warmup_bases
        progress = min(
            1.0,
            max(
                0.0,
                (processed_bases - self.warmup_bases)
                / (self.decay_bases - self.warmup_bases),
            ),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_lr + (self.peak_lr - self.minimum_lr) * cosine


class StreamBatchScheduler:
    """Deterministically interleave streams without reordering their segments."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        streams: Mapping[str, Sequence[StreamSegment]],
        *,
        batch_size: int,
        seed: int = 17,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Production TokenStreamDataset values are mmap-backed LazyTokenStreams.
        # Materialising them as tuples would create one Python StreamSegment per
        # 32-token window across the whole training corpus before the first
        # batch. Keep the indexed Sequence intact and fetch only scheduled
        # segments. This also keeps checkpoint state independent of corpus size.
        self.streams = dict(streams)
        if not self.streams or any(len(segments) == 0 for segments in self.streams.values()):
            raise ValueError("scheduler requires non-empty streams")
        for stream_id, segments in self.streams.items():
            first, last = segments[0], segments[-1]
            if first.stream_id != stream_id or last.stream_id != stream_id:
                raise ValueError("segment ownership does not match scheduler stream key")
            if first.segment_index != 0 or last.segment_index != len(segments) - 1:
                raise ValueError("segments must be contiguous and ordered within each stream")
            if not first.start_of_stream or not last.end_of_stream:
                raise ValueError("every scheduled stream requires explicit start/end markers")
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.order = sorted(self.streams)
        if shuffle:
            random.Random(seed).shuffle(self.order)
        self.next_stream_cursor = 0
        self.active: list[tuple[str, int] | None] = [None] * batch_size
        self.segments_yielded = 0
        for slot in range(batch_size):
            self._fill(slot)

    def _fill(self, slot: int) -> None:
        self.active[slot] = None
        if self.next_stream_cursor < len(self.order):
            self.active[slot] = (self.order[self.next_stream_cursor], 0)
            self.next_stream_cursor += 1

    @property
    def exhausted(self) -> bool:
        return all(item is None for item in self.active)

    def next_batch(self) -> tuple[StreamSegment, ...]:
        if self.exhausted:
            raise StopIteration
        batch: list[StreamSegment] = []
        for slot, item in enumerate(self.active):
            if item is None:
                continue
            stream_id, index = item
            segment = self.streams[stream_id][index]
            batch.append(segment)
            self.segments_yielded += 1
            if index + 1 == len(self.streams[stream_id]):
                self._fill(slot)
            else:
                self.active[slot] = (stream_id, index + 1)
        return tuple(batch)

    def to_state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "order": list(self.order),
            "next_stream_cursor": self.next_stream_cursor,
            "active": [list(item) if item is not None else None for item in self.active],
            "segments_yielded": self.segments_yielded,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported Stage C scheduler state version")
        if int(payload.get("batch_size", -1)) != self.batch_size:
            raise ValueError("scheduler batch size changed across resume")
        order = payload.get("order")
        active = payload.get("active")
        if not isinstance(order, list) or set(map(str, order)) != set(self.streams):
            raise ValueError("scheduler stream set changed across resume")
        if not isinstance(active, list) or len(active) != self.batch_size:
            raise ValueError("scheduler active-row payload is invalid")
        self.order = [str(value) for value in order]
        self.next_stream_cursor = int(payload.get("next_stream_cursor", 0))
        restored: list[tuple[str, int] | None] = []
        for item in active:
            if item is None:
                restored.append(None)
            elif isinstance(item, list) and len(item) == 2:
                stream_id, index = str(item[0]), int(item[1])
                if stream_id not in self.streams or not 0 <= index < len(self.streams[stream_id]):
                    raise ValueError("scheduler active stream cursor is invalid")
                restored.append((stream_id, index))
            else:
                raise ValueError("scheduler active-row entry is invalid")
        self.active = restored
        self.segments_yielded = int(payload.get("segments_yielded", 0))


class StatefulRotationScheduler:
    """Rotate contiguous accession bursts while retaining private stream state."""

    FORMAT_VERSION = 2
    POLICY = "stateful_rotation"

    def __init__(
        self,
        streams: Mapping[str, Sequence[StreamSegment]],
        *,
        batch_size: int,
        burst_segments: int = 96,
        seed: int = 17,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or burst_segments <= 0:
            raise ValueError("batch_size and burst_segments must be positive")
        self.streams = dict(streams)
        if not self.streams or any(not value for value in self.streams.values()):
            raise ValueError("scheduler requires non-empty streams")
        accessions: dict[str, list[str]] = {}
        for stream_id, segments in self.streams.items():
            first, last = segments[0], segments[-1]
            if (
                first.stream_id != stream_id
                or last.stream_id != stream_id
                or first.segment_index != 0
                or last.segment_index != len(segments) - 1
                or not first.start_of_stream
                or not last.end_of_stream
            ):
                raise ValueError("rotation scheduler requires complete ordered streams")
            accessions.setdefault(first.accession, []).append(stream_id)
        self.accession_streams = {
            accession: sorted(
                stream_ids,
                key=lambda stream_id: (-len(self.streams[stream_id]), stream_id),
            )
            for accession, stream_ids in accessions.items()
        }
        self.batch_size = batch_size
        self.burst_segments = burst_segments
        self.seed = seed
        self.shuffle = shuffle
        self.accession_order = sorted(self.accession_streams)
        if shuffle:
            random.Random(seed).shuffle(self.accession_order)
        self.ready = list(self.accession_order)
        self.active: list[str | None] = [None] * batch_size
        self.active_bursts = [0] * batch_size
        self.stream_cursors = {stream_id: 0 for stream_id in self.streams}
        self.accession_stream_cursors = {
            accession: 0 for accession in self.accession_streams
        }
        self.segments_yielded = 0
        for slot in range(batch_size):
            self._fill(slot)

    def _fill(self, slot: int) -> None:
        self.active[slot] = self.ready.pop(0) if self.ready else None
        self.active_bursts[slot] = 0

    def _current_stream(self, accession: str) -> str:
        index = self.accession_stream_cursors[accession]
        return self.accession_streams[accession][index]

    @property
    def exhausted(self) -> bool:
        return not self.ready and all(item is None for item in self.active)

    def next_batch(self) -> tuple[StreamSegment, ...]:
        if self.exhausted:
            raise StopIteration
        batch: list[StreamSegment] = []
        rotate_slots: list[int] = []
        for slot, accession in enumerate(self.active):
            if accession is None:
                continue
            stream_id = self._current_stream(accession)
            cursor = self.stream_cursors[stream_id]
            segment = self.streams[stream_id][cursor]
            batch.append(segment)
            self.segments_yielded += 1
            self.active_bursts[slot] += 1
            cursor += 1
            self.stream_cursors[stream_id] = cursor
            if cursor == len(self.streams[stream_id]):
                next_stream = self.accession_stream_cursors[accession] + 1
                self.accession_stream_cursors[accession] = next_stream
                if next_stream == len(self.accession_streams[accession]):
                    self.active[slot] = None
                    self.active_bursts[slot] = 0
                else:
                    self.active_bursts[slot] = 0
            elif self.active_bursts[slot] == self.burst_segments:
                rotate_slots.append(slot)
        for slot in rotate_slots:
            accession = self.active[slot]
            if accession is not None:
                self.ready.append(accession)
                self.active[slot] = None
                self.active_bursts[slot] = 0
        for slot, accession in enumerate(self.active):
            if accession is None:
                self._fill(slot)
        return tuple(batch)

    def to_state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.FORMAT_VERSION,
            "policy": self.POLICY,
            "batch_size": self.batch_size,
            "burst_segments": self.burst_segments,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "stream_ids": sorted(self.streams),
            "accession_order": list(self.accession_order),
            "accession_streams": {
                key: list(value) for key, value in sorted(self.accession_streams.items())
            },
            "ready": list(self.ready),
            "active": list(self.active),
            "active_bursts": list(self.active_bursts),
            "stream_cursors": dict(sorted(self.stream_cursors.items())),
            "accession_stream_cursors": dict(
                sorted(self.accession_stream_cursors.items())
            ),
            "segments_yielded": self.segments_yielded,
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if (
            payload.get("format_version") != self.FORMAT_VERSION
            or payload.get("policy") != self.POLICY
        ):
            raise ValueError("unsupported stateful-rotation scheduler state")
        expected = {
            "batch_size": self.batch_size,
            "burst_segments": self.burst_segments,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "stream_ids": sorted(self.streams),
            "accession_streams": {
                key: list(value) for key, value in sorted(self.accession_streams.items())
            },
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"rotation scheduler {key} changed across resume")
        active = payload.get("active")
        bursts = payload.get("active_bursts")
        ready = payload.get("ready")
        stream_cursors = payload.get("stream_cursors")
        accession_cursors = payload.get("accession_stream_cursors")
        if (
            not isinstance(active, list)
            or len(active) != self.batch_size
            or not isinstance(bursts, list)
            or len(bursts) != self.batch_size
            or not isinstance(ready, list)
            or not isinstance(stream_cursors, Mapping)
            or not isinstance(accession_cursors, Mapping)
        ):
            raise ValueError("rotation scheduler cursor payload is invalid")
        known_accessions = set(self.accession_streams)
        active_values = [None if value is None else str(value) for value in active]
        if (
            any(value not in known_accessions for value in active_values if value)
            or any(str(value) not in known_accessions for value in ready)
        ):
            raise ValueError("rotation scheduler references an unknown accession")
        restored_stream_cursors = {
            str(key): int(value) for key, value in stream_cursors.items()
        }
        if set(restored_stream_cursors) != set(self.streams) or any(
            not 0 <= value <= len(self.streams[key])
            for key, value in restored_stream_cursors.items()
        ):
            raise ValueError("rotation scheduler stream cursor is invalid")
        restored_accession_cursors = {
            str(key): int(value) for key, value in accession_cursors.items()
        }
        if set(restored_accession_cursors) != known_accessions or any(
            not 0 <= value <= len(self.accession_streams[key])
            for key, value in restored_accession_cursors.items()
        ):
            raise ValueError("rotation scheduler accession cursor is invalid")
        self.accession_order = [str(value) for value in payload["accession_order"]]
        self.ready = [str(value) for value in ready]
        self.active = active_values
        self.active_bursts = [int(value) for value in bursts]
        self.stream_cursors = restored_stream_cursors
        self.accession_stream_cursors = restored_accession_cursors
        self.segments_yielded = int(payload.get("segments_yielded", 0))


class StageCTrainer:
    """Train the Stage C LM while separating state persistence from gradients."""

    def __init__(
        self,
        model: StageCPaperMACForCausalLM,
        optimizer: torch.optim.Optimizer,
        *,
        device: torch.device | str = "cpu",
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.gradient_clip_norm = gradient_clip_norm
        self.stream_states: dict[str, BlockStates] = {}
        self.optimizer_step = 0
        self.processed_segments = 0
        self.processed_tokens = 0
        self.processed_bases = 0
        self.history: list[TrainingStepRecord] = []
        self.model.to(self.device)

    @staticmethod
    def _batch_tensors(batch: Sequence[StreamSegment], device: torch.device) -> dict[str, Tensor]:
        return {
            "input_ids": torch.tensor([item.input_ids for item in batch], dtype=torch.long, device=device),
            "labels": torch.tensor([item.labels for item in batch], dtype=torch.long, device=device),
            "valid_mask": torch.tensor([item.valid_mask for item in batch], dtype=torch.bool, device=device),
            "loss_mask": torch.tensor([item.loss_mask for item in batch], dtype=torch.bool, device=device),
            "represented_base_counts": torch.tensor(
                [item.represented_base_counts for item in batch], dtype=torch.long, device=device
            ),
        }

    def _states_for(self, batch: Sequence[StreamSegment]) -> tuple[BlockStates, ...]:
        states: list[BlockStates] = []
        for segment in batch:
            if segment.start_of_stream:
                if segment.stream_id in self.stream_states:
                    raise RuntimeError("a replacement stream inherited an existing state")
                self.stream_states[segment.stream_id] = self.model.initial_states(segment.stream_id)
            if segment.stream_id not in self.stream_states:
                raise RuntimeError("continuing stream has no functional state")
            states.append(self.stream_states[segment.stream_id])
        return tuple(states)

    def _commit_states(self, batch: Sequence[StreamSegment], states: Sequence[BlockStates]) -> None:
        for segment, state in zip(batch, states):
            if segment.end_of_stream:
                self.stream_states.pop(segment.stream_id, None)
            else:
                self.stream_states[segment.stream_id] = state

    def _detach_active_states(self) -> None:
        self.stream_states = {
            stream_id: detach_stream_states(states)
            for stream_id, states in self.stream_states.items()
        }

    def _require_finite(self, phase: str, tensors: Sequence[tuple[str, Tensor]]) -> None:
        for name, value in tensors:
            if not torch.isfinite(value).all():
                raise NonFiniteTrainingError(
                    f"non-finite {phase} at optimizer step {self.optimizer_step + 1}: {name}"
                )

    @staticmethod
    def _state_tensors(states: Sequence[BlockStates]) -> list[tuple[str, Tensor]]:
        tensors: list[tuple[str, Tensor]] = []
        for row, row_states in enumerate(states):
            for block, state in enumerate(row_states):
                tensors.extend(
                    (f"state[{row}][{block}].fast_weights.{name}", value)
                    for name, value in state.fast_weights.items()
                )
                tensors.extend(
                    (f"state[{row}][{block}].surprise.{name}", value)
                    for name, value in state.surprise.items()
                )
                if state.query_history is not None:
                    tensors.append(
                        (f"state[{row}][{block}].query_history", state.query_history)
                    )
                if state.write_history is not None:
                    tensors.append(
                        (f"state[{row}][{block}].write_history", state.write_history)
                    )
        return tensors

    def train(
        self,
        scheduler: StageCScheduler,
        *,
        max_valid_bases: int | None = None,
        max_optimizer_steps: int | None = None,
        memory_mode: MemoryMode | str | None = None,
        on_step: Callable[[TrainingStepRecord], None] | None = None,
        learning_rate_schedule: Callable[[int], float] | None = None,
    ) -> tuple[TrainingStepRecord, ...]:
        """Train to an explicit valid-base or optimizer-step budget."""

        if max_valid_bases is not None and max_valid_bases <= 0:
            raise ValueError("max_valid_bases must be positive")
        if max_optimizer_steps is not None and max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be positive")
        horizon = self.model.config.gradient_horizon
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        accumulation_losses: list[Tensor] = []
        accumulation_outputs: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        accumulation_segments = 0
        accumulation_examples = 0
        accumulation_tokens = 0
        accumulation_bases = 0
        retrieval_norm = 0.0
        update_norm = 0.0
        surprise_norm = 0.0
        state_drift_norm = 0.0
        gate_statistics = {
            key: 0.0
            for name in ("alpha", "eta", "theta")
            for key in (f"{name}_mean", f"{name}_std")
        }
        memory_gradient_statistics = {
            "raw_gradient_rms_max": 0.0,
            "conditioned_gradient_rms_max": 0.0,
            "gradient_scale_min": 1.0,
            "gradient_intervention_fraction": 0.0,
            "legacy_surprise_intervention_fraction": 0.0,
            "past_surprise_rms_max": 0.0,
            "momentary_surprise_rms_max": 0.0,
            "combined_surprise_rms_max": 0.0,
            "forgotten_weight_rms_max": 0.0,
            "past_momentary_cosine_mean": 0.0,
        }
        written_state_tensors: list[Tensor] = []

        def finish_step() -> None:
            nonlocal accumulation_losses, accumulation_outputs, accumulation_segments, accumulation_examples
            nonlocal accumulation_tokens, accumulation_bases, retrieval_norm, update_norm, started
            nonlocal surprise_norm, state_drift_norm, gate_statistics
            nonlocal memory_gradient_statistics
            nonlocal written_state_tensors
            if not accumulation_losses:
                return
            loss_sum = torch.stack(accumulation_losses).sum()
            (loss_sum / max(accumulation_tokens, 1)).backward()
            self._require_finite(
                "backward gradient",
                [
                    (name, parameter.grad)
                    for name, parameter in self.model.named_parameters()
                    if parameter.grad is not None
                ],
            )
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            if learning_rate_schedule is not None:
                scheduled_lr = float(learning_rate_schedule(self.processed_bases))
                for group in self.optimizer.param_groups:
                    group["lr"] = scheduled_lr
            else:
                scheduled_lr = float(self.optimizer.param_groups[0]["lr"])
            written_squares = [
                tensor.grad.detach().square().sum()
                for tensor in written_state_tensors
                if tensor.grad is not None
            ]
            written_gradient_norm = (
                float(torch.stack(written_squares).sum().sqrt().cpu())
                if written_squares
                else 0.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_step += 1
            self._detach_active_states()
            logits = torch.cat([item[0] for item in accumulation_outputs], dim=0)
            labels = torch.cat([item[1] for item in accumulation_outputs], dim=0)
            masks = torch.cat([item[2] for item in accumulation_outputs], dim=0)
            base_counts = torch.cat([item[3] for item in accumulation_outputs], dim=0)
            metrics = compute_stage_c_metrics(logits, labels, masks, base_counts)
            elapsed = time.perf_counter() - started
            record = TrainingStepRecord(
                optimizer_step=self.optimizer_step,
                segment_steps=accumulation_segments,
                segments=accumulation_examples,
                valid_tokens=accumulation_tokens,
                valid_bases=accumulation_bases,
                loss_per_token=metrics.loss_per_token,
                bits_per_base=metrics.bits_per_base,
                token_accuracy=metrics.token_accuracy,
                top_2_accuracy=metrics.top_2_accuracy,
                retrieval_norm=retrieval_norm,
                memory_update_norm=update_norm,
                surprise_norm=surprise_norm,
                state_drift_norm=state_drift_norm,
                alpha_mean=gate_statistics["alpha_mean"] / accumulation_segments,
                alpha_std=gate_statistics["alpha_std"] / accumulation_segments,
                eta_mean=gate_statistics["eta_mean"] / accumulation_segments,
                eta_std=gate_statistics["eta_std"] / accumulation_segments,
                theta_mean=gate_statistics["theta_mean"] / accumulation_segments,
                theta_std=gate_statistics["theta_std"] / accumulation_segments,
                raw_memory_gradient_rms_max=memory_gradient_statistics[
                    "raw_gradient_rms_max"
                ],
                conditioned_memory_gradient_rms_max=memory_gradient_statistics[
                    "conditioned_gradient_rms_max"
                ],
                memory_gradient_scale_min=memory_gradient_statistics["gradient_scale_min"],
                memory_gradient_intervention_fraction=memory_gradient_statistics[
                    "gradient_intervention_fraction"
                ] / accumulation_segments,
                legacy_surprise_intervention_fraction=memory_gradient_statistics[
                    "legacy_surprise_intervention_fraction"
                ] / accumulation_segments,
                gradient_norm=float(gradient_norm_tensor.detach().cpu()),
                written_state_gradient_norm=written_gradient_norm,
                elapsed_seconds=elapsed,
                bases_per_second=accumulation_bases / elapsed if elapsed else 0.0,
                learning_rate=scheduled_lr,
                past_surprise_rms_max=memory_gradient_statistics[
                    "past_surprise_rms_max"
                ],
                momentary_surprise_rms_max=memory_gradient_statistics[
                    "momentary_surprise_rms_max"
                ],
                combined_surprise_rms_max=memory_gradient_statistics[
                    "combined_surprise_rms_max"
                ],
                forgotten_weight_rms_max=memory_gradient_statistics[
                    "forgotten_weight_rms_max"
                ],
                past_momentary_cosine_mean=memory_gradient_statistics[
                    "past_momentary_cosine_mean"
                ] / accumulation_segments,
            )
            self.history.append(record)
            if on_step is not None:
                on_step(record)
            accumulation_losses = []
            accumulation_outputs = []
            accumulation_segments = 0
            accumulation_examples = 0
            accumulation_tokens = 0
            accumulation_bases = 0
            retrieval_norm = 0.0
            update_norm = 0.0
            surprise_norm = 0.0
            state_drift_norm = 0.0
            gate_statistics = {key: 0.0 for key in gate_statistics}
            memory_gradient_statistics = {
                "raw_gradient_rms_max": 0.0,
                "conditioned_gradient_rms_max": 0.0,
                "gradient_scale_min": 1.0,
                "gradient_intervention_fraction": 0.0,
                "legacy_surprise_intervention_fraction": 0.0,
                "past_surprise_rms_max": 0.0,
                "momentary_surprise_rms_max": 0.0,
                "combined_surprise_rms_max": 0.0,
                "forgotten_weight_rms_max": 0.0,
                "past_momentary_cosine_mean": 0.0,
            }
            written_state_tensors = []
            started = time.perf_counter()

        while not scheduler.exhausted:
            if max_optimizer_steps is not None and self.optimizer_step >= max_optimizer_steps:
                break
            if max_valid_bases is not None and self.processed_bases >= max_valid_bases:
                break
            batch = scheduler.next_batch()
            tensors = self._batch_tensors(batch, self.device)
            states = self._states_for(batch)
            output = self.model.forward_segment(states, memory_mode=memory_mode, **tensors)
            assert output.loss_sum is not None
            self._require_finite(
                "forward output",
                [("loss_sum", output.loss_sum), ("logits", output.logits)]
                + self._state_tensors(output.states),
            )
            self._commit_states(batch, output.states)
            accumulation_losses.append(output.loss_sum)
            accumulation_outputs.append(
                (
                    output.logits.detach(),
                    tensors["labels"].detach(),
                    tensors["loss_mask"].detach(),
                    tensors["represented_base_counts"].detach(),
                )
            )
            accumulation_segments += 1
            accumulation_examples += len(batch)
            accumulation_tokens += output.valid_tokens
            accumulation_bases += output.valid_bases
            retrieval_norm += output.retrieval_norm
            update_norm += output.memory_update_norm
            surprise_norm += output.surprise_norm
            state_drift_norm += output.state_drift_norm
            for key in gate_statistics:
                gate_statistics[key] += output.gate_statistics[key]
            memory_gradient_statistics["raw_gradient_rms_max"] = max(
                memory_gradient_statistics["raw_gradient_rms_max"],
                output.memory_gradient_statistics["raw_gradient_rms_max"],
            )
            memory_gradient_statistics["conditioned_gradient_rms_max"] = max(
                memory_gradient_statistics["conditioned_gradient_rms_max"],
                output.memory_gradient_statistics["conditioned_gradient_rms_max"],
            )
            memory_gradient_statistics["gradient_scale_min"] = min(
                memory_gradient_statistics["gradient_scale_min"],
                output.memory_gradient_statistics["gradient_scale_min"],
            )
            memory_gradient_statistics["gradient_intervention_fraction"] += (
                output.memory_gradient_statistics["gradient_intervention_fraction"]
            )
            memory_gradient_statistics["legacy_surprise_intervention_fraction"] += (
                output.memory_gradient_statistics["legacy_surprise_intervention_fraction"]
            )
            for key in (
                "past_surprise_rms_max",
                "momentary_surprise_rms_max",
                "combined_surprise_rms_max",
                "forgotten_weight_rms_max",
            ):
                memory_gradient_statistics[key] = max(
                    memory_gradient_statistics[key],
                    output.memory_gradient_statistics[key],
                )
            memory_gradient_statistics["past_momentary_cosine_mean"] += (
                output.memory_gradient_statistics["past_momentary_cosine_mean"]
            )
            if accumulation_segments < horizon:
                for row_states in output.states:
                    for state in row_states:
                        for value in state.fast_weights.values():
                            if value.requires_grad:
                                value.retain_grad()
                                written_state_tensors.append(value)
            self.processed_segments += len(batch)
            self.processed_tokens += output.valid_tokens
            self.processed_bases += output.valid_bases
            if accumulation_segments == horizon:
                finish_step()
        finish_step()
        return tuple(self.history)

    def state_metadata(self) -> dict[str, object]:
        return {
            "optimizer_step": self.optimizer_step,
            "processed_segments": self.processed_segments,
            "processed_tokens": self.processed_tokens,
            "processed_bases": self.processed_bases,
            "history": [record.to_dict() for record in self.history],
        }
