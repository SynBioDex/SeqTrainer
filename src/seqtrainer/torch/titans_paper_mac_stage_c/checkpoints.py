"""Versioned, atomic Stage C model/data-cursor/functional-state checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Mapping

import numpy as np
import torch

from seqtrainer.torch.titans_paper_mac import PaperMACStreamState

from .model import BlockStates
from .trainer import StageCScheduler, StageCTrainer, TrainingStepRecord


CHECKPOINT_FORMAT_VERSION = 2
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = frozenset({1, 2})


def checkpoint_parent_dataset_fingerprint(payload: Mapping[str, object]) -> str:
    """Return the immutable token dataset hash, excluding optional panel hashes."""

    components = payload.get("dataset_components")
    if isinstance(components, Mapping) and components.get("dataset"):
        return str(components["dataset"])
    return str(payload.get("dataset_fingerprint", ""))


def _cpu_byte_rng_state(value: object, *, name: str) -> torch.Tensor:
    """Normalize RNG metadata after a CUDA ``map_location`` checkpoint load."""

    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint {name} RNG state is invalid")
    return value.detach().to(device="cpu", dtype=torch.uint8)


def _serialize_states(states: Mapping[str, BlockStates]) -> dict[str, list[dict[str, object]]]:
    return {
        stream_id: [state.to_state_dict() for state in block_states]
        for stream_id, block_states in states.items()
    }


def _restore_states(
    payload: Mapping[str, object],
    *,
    device: torch.device,
) -> dict[str, BlockStates]:
    restored: dict[str, BlockStates] = {}
    for stream_id, raw_states in payload.items():
        if not isinstance(raw_states, list):
            raise ValueError("checkpoint functional state payload is invalid")
        restored[str(stream_id)] = tuple(
            PaperMACStreamState.from_state_dict(raw, device=device)
            for raw in raw_states
            if isinstance(raw, Mapping)
        )
        if len(restored[str(stream_id)]) != len(raw_states):
            raise ValueError("checkpoint contains an invalid block state")
    return restored


def save_stage_c_checkpoint(
    path: str | Path,
    trainer: StageCTrainer,
    scheduler: StageCScheduler,
    *,
    dataset_fingerprint: str,
    code_commit: str,
    dataset_components: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save only at the trainer's detached optimizer boundary."""

    if not dataset_fingerprint or not code_commit:
        raise ValueError("dataset fingerprint and code commit are required")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "format_version": min(CHECKPOINT_FORMAT_VERSION, trainer.model.config.format_version),
        "model_config": trainer.model.config.to_dict(),
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": scheduler.to_state_dict(),
        "stream_states": _serialize_states(trainer.stream_states),
        "trainer_state": trainer.state_metadata(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_components": dict(dataset_components or {}),
        "code_commit": code_commit,
    }
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_stage_c_checkpoint(
    path: str | Path,
    trainer: StageCTrainer,
    scheduler: StageCScheduler,
    *,
    dataset_fingerprint: str,
    trusted: bool = False,
) -> dict[str, object]:
    """Restore an explicitly trusted checkpoint and reject data/config drift."""

    if not trusted:
        raise ValueError("Stage C checkpoints contain Python RNG state; pass trusted=True for owned files")
    source = Path(path)
    try:
        payload = torch.load(source, map_location=trainer.device, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=trainer.device)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS
    ):
        raise ValueError("unsupported Stage C checkpoint format")
    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("dataset fingerprint changed across resume")
    if payload.get("model_config") != trainer.model.config.to_dict():
        raise ValueError("model/tokenizer/backend configuration changed across resume")
    trainer.model.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("checkpoint is missing scheduler state")
    scheduler.load_state_dict(scheduler_state)
    raw_states = payload.get("stream_states")
    if not isinstance(raw_states, Mapping):
        raise ValueError("checkpoint is missing functional stream states")
    trainer.stream_states = _restore_states(raw_states, device=trainer.device)
    raw_trainer = payload.get("trainer_state")
    if not isinstance(raw_trainer, Mapping):
        raise ValueError("checkpoint is missing trainer state")
    trainer.optimizer_step = int(raw_trainer.get("optimizer_step", 0))
    trainer.processed_segments = int(raw_trainer.get("processed_segments", 0))
    trainer.processed_tokens = int(raw_trainer.get("processed_tokens", 0))
    trainer.processed_bases = int(raw_trainer.get("processed_bases", 0))
    raw_history = raw_trainer.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("checkpoint history is invalid")
    trainer.history = [TrainingStepRecord(**row) for row in raw_history if isinstance(row, Mapping)]
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("checkpoint RNG state is invalid")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(rng.get("torch_cpu"), name="torch_cpu"))
    raw_cuda_rng = rng.get("torch_cuda", [])
    if torch.cuda.is_available() and raw_cuda_rng:
        if not isinstance(raw_cuda_rng, (list, tuple)):
            raise ValueError("checkpoint torch_cuda RNG state is invalid")
        torch.cuda.set_rng_state_all(
            [_cpu_byte_rng_state(value, name="torch_cuda") for value in raw_cuda_rng]
        )
    return dict(payload)


def warm_start_stage_c_checkpoint(
    path: str | Path,
    trainer: StageCTrainer,
    *,
    trusted: bool = False,
) -> dict[str, object]:
    """Continue slow weights/optimizer/RNG while starting a new stream panel."""

    if not trusted:
        raise ValueError("Stage C checkpoints contain Python RNG state; pass trusted=True")
    source = Path(path)
    try:
        payload = torch.load(source, map_location=trainer.device, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=trainer.device)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS
    ):
        raise ValueError("unsupported Stage C checkpoint format")
    if payload.get("model_config") != trainer.model.config.to_dict():
        raise ValueError("model/tokenizer/backend configuration changed across warm start")
    trainer.model.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    raw_trainer = payload.get("trainer_state")
    if not isinstance(raw_trainer, Mapping):
        raise ValueError("checkpoint is missing trainer state")
    trainer.optimizer_step = int(raw_trainer.get("optimizer_step", 0))
    trainer.processed_segments = int(raw_trainer.get("processed_segments", 0))
    trainer.processed_tokens = int(raw_trainer.get("processed_tokens", 0))
    trainer.processed_bases = int(raw_trainer.get("processed_bases", 0))
    # A warm start is a new exposure event, so old functional stream state and
    # step history are deliberately not imported.
    trainer.stream_states = {}
    trainer.history = []
    rng = payload.get("rng")
    if not isinstance(rng, Mapping):
        raise ValueError("checkpoint RNG state is invalid")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(rng.get("torch_cpu"), name="torch_cpu"))
    raw_cuda_rng = rng.get("torch_cuda", [])
    if torch.cuda.is_available() and raw_cuda_rng:
        if not isinstance(raw_cuda_rng, (list, tuple)):
            raise ValueError("checkpoint torch_cuda RNG state is invalid")
        torch.cuda.set_rng_state_all(
            [_cpu_byte_rng_state(value, name="torch_cuda") for value in raw_cuda_rng]
        )
    return dict(payload)
