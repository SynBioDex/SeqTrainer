"""Read-only deterministic reload/continuation verification for Stage C checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch

from seqtrainer.data.bacteria_titan import (
    StageCPanelManifest,
    TokenStreamDataset,
    validate_panel_against_dataset,
)

from .checkpoints import SUPPORTED_CHECKPOINT_FORMAT_VERSIONS, load_stage_c_checkpoint
from .config import StageCModelConfig
from .model import StageCPaperMACForCausalLM
from .trainer import StageCTrainer, StatefulRotationScheduler, StreamBatchScheduler


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_digest(digest: "hashlib._Hash", value: object) -> None:
    """Hash nested checkpoint state with explicit type, shape, and key boundaries."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(b"\0")
        digest.update(array.tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        digest.update(str(len(value)).encode())
        digest.update(b"\0")
        for item in value:
            _update_digest(digest, item)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")
    else:
        raise TypeError(f"unsupported digest value: {type(value).__name__}")


def _stable_digest(value: object) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def _read_owned_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    if payload.get("format_version") not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError("unsupported Stage C checkpoint format")
    return payload


def _stream_state_payload(trainer: StageCTrainer) -> dict[str, object]:
    return {
        stream_id: [state.to_state_dict() for state in states]
        for stream_id, states in sorted(trainer.stream_states.items())
    }


def _rng_payload() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _run_continuation(
    checkpoint: Path,
    dataset: TokenStreamDataset,
    dataset_fingerprint: str,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    expected_step: int | None,
    stream_ids: frozenset[str] | None,
) -> dict[str, object]:
    payload = _read_owned_checkpoint(checkpoint, device)
    raw_config = payload.get("model_config")
    raw_scheduler = payload.get("scheduler_state")
    if not isinstance(raw_config, Mapping) or not isinstance(raw_scheduler, Mapping):
        raise ValueError("checkpoint is missing model or scheduler configuration")
    config = StageCModelConfig.from_dict(raw_config)
    model = StageCPaperMACForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    streams = dataset.streams(split="train", stream_ids=stream_ids)
    if raw_scheduler.get("policy") == StatefulRotationScheduler.POLICY:
        scheduler = StatefulRotationScheduler(
            streams,
            batch_size=int(raw_scheduler["batch_size"]),
            burst_segments=int(raw_scheduler["burst_segments"]),
            seed=int(raw_scheduler["seed"]),
            shuffle=bool(raw_scheduler["shuffle"]),
        )
    else:
        scheduler = StreamBatchScheduler(
            streams,
            batch_size=int(raw_scheduler["batch_size"]),
            seed=int(raw_scheduler["seed"]),
            shuffle=bool(raw_scheduler["shuffle"]),
        )
    trainer = StageCTrainer(
        model,
        optimizer,
        device=device,
        gradient_clip_norm=gradient_clip_norm,
    )
    loaded = load_stage_c_checkpoint(
        checkpoint,
        trainer,
        scheduler,
        dataset_fingerprint=dataset_fingerprint,
        trusted=True,
    )
    loaded_step = trainer.optimizer_step
    if expected_step is not None and loaded_step != expected_step:
        raise ValueError(
            f"checkpoint optimizer step mismatch: expected {expected_step}, got {loaded_step}"
        )
    trainer.train(
        scheduler,
        max_optimizer_steps=loaded_step + 1,
        memory_mode=config.memory_mode,
    )
    if trainer.optimizer_step != loaded_step + 1:
        raise RuntimeError("deterministic continuation did not execute exactly one optimizer step")
    latest = trainer.history[-1].to_dict()
    if not all(
        np.isfinite(float(latest[key]))
        for key in (
            "loss_per_token",
            "bits_per_base",
            "gradient_norm",
            "memory_update_norm",
            "state_drift_norm",
        )
    ):
        raise RuntimeError("continuation produced a non-finite metric")
    deterministic_record = {
        key: value
        for key, value in latest.items()
        if key not in {"elapsed_seconds", "bases_per_second"}
    }
    components = {
        "model": _stable_digest(trainer.model.state_dict()),
        "optimizer": _stable_digest(trainer.optimizer.state_dict()),
        "scheduler": _stable_digest(scheduler.to_state_dict()),
        "functional_memory": _stable_digest(_stream_state_payload(trainer)),
        "rng": _stable_digest(_rng_payload()),
        "scientific_record": _stable_digest(deterministic_record),
    }
    return {
        "checkpoint_code_commit": loaded.get("code_commit"),
        "loaded_optimizer_step": loaded_step,
        "continued_optimizer_step": trainer.optimizer_step,
        "processed_bases": trainer.processed_bases,
        "latest_record": latest,
        "component_hashes": components,
        "combined_hash": _stable_digest(components),
        "model_config": config.to_dict(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-code-commit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be positive")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    checkpoint = args.checkpoint.resolve()
    before_hash = _sha256(checkpoint)
    manifest_path = args.dataset_dir / "token_stream_manifest.json"
    dataset = TokenStreamDataset(args.dataset_dir)
    panel = (
        StageCPanelManifest.from_path(args.panel_manifest)
        if args.panel_manifest
        else None
    )
    if panel:
        validate_panel_against_dataset(panel, dataset)
        if panel.payload["split"] != "train":
            raise ValueError("resume-verification panel must select train streams")
    parent_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    source_payload = _read_owned_checkpoint(checkpoint, device)
    if panel:
        components = source_payload.get("dataset_components")
        if (
            not isinstance(components, Mapping)
            or components.get("dataset") != parent_fingerprint
            or components.get("train_panel") != panel.hash
        ):
            raise ValueError("checkpoint, dataset, and training panel do not match")
        dataset_fingerprint = str(source_payload["dataset_fingerprint"])
    else:
        dataset_fingerprint = parent_fingerprint
    first = _run_continuation(
        checkpoint,
        dataset,
        dataset_fingerprint,
        device=device,
        gradient_clip_norm=args.gradient_clip_norm,
        expected_step=args.expected_step,
        stream_ids=panel.stream_ids if panel else None,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    second = _run_continuation(
        checkpoint,
        dataset,
        dataset_fingerprint,
        device=device,
        gradient_clip_norm=args.gradient_clip_norm,
        expected_step=args.expected_step,
        stream_ids=panel.stream_ids if panel else None,
    )
    after_hash = _sha256(checkpoint)
    deterministic = first["component_hashes"] == second["component_hashes"]
    source_unchanged = before_hash == after_hash
    observed_commit = first.get("checkpoint_code_commit")
    commit_matches = (
        True
        if args.expected_code_commit is None
        else observed_commit == args.expected_code_commit
    )
    # Older c9d checkpoints recorded ``unknown`` internally because training
    # resolved Git from Colab's working directory. The surrounding immutable
    # Colab manifest remains the authoritative clean commit for that run.
    commit_warning = (
        None
        if commit_matches
        else (
            f"checkpoint records {observed_commit!r}; expected "
            f"{args.expected_code_commit!r}. Verify the Colab run manifest."
        )
    )
    passed = deterministic and source_unchanged
    report = {
        "format_version": 1,
        "status": "passed" if passed else "failed",
        "read_only_source_checkpoint": source_unchanged,
        "deterministic_continuation": deterministic,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": before_hash,
        "checkpoint_sha256_after": after_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "expected_code_commit": args.expected_code_commit,
        "checkpoint_code_commit": observed_commit,
        "code_commit_matches": commit_matches,
        "code_commit_warning": commit_warning,
        "first_continuation": first,
        "second_continuation": second,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
