"""Reconstruct and document the exact Stage C model stored in a checkpoint.

The report intentionally derives its architecture from ``model_config`` and
``model_state`` in the checkpoint, instead of echoing notebook arguments.  It
therefore documents the model that was actually trained, including a compact
description of the stream-local functional-memory state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .config import StageCModelConfig
from .model import StageCPaperMACForCausalLM


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is invalid")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_groups(model: StageCPaperMACForCausalLM) -> dict[str, int]:
    """Count unique trainable parameters by top-level model component."""

    groups: defaultdict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        pieces = name.split(".")
        group = ".".join(pieces[:3]) if name.startswith("stack.blocks.") else pieces[0]
        groups[group] += parameter.numel()
    return dict(sorted(groups.items()))


def _tensor_shapes(values: Mapping[str, Tensor]) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in values.items()}


def _tensor_bytes(values: Sequence[Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def _memory_summary(model: StageCPaperMACForCausalLM) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for index, block in enumerate(model.stack.blocks):
        memory = block.memory
        state = memory.initial_state("architecture-inspection")
        histories = [
            history
            for history in (state.query_history, state.write_history)
            if history is not None
        ]
        state_tensors = [*state.fast_weights.values(), *state.surprise.values(), *histories]
        summaries.append(
            {
                "block": index,
                "memory_module": type(memory.memory_mlp).__name__,
                "gate_module": type(memory.gates).__name__,
                "d_model": memory.d_model,
                "segment_length": memory.segment_length,
                "architecture": memory.architecture,
                "expansion_factor": memory.expansion_factor,
                "projection_convolution_kernel": memory.projection_convolution_kernel,
                "normalize_queries_and_keys": memory.normalize_queries_and_keys,
                "associative_loss_reduction": memory.associative_loss_reduction,
                "max_surprise_norm": memory.max_surprise_norm,
                "max_gradient_rms": memory.max_gradient_rms,
                "max_gradient_rms_ratio": memory.max_gradient_rms_ratio,
                "fast_weight_shapes": _tensor_shapes(state.fast_weights),
                "surprise_shapes": _tensor_shapes(state.surprise),
                "projection_history_shapes": [list(history.shape) for history in histories],
                "functional_state_bytes_per_stream": _tensor_bytes(state_tensors),
            }
        )
    return summaries


def build_architecture_report(checkpoint: Path) -> tuple[dict[str, object], str]:
    """Load a checkpoint and return machine- and human-readable architecture data."""

    payload = _load_checkpoint(checkpoint)
    raw_config = payload.get("model_config")
    raw_state = payload.get("model_state")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint is missing model_config")
    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint is missing model_state")
    config = StageCModelConfig.from_dict(raw_config)
    model = StageCPaperMACForCausalLM(config)
    model.load_state_dict(raw_state, strict=True)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    memory = _memory_summary(model)
    report: dict[str, object] = {
        "format_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_format_version": payload.get("format_version"),
        "code_commit": payload.get("code_commit"),
        "model_config": config.to_dict(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_groups": _parameter_groups(model),
        "tied_input_output_embeddings": model.lm_head.weight is model.token_embeddings.weight,
        "functional_memory_by_block": memory,
        "functional_state_bytes_per_stream": sum(
            int(block["functional_state_bytes_per_stream"]) for block in memory
        ),
    }
    text = "\n".join(
        (
            "Stage C checkpoint architecture",
            f"checkpoint: {checkpoint}",
            f"checkpoint_sha256: {report['checkpoint_sha256']}",
            f"code_commit: {report['code_commit']}",
            f"total_parameters: {total_parameters:,}",
            f"trainable_parameters: {trainable_parameters:,}",
            f"tied_input_output_embeddings: {report['tied_input_output_embeddings']}",
            f"functional_state_bytes_per_stream: {report['functional_state_bytes_per_stream']:,}",
            "",
            "Model configuration:",
            json.dumps(config.to_dict(), indent=2, sort_keys=True),
            "",
            "Unique parameter groups:",
            json.dumps(report["parameter_groups"], indent=2, sort_keys=True),
            "",
            "Functional memory by block:",
            json.dumps(memory, indent=2, sort_keys=True),
            "",
            "PyTorch module tree:",
            repr(model),
            "",
        )
    )
    return report, text


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, text = build_architecture_report(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MODEL_ARCHITECTURE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "MODEL_ARCHITECTURE.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0
