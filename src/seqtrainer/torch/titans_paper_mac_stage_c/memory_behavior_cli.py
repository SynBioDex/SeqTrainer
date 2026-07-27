"""Probe delayed nonlinear associations in each trained functional-memory block."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from seqtrainer.torch.titans_paper_mac import ParameterGateValues, PaperMACStreamState

from .config import StageCModelConfig
from .model import StageCPaperMACForCausalLM
from .study import StudyProtocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260761)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if bool(args.protocol) != bool(args.run_id):
        parser.error("--protocol and --run-id must be supplied together")
    if args.pairs < 4:
        parser.error("--pairs must be at least four")
    return args


def _load(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is invalid")
    return payload


def _detach(state: PaperMACStreamState) -> PaperMACStreamState:
    return state.replace(
        fast_weights=OrderedDict(
            (name, value.detach().requires_grad_(True))
            for name, value in state.fast_weights.items()
        ),
        surprise=OrderedDict(
            (name, value.detach()) for name, value in state.surprise.items()
        ),
    )


def _token_gates(memory, token: Tensor):
    values = memory.gates(token.unsqueeze(0))
    if isinstance(values, ParameterGateValues):
        return tuple(
            OrderedDict((name, value[0]) for name, value in mapping.items())
            for mapping in (values.alpha, values.eta, values.theta)
        )
    return values.alpha[0], values.eta[0], values.theta[0]


def _mse(left: Tensor, right: Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).square().mean().cpu())


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol:
        StudyProtocol.from_path(args.protocol).validate_run_config(
            args.run_id, {"phase": "analysis"}
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    payload = _load(args.checkpoint, device)
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("checkpoint is missing model_config")
    config = StageCModelConfig.from_dict(raw_config)
    model = StageCPaperMACForCausalLM(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    keys = torch.randn(args.pairs, config.d_model, generator=generator).to(device)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    first = torch.randn(config.d_model, config.d_model, generator=generator).to(device)
    second = torch.randn(config.d_model, config.d_model, generator=generator).to(device)
    values = torch.tanh(keys @ first / config.d_model**0.5) + 0.25 * torch.sin(
        keys @ second / config.d_model**0.5
    )
    permutation = torch.randperm(args.pairs, generator=generator).to(device)
    block_results: list[dict[str, object]] = []
    for block_index, block in enumerate(model.stack.blocks):
        memory = block.memory
        state = memory.initial_state(f"controlled-block-{block_index}")
        immediate_before: list[float] = []
        immediate_after: list[float] = []
        for key, value in zip(keys, values):
            immediate_before.append(_mse(memory.retrieve(state, key), value))
            alpha, eta, theta = _token_gates(memory, key)
            state = memory.update_one(
                state,
                key,
                value,
                alpha=alpha,
                eta=eta,
                theta=theta,
            )
            immediate_after.append(_mse(memory.retrieve(state, key), value))
            state = _detach(state)
        delayed = memory.retrieve(state, keys)
        aligned_mse = _mse(delayed, values)
        shuffled_mse = _mse(delayed, values[permutation])
        before = sum(immediate_before) / len(immediate_before)
        after = sum(immediate_after) / len(immediate_after)
        block_results.append(
            {
                "block": block_index,
                "immediate_mse_before": before,
                "immediate_mse_after": after,
                "immediate_relative_improvement": (before - after) / max(before, 1e-12),
                "delayed_aligned_mse": aligned_mse,
                "delayed_shuffled_mse": shuffled_mse,
                "delayed_association_margin": shuffled_mse - aligned_mse,
            }
        )
    result = {
        "format_version": 1,
        "classification": "controlled_nonlinear_associative_memory_probe",
        "interpretation": (
            "Positive immediate improvement confirms that one memory write reduces "
            "the declared associative objective. A positive delayed margin indicates "
            "pair-specific recall after interference; neither metric is biological validation."
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_config": config.to_dict(),
        "pairs": args.pairs,
        "seed": args.seed,
        "blocks": block_results,
        "all_blocks_improve_immediately": all(
            float(row["immediate_relative_improvement"]) > 0 for row in block_results
        ),
        "blocks_with_positive_delayed_margin": sum(
            float(row["delayed_association_margin"]) > 0 for row in block_results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
