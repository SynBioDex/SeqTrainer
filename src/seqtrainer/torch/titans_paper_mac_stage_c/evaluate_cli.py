"""Evaluate trained Stage C runs and apply the held-out memory-ablation gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch

from seqtrainer.data.bacteria_titan import TokenStreamDataset

from .config import MemoryMode, StageCModelConfig
from .evaluation import EvaluationResult, evaluate_ordered_streams
from .model import StageCPaperMACForCausalLM
from .reporting import bar_svg


def _load_payload(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid Stage C checkpoint: {path}")
    return payload


def _bootstrap_difference(
    adaptive: Mapping[str, float],
    control: Mapping[str, float],
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, float | int]:
    keys = sorted(set(adaptive) & set(control))
    if not keys:
        return {"groups": 0, "mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    differences = [adaptive[key] - control[key] for key in keys]
    rng = random.Random(seed)
    bootstrapped = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "groups": len(keys),
        "mean": sum(differences) / len(differences),
        "lower_95": bootstrapped[int(0.025 * (samples - 1))],
        "upper_95": bootstrapped[int(0.975 * (samples - 1))],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=/path/to/latest.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-streams", type=int, default=0)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260747)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    streams = dataset.streams(split=args.split)
    fingerprint = hashlib.sha256(
        (args.dataset_dir / "token_stream_manifest.json").read_bytes()
    ).hexdigest()
    results: dict[str, EvaluationResult] = {}
    run_paths: dict[str, Path] = {}
    for specification in args.run:
        if "=" not in specification:
            raise ValueError("--run must use NAME=/checkpoint/path")
        name, raw_path = specification.split("=", 1)
        if name in run_paths:
            raise ValueError(f"duplicate run name: {name}")
        run_paths[name] = Path(raw_path).resolve()
    required_modes = {
        "adaptive": MemoryMode.ADAPTIVE,
        "reference": MemoryMode.REFERENCE,
        "frozen_memory": MemoryMode.FROZEN,
        "no_memory": MemoryMode.NONE,
    }
    missing = set(required_modes) - set(run_paths)
    if missing:
        raise ValueError(f"evaluation requires separately trained runs: {sorted(missing)}")
    if len(set(run_paths.values())) != len(run_paths):
        raise ValueError("every evaluation condition requires a distinct checkpoint path")
    for name, checkpoint_path in run_paths.items():
        payload = _load_payload(checkpoint_path, device)
        if payload.get("dataset_fingerprint") != fingerprint:
            raise ValueError(f"dataset fingerprint mismatch for run {name}")
        config_payload = payload.get("model_config")
        if not isinstance(config_payload, Mapping):
            raise ValueError(f"run {name} is missing model config")
        config = StageCModelConfig.from_dict(config_payload)
        expected_mode = required_modes.get(name)
        if expected_mode is not None and config.memory_mode is not expected_mode:
            raise ValueError(
                f"run {name} has memory mode {config.memory_mode.value}; expected {expected_mode.value}"
            )
        model = StageCPaperMACForCausalLM(config).to(device)
        model.load_state_dict(payload["model_state"])
        results[name] = evaluate_ordered_streams(
            model,
            streams,
            device=device,
            memory_mode=config.memory_mode,
            max_streams=None if args.max_streams == 0 else args.max_streams,
            max_segments=None if args.max_segments == 0 else args.max_segments,
        )
    comparisons = {}
    for control in ("frozen_memory", "no_memory"):
        if control not in results:
            continue
        improvement = results[control].bits_per_base - results["adaptive"].bits_per_base
        bootstrap = _bootstrap_difference(
            results["adaptive"].per_accession_bpb,
            results[control].per_accession_bpb,
            seed=args.seed,
        )
        comparisons[control] = {
            "adaptive_improvement_bpb": improvement,
            "paired_adaptive_minus_control": bootstrap,
            "passes_0_01_bpb": improvement >= 0.01,
            "bootstrap_direction_passed": float(bootstrap["upper_95"]) < 0,
        }
    gate_passed = all(
        comparison["passes_0_01_bpb"] and comparison["bootstrap_direction_passed"]
        for comparison in comparisons.values()
    ) and {"frozen_memory", "no_memory"}.issubset(comparisons)
    payload = {
        "format_version": 1,
        "split": args.split,
        "results": {name: result.to_dict() for name, result in results.items()},
        "comparisons": comparisons,
        "full_corpus_gate_passed": gate_passed,
        "threshold_bpb": 0.01,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evaluation_bpb.svg").write_text(
        bar_svg(
            list(results),
            [result.bits_per_base for result in results.values()],
            title=f"Stage C held-out {args.split} BPB",
            x_label="Bits per base (lower is better)",
        ),
        encoding="utf-8",
    )
    gc_labels = [
        f"{name}: {gc_bin.removeprefix('gc_')}"
        for name, result in results.items()
        for gc_bin in sorted(result.per_gc_bin_bpb)
    ]
    gc_values = [
        result.per_gc_bin_bpb[gc_bin]
        for result in results.values()
        for gc_bin in sorted(result.per_gc_bin_bpb)
    ]
    (args.output_dir / "evaluation_gc_bpb.svg").write_text(
        bar_svg(
            gc_labels,
            gc_values,
            title=f"Stage C held-out {args.split} BPB by whole-contig GC bin",
            x_label="Bits per base (lower is better)",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "evaluation_memory_diagnostics.svg").write_text(
        bar_svg(
            [
                f"{name}: {metric}"
                for name in results
                for metric in ("retrieval", "update", "surprise", "drift")
            ],
            [
                value
                for result in results.values()
                for value in (
                    result.retrieval_norm_mean,
                    result.memory_update_norm_mean,
                    result.surprise_norm_mean,
                    result.state_drift_norm_mean,
                )
            ],
            title="Held-out memory behavior",
            x_label="Mean segment norm",
        ),
        encoding="utf-8",
    )
    lines = [
        "# Stage C held-out evaluation",
        "",
        f"Split: `{args.split}`",
        "",
        "| Run | BPB | Perplexity | Token accuracy | Top-2 accuracy | Streams | Segments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result.bits_per_base:.4f} | {result.perplexity:.4f} | "
            f"{result.token_accuracy:.4f} | {result.top_2_accuracy:.4f} | "
            f"{result.streams} | {result.segments} |"
        )
    lines.extend(
        [
            "",
            "## GC-stratified BPB",
            "",
            "Whole-contig GC bins are reported in `evaluation.json` and `evaluation_gc_bpb.svg`.",
            "Memory retrieval/update/surprise/state-drift norms and gate distributions are also retained in the JSON evidence.",
        ]
    )
    lines.extend(
        [
            "",
            "## Full-corpus gate",
            "",
            f"**{'GREEN — passed' if gate_passed else 'RED — not passed'}**",
            "",
            "The gate requires adaptive memory to improve by at least 0.01 BPB over both separately trained controls and paired accession bootstrap intervals to support the direction.",
        ]
    )
    (args.output_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
