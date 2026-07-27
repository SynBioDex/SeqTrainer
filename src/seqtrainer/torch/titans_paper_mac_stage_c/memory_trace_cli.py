"""Compile segment-level hidden-state and functional-memory traces from a checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from seqtrainer.data.bacteria_titan import TokenStreamDataset

from .config import MemoryMode, StageCModelConfig
from .model import BlockStates, StageCPaperMACForCausalLM, detach_stream_states
from .study import StudyProtocol
from .trainer import StageCTrainer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--memory-mode", choices=[item.value for item in MemoryMode], default="adaptive")
    parser.add_argument("--max-streams", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=128)
    parser.add_argument("--samples-per-tensor", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-amendment", type=Path, action="append", default=[])
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if bool(args.protocol) != bool(args.run_id):
        parser.error("--protocol and --run-id must be supplied together")
    if args.protocol_amendment and not args.protocol:
        parser.error("--protocol-amendment requires --protocol and --run-id")
    if min(args.max_streams, args.max_segments, args.samples_per_tensor) <= 0:
        parser.error("trace limits must be positive")
    return args


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is invalid")
    return payload


def _tensor_features(value: Tensor, samples: int) -> list[float]:
    flat = value.detach().float().cpu().reshape(-1)
    if not flat.numel():
        return [0.0] * (samples + 3)
    indices = torch.linspace(0, flat.numel() - 1, samples).round().long()
    return [
        float(flat.mean()),
        float(flat.std(unbiased=False)),
        float(flat.square().mean().sqrt()),
        *[float(item) for item in flat[indices]],
    ]


def _state_features(states: BlockStates, samples: int) -> list[float]:
    values: list[float] = []
    for state in states:
        for mapping in (state.fast_weights, state.surprise):
            for tensor in mapping.values():
                values.extend(_tensor_features(tensor, samples))
        for history in (state.query_history, state.write_history):
            if history is not None:
                values.extend(_tensor_features(history, samples))
    return values


def _pca(features: np.ndarray) -> tuple[np.ndarray, list[float]]:
    centered = features - features.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    standardized = centered / np.where(scale > 1e-12, scale, 1.0)
    left, singular, _ = np.linalg.svd(standardized, full_matrices=False)
    components = left[:, :2] * singular[:2]
    variance = singular.square()
    fractions = (
        (variance[:2] / variance.sum()).tolist()
        if variance.sum() > 0
        else [0.0, 0.0]
    )
    if components.shape[1] == 1:
        components = np.column_stack((components, np.zeros(len(components))))
        fractions.append(0.0)
    return components, [float(item) for item in fractions[:2]]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _scatter_svg(points: np.ndarray, rows: list[dict[str, object]], variance: list[float]) -> str:
    width, height, margin = 900, 620, 70
    x, y = points[:, 0], points[:, 1]
    xspan = max(float(x.max() - x.min()), 1e-9)
    yspan = max(float(y.max() - y.min()), 1e-9)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    clades = {name: colors[index % len(colors)] for index, name in enumerate(sorted({str(row["clade_group"]) for row in rows}))}
    circles = []
    for index, row in enumerate(rows):
        px = margin + (float(x[index] - x.min()) / xspan) * (width - 2 * margin)
        py = height - margin - (float(y[index] - y.min()) / yspan) * (height - 2 * margin)
        circles.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{clades[str(row["clade_group"])]}" opacity="0.72"/>'
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="24" y="32" font-family="sans-serif" font-size="20">Deep-memory and hidden-state PCA</text>',
            f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif">PC1 ({100 * variance[0]:.1f}%)</text>',
            f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif">PC2 ({100 * variance[1]:.1f}%)</text>',
            *circles,
            "</svg>",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol:
        StudyProtocol.from_path(args.protocol).validate_run_config(
            args.run_id,
            {"phase": "analysis"},
            amendment_paths=args.protocol_amendment,
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
    payload = _load_checkpoint(args.checkpoint, device)
    fingerprint = hashlib.sha256(
        (args.dataset_dir / "token_stream_manifest.json").read_bytes()
    ).hexdigest()
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ValueError("checkpoint and trace dataset fingerprints differ")
    config_payload = payload.get("model_config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("checkpoint is missing model_config")
    config = StageCModelConfig.from_dict(config_payload)
    model = StageCPaperMACForCausalLM(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    rows: list[dict[str, object]] = []
    vectors: list[list[float]] = []
    streams = dataset.streams(split=args.split)
    for stream_number, stream_id in enumerate(sorted(streams)):
        if stream_number >= args.max_streams or len(rows) >= args.max_segments:
            break
        states = model.initial_states(stream_id)
        for segment in streams[stream_id]:
            if len(rows) >= args.max_segments:
                break
            tensors = StageCTrainer._batch_tensors((segment,), device)
            with torch.enable_grad():
                output = model.forward_segment(
                    (states,),
                    memory_mode=args.memory_mode,
                    **tensors,
                )
            if output.loss_sum is None or output.hidden_states is None:
                raise RuntimeError("trace forward did not produce losses and hidden states")
            hidden = output.hidden_states[0][tensors["valid_mask"][0]]
            vector = [
                *hidden.detach().float().mean(dim=0).cpu().tolist(),
                *hidden.detach().float().std(dim=0, unbiased=False).cpu().tolist(),
                *_state_features(output.states[0], args.samples_per_tensor),
            ]
            bpb = float(output.loss_sum.detach()) / (
                max(output.valid_bases, 1) * math.log(2.0)
            )
            row: dict[str, object] = {
                "stream_id": segment.stream_id,
                "accession": segment.accession,
                "contig_id": segment.contig_id,
                "clade_group": segment.clade_group,
                "gc_fraction": segment.gc_fraction,
                "segment_index": segment.segment_index,
                "base_start": segment.base_start,
                "bits_per_base": bpb,
                "retrieval_norm": output.retrieval_norm,
                "memory_update_norm": output.memory_update_norm,
                "surprise_norm": output.surprise_norm,
                "state_drift_norm": output.state_drift_norm,
                **output.gate_statistics,
                **{
                    f"memory_{key}": value
                    for key, value in output.memory_gradient_statistics.items()
                },
            }
            rows.append(row)
            vectors.append(vector)
            states = detach_stream_states(output.states[0])
    if len(rows) < 2:
        raise ValueError("memory trace requires at least two evaluated segments")
    feature_matrix = np.asarray(vectors, dtype=np.float64)
    points, variance = _pca(feature_matrix)
    for row, point in zip(rows, points):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])
    gc = np.asarray([float(row["gc_fraction"]) for row in rows])
    position = np.asarray([float(row["segment_index"]) for row in rows])
    summary = {
        "format_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_config": config.to_dict(),
        "split": args.split,
        "memory_mode": args.memory_mode,
        "segments": len(rows),
        "streams": len({str(row["stream_id"]) for row in rows}),
        "feature_count": int(feature_matrix.shape[1]),
        "pca_explained_variance": variance,
        "pc1_gc_correlation": _correlation(points[:, 0], gc),
        "pc1_stream_position_correlation": _correlation(points[:, 0], position),
        "mean_bits_per_base": float(np.mean([float(row["bits_per_base"]) for row in rows])),
        "mean_memory_update_norm": float(np.mean([float(row["memory_update_norm"]) for row in rows])),
        "mean_surprise_norm": float(np.mean([float(row["surprise_norm"]) for row in rows])),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "memory_trace.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "memory_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "memory_pca.svg").write_text(
        _scatter_svg(points, rows, variance),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
