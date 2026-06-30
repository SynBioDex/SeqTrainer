"""Shared split loading and summary helpers for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import BenchmarkConfig

REQUIRED_SPLITS = ("train", "validation", "test")


def resolve_split_paths(config: BenchmarkConfig, base_dir: str | Path | None = None) -> dict[str, Path]:
    """Resolve configured split paths for predefined split benchmarks."""
    if config.split.strategy != "predefined":
        raise ValueError("resolve_split_paths only supports split.strategy='predefined'")

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    paths = {}
    for split in REQUIRED_SPLITS:
        raw_path = config.dataset.split_files[split]
        path = Path(raw_path)
        paths[split] = path if path.is_absolute() else root / path
    return paths


def load_predefined_split_frames(
    config: BenchmarkConfig,
    base_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load train/validation/test CSV split files with basic schema checks."""
    paths = resolve_split_paths(config, base_dir=base_dir)
    frames: dict[str, pd.DataFrame] = {}
    required_columns = {config.dataset.sequence_field, config.dataset.label_field}
    if config.dataset.id_field:
        required_columns.add(config.dataset.id_field)

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split file: {path}")
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{split} split is missing required columns: {sorted(missing)}")
        frames[split] = frame
    return frames


def summarize_split_frames(config: BenchmarkConfig, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Summarize split sizes, class balance, and source files for manifests."""
    summary: dict[str, Any] = {}
    for split in REQUIRED_SPLITS:
        frame = frames[split]
        labels = frame[config.dataset.label_field]
        class_counts = labels.value_counts().sort_index().to_dict()
        split_summary: dict[str, Any] = {
            "rows": int(len(frame)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        }
        if config.dataset.sequence_field in frame:
            lengths = frame[config.dataset.sequence_field].astype(str).str.len()
            split_summary["sequence_length"] = {
                "min": int(lengths.min()) if len(lengths) else None,
                "median": float(lengths.median()) if len(lengths) else None,
                "max": int(lengths.max()) if len(lengths) else None,
            }
        if config.dataset.id_field and config.dataset.id_field in frame:
            split_summary["unique_ids"] = int(frame[config.dataset.id_field].nunique())
        summary[split] = split_summary
    return summary

