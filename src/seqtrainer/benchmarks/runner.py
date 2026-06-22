"""Shared CNN benchmark runner entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig, load_benchmark_config
from .splits import resolve_split_paths


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Result metadata returned by a benchmark run."""

    output_dir: Path
    status: str
    metrics: dict[str, dict[str, Any]]
    manifest: dict[str, Any]


class BenchmarkSkipped(RuntimeError):
    """Raised when an optional benchmark cannot run in the current environment."""


def run_benchmark(
    config_or_path: BenchmarkConfig | str | Path,
    *,
    base_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    allow_skip: bool = True,
) -> BenchmarkRunResult:
    """Run a configured benchmark and write the common artifact set."""
    config = load_benchmark_config(config_or_path) if not isinstance(config_or_path, BenchmarkConfig) else config_or_path
    family = config.model.family
    if family == "cnn":
        return _run_cnn(config, base_dir=base_dir, output_dir=output_dir)
    raise BenchmarkSkipped(f"Only CNN benchmarks are implemented in this PR branch; got model family {family!r}.")


def _run_cnn(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
) -> BenchmarkRunResult:
    from seqtrainer.torch.cnn_baseline import CnnCsvSplitConfig, run_cnn_csv_splits

    paths = _split_paths(config, base_dir)
    params = dict(config.training.params)
    model_params = dict(config.model.params)
    result = run_cnn_csv_splits(
        CnnCsvSplitConfig(
            train_csv=paths["train"],
            validation_csv=paths["validation"],
            test_csv=paths["test"],
            output_dir=Path(output_dir or config.outputs.output_dir),
            dataset_name=config.dataset.name,
            source_accession=config.dataset.source_accession,
            source_url=config.dataset.source_url,
            sequence_field=config.dataset.sequence_field,
            label_field=config.dataset.label_field,
            sequence_length=config.preprocessing.sequence_length or 300,
            seed=config.training.seed,
            batch_size=config.training.batch_size or 16,
            cycles=config.training.max_epochs or 10,
            learning_rate=config.training.learning_rate or 1e-3,
            weight_decay=float(params.get("weight_decay", 0.0)),
            optimizer_name=str(params.get("optimizer", "adam")).lower(),
            scheduler_name=str(params.get("scheduler", "none")).lower(),
            select_best_by_mcc=bool(params.get("select_best_by_mcc", False)),
            early_stopping_patience=_optional_int(params.get("early_stopping_patience")),
            model_variant=str(model_params.get("variant", "tiny")),
            dropout=float(model_params.get("dropout", 0.25)),
            class_weighting=bool(params.get("class_weighting", False)),
            device=_resolve_device(config.environment.device),
        )
    )
    return BenchmarkRunResult(
        output_dir=result.output_dir,
        status="completed",
        metrics=result.metrics,
        manifest=result.manifest,
    )


def _split_paths(config: BenchmarkConfig, base_dir: str | Path | None) -> dict[str, Path]:
    from .splits import resolve_split_paths

    return resolve_split_paths(config, base_dir=base_dir)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
