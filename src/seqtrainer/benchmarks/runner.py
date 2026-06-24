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
    reason = f"Only CNN benchmarks are implemented in this PR branch; got model family {family!r}."
    if not allow_skip:
        raise BenchmarkSkipped(reason)
    return _write_skipped_result(config, base_dir=base_dir, output_dir=output_dir, reason=reason)


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
            positive_label=config.label.positive_label,
            negative_label=config.label.negative_label,
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
            threshold_strategy=config.evaluation.threshold_strategy,
            device=_resolve_device(config.environment.device),
            save_json=config.outputs.save_json,
            save_csv=config.outputs.save_csv,
            save_predictions=config.outputs.save_predictions,
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


def _write_skipped_result(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
    reason: str,
) -> BenchmarkRunResult:
    from .artifacts import write_benchmark_outputs
    from .manifest import build_run_manifest
    from .splits import load_predefined_split_frames, summarize_split_frames

    out_dir = Path(output_dir or config.outputs.output_dir)
    try:
        frames = load_predefined_split_frames(config, base_dir=base_dir)
        split_summary: dict[str, Any] = summarize_split_frames(config, frames)
    except Exception as exc:
        split_summary = {"warning": f"Could not load configured splits: {exc}"}

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=None,
        model_metadata={"status": "skipped"},
        extra={"status": "skipped", "skip_reason": reason},
    )
    write_benchmark_outputs(out_dir, manifest=manifest, config=config)
    return BenchmarkRunResult(output_dir=out_dir, status="skipped", metrics={}, manifest=manifest)


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
