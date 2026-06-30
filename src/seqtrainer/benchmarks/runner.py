"""Shared benchmark runner entrypoints.

This module keeps CLI and notebook benchmark execution on the same path. CNN is
implemented as an in-package trainer. DNABERT2 and iPro-MP are dependency-gated
so the harness can be tested without downloading large external models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from seqtrainer.metrics import (
    best_threshold_by_metric,
    binary_classification_metrics,
    binary_classification_metrics_from_predictions,
)

from .artifacts import write_benchmark_outputs
from .config import BenchmarkConfig, load_benchmark_config
from .manifest import build_run_manifest
from .policy import decide_imbalance_policy, threshold_metric_from_strategy
from .splits import load_predefined_split_frames, resolve_split_paths, summarize_split_frames


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
    if family == "dnabert2":
        return _run_dnabert2(config, base_dir=base_dir, output_dir=output_dir, allow_skip=allow_skip)
    if family == "ipromp":
        return _run_ipromp(config, base_dir=base_dir, output_dir=output_dir, allow_skip=allow_skip)
    raise ValueError(f"Unsupported benchmark model family: {family}")


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
            sequence_length=_default_if_none(config.preprocessing.sequence_length, 300),
            seed=config.training.seed,
            batch_size=_default_if_none(config.training.batch_size, 16),
            cycles=_default_if_none(config.training.max_epochs, 10),
            learning_rate=_default_if_none(config.training.learning_rate, 1e-3),
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


def _run_dnabert2(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
    allow_skip: bool,
) -> BenchmarkRunResult:
    try:
        from seqtrainer.torch.dnabert2_benchmark import run_dnabert2_csv_splits

        return run_dnabert2_csv_splits(
            config,
            base_dir=base_dir,
            output_dir=Path(output_dir or config.outputs.output_dir),
        )
    except (BenchmarkSkipped, ModuleNotFoundError, ImportError, OSError) as exc:
        if not allow_skip:
            raise
        return _write_skipped_result(config, base_dir=base_dir, output_dir=output_dir, reason=str(exc))
    except RuntimeError as exc:
        if not allow_skip or not _looks_like_resource_error(exc):
            raise
        reason = f"DNABERT2 benchmark could not run with the available compute resources: {exc}"
        return _write_skipped_result(config, base_dir=base_dir, output_dir=output_dir, reason=reason)


def _run_ipromp(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
    allow_skip: bool,
) -> BenchmarkRunResult:
    params = dict(config.model.params)
    out_dir = Path(output_dir or config.outputs.output_dir)
    predictions_csv = params.get("predictions_csv")
    validation_predictions_csv = params.get("validation_predictions_csv")
    test_predictions_csv = params.get("test_predictions_csv")
    train_predictions_csv = params.get("train_predictions_csv")
    if train_predictions_csv and not _configured_path_exists(train_predictions_csv, base_dir):
        train_predictions_csv = None
    mapping_csv = params.get("mapping_csv") or (out_dir / "ipromp_id_mapping.csv")

    prediction_files_ready = (
        bool(predictions_csv)
        and _configured_path_exists(predictions_csv, base_dir)
    ) or (
        bool(validation_predictions_csv)
        and bool(test_predictions_csv)
        and _configured_path_exists(validation_predictions_csv, base_dir)
        and _configured_path_exists(test_predictions_csv, base_dir)
    )
    if prediction_files_ready:
        try:
            from seqtrainer.adapters.ipromp import normalize_ipromp_predictions, prepare_ipromp_inputs

            if not _configured_path_exists(mapping_csv, base_dir):
                prepared = prepare_ipromp_inputs(config, base_dir=base_dir, output_dir=out_dir)
                mapping_csv = prepared.mapping_csv
            predictions = normalize_ipromp_predictions(
                config,
                mapping_csv=mapping_csv,
                validation_predictions_csv=validation_predictions_csv,
                test_predictions_csv=test_predictions_csv,
                train_predictions_csv=train_predictions_csv,
                predictions_csv=predictions_csv,
                base_dir=base_dir,
            )
            return _evaluate_external_prediction_frame(
                config,
                predictions=predictions,
                base_dir=base_dir,
                output_dir=out_dir,
                model_metadata={
                    "adapter_mode": params.get("adapter_mode", "external_fasta_prediction"),
                    "species_id": params.get("species_id"),
                    "species_name": params.get("species_name"),
                    "mapping_csv": str(mapping_csv),
                    "validation_predictions_csv": str(validation_predictions_csv) if validation_predictions_csv else None,
                    "test_predictions_csv": str(test_predictions_csv) if test_predictions_csv else None,
                    "train_predictions_csv": str(train_predictions_csv) if train_predictions_csv else None,
                    "predictions_csv": str(predictions_csv) if predictions_csv else None,
                },
            )
        except Exception:
            if not allow_skip:
                raise
            raise

    try:
        from seqtrainer.adapters.ipromp import prepare_ipromp_inputs

        frames = load_predefined_split_frames(config, base_dir=base_dir)
        split_summary = summarize_split_frames(config, frames)
        imbalance_policy = decide_imbalance_policy(split_summary)
        prepared = prepare_ipromp_inputs(config, base_dir=base_dir, output_dir=out_dir)
        reason = (
            "FASTA prepared; run official iPro-MP externally and provide validation/test prediction CSVs."
        )
        manifest_extra = {
            "status": "skipped",
            "skip_reason": reason,
            "fasta_paths": {split: str(path) for split, path in prepared.fasta_paths.items()},
            "mapping_csv": str(prepared.mapping_csv),
            "command_script": str(prepared.command_script),
            "external_prediction_schema": str(prepared.prediction_schema),
            "imbalance_policy": {
                "apply_to_training": imbalance_policy.apply_to_training,
                "strategy": imbalance_policy.strategy,
                "class_counts": imbalance_policy.class_counts,
                "imbalance_ratio": imbalance_policy.imbalance_ratio,
                "reason": imbalance_policy.reason,
            },
        }
        return _write_skipped_result(
            config,
            base_dir=base_dir,
            output_dir=out_dir,
            reason=reason,
            extra=manifest_extra,
        )
    except Exception as exc:
        if not allow_skip:
            raise
        return _write_skipped_result(config, base_dir=base_dir, output_dir=out_dir, reason=str(exc))


def _evaluate_external_predictions(
    config: BenchmarkConfig,
    *,
    predictions_csv: Path,
    base_dir: str | Path | None,
    output_dir: Path,
    model_metadata: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    pred_path = predictions_csv if predictions_csv.is_absolute() else Path(base_dir or Path.cwd()) / predictions_csv
    if not pred_path.exists():
        raise FileNotFoundError(f"External prediction file not found: {pred_path}")

    frames = load_predefined_split_frames(config, base_dir=base_dir)
    split_summary = summarize_split_frames(config, frames)
    imbalance_policy = decide_imbalance_policy(split_summary)
    predictions = pd.read_csv(pred_path, sep=None, engine="python")
    required = {"split", "label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"External predictions are missing required columns: {sorted(missing)}")
    predictions = predictions.copy()
    metrics: dict[str, dict[str, Any]] = {}
    score_column = _prediction_score_column(predictions)
    threshold: float | None
    if score_column is not None:
        if score_column != "probability":
            predictions = predictions.rename(columns={score_column: "probability"})

        threshold = _select_threshold(config, predictions)
        predictions["threshold"] = float(threshold)
        predictions["prediction"] = (predictions["probability"].astype(float) >= threshold).astype(int)

        for split in ("train", "validation", "test"):
            split_predictions = predictions[predictions["split"] == split]
            if split_predictions.empty:
                continue
            metrics[split] = binary_classification_metrics(
                split_predictions["label"].to_numpy(),
                split_predictions["probability"].to_numpy(),
                threshold=threshold,
            )
    else:
        prediction_column = _prediction_label_column(predictions)
        if prediction_column is None:
            raise ValueError(
                "External predictions require either a probability/score column or a hard-label column such as "
                "`prediction`, `predicted_label`, `pred`, or `label_pred`."
            )
        threshold = None
        predictions["prediction"] = predictions[prediction_column].astype(int)
        predictions["threshold"] = pd.Series([None] * len(predictions), dtype="object")
        warning = (
            "Only hard labels were provided by the external model, so AUROC/AUPRC and validation threshold "
            "selection could not be computed."
        )
        for split in ("train", "validation", "test"):
            split_predictions = predictions[predictions["split"] == split]
            if split_predictions.empty:
                continue
            metrics[split] = binary_classification_metrics_from_predictions(
                split_predictions["label"].to_numpy(),
                split_predictions["prediction"].to_numpy(),
                threshold=None,
                warning=warning,
            )

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=threshold,
        model_metadata=model_metadata or {},
        extra={
            "status": "completed",
            "external_predictions_csv": str(pred_path),
            "imbalance_policy": {
                "apply_to_training": imbalance_policy.apply_to_training,
                "strategy": imbalance_policy.strategy,
                "class_counts": imbalance_policy.class_counts,
                "imbalance_ratio": imbalance_policy.imbalance_ratio,
                "reason": imbalance_policy.reason,
            },
        },
    )
    write_benchmark_outputs(output_dir, manifest=manifest, metrics=metrics, predictions=predictions, config=config)
    return BenchmarkRunResult(output_dir=output_dir, status="completed", metrics=metrics, manifest=manifest)


def _evaluate_external_prediction_frame(
    config: BenchmarkConfig,
    *,
    predictions: pd.DataFrame,
    base_dir: str | Path | None,
    output_dir: Path,
    model_metadata: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    frames = load_predefined_split_frames(config, base_dir=base_dir)
    split_summary = summarize_split_frames(config, frames)
    imbalance_policy = decide_imbalance_policy(split_summary)
    predictions = predictions.copy()
    metrics: dict[str, dict[str, Any]] = {}
    if "probability" in predictions.columns:
        threshold = _select_threshold(config, predictions)
        predictions["threshold"] = float(threshold)
        predictions["prediction"] = (predictions["probability"].astype(float) >= threshold).astype(int)
        for split in ("train", "validation", "test"):
            split_predictions = predictions[predictions["split"] == split]
            if split_predictions.empty:
                continue
            metrics[split] = binary_classification_metrics(
                split_predictions["label"].to_numpy(),
                split_predictions["probability"].to_numpy(),
                threshold=threshold,
            )
    else:
        prediction_column = _prediction_label_column(predictions)
        if prediction_column is None:
            raise ValueError("iPro-MP predictions require either probability or hard-label prediction columns.")
        threshold = None
        predictions["prediction"] = predictions[prediction_column].astype(int)
        predictions["threshold"] = pd.Series([None] * len(predictions), dtype="object")
        warning = (
            "Only hard labels were provided by the external iPro-MP model, so AUROC/AUPRC and validation "
            "threshold selection could not be computed."
        )
        for split in ("train", "validation", "test"):
            split_predictions = predictions[predictions["split"] == split]
            if split_predictions.empty:
                continue
            metrics[split] = binary_classification_metrics_from_predictions(
                split_predictions["label"].to_numpy(),
                split_predictions["prediction"].to_numpy(),
                threshold=None,
                warning=warning,
            )

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=threshold,
        model_metadata=model_metadata or {},
        extra={
            "status": "completed",
            "external_prediction_mode": "official_ipromp_or_seqtrainer_normalized",
            "imbalance_policy": {
                "apply_to_training": imbalance_policy.apply_to_training,
                "strategy": imbalance_policy.strategy,
                "class_counts": imbalance_policy.class_counts,
                "imbalance_ratio": imbalance_policy.imbalance_ratio,
                "reason": imbalance_policy.reason,
            },
        },
    )
    write_benchmark_outputs(output_dir, manifest=manifest, metrics=metrics, predictions=predictions, config=config)
    return BenchmarkRunResult(output_dir=output_dir, status="completed", metrics=metrics, manifest=manifest)


def _prediction_score_column(predictions: pd.DataFrame) -> str | None:
    for column in ("probability", "score", "positive_score", "promoter_score"):
        if column in predictions.columns:
            return column
    return None


def _prediction_label_column(predictions: pd.DataFrame) -> str | None:
    for column in ("prediction", "predicted_label", "pred", "label_pred"):
        if column in predictions.columns:
            return column
    return None


def _write_skipped_result(
    config: BenchmarkConfig,
    *,
    base_dir: str | Path | None,
    output_dir: str | Path | None,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> BenchmarkRunResult:
    out_dir = Path(output_dir or config.outputs.output_dir)
    split_summary: dict[str, Any] = {}
    try:
        frames = load_predefined_split_frames(config, base_dir=base_dir)
        split_summary = summarize_split_frames(config, frames)
    except Exception as exc:  # keep dependency skips informative even without data files
        split_summary = {"warning": f"Could not load configured splits: {exc}"}

    manifest = build_run_manifest(
        config,
        split_summary=split_summary,
        threshold=None,
        model_metadata={"status": "skipped"},
        extra=extra
        or {
            "status": "skipped",
            "skip_reason": reason,
            "imbalance_policy": _imbalance_policy_payload(split_summary),
        },
    )
    write_benchmark_outputs(out_dir, manifest=manifest, config=config)
    return BenchmarkRunResult(output_dir=out_dir, status="skipped", metrics={}, manifest=manifest)


def _split_paths(config: BenchmarkConfig, base_dir: str | Path | None) -> dict[str, Path]:
    return resolve_split_paths(config, base_dir=base_dir)


def _select_threshold(config: BenchmarkConfig, predictions: pd.DataFrame) -> float:
    if config.evaluation.threshold_strategy == "fixed_0_5":
        return 0.5

    metric = threshold_metric_from_strategy(config.evaluation.threshold_strategy)
    if metric is None:
        return 0.5
    validation = predictions[predictions["split"] == "validation"]
    if validation.empty:
        raise ValueError("Validation predictions are required for validation-threshold selection")
    threshold, _ = best_threshold_by_metric(
        validation["label"].to_numpy(),
        validation["probability"].to_numpy(),
        metric=metric,
    )
    return float(threshold)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _looks_like_resource_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(fragment in message for fragment in ("out of memory", "cuda", "cudnn", "mps"))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _default_if_none(value: Any, default: Any) -> Any:
    return default if value is None else value


def _configured_path_exists(path: Any, base_dir: str | Path | None) -> bool:
    resolved = Path(str(path))
    if not resolved.is_absolute():
        resolved = Path(base_dir or Path.cwd()) / resolved
    return resolved.exists()


def _imbalance_policy_payload(split_summary: dict[str, Any]) -> dict[str, Any]:
    policy = decide_imbalance_policy(split_summary)
    return {
        "apply_to_training": policy.apply_to_training,
        "strategy": policy.strategy,
        "class_counts": policy.class_counts,
        "imbalance_ratio": policy.imbalance_ratio,
        "reason": policy.reason,
    }
