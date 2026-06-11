"""Artifact writers shared by benchmark runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import to_plain_data


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact with stable formatting."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_plain_data(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def write_metrics_csv(metrics_by_split: dict[str, dict[str, Any]], path: str | Path) -> Path:
    """Write split-wise metrics to a flat CSV table."""
    rows = []
    for split, metrics in metrics_by_split.items():
        row = {"split": split}
        for key, value in metrics.items():
            if key == "confusion_matrix" and isinstance(value, dict):
                row.update({str(cm_key): cm_value for cm_key, cm_value in value.items()})
            else:
                row[key] = value
        rows.append(row)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def write_table_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a tabular benchmark artifact."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path


def write_benchmark_outputs(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    metrics: dict[str, dict[str, Any]] | None = None,
    predictions: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    config: Any | None = None,
) -> dict[str, Path]:
    """Write the common benchmark artifact set and return written paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {"manifest": write_json(out_dir / "manifest.json", manifest)}
    if config is not None:
        written["config"] = write_json(out_dir / "config.json", config)
    if metrics is not None:
        written["metrics_json"] = write_json(out_dir / "metrics.json", metrics)
        written["metrics_csv"] = write_metrics_csv(metrics, out_dir / "metrics.csv")
    if predictions is not None:
        written["predictions"] = write_table_csv(predictions, out_dir / "predictions.csv")
    if history is not None:
        written["history"] = write_table_csv(history, out_dir / "history.csv")
    return written

