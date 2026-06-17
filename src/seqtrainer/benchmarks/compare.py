"""Compare completed benchmark artifact folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def compare_benchmark_outputs(
    artifact_dirs: Iterable[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Rank benchmark artifact folders by held-out test MCC and AUPRC."""
    rows: list[dict[str, Any]] = []
    for artifact_dir_raw in artifact_dirs:
        artifact_dir = Path(artifact_dir_raw)
        metrics_path = artifact_dir / "metrics.csv"
        manifest_path = artifact_dir / "manifest.json"
        if not metrics_path.exists():
            continue

        metrics = pd.read_csv(metrics_path)
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        model = manifest.get("model", {})
        experiment = manifest.get("experiment", {})
        threshold = manifest.get("evaluation", {}).get("selected_threshold")

        for _, metric_row in metrics.iterrows():
            row = {
                "artifact_dir": str(artifact_dir),
                "experiment": experiment.get("name", artifact_dir.name),
                "model_family": model.get("family"),
                "model_name": model.get("name"),
                "split": metric_row.get("split"),
                "selected_threshold": threshold,
            }
            row.update(metric_row.to_dict())
            rows.append(row)

    if not rows:
        raise ValueError("No benchmark metrics.csv files were found in the provided artifact directories")

    comparison = pd.DataFrame(rows).sort_values(
        by=["split", "mcc", "auprc"],
        ascending=[True, False, False],
        na_position="last",
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = out_dir / "comparison_metrics.csv"
    summary_out = out_dir / "comparison_summary.md"
    comparison.to_csv(metrics_out, index=False)
    summary_out.write_text(_summary_markdown(comparison), encoding="utf-8")
    return {"comparison_metrics": metrics_out, "comparison_summary": summary_out}


def _summary_markdown(comparison: pd.DataFrame) -> str:
    test_rows = comparison[comparison["split"] == "test"].copy()
    if not test_rows.empty:
        test_rows = test_rows.sort_values(["mcc", "auprc"], ascending=[False, False], na_position="last")

    lines = [
        "# Benchmark Comparison Summary",
        "",
        "Models are ranked using held-out test MCC first and test AUPRC second.",
        "Thresholds must be selected on the validation split only; the test split is final reporting only.",
        "",
        "## Test Ranking",
        "",
    ]
    if test_rows.empty:
        lines.append("No test split rows were found in the supplied benchmark metrics.")
    else:
        lines.append("| Rank | Experiment | Model | Test MCC | Test AUPRC | Test Accuracy | Threshold |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for rank, (_, row) in enumerate(test_rows.iterrows(), start=1):
            lines.append(
                "| {rank} | {experiment} | {model} | {mcc} | {auprc} | {accuracy} | {threshold} |".format(
                    rank=rank,
                    experiment=_display(row.get("experiment")),
                    model=_display(row.get("model_name")),
                    mcc=_format_float(row.get("mcc")),
                    auprc=_format_float(row.get("auprc")),
                    accuracy=_format_float(row.get("accuracy")),
                    threshold=_format_float(row.get("selected_threshold")),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Prefer MCC and AUPRC over plain accuracy for promoter prediction comparisons.",
            "- A higher test score is only meaningful when all models used the same split files.",
            "- If a model run was skipped, rerun it after installing dependencies or providing external predictions.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _display(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
