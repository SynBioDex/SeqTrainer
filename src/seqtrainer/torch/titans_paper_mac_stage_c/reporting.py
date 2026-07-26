"""Deterministic, dependency-light Stage C evidence reports and plots."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .baselines import BaselineResult
from .tokenizers import TokenizerMetrics
from .trainer import TrainingStepRecord


def bar_svg(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    x_label: str,
) -> str:
    width, height = 900, max(260, 90 + 52 * len(labels))
    left, right, top = 250, 40, 55
    plot_width = width - left - right
    maximum = max(values, default=1.0) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * 52
        bar_width = value / maximum * plot_width
        color = "#2563eb" if index % 2 == 0 else "#0f766e"
        lines.extend(
            [
                f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" font-family="sans-serif" font-size="13">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="28" fill="{color}" rx="3"/>',
                f'<text x="{left + bar_width + 7:.2f}" y="{y + 20}" font-family="sans-serif" font-size="12">{value:.4f}</text>',
            ]
        )
    lines.append(
        f'<text x="{left + plot_width / 2}" y="{height - 14}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def line_svg(
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    y_label: str,
) -> str:
    width, height = 900, 430
    left, right, top, bottom = 80, 35, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for values in series.values() for value in values]
    minimum = min(all_values, default=0.0)
    maximum = max(all_values, default=1.0)
    if maximum == minimum:
        maximum = minimum + 1.0
    maximum_points = max((len(values) for values in series.values()), default=1)
    colors = ("#2563eb", "#0f766e", "#c2410c", "#7c3aed", "#be123c")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334155"/>',
    ]
    for tick in range(5):
        value = minimum + (maximum - minimum) * tick / 4
        y = top + plot_height - plot_height * tick / 4
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0"/>',
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.4g}</text>',
            ]
        )
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = []
        for point, value in enumerate(values):
            x = left + (point / max(maximum_points - 1, 1)) * plot_width
            y = top + (maximum - value) / (maximum - minimum) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        if points:
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        legend_x = left + index * 165
        lines.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 25}" x2="{legend_x + 22}" y2="{height - 25}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 28}" y="{height - 20}" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 44}" text-anchor="middle" font-family="sans-serif" font-size="13">Optimizer step</text>',
            f'<text transform="translate(18 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(y_label)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def write_tokenizer_report(
    metrics: Sequence[TokenizerMetrics],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "tokenizer_metrics.json"
    csv_path = output / "tokenizer_metrics.csv"
    plot_path = output / "tokenizer_bases_per_token.svg"
    report_path = output / "TOKENIZER_REPORT.md"
    rows = [item.to_dict() for item in metrics]
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_path.write_text(
        bar_svg(
            [f"{item.family}: {item.name}" for item in metrics],
            [item.bases_per_token for item in metrics],
            title="Stage C tokenizer compression",
            x_label="Bases per token (higher means wider effective base context)",
        ),
        encoding="utf-8",
    )
    gc_plot_path = output / "tokenizer_gc_compression.svg"
    gc_labels: list[str] = []
    gc_values: list[float] = []
    for item in metrics:
        for gc_bin, value in item.gc_bin_bases_per_token.items():
            gc_labels.append(f"{item.name}: {gc_bin.removeprefix('gc_')}")
            gc_values.append(value)
    gc_plot_path.write_text(
        bar_svg(
            gc_labels,
            gc_values,
            title="Tokenizer compression across whole-contig GC bins",
            x_label="Bases per token",
        ),
        encoding="utf-8",
    )
    lines = [
        "# Stage C tokenizer intrinsic assessment",
        "",
        "Eligibility here covers deterministic intrinsic behavior only; final selection also requires the matched CPU LM BPB study.",
        "",
        "| Tokenizer | Family | Bases/token | Vocabulary used | UNK rate | Round trip | Eligible | Verification |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in metrics:
        lines.append(
            f"| {item.name} | {item.family} | {item.bases_per_token:.4f} | "
            f"{item.vocabulary_used} | {item.unknown_rate:.6f} | "
            f"{'PASS' if item.round_trip_passed else 'FAIL'} | "
            f"{'yes' if item.eligible else 'no'} | {item.verification} |"
        )
    lines.extend(
        [
            "",
            "## Assessment",
            "",
            "- Green: round trip passes, no unknown tokens, and official/native verification is available.",
            "- Yellow: behavior is measurable but external tokenizer parity is not yet verified.",
            "- Red: sequence reconstruction or coverage fails.",
            "",
            "See `tokenizer_bases_per_token.svg` for the single- versus multi-nucleotide context comparison.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "plot": plot_path,
        "gc_plot": gc_plot_path,
        "report": report_path,
    }


def write_cpu_pilot_report(
    *,
    baselines: Sequence[BaselineResult],
    tokenizer_runs: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "baselines": [item.to_dict() for item in baselines],
        "tokenizer_runs": [dict(item) for item in tokenizer_runs],
        "tokenizer_selection": dict(selection),
    }
    json_path = output / "cpu_pilot.json"
    report_path = output / "CPU_PILOT_REPORT.md"
    plot_path = output / "cpu_pilot_bpb.svg"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    labels = [item.name for item in baselines] + [
        f"{item['tokenizer']} ({item['regime']})" for item in tokenizer_runs
    ]
    values = [item.bits_per_base for item in baselines] + [float(item["validation_bpb"]) for item in tokenizer_runs]
    plot_path.write_text(
        bar_svg(labels, values, title="CPU basal capability", x_label="Held-out bits per base (lower is better)"),
        encoding="utf-8",
    )
    lines = [
        "# Stage C CPU basal capability",
        "",
        "| Model/tokenizer | Regime | Validation BPB | Valid bases | Train bases | Parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in baselines:
        lines.append(
            f"| {item.name} | statistical | {item.bits_per_base:.4f} | {item.bases} | 0 | 0 |"
        )
    for item in tokenizer_runs:
        lines.append(
            f"| {item['tokenizer']} | {item['regime']} | {float(item['validation_bpb']):.4f} | "
            f"{int(item['validation_bases'])} | {int(item['train_bases'])} | "
            f"{int(item['parameter_count'])} |"
        )
    best_frequency = next((item.bits_per_base for item in baselines if item.name == "nucleotide_frequency"), float("inf"))
    neural_pass = any(float(item["validation_bpb"]) < best_frequency for item in tokenizer_runs)
    lines.extend(
        [
            "",
            "## Traffic-light assessment",
            "",
            f"- {'Green' if neural_pass else 'Red'}: at least one tiny paper-MAC run beats the nucleotide-frequency baseline.",
            "- This CPU gate validates pipeline learning and informativeness; it is not a capacity claim.",
            f"- Selected tokenizer: `{selection['selected_tokenizer']}` under the equal-base BPB rule.",
            f"- Selection threshold: {float(selection['minimum_bpb_improvement']):.4f} BPB over the SeqTrainer base tokenizer.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    selection_path = output / "tokenizer_selection.json"
    selection_path.write_text(
        json.dumps(dict(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "json": json_path,
        "plot": plot_path,
        "report": report_path,
        "selection": selection_path,
    }


def write_training_history(
    history: Iterable[TrainingStepRecord],
    output_dir: str | Path,
    *,
    write_plots: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = list(history)
    rows = [record.to_dict() for record in records]
    json_path = output / "training_history.json"
    csv_path = output / "training_history.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_specs = {
        "training_bpb.svg": ({"BPB": [item.bits_per_base for item in records]}, "Training BPB", "Bits per base"),
        "memory_diagnostics.svg": (
            {
                "retrieval": [item.retrieval_norm for item in records],
                "update": [item.memory_update_norm for item in records],
                "surprise": [item.surprise_norm for item in records],
                "state drift": [item.state_drift_norm for item in records],
            },
            "Memory retrieval and update norms",
            "Norm",
        ),
        "gate_diagnostics.svg": (
            {
                "alpha": [item.alpha_mean for item in records],
                "eta": [item.eta_mean for item in records],
                "theta": [item.theta_mean for item in records],
            },
            "Adaptive memory gate means",
            "Gate value",
        ),
        "gradient_diagnostics.svg": (
            {
                "parameters": [item.gradient_norm for item in records],
                "written state": [item.written_state_gradient_norm for item in records],
                "raw memory RMS max": [
                    item.raw_memory_gradient_rms_max for item in records
                ],
                "conditioned memory RMS max": [
                    item.conditioned_memory_gradient_rms_max for item in records
                ],
            },
            "Gradient health",
            "Gradient norm",
        ),
        "memory_conditioning.svg": (
            {
                "minimum gradient scale": [
                    item.memory_gradient_scale_min for item in records
                ],
                "gradient intervention fraction": [
                    item.memory_gradient_intervention_fraction for item in records
                ],
                "legacy cap intervention fraction": [
                    item.legacy_surprise_intervention_fraction for item in records
                ],
            },
            "Neural-memory conditioning",
            "Fraction / scale",
        ),
        "training_throughput.svg": (
            {"bases/s": [item.bases_per_second for item in records]},
            "Training throughput",
            "Bases per second",
        ),
    }
    paths = {"json": json_path, "csv": csv_path}
    if not write_plots:
        return paths
    for filename, (series, title, y_label) in plot_specs.items():
        path = output / filename
        path.write_text(line_svg(series, title=title, y_label=y_label), encoding="utf-8")
        paths[filename.removesuffix(".svg")] = path
    return paths
