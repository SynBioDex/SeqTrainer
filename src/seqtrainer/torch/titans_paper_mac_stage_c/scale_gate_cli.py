"""Compile the frozen Stage C v3 E25/E100 scale decision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Mapping, Sequence


def _read(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _evaluation(path: Path, name: str | None) -> Mapping[str, object]:
    payload = _read(path)
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return payload
    if name:
        value = results.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"evaluation has no result named {name!r}")
        return value
    values = [value for value in results.values() if isinstance(value, Mapping)]
    if len(values) != 1:
        raise ValueError("use --baseline-name/--candidate-name for multi-run evaluations")
    return values[0]


def _paired_bootstrap(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, object]:
    accessions = sorted(set(candidate) & set(baseline))
    if len(accessions) < 2:
        raise ValueError("paired gate requires at least two shared held-out accessions")
    differences = [float(candidate[key]) - float(baseline[key]) for key in accessions]
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(samples)
    )
    return {
        "accessions": len(accessions),
        "mean_candidate_minus_baseline_bpb": statistics.fmean(differences),
        "lower_95": draws[int(0.025 * (samples - 1))],
        "upper_95": draws[int(0.975 * (samples - 1))],
        "accessions_improved": sum(value < 0 for value in differences),
        "fraction_improved": sum(value < 0 for value in differences) / len(differences),
    }


def _group(report: Mapping[str, object], name: str) -> Mapping[str, float]:
    summary = report.get("distribution_summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get(name), Mapping):
        raise ValueError(f"generation report has no distribution group {name!r}")
    return summary[name]  # type: ignore[return-value]


def _generation_comparison(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    group: str,
) -> dict[str, object]:
    baseline_reference = _group(baseline, "reference")
    candidate_reference = _group(candidate, "reference")
    baseline_group = _group(baseline, group)
    candidate_group = _group(candidate, group)

    def error(sample: Mapping[str, float], reference: Mapping[str, float], key: str) -> float:
        return abs(float(sample[key]) - float(reference[key]))

    baseline_jsd = float(
        baseline["kmer_jsd_to_heldout_reference"][group]["6"]  # type: ignore[index]
    )
    candidate_jsd = float(
        candidate["kmer_jsd_to_heldout_reference"][group]["6"]  # type: ignore[index]
    )
    aligned_ratio = float(candidate_group["aligned_unique_6mer_fraction"]) / max(
        float(candidate_reference["aligned_unique_6mer_fraction"]), 1e-12
    )
    families = {
        "six_mer_jsd": candidate_jsd < baseline_jsd,
        "gc_fraction": error(candidate_group, candidate_reference, "gc_fraction")
        < error(baseline_group, baseline_reference, "gc_fraction"),
        "base_entropy": error(candidate_group, candidate_reference, "base_entropy_bits")
        < error(baseline_group, baseline_reference, "base_entropy_bits"),
        "orf_count": error(
            candidate_group, candidate_reference, "heuristic_orfs_at_least_90bp"
        )
        < error(baseline_group, baseline_reference, "heuristic_orfs_at_least_90bp"),
        "longest_orf": error(
            candidate_group, candidate_reference, "heuristic_longest_orf_bases"
        )
        < error(baseline_group, baseline_reference, "heuristic_longest_orf_bases"),
    }
    return {
        "group": group,
        "baseline_six_mer_jsd": baseline_jsd,
        "candidate_six_mer_jsd": candidate_jsd,
        "six_mer_jsd_relative_improvement": (
            baseline_jsd - candidate_jsd
        ) / max(baseline_jsd, 1e-12),
        "candidate_aligned_six_mer_diversity_ratio": aligned_ratio,
        "candidate_gc_absolute_error": error(
            candidate_group, candidate_reference, "gc_fraction"
        ),
        "baseline_gc_absolute_error": error(
            baseline_group, baseline_reference, "gc_fraction"
        ),
        "metric_families": families,
        "families_improved": sum(families.values()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("e25", "e100"), required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-name")
    parser.add_argument("--candidate-name")
    parser.add_argument("--memory-behavior", type=Path, required=True)
    parser.add_argument("--baseline-generation", type=Path, required=True)
    parser.add_argument("--candidate-generation", type=Path, required=True)
    parser.add_argument("--generation-group", default="temperature_0.6")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260751)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = _evaluation(args.baseline_evaluation, args.baseline_name)
    candidate = _evaluation(args.candidate_evaluation, args.candidate_name)
    baseline_by_accession = baseline["per_accession_bpb"]
    candidate_by_accession = candidate["per_accession_bpb"]
    if not isinstance(baseline_by_accession, Mapping) or not isinstance(
        candidate_by_accession, Mapping
    ):
        raise ValueError("evaluation is missing per_accession_bpb")
    paired = _paired_bootstrap(
        candidate_by_accession, baseline_by_accession, seed=args.seed  # type: ignore[arg-type]
    )
    median_bpb = statistics.median(map(float, candidate_by_accession.values()))
    mean_improvement = float(baseline["bits_per_base"]) - float(candidate["bits_per_base"])
    behavior = _read(args.memory_behavior)
    blocks = behavior.get("blocks")
    if not isinstance(blocks, list):
        raise ValueError("memory behavior report is missing blocks")
    immediate = sum(
        float(row["immediate_relative_improvement"]) > 0
        for row in blocks
        if isinstance(row, Mapping)
    )
    delayed = sum(
        float(row["delayed_association_margin"]) > 0
        for row in blocks
        if isinstance(row, Mapping)
    )
    gradient = candidate.get("memory_gradient_statistics", {})
    intervention = (
        float(gradient.get("gradient_intervention_fraction", math.inf))
        if isinstance(gradient, Mapping)
        else math.inf
    )
    generation = _generation_comparison(
        _read(args.baseline_generation),
        _read(args.candidate_generation),
        group=args.generation_group,
    )
    if args.stage == "e25":
        prediction_pass = (
            median_bpb <= 1.96
            and mean_improvement >= 0.02
            and float(paired["upper_95"]) < 0
            and float(paired["fraction_improved"]) >= 0.75
        )
        generation_families = 3
    else:
        prediction_pass = (
            median_bpb <= 1.92
            and int(paired["accessions_improved"]) >= 10
        )
        generation_families = 4
    memory_pass = immediate >= 10 and delayed >= 10 and intervention == 0.0
    generation_pass = (
        0.95 <= float(generation["candidate_aligned_six_mer_diversity_ratio"]) <= 1.05
        and float(generation["six_mer_jsd_relative_improvement"]) >= 0.10
        and float(generation["candidate_gc_absolute_error"])
        < float(generation["baseline_gc_absolute_error"])
        and int(generation["families_improved"]) >= generation_families
    )
    report = {
        "format_version": 1,
        "classification": "frozen_stage_c_v3_scale_gate",
        "stage": args.stage,
        "prediction": {
            "candidate_median_accession_bpb": median_bpb,
            "candidate_mean_bpb": candidate["bits_per_base"],
            "baseline_mean_bpb": baseline["bits_per_base"],
            "mean_improvement_bpb": mean_improvement,
            "paired_bootstrap": paired,
            "passed": prediction_pass,
        },
        "memory": {
            "blocks_improving_immediately": immediate,
            "blocks_with_positive_delayed_margin": delayed,
            "gradient_intervention_fraction": intervention,
            "passed": memory_pass,
        },
        "generation": {**generation, "passed": generation_pass},
        "proceed": prediction_pass and memory_pass and generation_pass,
        "claim_limit": (
            "This adaptive-only exploratory gate supports scale allocation, not an "
            "adaptive-memory benefit or biological-function claim."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scale_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Stage C v3 {args.stage.upper()} scale gate",
        "",
        f"Decision: `{'PROCEED' if report['proceed'] else 'STOP'}`",
        "",
        f"- Prediction gate: `{prediction_pass}`",
        f"- Memory gate: `{memory_pass}`",
        f"- Generation gate: `{generation_pass}`",
        f"- Median accession BPB: `{median_bpb:.6f}`",
        f"- Mean BPB improvement: `{mean_improvement:.6f}`",
        f"- Paired bootstrap upper 95% bound: `{float(paired['upper_95']):.6f}`",
        f"- Held-out accessions improved: `{float(paired['fraction_improved']):.1%}`",
        f"- Memory blocks immediate/delayed: `{immediate}/{delayed}`",
        f"- Six-mer JSD relative improvement: "
        f"`{float(generation['six_mer_jsd_relative_improvement']):.1%}`",
        f"- Generation metric families improved: "
        f"`{generation['families_improved']}`",
        "",
        str(report["claim_limit"]),
    ]
    (args.output_dir / "SCALE_GATE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
