"""Command line interface for common SeqTrainer workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from seqtrainer.data.sbol import build_dataset_from_files, get_sequence_from_sbol
from seqtrainer.sparql.prefixes import format_prefixes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seqtrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_sbol = subparsers.add_parser("inspect-sbol", help="Inspect one SBOL file")
    inspect_sbol.add_argument("file", type=Path)

    build_dataset = subparsers.add_parser("build-dataset", help="Build simple sequence/target dataset")
    build_dataset.add_argument("files", nargs="+", type=Path)
    build_dataset.add_argument("--y-uri", default="http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue")

    cnn_baseline = subparsers.add_parser("reproduce-cnn-baseline", help="Reproduce the tutorial CNN baseline")
    cnn_baseline.add_argument("--data-dir", type=Path, default=Path("data/sbol_data"))
    cnn_baseline.add_argument("--output-dir", type=Path, default=Path("outputs/cnn_baseline_reference"))
    cnn_baseline.add_argument("--max-files", type=int, default=40)
    cnn_baseline.add_argument("--sequence-length", type=int, default=120)
    cnn_baseline.add_argument("--seed", type=int, default=42)
    cnn_baseline.add_argument("--batch-size", type=int, default=16)
    cnn_baseline.add_argument("--cycles", type=int, default=10)
    cnn_baseline.add_argument("--learning-rate", type=float, default=1e-3)
    cnn_baseline.add_argument("--device", default="cpu")

    cnn_csv = subparsers.add_parser("run-cnn-benchmark", help="Train CNN on predefined CSV split files")
    cnn_csv.add_argument("--config", type=Path, default=Path("config-examples/benchmarks/cnn.toml"))
    cnn_csv.add_argument("--train-csv", type=Path)
    cnn_csv.add_argument("--validation-csv", "--validate-csv", "--eval-csv", dest="validation_csv", type=Path)
    cnn_csv.add_argument("--test-csv", type=Path)
    cnn_csv.add_argument("--output-dir", type=Path)
    cnn_csv.add_argument("--sequence-length", type=int)
    cnn_csv.add_argument("--seed", type=int)
    cnn_csv.add_argument("--batch-size", type=int)
    cnn_csv.add_argument("--cycles", type=int)
    cnn_csv.add_argument("--learning-rate", type=float)
    cnn_csv.add_argument("--device")

    benchmark_manifest = subparsers.add_parser(
        "benchmark-manifest",
        help="Validate a benchmark config and write shared manifest artifacts",
    )
    benchmark_manifest.add_argument("--config", type=Path, required=True)
    benchmark_manifest.add_argument("--output-dir", type=Path)
    benchmark_manifest.add_argument("--base-dir", type=Path, default=Path.cwd())

    benchmark = subparsers.add_parser("benchmark", help="Benchmark harness commands")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_sub.add_parser("run", help="Run a configured benchmark")
    benchmark_run.add_argument("config", type=Path)
    benchmark_run.add_argument("--output-dir", type=Path)
    benchmark_run.add_argument("--base-dir", type=Path, default=Path.cwd())
    benchmark_run.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of writing a skipped manifest for unsupported benchmark configs",
    )
    benchmark_manifest_nested = benchmark_sub.add_parser("manifest", help="Validate a config and write manifest artifacts")
    benchmark_manifest_nested.add_argument("config", type=Path)
    benchmark_manifest_nested.add_argument("--output-dir", type=Path)
    benchmark_manifest_nested.add_argument("--base-dir", type=Path, default=Path.cwd())
    benchmark_compare = benchmark_sub.add_parser("compare", help="Compare completed benchmark artifact folders")
    benchmark_compare.add_argument("artifact_dirs", nargs="+", type=Path)
    benchmark_compare.add_argument("--output-dir", type=Path, required=True)

    sparql = subparsers.add_parser("sparql", help="SPARQL helpers")
    sparql_sub = sparql.add_subparsers(dest="sparql_command", required=True)
    sparql_sub.add_parser("prefixes", help="Print default prefixes")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-sbol":
        seq = get_sequence_from_sbol(args.file)
        print(f"sequence_length={len(seq) if seq else 0}")
        return 0

    if args.command == "build-dataset":
        frame = build_dataset_from_files(args.files, args.y_uri)
        print(frame.to_csv(index=False))
        return 0

    if args.command == "reproduce-cnn-baseline":
        from seqtrainer.torch.cnn_baseline import CnnBaselineConfig, run_cnn_baseline

        result = run_cnn_baseline(
            CnnBaselineConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                max_files=args.max_files,
                sequence_length=args.sequence_length,
                seed=args.seed,
                batch_size=args.batch_size,
                cycles=args.cycles,
                learning_rate=args.learning_rate,
                device=args.device,
            )
        )
        print(f"output_dir={result.output_dir}")
        for split, metrics in result.metrics.items():
            print(
                f"{split}: "
                f"accuracy={metrics['accuracy']:.3f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                f"mcc={metrics['mcc']:.3f}"
            )
        return 0

    if args.command == "run-cnn-benchmark":
        from seqtrainer.benchmarks import load_benchmark_config
        from seqtrainer.torch.cnn_baseline import CnnCsvSplitConfig, run_cnn_csv_splits

        benchmark = load_benchmark_config(args.config)
        split_files = benchmark.dataset.split_files
        training_params = dict(benchmark.training.params)
        model_params = dict(benchmark.model.params)
        result = run_cnn_csv_splits(
            CnnCsvSplitConfig(
                train_csv=args.train_csv or Path(split_files["train"]),
                validation_csv=args.validation_csv or Path(split_files["validation"]),
                test_csv=args.test_csv or Path(split_files["test"]),
                output_dir=args.output_dir or Path(benchmark.outputs.output_dir),
                dataset_name=benchmark.dataset.name,
                source_accession=benchmark.dataset.source_accession,
                source_url=benchmark.dataset.source_url,
                sequence_field=benchmark.dataset.sequence_field,
                label_field=benchmark.dataset.label_field,
                positive_label=benchmark.label.positive_label,
                negative_label=benchmark.label.negative_label,
                sequence_length=args.sequence_length or benchmark.preprocessing.sequence_length or 300,
                seed=args.seed or benchmark.experiment.seed,
                batch_size=args.batch_size or benchmark.training.batch_size or 16,
                cycles=args.cycles or benchmark.training.max_epochs or 10,
                learning_rate=args.learning_rate or benchmark.training.learning_rate or 1e-3,
                weight_decay=float(training_params.get("weight_decay", 0.0)),
                optimizer_name=str(training_params.get("optimizer", "adam")).lower(),
                scheduler_name=str(training_params.get("scheduler", "none")).lower(),
                select_best_by_mcc=bool(training_params.get("select_best_by_mcc", False)),
                early_stopping_patience=_optional_int(training_params.get("early_stopping_patience")),
                model_variant=str(model_params.get("variant", "tiny")),
                dropout=float(model_params.get("dropout", 0.25)),
                class_weighting=bool(training_params.get("class_weighting", False)),
                threshold_strategy=benchmark.evaluation.threshold_strategy,
                device=args.device or _resolve_device(benchmark.environment.device),
                save_json=benchmark.outputs.save_json,
                save_csv=benchmark.outputs.save_csv,
                save_predictions=benchmark.outputs.save_predictions,
            )
        )
        print(f"output_dir={result.output_dir}")
        print(f"threshold={result.manifest['threshold_selection']['threshold']:.3f}")
        for split, metrics in result.metrics.items():
            print(
                f"{split}: "
                f"accuracy={metrics['accuracy']:.3f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                f"mcc={metrics['mcc']:.3f}"
        )
        return 0

    if args.command == "benchmark":
        if args.benchmark_command == "run":
            from seqtrainer.benchmarks import run_benchmark

            result = run_benchmark(
                args.config,
                base_dir=args.base_dir,
                output_dir=args.output_dir,
                allow_skip=not args.strict,
            )
            print(f"status={result.status}")
            print(f"output_dir={result.output_dir}")
            if result.metrics:
                for split, metrics in result.metrics.items():
                    print(
                        f"{split}: "
                        f"accuracy={metrics['accuracy']:.3f} "
                        f"balanced_accuracy={metrics['balanced_accuracy']:.3f} "
                        f"mcc={metrics['mcc']:.3f}"
                    )
            else:
                reason = result.manifest.get("extra", {}).get("skip_reason", "not run")
                print(f"skip_reason={reason}")
            return 0

        if args.benchmark_command == "manifest":
            return _write_benchmark_manifest(args.config, args.output_dir, args.base_dir)

        if args.benchmark_command == "compare":
            from seqtrainer.benchmarks import compare_benchmark_outputs

            written = compare_benchmark_outputs(args.artifact_dirs, output_dir=args.output_dir)
            print(f"comparison_metrics={written['comparison_metrics']}")
            print(f"comparison_summary={written['comparison_summary']}")
            return 0

    if args.command == "benchmark-manifest":
        return _write_benchmark_manifest(args.config, args.output_dir, args.base_dir)

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    parser.error("Unhandled command")
    return 2


def _write_benchmark_manifest(config_path: Path, output_dir_arg: Path | None, base_dir: Path) -> int:
    from seqtrainer.benchmarks import (
        build_run_manifest,
        load_benchmark_config,
        load_predefined_split_frames,
        summarize_split_frames,
        write_benchmark_outputs,
    )

    benchmark = load_benchmark_config(config_path)
    frames = load_predefined_split_frames(benchmark, base_dir=base_dir)
    split_summary = summarize_split_frames(benchmark, frames)
    manifest = build_run_manifest(benchmark, split_summary=split_summary)
    output_dir = output_dir_arg or Path(benchmark.outputs.output_dir)
    written = write_benchmark_outputs(output_dir, manifest=manifest, config=benchmark)

    print(f"output_dir={output_dir}")
    print(f"manifest={written['manifest']}")
    for split, summary in split_summary.items():
        print(f"{split}: rows={summary['rows']} class_counts={summary['class_counts']}")
    return 0


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
