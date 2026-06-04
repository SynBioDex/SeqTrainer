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
                sequence_length=args.sequence_length or benchmark.preprocessing.sequence_length or 300,
                seed=args.seed or benchmark.experiment.seed,
                batch_size=args.batch_size or benchmark.training.batch_size or 16,
                cycles=args.cycles or benchmark.training.max_epochs or 10,
                learning_rate=args.learning_rate or benchmark.training.learning_rate or 1e-3,
                device=args.device or "cpu",
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

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
