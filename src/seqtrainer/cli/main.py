"""Command line interface for common SeqTrainer workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TypeVar

from seqtrainer.data.sbol import build_dataset_from_files, get_sequence_from_sbol
from seqtrainer.sparql.prefixes import format_prefixes


_T = TypeVar("_T")


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
        help="Fail instead of writing a skipped manifest when optional model dependencies are unavailable",
    )
    benchmark_manifest_nested = benchmark_sub.add_parser("manifest", help="Validate a config and write manifest artifacts")
    benchmark_manifest_nested.add_argument("config", type=Path)
    benchmark_manifest_nested.add_argument("--output-dir", type=Path)
    benchmark_manifest_nested.add_argument("--base-dir", type=Path, default=Path.cwd())
    benchmark_compare = benchmark_sub.add_parser("compare", help="Compare completed benchmark artifact folders")
    benchmark_compare.add_argument("artifact_dirs", nargs="+", type=Path)
    benchmark_compare.add_argument("--output-dir", type=Path, required=True)
    benchmark_prepare_dnabert2 = benchmark_sub.add_parser(
        "prepare-dnabert2",
        help="Tokenize configured split CSVs for DNABERT2 smoke checks",
    )
    benchmark_prepare_dnabert2.add_argument("config", type=Path)
    benchmark_prepare_dnabert2.add_argument("--output-dir", type=Path)
    benchmark_prepare_dnabert2.add_argument("--base-dir", type=Path, default=Path.cwd())
    benchmark_prepare_ipromp = benchmark_sub.add_parser(
        "prepare-ipromp",
        help="Write FASTA, mapping, and command files for external iPro-MP prediction",
    )
    benchmark_prepare_ipromp.add_argument("config", type=Path)
    benchmark_prepare_ipromp.add_argument("--output-dir", type=Path)
    benchmark_prepare_ipromp.add_argument("--base-dir", type=Path, default=Path.cwd())
    benchmark_prepare_ai_x_bio = benchmark_sub.add_parser(
        "prepare-ai-x-bio",
        help="Prepare an ai x bio Drive file as sequence,label,id train/validation/test CSVs",
    )
    benchmark_prepare_ai_x_bio.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive"))
    benchmark_prepare_ai_x_bio.add_argument("--source-file", type=Path)
    benchmark_prepare_ai_x_bio.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/ai_x_bio"))
    benchmark_prepare_ai_x_bio.add_argument("--seed", type=int, default=42)

    annotate = subparsers.add_parser("annotate", help="Annotation workflows")
    annotate_sub = annotate.add_subparsers(dest="annotate_command", required=True)
    annotate_promoters = annotate_sub.add_parser(
        "promoters", aliases=["promoter"], help="Annotate predicted promoters in GenBank files"
    )
    annotate_promoters.add_argument("input", type=Path)
    annotate_promoters.add_argument("--model-family", choices=("dnabert2", "cnn_v2", "dummy"), default="dummy")
    annotate_promoters.add_argument("--checkpoint", type=Path)
    annotate_promoters.add_argument("--benchmark-manifest", type=Path)
    annotate_promoters.add_argument("--threshold", type=float)
    annotate_promoters.add_argument("--window-size", type=int)
    annotate_promoters.add_argument("--step-size", type=int, default=25)
    annotate_promoters.add_argument("--scan-both-strands", action=argparse.BooleanOptionalAction, default=True)
    annotate_promoters.add_argument("--merge-distance", type=int, default=25)
    annotate_promoters.add_argument("--min-score", type=float)
    annotate_promoters.add_argument("--predictions-csv", type=Path)
    annotate_promoters.add_argument("--manifest", type=Path)
    annotate_promoters.add_argument("--output", type=Path)
    annotate_promoters.add_argument("--preserve-existing-features", action=argparse.BooleanOptionalAction, default=True)
    annotate_promoters.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove this run's existing annotation output files before writing fresh artifacts",
    )
    annotate_promoters.add_argument(
        "--open-output-folder",
        action="store_true",
        help="Open the annotation output folder after a successful run",
    )
    annotate_promoters.add_argument("--gold-csv", type=Path)
    annotate_promoters.add_argument("--evaluation-dir", type=Path)
    annotate_promoters.add_argument("--sbol-output", type=Path)
    annotate_promoters.add_argument("--sbol-namespace", default="https://seqtrainer.org/designs")
    annotate_promoters.add_argument("--promoter-label-mode", choices=("strict", "labelled"), default="labelled")
    annotate_promoters.add_argument("--annotation-completeness", choices=("verified_complete", "partial", "unknown"), default="unknown")
    annotate_promoters.add_argument("--iou-threshold", type=float, default=0.50)

    annotate_collection = annotate_sub.add_parser("promoter-collection", help="Evaluate a collection of labelled GenBank plasmids")
    annotate_collection.add_argument("--manifest", type=Path, required=True)
    annotate_collection.add_argument("--input-dir", type=Path, required=True)
    annotate_collection.add_argument("--output-dir", type=Path, required=True)
    annotate_collection.add_argument("--predictor", "--model-family", dest="predictor", choices=("dnabert2", "cnn_v2", "dummy"), default="dummy")
    annotate_collection.add_argument("--model-path", "--checkpoint", dest="model_path", type=Path)
    annotate_collection.add_argument("--benchmark-manifest", type=Path)
    annotate_collection.add_argument("--sbol-namespace", default="https://seqtrainer.org/designs")
    annotate_collection.add_argument("--promoter-label-mode", choices=("strict", "labelled"), default="labelled")
    annotate_collection.add_argument("--annotation-completeness", choices=("verified_complete", "partial", "unknown"), default="unknown")
    annotate_collection.add_argument("--write-sbol3", action="store_true")
    annotate_collection.add_argument("--continue-on-error", action="store_true")

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
                sequence_length=_first_not_none(args.sequence_length, benchmark.preprocessing.sequence_length, 300),
                seed=_first_not_none(args.seed, benchmark.training.seed, benchmark.experiment.seed),
                batch_size=_first_not_none(args.batch_size, benchmark.training.batch_size, 16),
                cycles=_first_not_none(args.cycles, benchmark.training.max_epochs, 10),
                learning_rate=_first_not_none(args.learning_rate, benchmark.training.learning_rate, 1e-3),
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

        if args.benchmark_command == "prepare-dnabert2":
            from seqtrainer.benchmarks import prepare_dnabert2_tokenized_splits

            result = prepare_dnabert2_tokenized_splits(
                args.config,
                base_dir=args.base_dir,
                output_dir=args.output_dir,
            )
            print(f"output_dir={result.output_dir}")
            print(f"metadata={result.metadata_path}")
            for split, path in result.tokenized_paths.items():
                print(f"{split}={path}")
            return 0

        if args.benchmark_command == "prepare-ipromp":
            from seqtrainer.adapters.ipromp import prepare_ipromp_inputs

            result = prepare_ipromp_inputs(
                args.config,
                base_dir=args.base_dir,
                output_dir=args.output_dir,
            )
            print(f"output_dir={result.output_dir}")
            print(f"mapping_csv={result.mapping_csv}")
            print(f"command_script={result.command_script}")
            print(f"external_prediction_schema={result.prediction_schema}")
            for split, path in result.fasta_paths.items():
                print(f"{split}={path}")
            return 0

        if args.benchmark_command == "prepare-ai-x-bio":
            from seqtrainer.benchmarks import prepare_ai_x_bio_splits

            result = prepare_ai_x_bio_splits(
                drive_root=args.drive_root,
                source_file=args.source_file,
                output_dir=args.output_dir,
                seed=args.seed,
            )
            print(f"output_dir={result.output_dir}")
            print(f"metadata={result.metadata_path}")
            for split, path in result.split_paths.items():
                print(f"{split}={path}")
            return 0

    if args.command == "annotate":
        if args.annotate_command == "promoters":
            from seqtrainer.annotation import PromoterAnnotationConfig, run_promoter_annotation

            if args.clean_output:
                for path in (args.output, args.predictions_csv, args.manifest):
                    if path is not None and path.exists():
                        path.unlink()

            manifest = run_promoter_annotation(
                PromoterAnnotationConfig(
                    input_file=args.input,
                    output_file=args.output,
                    predictions_csv=args.predictions_csv,
                    manifest=args.manifest,
                    model_family=args.model_family,
                    checkpoint=args.checkpoint,
                    benchmark_manifest=args.benchmark_manifest,
                    threshold=args.threshold,
                    window_size=args.window_size,
                    step_size=args.step_size,
                    scan_both_strands=args.scan_both_strands,
                    merge_distance=args.merge_distance,
                    min_score=args.min_score,
                    preserve_existing_features=args.preserve_existing_features,
                    gold_csv=args.gold_csv,
                    evaluation_dir=args.evaluation_dir,
                    sbol_output=args.sbol_output,
                    sbol_namespace=args.sbol_namespace,
                    promoter_label_mode=args.promoter_label_mode,
                    annotation_completeness=args.annotation_completeness,
                    iou_threshold=args.iou_threshold,
                )
            )
            print(f"output_file={manifest['output_file']}")
            print(f"predictions_csv={manifest['predictions_csv']}")
            print(f"manifest={manifest['manifest_file']}")
            print(f"predicted_promoters_added={manifest['predicted_promoters_added']}")
            if args.open_output_folder:
                _open_output_folder(Path(manifest["output_file"]).parent)
            return 0

        if args.annotate_command == "promoter-collection":
            from seqtrainer.annotation.collection import run_promoter_collection

            result = run_promoter_collection(
                args.manifest,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                predictor=args.predictor,
                model_path=args.model_path,
                benchmark_manifest=args.benchmark_manifest,
                sbol_namespace=args.sbol_namespace,
                promoter_label_mode=args.promoter_label_mode,
                write_sbol3=args.write_sbol3,
                continue_on_error=args.continue_on_error,
                annotation_completeness=args.annotation_completeness,
            )
            print(f"collection_manifest={args.output_dir / 'collection_manifest.json'}")
            print(f"included_count={result['included_count']}")
            print(f"excluded_count={result['excluded_count']}")
            return 0

    if args.command == "benchmark-manifest":
        return _write_benchmark_manifest(args.config, args.output_dir, args.base_dir)

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    parser.error("Unhandled command")
    return 2


def _open_output_folder(path: Path) -> None:
    folder = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(folder)], check=False)
    else:
        subprocess.run(["xdg-open", str(folder)], check=False)


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


def _first_not_none(*values: _T | None) -> _T:
    for value in values:
        if value is not None:
            return value
    raise ValueError("At least one fallback value is required")


if __name__ == "__main__":
    raise SystemExit(main())
