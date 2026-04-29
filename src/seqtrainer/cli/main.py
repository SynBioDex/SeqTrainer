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

    if args.command == "sparql" and args.sparql_command == "prefixes":
        print(format_prefixes())
        return 0

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
