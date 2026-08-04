"""Freeze or validate nested whole-replicon Stage C E. coli panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from seqtrainer.data.bacteria_titan import (
    StageCPanelManifest,
    TokenStreamDataset,
    freeze_ecoli_panels,
    read_table,
    sha256_file,
    validate_panel_against_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--dataset-dir", type=Path, required=True)
    freeze.add_argument("--accession-manifest", type=Path, required=True)
    freeze.add_argument("--ani-membership", type=Path, required=True)
    freeze.add_argument("--ani-pairs", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--ncbi-zip-dir", type=Path)
    freeze.add_argument("--seed", type=int, default=20260751)
    freeze.add_argument("--e25-bases", type=int, default=25_000_000)
    freeze.add_argument("--e100-bases", type=int, default=100_000_000)
    freeze.add_argument("--e250-bases", type=int, default=250_000_000)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--panel-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return read_table(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "validate":
        dataset = TokenStreamDataset(args.dataset_dir, verify_checksums=True)
        panel = StageCPanelManifest.from_path(args.panel_manifest)
        validate_panel_against_dataset(panel, dataset)
        print(json.dumps({"panel_hash": panel.hash, "valid": True}, indent=2))
        return 0
    paths = freeze_ecoli_panels(
        dataset_dir=args.dataset_dir,
        accession_manifest=_read(args.accession_manifest),
        ani_membership=_read(args.ani_membership),
        ani_pairs=pd.read_csv(args.ani_pairs, sep="\t"),
        output_dir=args.output_dir,
        ncbi_zip_dir=args.ncbi_zip_dir,
        targets={
            "e25": args.e25_bases,
            "e100": args.e100_bases,
            "e250": args.e250_bases,
        },
        seed=args.seed,
        source_provenance={
            "accession_manifest": {
                "path": str(args.accession_manifest),
                "sha256": sha256_file(args.accession_manifest),
            },
            "ani_membership": {
                "path": str(args.ani_membership),
                "sha256": sha256_file(args.ani_membership),
            },
            "ani_pairs": {
                "path": str(args.ani_pairs),
                "sha256": sha256_file(args.ani_pairs),
            },
        },
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
