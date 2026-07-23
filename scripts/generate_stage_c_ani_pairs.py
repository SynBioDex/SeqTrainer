#!/usr/bin/env python3
"""Generate the E. coli Skani triangle evidence required by Stage C splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd

from seqtrainer.data.bacteria_titan import read_table, run_skani_triangle


FASTA_SUFFIXES = (".fna", ".fa", ".fasta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Local temporary directory for one FASTA per selected E. coli accession",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skani", default="skani")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser.parse_args()


def materialize_ecoli_fastas(
    zip_dir: Path,
    accessions: set[str],
    work_dir: Path,
) -> list[Path]:
    """Extract each selected accession into one Skani-ready multi-FASTA file."""

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    written: set[str] = set()
    for zip_path in sorted(zip_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for member in sorted(archive.namelist()):
                if not member.endswith(FASTA_SUFFIXES):
                    continue
                accession = next(
                    (part for part in member.split("/") if part in accessions), None
                )
                if accession is None:
                    continue
                destination = work_dir / f"{accession}.fna"
                payload = archive.read(member)
                with destination.open("ab") as handle:
                    handle.write(payload)
                    if not payload.endswith(b"\n"):
                        handle.write(b"\n")
                written.add(accession)
    missing = sorted(accessions - written)
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"could not locate FASTA members for {len(missing)} selected E. coli accessions: {preview}"
        )
    return [work_dir / f"{accession}.fna" for accession in sorted(written)]


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    manifest_path = args.source_root / "manifests" / "accession_manifest.parquet"
    zip_dir = args.source_root / "raw" / "ncbi_dataset_zips"
    if not manifest_path.exists() or not zip_dir.is_dir():
        raise FileNotFoundError("source root must contain the accession manifest and NCBI ZIP directory")
    manifest = read_table(manifest_path)
    accessions = set(
        manifest.loc[manifest["scope"].eq("ecoli_species"), "accession"].astype(str)
    )
    if not accessions:
        raise ValueError("source accession manifest contains no ecoli_species accessions")
    fasta_paths = materialize_ecoli_fastas(zip_dir, accessions, args.work_dir)
    try:
        output = run_skani_triangle(
            fasta_paths,
            args.output,
            threads=args.threads,
            command=args.skani,
        )
        columns = set(pd.read_csv(output, sep="\t", nrows=0).columns)
        required = {"Ref_file", "Query_file", "ANI"}
        if missing := required - columns:
            raise ValueError(f"Skani output is missing required columns: {sorted(missing)}")
        print(
            json.dumps(
                {
                    "ani_pairs": str(output),
                    "ecoli_accessions": len(accessions),
                    "fasta_inputs": len(fasta_paths),
                    "screening_ani": 95,
                    "cluster_ani": 99,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
