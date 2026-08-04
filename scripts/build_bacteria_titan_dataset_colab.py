#!/usr/bin/env python3
"""Build a resumable GTDB R220/NCBI bacterial Titan dataset on Google Drive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from seqtrainer.data.bacteria_titan import (
    DEFAULT_SCOPE_FRACTIONS,
    accession_stats,
    download_gtdb_r220,
    download_ncbi_batches,
    prepare_candidates,
    read_gtdb_metadata,
    regenerate_fasta_shards,
    sample_accessions,
    split_accessions,
    tokenize_fasta_shards,
    validate_accession_manifest,
    write_dataset_metadata,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dataset-class", default="bacteria_titan_v1_ecoli_related_15gbp")
    parser.add_argument("--target-bp", type=int, default=15_000_000_000)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ncbi-batch-size", type=int, default=500)
    parser.add_argument("--fasta-shard-bp", type=int, default=500_000_000)
    parser.add_argument("--windows-per-shard", type=int, default=100_000)
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing NCBI ZIP cache only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.drive_root / args.dataset_class
    manifests = root / "manifests"
    gtdb_paths = download_gtdb_r220(root / "raw" / "gtdb_r220")
    metadata = read_gtdb_metadata(gtdb_paths["metadata"], gtdb_paths["taxonomy"])
    candidates = prepare_candidates(metadata)
    selected = sample_accessions(candidates, args.target_bp, DEFAULT_SCOPE_FRACTIONS, args.seed)
    selected = split_accessions(selected, seed=args.seed)
    validate_accession_manifest(selected)

    write_table(selected, manifests / "accession_manifest")
    for split in ("train", "val", "test"):
        selected.loc[selected["split"].eq(split)].to_parquet(
            manifests / f"accession_manifest_{split}.parquet", index=False
        )

    zip_dir = root / "raw" / "ncbi_dataset_zips"
    ncbi_manifest = download_ncbi_batches(
        selected["accession"],
        zip_dir,
        args.ncbi_batch_size,
        allow_download=not args.skip_download,
    )
    write_table(ncbi_manifest, manifests / "ncbi_batch_manifest")

    fasta_manifest = regenerate_fasta_shards(
        zip_dir, selected, root, args.dataset_class, target_shard_bp=args.fasta_shard_bp
    )
    write_table(fasta_manifest, manifests / "fasta_shard_manifest")
    token_manifest = tokenize_fasta_shards(
        fasta_manifest,
        root,
        args.dataset_class,
        context_length=args.context_length,
        windows_per_shard=args.windows_per_shard,
    )
    write_table(token_manifest, manifests / "token_shard_manifest")

    stats = accession_stats(selected)
    manifest = {
        "dataset_class": args.dataset_class,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gtdb_release": "R220",
        "target_bp": args.target_bp,
        "actual_selected_bp": stats["total_bp"],
        "context_length": args.context_length,
        "window_width": args.context_length + 1,
        "token_dtype": "uint8",
        "token_contract": {"PAD": 0, "N/UNK": 1, "A": 2, "C": 3, "G": 4, "T": 5},
        "scope_fractions": DEFAULT_SCOPE_FRACTIONS,
        "split_fractions": {"train": 0.90, "val": 0.05, "test": 0.05},
        "sampling_seed": args.seed,
        "quality_filters": {
            "minimum_completeness": 90,
            "maximum_contamination": 5,
            "genome_size_bp": [500_000, 12_000_000],
        },
        "accession_stats": stats,
        "fasta_shards": len(fasta_manifest),
        "token_shards": len(token_manifest),
    }
    readme = f"""# {args.dataset_class}

GTDB R220 bacterial genomes selected to approximately {args.target_bp:,} bp and
downloaded from NCBI by assembly accession. Train, validation, and test were
assigned at accession level before FASTA and token materialization. Never create
fallback token-level splits from this dataset.

Token contract: `PAD=0, N/UNK=1, A=2, C=3, G=4, T=5`.
"""
    write_dataset_metadata(root, manifest, readme)
    print(f"Dataset: {root}")
    print(f"Selected bp: {stats['total_bp']:,}; accessions: {stats['accessions']:,}")
    print(f"FASTA shards: {len(fasta_manifest)}; token shards: {len(token_manifest)}")


if __name__ == "__main__":
    main()
