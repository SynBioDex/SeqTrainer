"""Reproducible genome-level bacterial datasets for Titan MAC training."""

from .manifests import (
    GTDB_R220_METADATA_URL,
    GTDB_R220_TAXONOMY_URL,
    download_gtdb_r220,
    download_ncbi_batches,
    read_gtdb_metadata,
    validate_accession_manifest,
    write_dataset_metadata,
    write_table,
)
from .sampling import DEFAULT_SCOPE_FRACTIONS, classify_scope, prepare_candidates, sample_accessions
from .splits import DEFAULT_SPLIT_FRACTIONS, assert_no_accession_leakage, split_accessions
from .stats import accession_stats, token_shard_stats
from .token_shards import TokenShardDataset, regenerate_fasta_shards, tokenize_fasta_shards

__all__ = [
    "GTDB_R220_METADATA_URL",
    "GTDB_R220_TAXONOMY_URL",
    "DEFAULT_SCOPE_FRACTIONS",
    "DEFAULT_SPLIT_FRACTIONS",
    "download_gtdb_r220",
    "download_ncbi_batches",
    "read_gtdb_metadata",
    "prepare_candidates",
    "classify_scope",
    "sample_accessions",
    "split_accessions",
    "assert_no_accession_leakage",
    "regenerate_fasta_shards",
    "tokenize_fasta_shards",
    "TokenShardDataset",
    "accession_stats",
    "token_shard_stats",
    "validate_accession_manifest",
    "write_table",
    "write_dataset_metadata",
]
