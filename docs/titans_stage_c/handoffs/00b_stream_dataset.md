# Handoff 00b — clade-safe ordered stream dataset

Notebook: `notebooks/titans_stage_c/00b_stage_c_stream_dataset.ipynb`

## Prerequisites and editable values

Run this after Handoff 00 freezes `tokenizer_selection.json`. Supply the Stage
B source root, the skani extended triangle TSV for the selected *E. coli*
assemblies, and the frozen tokenizer artifact paths. The skani screen must have
been below the 99% clustering boundary (the repository helper uses 95%) and the
table must retain `Ref_file`, `Query_file`, and `ANI` columns.

Edit only the tagged configuration cell. `GIT_REF` must be a pushed immutable
commit. The notebook loads `tokenizer_selection.json` and refuses a tokenizer
name or checksum mismatch, so it cannot silently retrain or substitute BPE.

## Success and returned evidence

Success produces ANI membership, hybrid clade/accession and FASTA manifests,
tokenizer identity, a contig-level JSONL stream index, memory-mapped token/base-
length shards, and `token_stream_manifest.json` with checksums and provenance.
The output format lazily creates 32-token segments while preserving raw-base
counts and complete contig state boundaries. `split_summary.json`,
`split_bases.svg`, and `split_clade_groups.svg` make imbalance or unexpectedly
small held-out partitions visible before accelerator spending.

Return both printed directories. The run directory contains continuously
persisted logs and the Colab manifest; the dataset directory is the immutable
input for T4/A100 work. If `FAILED.txt` exists, stop before Handoff 01.
