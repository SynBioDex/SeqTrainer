# Handoff 00b — clade-safe ordered stream dataset

Notebook: `notebooks/titans_stage_c/00b_stage_c_stream_dataset.ipynb`

## Prerequisites and editable values

Run this after Handoff 00 freezes `tokenizer_selection.json`. Add the shared
`bacteria_titan_v1_ecoli_related_15gbp` Stage B source folder to My Drive as a
shortcut and leave `SOURCE_ROOT` pointed to it. Notebook 00b materializes one
temporary FASTA per selected *E. coli* assembly from the cached NCBI ZIPs, runs
the Skani extended triangle, and writes `inputs/ecoli_skani_triangle.tsv` to
the Stage C Drive workspace. The 95% screen is deliberately below the 99% split
clustering boundary so near-threshold pairs are retained. The resulting table
must retain `Ref_file`, `Query_file`, and `ANI` columns.

The notebook uses `virtualenv` to run Stage C commands in
`/content/seqtrainer-stage-c-venv`, with its own pinned NumPy/Pandas/PyArrow
ABI. This avoids replacing compiled packages inside Colab's already-running
notebook kernel, and does not depend on Colab shipping the optional standard
library `venv` component.

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
