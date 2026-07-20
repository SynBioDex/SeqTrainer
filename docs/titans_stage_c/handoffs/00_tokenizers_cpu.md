# Handoff 00 — tokenizers and CPU basal gate

Notebook: `notebooks/titans_stage_c/00_stage_c_tokenizers_and_cpu_baseline.ipynb`

## Purpose and inputs

This CPU notebook compares the SeqTrainer base, Evo 2 character, fixed 6-mer,
official DNABERT-2 BPE, and train-only bacterial BPE tokenizers. It then runs
the matched tiny paper-MAC and statistical baselines in both equal-optimizer-
step and equal-raw-base regimes. Place leakage-safe train and validation FASTAs
on Drive before starting.

Edit only the tagged configuration cell: reviewed commit, Drive root, FASTA
paths, and run name. The commit must be a pushed immutable SHA. CPU runtime is
expected; this gate should consume no GPU compute units. The notebook installs
the small official `vtx` package so the Evo 2 `CharLevelTokenizer(512)` adapter
is parity-verified without downloading Evo 2 model weights.

## Success, recovery, and return

Success requires `TOKENIZER_REPORT.md`, `CPU_PILOT_REPORT.md`,
`tokenizer_selection.json`, their JSON/CSV
sources, SVG plots, per-tokenizer training histories, and frozen tokenizer
artifacts. The Evo 2 fallback is ineligible until official Vortex parity is
available. The selection uses held-out equal-base BPB, intrinsic eligibility,
and the documented 0.01-BPB promotion threshold over the base tokenizer. A
rerun may reuse downloaded DNABERT-2 artifacts and overwrite only the explicitly
named run directory. Return the complete directory printed by the final cell;
on failure return `FAILED.txt`, `colab_run_manifest.json`, and `logs/`.
