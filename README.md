# SeqTrainer

SeqTrainer is a synthetic biology ML domain toolkit focused on connecting **SBOL / SynBioHub data** to modern model workflows.

It is designed to be complementary to Keras and PyTorch rather than replacing them.

## What this refactor introduces

- Clear package layering under `seqtrainer/`
- Framework-neutral core (`clients`, `sparql`, `data`, `transforms`, `models`)
- Optional framework adapters (`seqtrainer.keras`, `seqtrainer.torch`)
- Graph-focused utilities in `seqtrainer.graph`
- Application-level API entrypoints in `seqtrainer.applications`
- CLI foundation (`seqtrainer` command)
- Lightweight KNN retrieval utilities for scalar-label sequence design workflows

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[torch]'
pip install -e '.[keras]'
pip install -e '.[gnn]'
pip install -e '.[dev]'
```

## Package layout

- `seqtrainer/clients`: SynBioHub and remote clients
- `seqtrainer/sparql`: prefixes, builders, and query recipes
- `seqtrainer/data`: SBOL loaders, recipes, materialized datasets
- `seqtrainer/transforms`: DNA transforms and feature extraction
- `seqtrainer/models`: framework-neutral backbone/head registry stubs and KNN retrieval helpers
- `seqtrainer/keras`: Keras adapters/factories (optional dependency)
- `seqtrainer/torch`: PyTorch adapters/fine-tune helpers (optional dependency)
- `seqtrainer/graph`: RDF/SBOL graph conversion utilities
- `seqtrainer/applications`: task-oriented blueprints
- `seqtrainer/cli`: command-line entrypoints

## CLI examples

```bash
seqtrainer sparql prefixes
seqtrainer inspect-sbol data/sbol_data/sample_design_0.xml
seqtrainer build-dataset data/sbol_data/sample_design_0.xml
```

## Status

This is the first architecture-focused cleanup. Some framework integrations are intentionally placeholders with TODOs to keep a stable, minimal public surface.


## Tutorial notebooks

A starter notebook series is available in `notebooks/tutorials/`:

- `00_quickstart.ipynb`: core API objects and basic transforms
- `01_sbol_to_dataset.ipynb`: extract sequences/targets from SBOL files
- `02_dna_features.ipynb`: one-hot, GC content, and k-mer features
- `03_dataset_splits.ipynb`: reproducible train/val/test splits
- `04_end_to_end_cnn_classification.ipynb`: end-to-end mini CNN classifier demo (10 cycles)
- `05_end_to_end_cnn_regression.ipynb`: end-to-end mini CNN regressor for promoter activity (10 cycles)
- `06_nucleotide_transformer_v2_promoter_tasks.ipynb`: SeqTrainer-based Nucleotide Transformer v2 fine-tuning for promoter classification and regression
- `07_hyena2_promoter_classification_regression.ipynb`: SeqTrainer-based Hyena2/HyenaDNA workflow for promoter classification and regression
- `08_evo2_promoter_classification_regression.ipynb`: SeqTrainer-based Evo 2 workflow for promoter classification and regression
- `09_gemma3_4b_promoter_classification_regression.ipynb`: SeqTrainer-based Gemma 3 4B workflow for promoter classification and regression
- `10_promoter_activity_knn_retrieval.ipynb`: KNN retrieval workflow that returns ranked DNA sequences closest to a requested promoter activity
- `11_titans_miras_memory_context_promoter_classification.ipynb`: simplified Titans/MIRAS Memory-as-Context promoter classification workflow
