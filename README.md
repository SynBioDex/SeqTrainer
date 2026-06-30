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
- `seqtrainer/models`: framework-neutral backbone/head registry stubs
- `seqtrainer/keras`: Keras adapters/factories (optional dependency)
- `seqtrainer/torch`: PyTorch adapters/fine-tune helpers (optional dependency)
- `seqtrainer/graph`: RDF/SBOL graph conversion utilities
- `seqtrainer/applications`: task-oriented blueprints
- `seqtrainer/cli`: command-line entrypoints

## CLI examples

The promoter benchmark CSVs are bundled as a ZIP archive. Extract them once in
a fresh checkout before running the benchmark commands:

```bash
python -m zipfile -e data/data_DNABERT/promoter_classification_DNABERT.zip data/promoter_classification
```

Then run:

```bash
seqtrainer sparql prefixes
seqtrainer inspect-sbol data/sbol_data/sample_design_0.xml
seqtrainer build-dataset data/sbol_data/sample_design_0.xml
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
seqtrainer benchmark compare outputs/benchmarks/cnn* --output-dir outputs/benchmarks/comparison
```

CNN benchmark configs live in `config-examples/benchmarks/`. The benchmark
runner writes reproducible artifacts such as `metrics.csv`, `metrics.json`,
`predictions.csv`, `manifest.json`, `history.csv`, and model checkpoints when a
model is trained. See `notebooks/benchmarks/cnn_benchmark/README.md` for the
Colab-ready CNN workflow and comparison rules.

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

Benchmark notebooks are grouped by model family under `notebooks/benchmarks/`.
The current CNN benchmark suite is available in `notebooks/benchmarks/cnn_benchmark/`:

- `notebooks/benchmarks/cnn_benchmark/cnn_reference_benchmark_colab.ipynb`: Colab-ready reference CNN benchmark on predefined train/eval/test CSV splits
- `notebooks/benchmarks/cnn_benchmark/cnn_v2_final_benchmark_colab.ipynb`: Colab-ready CNN-v2 benchmark with stronger regularized candidates and full metrics
