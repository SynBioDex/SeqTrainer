# Benchmark Notebooks

This folder contains benchmark notebooks for the current promoter-classification work.

Current focus:

1. CNN reference and CNN-v2 baselines.
2. DNABERT2 frozen baseline and HPC workflow.

iPro-MP is intentionally not included here for now. We will return to it after the DNABERT path is stable.

## Folders

- [`cnn_benchmark/`](cnn_benchmark/): CNN reference and CNN-v2 notebooks on the shared train/validation/test split.
- [`dnabert_benchmark/`](dnabert_benchmark/): DNABERT2 smoke, Colab, and Alpine/HPC notebooks on the same shared split.

## Shared Rules

- Use the same train/validation/test CSVs across models.
- Use seed `42`.
- Select threshold on validation MCC only.
- Use the held-out test split only for final reporting.
- Compare primarily by test MCC and secondarily by test AUPRC.
- Do not compare tiny smoke-test metrics with full benchmark metrics.

## Useful Links

- [CNN reference Colab](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/cnn_benchmark/cnn_reference_benchmark_colab.ipynb)
- [CNN-v2 Colab](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/cnn_benchmark/cnn_v2_final_benchmark_colab.ipynb)
- [DNABERT2 Colab](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_ai_x_bio_frozen_colab.ipynb)
- [DNABERT2 Alpine/HPC notebook](dnabert_benchmark/dnabert2_alpine_hpc_benchmark.ipynb)

## Main Commands

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
seqtrainer benchmark prepare-dnabert2 config-examples/benchmarks/dnabert2_smoke.toml
seqtrainer benchmark run config-examples/benchmarks/dnabert2_frozen.toml
```

For full DNABERT2, use Alpine/HPC rather than a CPU-only local machine.
