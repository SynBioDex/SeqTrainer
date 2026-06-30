# Benchmark Notebooks

Benchmark notebooks are grouped by model family so experiments stay separate
from the starter tutorials.

## CNN Benchmark

- [`cnn_benchmark/`](cnn_benchmark/): CNN reference and CNN-v2 promoter
  classification benchmarks on the predefined train/eval/test split.

The package benchmark harness currently supports the CNN benchmark path in this
PR:

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
seqtrainer benchmark compare outputs/benchmarks/cnn* --output-dir outputs/benchmarks/comparison
```

Use `--base-dir` when running from outside the repository root, and
`--output-dir` to override the configured output folder:

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml \
  --output-dir outputs/benchmarks/cnn_v2_trial
```

`cnn.toml` is the reference CNN configuration. `cnn_v2.toml` is the regularized
CNN-v2 candidate using AdamW, OneCycleLR, dropout, validation-MCC checkpoint
selection, and early stopping.

## Colab Notebook Links

- [CNN reference benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-cnn-baseline-reproduction/notebooks/benchmarks/cnn_benchmark/cnn_reference_benchmark_colab.ipynb)
- [CNN-v2 final benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-cnn-baseline-reproduction/notebooks/benchmarks/cnn_benchmark/cnn_v2_final_benchmark_colab.ipynb)

After the notebook branch merges, replace the branch name in those URLs with
`dev`.

## Shared Outputs

Completed CNN benchmark runs should write:

- `metrics.csv`
- `metrics.json`
- `predictions.csv`
- `manifest.json`
- `history.csv`
- `checkpoints/`

## Comparison Table Template

| Model | Config | Selection Metric | Test MCC | Test AUPRC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| CNN reference | `cnn.toml` | validation MCC |  |  | exact CNN baseline |
| CNN-v2 | `cnn_v2.toml` | validation MCC |  |  | regularized CNN candidate |

## Reproducibility Rule

Benchmark notebooks should record:

- dataset source and split files
- model configuration
- random seed
- selected validation threshold
- validation metrics used for model choice
- held-out test metrics used for final reporting
