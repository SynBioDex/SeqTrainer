# DNABERT2 CPU Smoke Result

This asset records the local no-CUDA DNABERT2 smoke run from `dnabert2_local_hpc_benchmark.ipynb`.

The run used a small stratified subset only:

- train: 16 rows
- validation: 8 rows
- test: 8 rows
- batch size: 1
- max epochs: 3
- threshold: selected on validation MCC

These metrics are real, but they are not a full scientific comparison against CNN-v2 because the full train/validation/test split was not used.

| Split | Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1 | MCC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.505 | 0.750 | 0.750 | 0.833 | 0.625 | 0.714 | 0.516 | 0.781 | 0.715 |
| validation | 0.505 | 0.625 | 0.625 | 0.667 | 0.500 | 0.571 | 0.258 | 0.438 | 0.542 |
| test | 0.505 | 0.625 | 0.625 | 0.600 | 0.750 | 0.667 | 0.258 | 0.500 | 0.567 |

Use this result to verify that DNABERT2 artifact generation works on systems without CUDA. For final comparison, rerun `config-examples/benchmarks/dnabert2_frozen.toml` on a CUDA GPU or HPC node using the full shared split.
