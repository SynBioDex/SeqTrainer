# Benchmark Notebooks

Benchmark notebooks are grouped by model family so future experiments do not get mixed with the starter tutorials.

## Folders

- [`cnn_benchmark/`](cnn_benchmark/): CNN reference and CNN-v2 promoter classification benchmarks on the shared train/eval/test split.
- [`dnabert_benchmark/`](dnabert_benchmark/): DNABERT2 frozen embedding benchmark and optional fine-tuning notebook on the same shared split.
- [`ipromp_benchmark/`](ipromp_benchmark/): iPro-MP external FASTA preparation, official prediction setup, and normalized evaluation path.

Future model families should get their own folders while reusing the same dataset split and metric policy whenever possible.

## Benchmark CLI

The package benchmark harness uses one command shape for all model families:

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
seqtrainer benchmark prepare-dnabert2 config-examples/benchmarks/dnabert2_smoke.toml
seqtrainer benchmark run config-examples/benchmarks/dnabert2_frozen.toml
seqtrainer benchmark run config-examples/benchmarks/dnabert2_finetune.toml
seqtrainer benchmark prepare-ipromp config-examples/benchmarks/ipromp_external.toml
python notebooks/benchmarks_sg/prepare_ai_x_bio_splits.py --drive-root /content/drive/MyDrive
seqtrainer benchmark run config-examples/benchmarks/dnabert2_ai_x_bio_frozen.toml
seqtrainer benchmark run config-examples/benchmarks/ipromp_external.toml
seqtrainer benchmark compare outputs/benchmarks/* --output-dir outputs/benchmarks/comparison
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

DNABERT2 and iPro-MP are dependency-gated. If optional dependencies, cached model
files, or external iPro-MP predictions are unavailable, the command writes a
`manifest.json` with a skipped status instead of silently producing incomparable
metrics. For final DNABERT2/iPro-MP benchmark runs, install the needed external
dependencies and model files first, then rerun the same config on the same split.

The iPro-MP path should initially be treated as an external baseline/adaptation
path, following the promoter-modeling motivation in the 2025 Genome Biology
iPro-MP paper: <https://link.springer.com/article/10.1186/s13059-025-03819-9>.
SeqTrainer should compare it with CNN and DNABERT2 using the shared split and
metric policy before considering any fine-tuning work.

## Colab Notebook Links

- [CNN reference benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/cnn_benchmark/cnn_reference_benchmark_colab.ipynb)
- [CNN-v2 final benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/cnn_benchmark/cnn_v2_final_benchmark_colab.ipynb)
- [DNABERT2 ai x bio frozen benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_ai_x_bio_frozen_colab.ipynb)
- [DNABERT2 local/HPC benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_local_hpc_benchmark.ipynb)
- [DNABERT2 Alpine HPC benchmark](https://github.com/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_alpine_hpc_benchmark.ipynb)
- [DNABERT2 benchmark workflow](https://github.com/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_benchmark_workflow.md)
- [iPro-MP external benchmark](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/ipromp_benchmark/ipromp_external_benchmark_colab.ipynb)
- [iPro-MP benchmark README](https://github.com/simplyshree/SeqTrainer/blob/issue-3-all-model-baselines/notebooks/benchmarks_sg/ipromp_benchmark/README.md)

After the notebook branches merge, replace the branch names in those URLs with
`dev`.

## Shared Outputs

Completed benchmark runs should write:

- `metrics.csv`
- `metrics.json`
- `predictions.csv`
- `manifest.json`
- `history.csv` when the runner trains a model
- `checkpoints/` when the runner trains or selects a model checkpoint

Skipped dependency-gated runs still write `manifest.json` and `config.json` so
the missing dependency or model-file reason is recorded.

## Comparison Table Template

| Model | Config | Selection Metric | Test MCC | Test AUPRC | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| CNN reference | `cnn.toml` | validation MCC |  |  | exact CNN baseline |
| CNN-v2 | `cnn_v2.toml` | validation MCC |  |  | regularized CNN candidate |
| DNABERT2 frozen | `dnabert2_frozen.toml` | validation MCC |  |  | dependency-gated until model files are available |
| DNABERT2 fine-tune | `dnabert2_finetune.toml` | validation MCC |  |  | optional compute-heavy run |
| iPro-MP external | `ipromp_external.toml` | validation MCC |  |  | FASTA adapter / external prediction path |

The full Colab-ready runbook lives at
[`docs/benchmarks/promoter_benchmark.md`](../../docs/benchmarks/promoter_benchmark.md).

## Reproducibility Rule

Benchmark notebooks should record:

- dataset source and split files
- model configuration
- random seed
- selected validation threshold
- validation metrics used for model choice
- held-out test metrics used for final reporting

Local CPU smoke runs are allowed for checking imports, data loading, artifact
writing, and metric formatting. They must be labelled as smoke runs and should
not be compared against full-split CNN-v2 metrics.

