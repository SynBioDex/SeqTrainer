# DNABERT2 Promoter Benchmark

This folder contains DNABERT2 benchmark notebooks for the shared E. coli promoter classification split used by the CNN benchmarks.

The goal is to keep DNABERT2 directly comparable with CNN-v2:

- same train/validation/test CSV files
- seed `42`
- validation-only threshold selection using MCC
- held-out test set used only for final reporting
- primary metric: MCC
- secondary metric: AUPRC
- same artifact contract: `metrics.csv`, `metrics.json`, `predictions.csv`, `manifest.json`, `history.csv`, `checkpoints/`, and `embeddings/`

## Files

- `dnabert2_local_hpc_benchmark.ipynb`: local/HPC notebook for the shared split. It runs a tiny CPU smoke benchmark when CUDA is unavailable and the full DNABERT2 benchmark when a CUDA GPU is visible.
- `dnabert2_local_hpc_benchmark_executed.ipynb`: executed reference copy showing the CPU-smoke path on a no-CUDA machine.
- `dnabert2_alpine_hpc_benchmark.ipynb`: fresh CURC Alpine runbook that writes SLURM scripts for an A100/L40 GPU smoke job and the full frozen DNABERT2 benchmark.
- `dnabert2_alpine_hpc_benchmark_executed.ipynb`: executed script-generation copy. It does not contain full model metrics because those must be produced on Alpine.
- `alpine_dnabert2_smoke.sbatch`: short Alpine tokenization/download smoke job.
- `alpine_dnabert2_frozen.sbatch`: full Alpine frozen DNABERT2 benchmark job.
- `dnabert2_ai_x_bio_frozen_colab.ipynb`: Colab-oriented DNABERT2 frozen benchmark notebook for the `ai x bio`/Drive preparation path.

## Dataset

Task: binary bacterial promoter prediction from DNA sequence windows.

Source accession: [GSE144621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144621)

The benchmark uses the same predefined split files as CNN-v2:

- `train_EP_DNA_BERT2_genomic_order.csv`
- `eval_EP_DNA_BERT2_genomic_order.csv`
- `test_EP_DNA_BERT2_genomic_order.csv`

These are extracted into:

```text
data/promoter_classification/
```

## Model

The primary DNABERT2 run is:

```text
zhihan1996/DNABERT-2-117M
```

The default path is a frozen encoder plus a small classifier head. This is the first fair comparison against CNN-v2 because it tests whether pretrained DNABERT2 representations improve promoter classification on the same split.

## Hardware Check

On the local test machine, Windows/WSL detected:

```text
GPU: Intel(R) Iris(R) Xe Graphics
nvidia-smi: unavailable
torch.cuda.is_available(): False
```

That means there is no NVIDIA CUDA GPU available locally. The notebook therefore runs a small CPU smoke benchmark so the code path and artifacts can be verified. Full DNABERT2 metrics should be produced on a CUDA GPU machine or HPC node.

## CPU Smoke Result

The CPU smoke run uses a small stratified subset:

- train: 16 rows
- validation: 8 rows
- test: 8 rows
- batch size: 1
- max epochs: 3
- threshold selected on validation MCC

This run produced the expected artifacts under:

```text
outputs/benchmarks/dnabert2_cpu_smoke_ep_genomic_order/
```

Metrics from the CPU smoke run:

Committed asset files:

- [`assets/dnabert2_cpu_smoke_metrics.csv`](assets/dnabert2_cpu_smoke_metrics.csv)
- [`assets/dnabert2_cpu_smoke_result.md`](assets/dnabert2_cpu_smoke_result.md)

| Split | Threshold | Accuracy | Balanced Accuracy | Precision | Recall/Sensitivity | F1 | MCC | Specificity | AUROC | AUPRC | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 0.505 | 0.750 | 0.750 | 0.833 | 0.625 | 0.714 | 0.516 | 0.875 | 0.781 | 0.715 | 0.690 |
| validation | 0.505 | 0.625 | 0.625 | 0.667 | 0.500 | 0.571 | 0.258 | 0.750 | 0.438 | 0.542 | 0.693 |
| test | 0.505 | 0.625 | 0.625 | 0.600 | 0.750 | 0.667 | 0.258 | 0.500 | 0.500 | 0.567 | 0.694 |

These are real metrics, but they are only a smoke-test result. They should not be compared against the full CNN-v2 benchmark as a scientific result because the CPU smoke run uses a tiny subset.

## Final DNABERT2 Run

For final model comparison, run the same config on a CUDA GPU/HPC node:

```bash
seqtrainer benchmark run config-examples/benchmarks/dnabert2_frozen.toml
```

On CURC Alpine, use the fresh Alpine notebook to write job scripts, then edit
`#SBATCH --account=<YOUR_ACCOUNT>` in both `.sbatch` files before submitting:

```bash
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_smoke.sbatch
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_frozen.sbatch
```

Use `aa100` first because it provides NVIDIA A100 GPUs. `al40` is a reasonable
fallback for NVIDIA L40 GPUs. Avoid `ami100` for this workflow unless you are
intentionally porting the environment to ROCm, because this SeqTrainer path uses
the PyTorch CUDA/NVIDIA stack.

The full run should write:

```text
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/
```

Use:

```bash
seqtrainer benchmark compare \
  outputs/benchmarks/cnn_v2_regularized_ep_genomic_order \
  outputs/benchmarks/dnabert2_frozen_ep_genomic_order \
  --output-dir outputs/benchmarks/comparison_cnn_v2_dnabert2
```

## Why FlashAttention Is Disabled

The official DNABERT2 remote code prefers a Triton FlashAttention path. On Colab and local WSL this produced two recurring failures:

- CUDA out-of-memory when the whole split was embedded at once
- Triton compatibility errors such as `dot() got an unexpected keyword argument 'trans_b'`

The runner now supports batched embedding extraction and can disable DNABERT2 FlashAttention so the model uses its standard PyTorch attention fallback. This keeps the benchmark reproducible across local CPU smoke runs, Colab, and HPC.

## Scientific Rule

Never tune the threshold on the test set. The validation split chooses the threshold by MCC; the test split reports final held-out metrics using that validation-selected threshold.
