# DNABERT2 Benchmark

This folder is for the DNABERT2 promoter-classification benchmark on the same split used by CNN-v2.

## What Stays Fixed

- Dataset: GSE144621 promoter classification split.
- Split files:
  - `data/promoter_classification/train_EP_DNA_BERT2_genomic_order.csv`
  - `data/promoter_classification/eval_EP_DNA_BERT2_genomic_order.csv`
  - `data/promoter_classification/test_EP_DNA_BERT2_genomic_order.csv`
- Seed: `42`.
- Threshold: selected on validation MCC only.
- Final comparison: held-out test MCC first, held-out test AUPRC second.

## Files

- `dnabert2_ai_x_bio_frozen_colab.ipynb`: Colab-oriented DNABERT2 frozen benchmark notebook.
- `dnabert2_local_hpc_benchmark.ipynb`: local/HPC notebook. On CPU-only machines, use it only for smoke checks.
- `dnabert2_local_hpc_benchmark_executed.ipynb`: executed CPU-smoke reference from a no-CUDA machine.
- `dnabert2_alpine_hpc_benchmark.ipynb`: Alpine runbook that writes SLURM scripts.
- `dnabert2_alpine_hpc_benchmark_executed.ipynb`: executed script-generation copy.
- `alpine_dnabert2_smoke.sbatch`: short Alpine smoke job.
- `alpine_dnabert2_frozen.sbatch`: full frozen DNABERT2 Alpine job.
- `assets/dnabert2_cpu_smoke_metrics.csv`: CPU smoke metrics.
- `assets/dnabert2_cpu_smoke_result.md`: explanation of the CPU smoke result.

## Current Colab Notebook

- Colab notebook: <https://colab.research.google.com/drive/1DookxBKrzfY2Hm56Cy7aAKMBInuE-wCg>
- Google Drive file: <https://drive.google.com/file/d/1DookxBKrzfY2Hm56Cy7aAKMBInuE-wCg/view?usp=drivesdk>

Use this notebook as the current DNABERT working copy for the shared-split
benchmark. Keep its dataset split, seed, validation-threshold policy, and metric
set aligned with the CNN-v2 benchmark before comparing results.

## Ubuntu/Local Result

The Ubuntu/WSL work was only a smoke test because the local machine did not expose an NVIDIA CUDA GPU:

```text
nvidia-smi: unavailable
torch.cuda.is_available(): False
```

That smoke test proves the code path works: imports, shared split loading, DNABERT2 runner calls, metric formatting, and artifact writing. It does **not** prove DNABERT2 model performance and must not be compared with full CNN-v2 metrics.

## Full DNABERT2 Run

Use Alpine or another NVIDIA CUDA GPU system.

Preferred Alpine partition:

```text
aa100
```

Fallback:

```text
al40
```

Avoid `ami100` for now because this workflow expects the PyTorch CUDA/NVIDIA stack, not ROCm.

Before submitting, edit both SLURM files and replace:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
```

Then run:

```bash
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_smoke.sbatch
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_frozen.sbatch
```

## Compare With CNN-v2

After the full DNABERT2 run finishes:

```bash
seqtrainer benchmark compare \
  outputs/benchmarks/cnn_v2_regularized_ep_genomic_order \
  outputs/benchmarks/dnabert2_frozen_ep_genomic_order \
  --output-dir outputs/benchmarks/comparison_cnn_v2_dnabert2
```

## DNABERT Family Plan

Start with DNABERT2 frozen. If that path is stable, add a separate frozen-family comparison for DNABERT-6, DNABERT2, and DNABERT-S. Fine-tune only the best frozen candidate later.
