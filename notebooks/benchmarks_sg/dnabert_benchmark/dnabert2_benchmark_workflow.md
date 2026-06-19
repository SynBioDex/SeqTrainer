# DNABERT2 Benchmark Workflow

This workflow describes the updated DNABERT2 plan for SeqTrainer. It keeps DNABERT2 comparable with CNN-v2 by using the same dataset split, metrics, and validation-only threshold policy.

## What Today's Ubuntu Work Proved

The local Ubuntu/WSL setup was used for environment and smoke validation, not for final scientific DNABERT2 metrics.

The local machine did not expose an NVIDIA CUDA GPU:

```text
nvidia-smi: unavailable
torch.cuda.is_available(): False
```

Because full DNABERT2 on the complete split is too heavy for CPU-only local execution, the local run used a tiny CPU smoke subset. That smoke run proved:

- SeqTrainer can import and call the DNABERT2 benchmark runner.
- The shared split loader works.
- DNABERT2 model loading issues were handled more robustly.
- The runner can produce benchmark artifacts.
- Validation-only threshold selection works.
- The metrics table format matches CNN-v2.

It did not prove that DNABERT2 beats CNN-v2. The smoke dataset is intentionally tiny and is not a full benchmark.

## Scientific Comparison Rules

- Use the same train/validation/test CSV files as CNN-v2.
- Use seed `42`.
- Do not tune on the test set.
- Select threshold only on validation, using MCC.
- Report held-out test metrics using the validation-selected threshold.
- Primary metric: MCC.
- Secondary metric: AUPRC.
- Also report accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, AUROC, and confusion matrix.
- Label CPU/no-GPU runs as smoke runs only.

## Shared Dataset

DNABERT2 must use the same split files as CNN-v2:

```text
data/promoter_classification/train_EP_DNA_BERT2_genomic_order.csv
data/promoter_classification/eval_EP_DNA_BERT2_genomic_order.csv
data/promoter_classification/test_EP_DNA_BERT2_genomic_order.csv
```

These files come from the GSE144621 promoter benchmark workflow already used by the CNN benchmarks.

## Model Plan

### 1. Smoke Test

Purpose: verify setup, imports, tokenization, model loading, metrics, and artifact writing.

Config:

```text
config-examples/benchmarks/dnabert2_smoke.toml
```

Run:

```bash
seqtrainer benchmark prepare-dnabert2 config-examples/benchmarks/dnabert2_smoke.toml
```

Use this locally or on Alpine before running the full benchmark.

### 2. Frozen DNABERT2 Baseline

Purpose: first real DNABERT2 comparison against CNN-v2.

Config:

```text
config-examples/benchmarks/dnabert2_frozen.toml
```

Run:

```bash
seqtrainer benchmark run config-examples/benchmarks/dnabert2_frozen.toml
```

This uses DNABERT2 as a pretrained encoder and trains/evaluates a lightweight classifier path using the shared benchmark artifact contract.

Expected artifacts:

```text
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/metrics.csv
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/metrics.json
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/predictions.csv
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/manifest.json
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/history.csv
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/checkpoints/
outputs/benchmarks/dnabert2_frozen_ep_genomic_order/embeddings/
```

### 3. Optional Fine-Tuning

Purpose: only after frozen DNABERT2 is stable.

Config:

```text
config-examples/benchmarks/dnabert2_finetune.toml
```

Fine-tuning is compute-heavy and should be run only on suitable GPU/HPC hardware. It should still use the same split and validation-only thresholding.

## Recommended Hardware

Use CURC Alpine or another NVIDIA CUDA GPU system for full DNABERT2 results.

Preferred Alpine target:

```text
aa100 partition, NVIDIA A100 GPU
```

Fallback:

```text
al40 partition, NVIDIA L40 GPU
```

Avoid `ami100` for now unless intentionally porting to ROCm, because the current SeqTrainer DNABERT2 workflow expects the PyTorch CUDA/NVIDIA stack.

## Alpine Workflow

Use the Alpine notebook:

```text
notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_alpine_hpc_benchmark.ipynb
```

It writes two SLURM scripts:

```text
notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_smoke.sbatch
notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_frozen.sbatch
```

Before submitting, edit both files and replace:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
```

Submit smoke first:

```bash
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_smoke.sbatch
```

If the smoke job succeeds, submit the full frozen benchmark:

```bash
sbatch notebooks/benchmarks_sg/dnabert_benchmark/alpine_dnabert2_frozen.sbatch
```

Check logs:

```bash
ls logs/
cat logs/seqtrainer-dnabert2-smoke-*.out
cat logs/seqtrainer-dnabert2-frozen-*.out
```

## Compare With CNN-v2

After the full frozen DNABERT2 run finishes:

```bash
seqtrainer benchmark compare \
  outputs/benchmarks/cnn_v2_regularized_ep_genomic_order \
  outputs/benchmarks/dnabert2_frozen_ep_genomic_order \
  --output-dir outputs/benchmarks/comparison_cnn_v2_dnabert2
```

Rank by held-out test MCC first and held-out test AUPRC second.

## How To Interpret Results

- If DNABERT2 improves test MCC and AUPRC over CNN-v2, it becomes the stronger model path.
- If DNABERT2 only improves AUROC but not MCC/AUPRC, inspect thresholding and class-wise behavior before claiming improvement.
- If DNABERT2 is only slightly better but much slower, report the tradeoff honestly.
- If DNABERT2 does not beat CNN-v2, keep CNN-v2 as the stronger baseline and use DNABERT2 as a documented comparison.

## What Not To Do

- Do not compare the CPU smoke result against full CNN-v2 metrics.
- Do not tune thresholds on the test set.
- Do not change the split files between CNN-v2 and DNABERT2.
- Do not report skipped or smoke runs as final model performance.
