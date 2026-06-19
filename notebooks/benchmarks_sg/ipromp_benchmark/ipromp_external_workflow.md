# iPro-MP External Benchmark Workflow

This workflow keeps iPro-MP comparable with the SeqTrainer CNN-v2 and DNABERT2 benchmarks.

## Goal

Run official iPro-MP predictions on the same shared E. coli promoter split used by CNN-v2 and DNABERT2, then evaluate those predictions with SeqTrainer's shared metrics.

## Shared Benchmark Rules

- Use the same train/validation/test CSV files as CNN-v2 and DNABERT2.
- Use species ID `10` for `Escherichia coli str K-12 substr. MG1655`.
- Do not tune on the test set.
- Select the classification threshold only on validation predictions, using MCC.
- Report final test metrics only after validation threshold selection.
- Do not fake metrics if official iPro-MP weights or prediction files are missing.

## Step 1: Prepare SeqTrainer Inputs

From the SeqTrainer repo root:

```bash
python -m pip install -e ".[torch]"
seqtrainer benchmark prepare-ipromp config-examples/benchmarks/ipromp_external.toml
```

This writes:

```text
outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_fasta/train.fasta
outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_fasta/validation.fasta
outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_fasta/test.fasta
outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_id_mapping.csv
outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_run_commands.sh
outputs/benchmarks/ipromp_external_ep_genomic_order/external_prediction_schema.md
```

## Step 2: Set Up Official iPro-MP

```bash
mkdir -p external
cd external
git clone https://github.com/Jackie-Suv/iPro-MP.git
cd iPro-MP

conda create -n ipromp python=3.8 -y
conda activate ipromp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Step 3: Add External Model Files

Download the official pretrained model archive from Zenodo:

```text
https://doi.org/10.5281/zenodo.15180138
```

The archive is large, about 38.3 GB, so prefer HPC or a high-storage machine.

After extraction, place the E. coli fold files under:

```text
external/iPro-MP/models/
```

The official prediction script expects files like:

```text
10_fold_1.pth
10_fold_2.pth
10_fold_3.pth
10_fold_4.pth
10_fold_5.pth
```

Also place DNABERT-6 under:

```text
external/iPro-MP/DNABERT-6
```

## Step 4: Run Official iPro-MP Predictions

From the SeqTrainer repo root:

```bash
bash outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_run_commands.sh
```

This should produce:

```text
outputs/benchmarks/ipromp_external_ep_genomic_order/external_predictions/train_predictions.csv
outputs/benchmarks/ipromp_external_ep_genomic_order/external_predictions/validation_predictions.csv
outputs/benchmarks/ipromp_external_ep_genomic_order/external_predictions/test_predictions.csv
```

The expected official output columns are:

```text
Sequence,Prediction,Probability
```

SeqTrainer also accepts normalized prediction files with:

```text
split,sequence_id,label,probability
```

## Step 5: Evaluate With SeqTrainer

After prediction files exist:

```bash
seqtrainer benchmark run config-examples/benchmarks/ipromp_external.toml
```

Expected output artifacts:

```text
outputs/benchmarks/ipromp_external_ep_genomic_order/metrics.csv
outputs/benchmarks/ipromp_external_ep_genomic_order/metrics.json
outputs/benchmarks/ipromp_external_ep_genomic_order/predictions.csv
outputs/benchmarks/ipromp_external_ep_genomic_order/manifest.json
outputs/benchmarks/ipromp_external_ep_genomic_order/config.json
```

If prediction files are missing, SeqTrainer writes a skipped manifest instead of fake metrics.

## Step 6: Compare With CNN-v2 And DNABERT2

```bash
seqtrainer benchmark compare \
  outputs/benchmarks/cnn_v2_regularized_ep_genomic_order \
  outputs/benchmarks/dnabert2_frozen_ep_genomic_order \
  outputs/benchmarks/ipromp_external_ep_genomic_order \
  --output-dir outputs/benchmarks/comparison
```

Rank models primarily by held-out test MCC and secondarily by held-out test AUPRC.
