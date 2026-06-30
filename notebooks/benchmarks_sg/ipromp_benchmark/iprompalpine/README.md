# Run iPro-MP On Alpine

This folder is a self-contained Alpine workflow for the official E. coli
iPro-MP five-fold ensemble. Code and environments live under `/projects`; the
job uses `/scratch/alpine` for temporary data and copies final artifacts back to
`/projects`.

## 1. Put Code And Data On Alpine

From an Alpine login node:

```bash
cd /projects/$USER
git clone --branch issue-3-all-model-baselines \
  https://github.com/simplyshree/SeqTrainer.git
cd SeqTrainer

mkdir -p data/promoter_classification
```

Place these exact shared files in `data/promoter_classification/`:

```text
train_EP_DNA_BERT2_genomic_order.csv
eval_EP_DNA_BERT2_genomic_order.csv
test_EP_DNA_BERT2_genomic_order.csv
```

Each CSV must contain `sequence` and `label`. These are the same rows used by
CNN and DNABERT2; do not regenerate or reshuffle them.

## 2. Build The Environment Once

Run setup on an `acompile` or other compute session, not on the login node:

```bash
acompile
cd /projects/$USER/SeqTrainer
bash notebooks/benchmarks_sg/ipromp_benchmark/iprompalpine/setup_ipromp_alpine.sh
exit
```

Setup creates `/projects/$USER/seqtrainer_ipromp/env`, downloads DNABERT-6,
and selectively downloads the five E. coli checkpoints from Zenodo. It does
not download the complete 38.3 GB all-species archive.

## 3. Submit The A100 Job

```bash
cd /projects/$USER/SeqTrainer/notebooks/benchmarks_sg/ipromp_benchmark/iprompalpine
mkdir -p logs

sbatch \
  --account=<YOUR_ALPINE_ALLOCATION> \
  --export=ALL,DATA_DIR=/projects/$USER/SeqTrainer/data/promoter_classification \
  run_ipromp_alpine.sbatch
```

The script requests one NVIDIA A100, 64 GB RAM, eight CPU cores, and up to 12
hours. Override `BATCH_SIZE` at submission if memory is tight:

```bash
sbatch \
  --account=<YOUR_ALPINE_ALLOCATION> \
  --export=ALL,DATA_DIR=/projects/$USER/SeqTrainer/data/promoter_classification,BATCH_SIZE=8 \
  run_ipromp_alpine.sbatch
```

## 4. Monitor And Collect Results

```bash
squeue --me
tail -f logs/ipromp-<JOB_ID>.out
```

Persistent results are copied to:

```text
/projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/
```

Verify:

```bash
cat /projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/metrics.csv
cat /projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/manifest.json
```

## Scientific Comparison Contract

- Same predefined train/validation/test CSVs as CNN-v2.
- Same label meaning: promoter `1`, background `0`.
- Same seed: 42.
- No test-set threshold tuning.
- Validation selects the MCC-maximizing threshold.
- Test MCC is the primary final result; test AUPRC is secondary.
- Official fold probabilities are retained in `external_predictions/`.

The paper's 5-fold training design explains why five pretrained checkpoints are
ensembled. This run does not retrain those folds on SeqTrainer data; it measures
the external pretrained E. coli model on SeqTrainer's held-out split.
