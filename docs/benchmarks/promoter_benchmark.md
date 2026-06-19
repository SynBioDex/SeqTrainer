# Promoter Benchmark Workflow

This benchmark compares promoter-sequence classifiers on the same bacterial
promoter prediction split. The current model families are:

- CNN reference baseline
- CNN-v2 regularized candidate
- DNABERT2 frozen encoder and optional fine-tuning
- iPro-MP/iPromoter external prediction evaluation

The expected research trajectory is CNN baseline < CNN-v2 < DNABERT2 <
iPro-MP/iPromoter, but the actual ranking must come from the shared benchmark
artifacts. Do not fabricate metrics or tune on the test set.

## Why The Rules Matter

All models must use the same train/validation/test CSV split because otherwise
model differences can be caused by different data, not better learning.

Thresholds are selected on the validation split only, usually by MCC. This
prevents test leakage. The test split is held back for final reporting after the
model and threshold have already been chosen.

MCC and AUPRC are primary metrics because promoter datasets can become
imbalanced. Accuracy is still reported, but it can look good even when a model
misses many positives. MCC summarizes all four confusion-matrix cells, and AUPRC
focuses on positive-class ranking quality.

## Input CSV Format

Each split CSV must contain at least:

```text
sequence,label
ACGT...,1
TGCA...,0
```

The benchmark configs currently point to:

- `data/promoter_classification/train_EP_DNA_BERT2_genomic_order.csv`
- `data/promoter_classification/eval_EP_DNA_BERT2_genomic_order.csv`
- `data/promoter_classification/test_EP_DNA_BERT2_genomic_order.csv`

These files are the shared split for CNN, DNABERT2, and iPro-MP/iPromoter.

## Colab Setup

```bash
git clone https://github.com/simplyshree/SeqTrainer.git
cd SeqTrainer
git checkout issue-3-all-model-baselines
python -m pip install --upgrade pip
python -m pip install -e ".[torch]"
```

If using Google Drive data, mount Drive in a notebook and copy the three CSVs to
`data/promoter_classification/`. The shared Drive folder used for the CNN and
DNABERT2 runs is:

```text
/content/drive/MyDrive/AIxBio/Promoter Classification/Data
```

Drive folder link:
`https://drive.google.com/drive/folders/1rH47oJEjQjkJvHXKX_rwDjDb--dGPGx2`

It should contain the same three split files listed above. If using the bundled
archive, extract `data/data_DNABERT/promoter_classification_DNABERT.zip` so the
same three CSV paths exist.

## Run CNN Benchmarks

```bash
seqtrainer benchmark run config-examples/benchmarks/cnn.toml
seqtrainer benchmark run config-examples/benchmarks/cnn_v2.toml
```

`cnn.toml` is the reproducible reference baseline. `cnn_v2.toml` uses AdamW,
OneCycleLR, dropout, early stopping, validation-MCC threshold selection, and
best-checkpoint selection.

## Run DNABERT2

Run a tokenization smoke check before training:

```bash
seqtrainer benchmark prepare-dnabert2 config-examples/benchmarks/dnabert2_smoke.toml
```

This writes `dnabert2_tokenized/train.csv`, `validation.csv`, `test.csv`, and
`dnabert2_tokenization_metadata.json`. It verifies that DNABERT2 is reading the
same split rows as CNN and records tokenizer/model settings.

Frozen encoder first:

```bash
seqtrainer benchmark run config-examples/benchmarks/dnabert2_frozen.toml
```

The frozen run loads `AutoTokenizer` and `AutoModel`, freezes DNABERT2, extracts
train/validation/test embeddings, caches them under the run's `embeddings/`
folder, and trains only a small classifier head. This is the first DNABERT2
baseline to compare against CNN-v2 because it keeps transfer learning useful
without immediately paying the full fine-tuning cost.

Optional full fine-tuning:

```bash
seqtrainer benchmark run config-examples/benchmarks/dnabert2_finetune.toml
```

The fine-tuning config is compute-heavy and GPU-gated. It uses a lower learning
rate, AdamW, warmup, early stopping by validation MCC, and best-checkpoint
selection. Inspect `history.csv` for train loss, validation loss, validation MCC,
selected threshold, and learning rate to decide whether the model is overfitting.

DNABERT2 is dependency-gated. If `transformers`, `torch`, model files, tokenizer
files, or compute resources are unavailable, the runner writes a skipped
`manifest.json` instead of pretending metrics exist. To allow model/tokenizer
download in Colab, set `model.params.allow_download = true` in the config for
that run. Keep the downloaded model revision recorded in the run artifacts before
using the results in a report.

The runner first tries explicit `AutoTokenizer` + `AutoConfig` + `AutoModel`
loading with `pad_token_id` patched from the tokenizer. If the model still loads
with meta-device parameters in Colab, it automatically falls back to the CPU
state-dict loader before skipping. This keeps the fix in the package runner
rather than in notebook-only retry code.

## iPro-MP Setup And External Benchmark

iPro-MP is a DNABERT-based prokaryotic promoter model. The official paper
evaluates it across 23 species and reports Acc, AUC, AUPRC, and MCC. SeqTrainer
keeps iPro-MP behind an external adapter so the official code, DNABERT-6
dependency, and pretrained weights do not get hardcoded into this package.

Prepare SeqTrainer FASTA inputs:

```bash
git clone --branch issue-3-all-model-baselines https://github.com/simplyshree/SeqTrainer.git
cd SeqTrainer
python -m pip install --upgrade pip
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

Set up official iPro-MP separately:

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

Then:

- download DNABERT-6 into `external/iPro-MP/DNABERT-6`
- download pretrained iPro-MP model files from
  `https://doi.org/10.5281/zenodo.15180138`
- do not commit large weights
- for `Escherichia coli str K-12 substr. MG1655`, use `species_id = 10`
- verify the exact downloaded model filenames; they may follow a fold pattern
  such as `10_fold_1.pth` through `10_fold_5.pth`

Official prediction command shape:

```bash
python iPro-MP_predict.py -i example.fasta -s species_ID -o outputfile
```

For SeqTrainer:

```bash
python iPro-MP_predict.py \
  -i ../../outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_fasta/validation.fasta \
  -s 10 \
  -o ../../outputs/benchmarks/ipromp_external_ep_genomic_order/external_predictions/validation_predictions.csv

python iPro-MP_predict.py \
  -i ../../outputs/benchmarks/ipromp_external_ep_genomic_order/ipromp_fasta/test.fasta \
  -s 10 \
  -o ../../outputs/benchmarks/ipromp_external_ep_genomic_order/external_predictions/test_predictions.csv
```

Official iPro-MP output is expected to include:

```text
Sequence,Prediction,Probability
```

SeqTrainer also accepts normalized prediction files with:

```text
split,sequence_id,label,probability
```

If only hard labels are available, SeqTrainer computes hard-label metrics and
records that AUROC/AUPRC and validation-threshold selection are unavailable.

After prediction files exist at the paths configured in
`config-examples/benchmarks/ipromp_external.toml`, evaluate them:

```bash
seqtrainer benchmark run config-examples/benchmarks/ipromp_external.toml
```

## Compare Completed Runs

After model runs finish:

```bash
seqtrainer benchmark compare outputs/benchmarks/* --output-dir outputs/benchmarks/comparison
```

This creates:

- `comparison_metrics.csv`
- `comparison_summary.md`

The summary ranks models by held-out test MCC first and test AUPRC second. This
ranking is only valid if every model used the same split files and selected its
threshold on validation data.

## Output Artifacts

Each completed run should contain:

- `metrics.csv`: flat split-wise metrics for quick reading
- `metrics.json`: complete split-wise metrics
- `predictions.csv`: row-level probabilities, selected threshold, and predicted labels
- `manifest.json`: dataset, split, seed, model, threshold, runtime, and git metadata
- `history.csv`: training history when training happens
- `checkpoints/`: saved model state when training happens

Skipped dependency-gated runs still write `manifest.json` and `config.json` so
the missing dependency or model-file reason is reproducible.
