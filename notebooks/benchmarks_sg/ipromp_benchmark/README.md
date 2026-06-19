# iPro-MP External Promoter Benchmark

This folder documents the iPro-MP/iPromoter external benchmark path for SeqTrainer.

iPro-MP is treated as an external model family, not as code vendored into SeqTrainer. SeqTrainer prepares the same shared train/validation/test promoter split as FASTA, records a stable row mapping, and then evaluates official iPro-MP predictions with the same metrics used for CNN-v2 and DNABERT2.

## Why iPro-MP

iPro-MP is a DNABERT-based model for prokaryotic promoter prediction across 23 species. The Genome Biology paper reports that iPro-MP uses multi-head self-attention to learn promoter sequence patterns and reports AUC greater than 0.9 in 18 of 23 species. It also emphasizes species-specific modeling, which is why the SeqTrainer config uses the E. coli K-12 MG1655 species ID when comparing against the current E. coli promoter split.

Sources:

- Paper: <https://link.springer.com/article/10.1186/s13059-025-03819-9>
- PubMed summary: <https://pubmed.ncbi.nlm.nih.gov/41083998/>
- Official code: <https://github.com/Jackie-Suv/iPro-MP>
- Pretrained models: <https://doi.org/10.5281/zenodo.15180138>

## Shared Benchmark Rules

- Use the exact same CSV split files as CNN-v2 and DNABERT2.
- Do not regenerate train/validation/test splits for iPro-MP.
- Select the threshold only on validation predictions, usually by MCC.
- Apply that validation-selected threshold unchanged to test.
- Compare models primarily by held-out test MCC and secondarily by test AUPRC.
- Do not fake metrics if official iPro-MP weights or predictions are missing.

Shared split files:

```text
data/promoter_classification/train_EP_DNA_BERT2_genomic_order.csv
data/promoter_classification/eval_EP_DNA_BERT2_genomic_order.csv
data/promoter_classification/test_EP_DNA_BERT2_genomic_order.csv
```

## Prepare iPro-MP Inputs

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

FASTA headers are deterministic:

```text
>seqtrainer|split=validation|row_index=17|sequence_id=validation_000017|label=1
```

The mapping file preserves:

```text
split,row_index,sequence_id,label,sequence
```

Official iPro-MP output joins back by exact sequence. If duplicate sequences exist, use SeqTrainer-normalized predictions with `sequence_id` to avoid ambiguous joins.

## External iPro-MP Setup

Run iPro-MP outside SeqTrainer:

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

- Download DNABERT-6 into `external/iPro-MP/DNABERT-6`.
- Download the E. coli pretrained iPro-MP model files from Zenodo.
- Do not commit large model weights.
- For `Escherichia coli str K-12 substr. MG1655`, use `species_id = 10`.
- Verify model filenames after extraction. They may follow a fold pattern such as `10_fold_1.pth` through `10_fold_5.pth`, but the exact names should be confirmed from the downloaded archive.

## Run Official Prediction

Official command shape:

```bash
python iPro-MP_predict.py -i example.fasta -s species_ID -o outputfile
```

For this benchmark:

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

The official output is expected to include:

```text
Sequence,Prediction,Probability
```

SeqTrainer also accepts normalized predictions:

```text
split,sequence_id,label,probability
```

Hard-label-only predictions are allowed, but AUROC/AUPRC and validation-threshold selection are unavailable without probabilities.

## Evaluate iPro-MP Predictions

After official prediction files exist at the configured paths:

```bash
seqtrainer benchmark run config-examples/benchmarks/ipromp_external.toml
```

Expected benchmark artifacts:

```text
metrics.csv
metrics.json
predictions.csv
manifest.json
config.json
```

If prediction files are missing, the run writes a skipped manifest and records the FASTA/mapping/script paths instead of producing fake metrics.

## Compare With CNN-v2 And DNABERT2

```bash
seqtrainer benchmark compare \
  outputs/benchmarks/cnn_v2_regularized_ep_genomic_order \
  outputs/benchmarks/dnabert2_frozen_ep_genomic_order \
  outputs/benchmarks/ipromp_external_ep_genomic_order \
  --output-dir outputs/benchmarks/comparison
```

The comparison ranks primarily by test MCC and then by test AUPRC, while preserving the rule that thresholds come from validation only.
