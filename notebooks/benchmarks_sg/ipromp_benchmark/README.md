# iPro-MP Benchmark

This benchmark evaluates the official iPro-MP E. coli model on the same
GSE144621 train, validation, and test CSV files used by the SeqTrainer CNN.
The pretrained model is used for inference only. SeqTrainer selects one
probability threshold from the validation split by MCC and applies that fixed
threshold to the held-out test split.

## Current Result Status

| Model/run | Test MCC | Test AUPRC | Status |
| --- | ---: | ---: | --- |
| CNN-v2, 50 cycles | 0.220884 | 0.645976 | Current benchmark to beat |
| DNABERT2 full fine-tuning, Colab T4 | 0.147631 | 0.365169 | Completed workflow check |
| iPro-MP E. coli model 10 / five-fold ensemble, Colab T4 | 0.068364 | 0.372180 | Completed pretrained inference check |

Conclusion: **iPro-MP has a completed Colab T4 inference result, but it does not
yet beat CNN-v2**. The run selected threshold `0.327886` on validation MCC and
reported held-out test MCC `0.068364` and test AUPRC `0.372180`. Treat this as a
resource-constrained pretrained-inference check. The final same-split claim
should be rerun with the explicit `AIxBio/Promoter Classification/Data` path and
then confirmed on the Alpine/A100 workflow.

## What Is Reproduced

- Model: iPro-MP DNABERT-6 classifier for species ID 10, E. coli K-12 MG1655.
- Tokenization: overlapping 6-mers.
- Ensemble: five official E. coli fold checkpoints, each evaluated on the same
  SeqTrainer validation/test sequences and averaged by positive-class
  probability. These are iPro-MP's pretrained fold models, not new SeqTrainer
  train/validation/test folds.
- Seed: 42.
- Input sequence length: the shared CSV sequences remain unchanged at 300 bp.
- Model token limit: 300 tokens, so all overlapping 6-mers from each shared
  300 bp sequence are retained.
- Primary comparison metric: test MCC; secondary metric: test AUPRC.
- All metrics: accuracy, balanced accuracy, precision, recall/sensitivity,
  specificity, F1, MCC, AUROC, AUPRC, and confusion-matrix counts.

The official paper trained species-specific DNABERT models using 6-mers and
fivefold cross-validation. Its 128-token default was designed for 81 bp source
windows. SeqTrainer raises the limit to 300 for this dataset so the shared
window is not silently truncated. The official prediction code averages five
fold models at a fixed 0.5 threshold. SeqTrainer preserves those probabilities
but replaces the final decision threshold with the validation-MCC threshold so
the comparison follows the same policy as CNN-v2 without tuning on test data.

### How The Five Folds Are Used

For the current external-inference baseline, SeqTrainer does not retrain or
refit the five iPro-MP folds. The notebook/script loads `10_fold_1.pth` through
`10_fold_5.pth` one at a time to reduce GPU memory. Each fold scores every
sequence in the validation and test FASTA files. The five fold probabilities are
then averaged per sequence to produce one ensemble probability. Validation MCC
selects one threshold from the averaged validation probabilities; that same
threshold is applied unchanged to the averaged test probabilities.

## Why A Wrapper Is Needed

The upstream prediction script assumes `./DNABERT-6` and `./models`, omits the
`BertModel` import, drops FASTA IDs, and joins output paths to a fixed
`Predict_Results` directory. SeqTrainer's wrapper keeps the official model
architecture and checkpoint values while making paths explicit, preserving
stable IDs, loading folds sequentially to reduce GPU memory, and writing the
shared benchmark artifact format.

## Alpine Bundle

The runnable files are in [`iprompalpine`](iprompalpine/README.md):

- `setup_ipromp_alpine.sh` creates the pinned environment and downloads models.
- `download_ecoli_weights.py` range-downloads only the five E. coli checkpoints.
- `run_ipromp_alpine.sbatch` runs inference and shared evaluation on an A100.
- `config/ipromp_external.toml` records the fixed scientific settings.

Model weights are not committed. The official Zenodo ZIP is 38.3 GB and holds
all 23 species. The selective downloader retrieves only `10_fold_1.pth` through
`10_fold_5.pth`, approximately 1.8 GB total.

## Outputs

A completed run contains:

```text
metrics.csv
metrics.json
predictions.csv
manifest.json
ipromp_id_mapping.csv
external_predictions/
ipromp_fasta/
```

The fold-inference CSVs also have adjacent `*.metadata.json` files containing
runtime, device, seed, model paths, and inference settings.

## Sources

- [iPro-MP paper](https://link.springer.com/article/10.1186/s13059-025-03819-9)
- [Official iPro-MP repository](https://github.com/Jackie-Suv/iPro-MP)
- [Official model record](https://doi.org/10.5281/zenodo.15180139)
- [Alpine hardware documentation](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
