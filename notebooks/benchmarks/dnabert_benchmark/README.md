# DNABERT2 Promoter Benchmark

This folder contains the Colab-ready DNABERT2 benchmark notebook for the shared
E. coli promoter classification split.

The purpose is to compare DNABERT2 against the CNN reference and CNN-v2 results
without changing the dataset split or evaluation policy.

## Files

- `dnabert2_shared_split_benchmark_colab.ipynb`: frozen DNABERT2 embedding
  baseline first, with an optional full fine-tuning cell.

## Dataset

Task: binary bacterial promoter prediction from fixed-length DNA sequence
windows.

Source accession: [GSE144621](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144621)

The notebook uses the same predefined split files as the CNN benchmark:

- `train_EP_DNA_BERT2_genomic_order.csv`
- `eval_EP_DNA_BERT2_genomic_order.csv`
- `test_EP_DNA_BERT2_genomic_order.csv`

The files are materialized to:

```text
data/promoter_classification/
```

The notebook extracts the bundled repo archive by default. This avoids Colab
Drive mount/fetch failures and is the most reproducible path for reviewers.

If you specifically want to use Google Drive, set `USE_GOOGLE_DRIVE = True` in
the data-preparation cell. The notebook can then copy the files from Drive or
fall back to the public Drive folder:

[AIxBio promoter classification data](https://drive.google.com/drive/folders/1rH47oJEjQjkJvHXKX_rwDjDb--dGPGx2)

## Model Order

Run the frozen DNABERT2 benchmark first.

The frozen run loads DNABERT2, extracts embeddings for the fixed split, caches
those embeddings, and trains only a small classifier head. This gives a strong
transfer-learning baseline while keeping compute lower than full fine-tuning.

Full fine-tuning is optional and GPU-heavy. Use it only after the frozen run is
working and the artifact outputs look correct.

## Metrics

Use the same benchmark metrics as CNN:

- MCC
- AUPRC
- AUROC
- accuracy
- balanced accuracy
- precision
- recall / sensitivity
- specificity
- F1
- confusion matrix

The validation split selects the classification threshold by MCC. The test split
is used only after the model and threshold are chosen.

## Colab

Open:

- [DNABERT2 shared split benchmark in Colab](https://colab.research.google.com/github/simplyshree/SeqTrainer/blob/issue-3-cnn-baseline-reproduction/notebooks/benchmarks/dnabert_benchmark/dnabert2_shared_split_benchmark_colab.ipynb)

Use a GPU runtime:

```text
Runtime > Change runtime type > GPU
```

After the branch is merged, replace `issue-3-cnn-baseline-reproduction` in the
Colab URL with `dev`.

## Output Artifacts

Frozen DNABERT2 writes:

```text
outputs/benchmarks/dnabert2_frozen_colab/
```

Optional fine-tuning writes:

```text
outputs/benchmarks/dnabert2_finetune_colab/
```

Each completed run should include:

- `metrics.csv`
- `metrics.json`
- `predictions.csv`
- `manifest.json`
- `history.csv`
- `checkpoints/best_model.pt`
- `embeddings/*.pt` for the frozen embedding run

Compare these artifacts against the CNN benchmark artifacts, prioritizing test
MCC and test AUPRC after validation-MCC model selection.
