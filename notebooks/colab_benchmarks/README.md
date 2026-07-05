# Colab Benchmarks

This folder contains reproducible Google Colab entry points for SeqTrainer
benchmarks that need hosted accelerators.

## DNABERT2 Full Fine-Tuning

[`dnabert2_finetune_a100_colab.ipynb`](dnabert2_finetune_a100_colab.ipynb)
mirrors the DNABERT2 Alpine full-fine-tuning experiment on a Colab A100.

The notebook keeps the scientific contract fixed:

- the same predefined train, validation, and test CSV files as CNN-v2;
- seed `42`;
- full DNABERT2 encoder fine-tuning;
- AdamW at `3e-5` with `0.1` warmup and `0.01` weight decay;
- physical batch size `4`, gradient accumulation `8`, effective batch size `32`;
- four maximum epochs with validation-MCC early stopping;
- validation-only threshold selection and final held-out test reporting;
- the same metrics and benchmark artifact format as the Alpine run.

The notebook requires an A100 because the canonical configuration uses BF16.
Dataset files are read from Google Drive, while model downloads and caches use
Colab-local storage. Checkpoints and final artifacts are written back to Drive.

The notebook pins the SeqTrainer implementation commit used by the Alpine
bundle, keeping the Colab and Alpine executions tied to the same tested model
runner and configuration.

## DNABERT2 T4 Profile

[`dnabert2_finetune_t4_colab.ipynb`](dnabert2_finetune_t4_colab.ipynb)
runs the same pinned DNABERT2-117M backbone, shared split, seed, optimizer,
learning rate, pooling, validation-only threshold policy, metrics, and artifact
schema on a 16 GB T4.

The T4-specific changes are explicit in
[`config/dnabert2_finetune_t4.toml`](config/dnabert2_finetune_t4.toml):

- FP16 instead of A100 BF16;
- physical batch size `2` with gradient accumulation `16`, preserving effective
  batch size `32`;
- activation checkpointing to reduce memory;
- two maximum epochs instead of four, with validation-MCC early stopping.

This is a resource-constrained candidate, not a numerically identical replay of
the A100 run. Use it to obtain a real full-dataset T4 result, then confirm a
promising configuration with the canonical A100 or Alpine profile.

## iPro-MP T4 Inference

[`ipromp_t4_colab.ipynb`](ipromp_t4_colab.ipynb) evaluates the same official
five-fold E. coli iPro-MP ensemble as the Alpine workflow. iPro-MP is
inference-only here, so reducing epochs does not apply. T4 memory and runtime
are controlled by:

- loading one official fold checkpoint at a time;
- inference batch size `4`;
- running validation and test by default;
- saving each completed split to Drive so a disconnected session can resume.

Train-split inference is optional because it is not used for threshold selection
or final held-out comparison. The five folds, 6-mer tokenization, sequence
coverage, validation-MCC threshold, test metrics, and output schema are not
reduced. Model weights require roughly 1.8 GB for the five iPro-MP checkpoints,
plus the DNABERT-6 backbone.

## Comparison Rule

The T4 notebooks keep the scientific comparison surface fixed: identical input
CSVs and labels, seed `42`, no test-set tuning, validation-selected threshold,
and held-out test MCC/AUPRC. Resource settings such as precision, physical batch
size, checkpointing, and epoch budget must be reported alongside the metrics.
