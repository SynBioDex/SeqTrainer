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

Latest recorded T4 inference result: validation-selected threshold `0.327886`,
held-out test MCC `0.068364`, test AUPRC `0.372180`, recall `0.234484`, and
specificity `0.823207`. The five official E. coli fold checkpoints are loaded
one at a time, each produces probabilities for the same validation/test rows,
and the final score is the arithmetic mean of those five probabilities.

## iPro-MP A100 Inference

[`ipromp_a100_colab.ipynb`](ipromp_a100_colab.ipynb) mirrors the Alpine
iPro-MP external-inference workflow on a Colab A100. It keeps the same official
E. coli model 10, five-fold ensemble, 6-mer tokenization, shared split, seed
`42`, validation-MCC threshold, and held-out test metrics.

iPro-MP is not trained in this benchmark, so epochs remain `0`. The A100 profile
only increases the inference batch size from `4` to `16`, matching the Alpine
profile and reducing wall-clock time without changing the model or scientific
comparison policy.

## Comparison Rule

The Colab notebooks keep the scientific comparison surface fixed: identical input
CSVs and labels, seed `42`, no test-set tuning, validation-selected threshold,
and held-out test MCC/AUPRC. Resource settings such as precision, physical batch
size, checkpointing, and epoch budget must be reported alongside the metrics.
