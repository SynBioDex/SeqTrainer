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
A T4 fallback is intentionally not included: changing precision or batch
behavior would make it a different execution profile. Dataset files are read
from Google Drive, while model downloads and caches use Colab-local storage.
Checkpoints and final artifacts are written back to Drive.

The notebook pins the SeqTrainer implementation commit used by the Alpine
bundle, keeping the Colab and Alpine executions tied to the same tested model
runner and configuration.
