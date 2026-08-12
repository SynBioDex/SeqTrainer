# SeqTrainer Alpine HPC benchmark bundle

This folder contains only the Alpine execution layer for the model baselines.
It is intentionally separate from the repository's shared benchmark runners and
does not change the default workflow or model comparison semantics.

The bundle is designed to be added to `issue-3-all-model-baselines`. It uses the
existing SeqTrainer benchmark CLI and the same contract as the CNN benchmark:

- identical predefined `train`, `validation`, and `test` CSV files;
- seed `42` and binary labels (`0` background, `1` promoter);
- threshold selected on validation MCC only;
- final test reporting uses that frozen validation threshold;
- metrics include accuracy, balanced accuracy, precision, recall, F1, MCC,
  AUROC, AUPRC, sensitivity, specificity, and confusion counts;
- outputs include `metrics.csv`, `metrics.json`, `predictions.csv`,
  `manifest.json`, `history.csv` for training, and `checkpoints/`.

## Bundle layout

Prepare a private Alpine bundle with this layout. Large data and model files are
not committed to Git.

```text
alpine_bundle/
  image/seqtrainer-alpine-gpu.sif
  repository/SeqTrainer/
  data/promoter_classification/
    train_EP_DNA_BERT2_genomic_order.csv
    eval_EP_DNA_BERT2_genomic_order.csv
    test_EP_DNA_BERT2_genomic_order.csv
  models/DNABERT-2-117M/
  models/DNABERT-6/
  models/ipromp_ecoli/
    10_fold_1.pth ... 10_fold_5.pth
  manifests/
```

The repository directory must be the same commit used for the run. Copy the
branch checkout into `repository/SeqTrainer`, then run the bundle validator:

```bash
python tools/prepare_offline_bundle.py --repo-dir /path/to/SeqTrainer --bundle-root /path/to/alpine_bundle
python tools/validate_offline_bundle.py --bundle-root /path/to/alpine_bundle
```

The preparation script creates directories and a `SHA256SUMS` file. It does not
download data or model weights. Stage those files from the approved internal
Drive/Zenodo/Hugging Face downloads before submitting jobs.

## Build the container

On a machine with Apptainer/Singularity and internet access, build once:

```bash
apptainer build image/seqtrainer-alpine-gpu.sif container/seqtrainer_alpine_gpu.def
```

Copy the resulting image into the bundle's `image/` directory. The runtime
sets Hugging Face offline variables and exposes `/repo/src` through
`PYTHONPATH`, so no package installation occurs on the compute node.

## Submit on Alpine

From the bundle directory, edit only the paths at the top of each job script if
your bundle is elsewhere. Submit the two model workflows separately:

```bash
sbatch dnabert2/run.sbatch
sbatch ipromp/run.sbatch
```

Replace `<JOBID>` with the numeric ID printed by `sbatch`:

```bash
tail -f seqtrainer-dnabert2-<JOBID>.out
tail -f seqtrainer-ipromp-<JOBID>.out
squeue -j <JOBID>
```

The DNABERT2 job is full fine-tuning: seed 42, maximum sequence token length
104, batch size 4, gradient accumulation 8 (effective batch size 32), AdamW,
learning rate `2e-5`, weight decay `0.01`, five epochs, warmup ratio `0.10`,
early stopping patience 2, gradient checkpointing, and bf16 when supported.
The checkpoint and model are selected using validation MCC; the test set is
never used for selection.

iPro-MP is inference-only. It has no epochs or learning rate: the official
E. coli species-10 model is an ensemble of five pretrained fold checkpoints.
The job runs all five folds, averages positive-class probabilities, and then
lets the shared benchmark runner select the validation MCC threshold. Its
batch size is 16 and its token limit is 128.

## Outputs

Completed output folders are copied back under the job's `RUN_ROOT`:

```text
runs/dnabert2/metrics.csv
runs/dnabert2/metrics.json
runs/dnabert2/predictions.csv
runs/dnabert2/manifest.json
runs/dnabert2/history.csv
runs/dnabert2/checkpoints/
runs/ipromp/metrics.csv
runs/ipromp/metrics.json
runs/ipromp/predictions.csv
runs/ipromp/manifest.json
runs/ipromp/ipromp_fasta/
runs/ipromp/external_predictions/
```

Compare completed folders with the existing command from the repository root:

```bash
python -m seqtrainer.cli.main benchmark compare runs/dnabert2 runs/ipromp --output-dir runs/comparison
```

This folder is an Alpine submission bundle only. It does not replace the
shared CNN/DNABERT2/iPro-MP implementations under `src/`.
