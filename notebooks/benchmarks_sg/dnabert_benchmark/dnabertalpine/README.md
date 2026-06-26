# DNABERT2 Alpine HPC Run

This folder contains the Alpine HPC version of the SeqTrainer DNABERT2 benchmark. Use this when Colab is too slow, times out, or fails with CUDA/Triton/FlashAttention errors.

The goal is to run DNABERT2 full fine-tuning in a way that is comparable to the CNN-v2 benchmark:

- same train/eval/test CSV files
- same seed: `42`
- same held-out test set
- threshold selected on validation MCC only
- final comparison by held-out test MCC and AUPRC
- same benchmark artifact style: `metrics.csv`, `metrics.json`, `predictions.csv`, `manifest.json`, `history.csv`, `checkpoints/`

## Folder Contents

```text
dnabertalpine/
  README.md
  run_dnabert2_finetune_alpine.sbatch
  config/
    dnabert2_finetune.toml
  src_patch/
    dnabert2_benchmark.py
  data/
    promoter_classification/
```

What each file does:

- `run_dnabert2_finetune_alpine.sbatch`: the Slurm job script you submit on Alpine.
- `config/dnabert2_finetune.toml`: the benchmark settings: model, seed, split files, epochs, learning rate, batch size, metrics.
- `src_patch/dnabert2_benchmark.py`: the Python runner that actually loads DNABERT2, trains/evaluates it, and writes artifacts.
- `data/promoter_classification/`: optional place to put the three CSV split files if the repo zip is not available on Alpine.

## Where The Python Code Is

The Python model/training code is here in this folder:

```text
dnabertalpine/src_patch/dnabert2_benchmark.py
```

When the Slurm script runs, it clones SeqTrainer on Alpine and copies that file into the real package location:

```text
SeqTrainer/src/seqtrainer/torch/dnabert2_benchmark.py
```

So the `.sbatch` file does not contain the model logic. It only prepares the HPC environment and launches the benchmark. The actual DNABERT2 logic is in `dnabert2_benchmark.py`, controlled by `dnabert2_finetune.toml`.

## Data Files Expected

The run expects the same three CSV split files used for the CNN benchmark:

```text
train_EP_DNA_BERT2_genomic_order.csv
eval_EP_DNA_BERT2_genomic_order.csv
test_EP_DNA_BERT2_genomic_order.csv
```

The script first tries to extract them from:

```text
SeqTrainer/data/data_DNABERT/promoter_classification_DNABERT.zip
```

If that zip is not available, put the three CSVs here before uploading this folder to Alpine:

```text
dnabertalpine/data/promoter_classification/
```

During the job, the files are copied/extracted into:

```text
/scratch/alpine/$USER/seqtrainer_dnabert2_work/SeqTrainer/data/promoter_classification/
```

## How To Run On Alpine

### 1. Upload This Folder

Upload the whole `dnabertalpine/` folder to Alpine, for example:

```bash
/scratch/alpine/$USER/dnabertalpine
```

You can use Alpine OnDemand, Globus, or another approved transfer method.

### 2. Go To The Folder

```bash
cd /scratch/alpine/$USER/dnabertalpine
```

### 3. Edit The Account Line

Open the Slurm script:

```bash
nano run_dnabert2_finetune_alpine.sbatch
```

Find:

```bash
##SBATCH --account=<YOUR_ALPINE_ACCOUNT>
```

If your Alpine allocation requires an account, change it to:

```bash
#SBATCH --account=your_account_name
```

If your allocation does not require it, leave it commented.

### 4. Submit The Job

```bash
sbatch run_dnabert2_finetune_alpine.sbatch
```

### 5. Check Job Status

```bash
squeue -u $USER
```

### 6. Watch The Log

Replace `<JOBID>` with the job id printed by `sbatch`:

```bash
tail -f seqtrainer-dnabert2-ft-<JOBID>.out
```

If the job fails:

```bash
cat seqtrainer-dnabert2-ft-<JOBID>.err
```

## What The Script Does

The script will:

1. create a working directory under `/scratch/alpine/$USER/seqtrainer_dnabert2_work`
2. clone `https://github.com/simplyshree/SeqTrainer.git`
3. check out branch `issue-3-all-model-baselines`
4. copy `src_patch/dnabert2_benchmark.py` into `SeqTrainer/src/seqtrainer/torch/dnabert2_benchmark.py`
5. copy `config/dnabert2_finetune.toml` into `SeqTrainer/config-examples/benchmarks/dnabert2_finetune.toml`
6. locate or extract the three shared CSV split files
7. create a Python virtual environment
8. install PyTorch, Transformers, pandas, scikit-learn, and SeqTrainer
9. run:

```bash
seqtrainer benchmark run config-examples/benchmarks/dnabert2_finetune.toml \
  --base-dir /scratch/alpine/$USER/seqtrainer_dnabert2_work/SeqTrainer \
  --output-dir /scratch/alpine/$USER/seqtrainer_dnabert2_work/runs/dnabert2_finetune_seed42 \
  --strict
```

## Resource Request

The script currently requests:

```bash
#SBATCH --partition=aa100
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
```

This is intended for a real DNABERT2 fine-tuning run, not a quick smoke test.

For a short test, reduce the config epochs and time limit. Do not use smoke-test metrics as final scientific results.

## Output Location

After the run, outputs should be here:

```text
/scratch/alpine/$USER/seqtrainer_dnabert2_work/runs/dnabert2_finetune_seed42/
```

Expected files:

```text
metrics.csv
metrics.json
predictions.csv
manifest.json
history.csv
checkpoints/
```

## What To Compare Against CNN-v2

Use these values for fair comparison:

- test MCC: primary decision metric
- test AUPRC: secondary decision metric
- validation-selected threshold: must come only from validation, not test
- same train/eval/test split files
- same seed: `42`

The scientific question is:

```text
Does DNABERT2 improve held-out test MCC/AUPRC over CNN-v2 enough to justify the extra compute?
```

## Current Status

This folder is a run bundle. It prepares Alpine and runs the DNABERT2 benchmark. Final DNABERT2 performance should only be reported after the Alpine job completes and the artifact files are available.
