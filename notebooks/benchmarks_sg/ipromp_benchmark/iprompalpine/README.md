# Run iPro-MP On Alpine

This folder is a self-contained Alpine workflow for the official E. coli
iPro-MP five-fold ensemble. Code and environments live under `/projects`; the
job uses `/scratch/alpine` for temporary data and copies final artifacts back to
`/projects`.

## Values You Must Provide

Only two site-specific values are required:

1. Your Alpine allocation/account name, used in `--account=<YOUR_ALPINE_ALLOCATION>`.
   Replace only the text inside angle brackets. You can find the allocation in
   the Alpine portal or ask the allocation owner/mentor.
2. The directory containing the three shared benchmark CSV files, passed as
   `DATA_DIR=...` when submitting the job.

Do not replace `$USER`: Alpine expands it automatically to your login name.
Do not invent a job ID: `sbatch` prints the real ID after submission. The
Zenodo record and Hugging Face model IDs are already encoded in the setup
scripts and do not need to be entered manually.

Example placeholders used below:

```text
<YOUR_ALPINE_ALLOCATION>  -> your project allocation, for example ucb-general
<JOB_ID>                  -> the number printed by sbatch, for example 12345678
```

Before setup, load Alpine's Slurm commands and confirm that your account can
see the requested A100 partition:

```bash
module load slurm/alpine
sinfo --Format=Partition,Gres | grep aa100
```

The production job uses Alpine's supported `aa100` partition, requests one GPU
with `--gres=gpu:1`, and uses the `normal` QoS. The current 12-hour request is
within the `normal` QoS 24-hour limit. If your allocation cannot submit to
`aa100`, ask the allocation owner or CURC support before changing the script.

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

Verify the location before continuing:

```bash
ls -lh /projects/$USER/SeqTrainer/data/promoter_classification/*.csv
```

The command must show all three files. If your data is stored elsewhere, keep
it there and use that absolute directory as `DATA_DIR` during submission.

## 2. Build The Environment Once

Run setup on an `acompile` session, not on the login node. Alpine documents
`acompile` as its CPU-only environment-building partition; two cores and two
hours are sufficient for the initial package and model downloads in normal
conditions:

```bash
acompile --ntasks=2 --time=02:00:00
cd /projects/$USER/SeqTrainer
bash notebooks/benchmarks_sg/ipromp_benchmark/iprompalpine/setup_ipromp_alpine.sh
exit
```

Setup creates `/projects/$USER/seqtrainer_ipromp/env`, downloads DNABERT-6,
and selectively downloads the five E. coli checkpoints from Zenodo. It does
not download the complete 38.3 GB all-species archive.

Run setup only once unless the environment or model files are removed. Verify
the downloaded checkpoints with:

```bash
ls -lh /projects/$USER/seqtrainer_ipromp/models/ipromp_ecoli/10_fold_*.pth
ls -lh /projects/$USER/seqtrainer_ipromp/models/DNABERT-6/pytorch_model.bin
```

## 3. Submit The A100 Job

```bash
cd /projects/$USER/SeqTrainer/notebooks/benchmarks_sg/ipromp_benchmark/iprompalpine
mkdir -p logs

sbatch \
  --account=<YOUR_ALPINE_ALLOCATION> \
  --export=ALL,DATA_DIR=/projects/$USER/SeqTrainer/data/promoter_classification \
  run_ipromp_alpine.sbatch
```

Replace `<YOUR_ALPINE_ALLOCATION>` in that command with the allocation name.
For example, if the allocation is `ucb-general`, use
`--account=ucb-general`. Do not add the angle brackets to the real command.
The `--account` value is the allocation name, not your Alpine username.

On success, Alpine prints something similar to:

```text
Submitted batch job 12345678
```

Here, `12345678` is the `<JOB_ID>` used in the monitoring and result commands.

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

For the example job above, the log command would be:

```bash
tail -f logs/ipromp-12345678.out
```

Use `Ctrl+C` to stop following the log; this does not cancel the job. To see a
failure log, run `cat logs/ipromp-<JOB_ID>.err`. To cancel a submitted job, use
`scancel <JOB_ID>`.

Persistent results are copied to:

```text
/projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/
```

Verify:

```bash
cat /projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/metrics.csv
cat /projects/$USER/seqtrainer_ipromp/results/<JOB_ID>/manifest.json
```

The completed result directory should contain at least:

```text
metrics.csv
metrics.json
predictions.csv
manifest.json
external_predictions/
ipromp_fasta/
```

If setup cannot access Zenodo or Hugging Face, do not run the benchmark job
until the five `10_fold_*.pth` files and DNABERT-6 model are present in the
paths shown above. The job performs explicit file checks and exits rather than
producing incomplete metrics.

The optional `atesting_a100` partition is suitable only for a short workflow
check. It provides a 20 GB A100 MIG slice and a maximum one-hour testing job, so
it should not be used for the complete five-model benchmark or for reporting
final scientific metrics.

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

## Alpine References

- [Alpine hardware, partitions, GPUs, and QoS](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html)
- [CU DBMI Python and Anaconda workflow](https://cu-dbmi.github.io/set-website/2023/07/07/Using-Python-and-Anaconda-with-the-Alpine-HPC-Cluster.html)
- [CU DBMI Alpine Python example](https://github.com/CU-DBMI/example-hpc-alpine-python)
