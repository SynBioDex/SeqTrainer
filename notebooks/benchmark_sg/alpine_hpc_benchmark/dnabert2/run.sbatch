#!/bin/bash
#SBATCH --job-name=seqtrainer-dnabert2
#SBATCH --partition=alpine-gpu
#SBATCH --qos=normal
#SBATCH --account=ACCOUNT_ID
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=seqtrainer-dnabert2-%j.out
#SBATCH --error=seqtrainer-dnabert2-%j.err

set -euo pipefail

# Edit these two paths for your Alpine project allocation and bundle location.
BUNDLE_DIR="${BUNDLE_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-${TMPDIR:-/scratch/$USER}/seqtrainer_dnabert2_${SLURM_JOB_ID}}"
IMAGE="$BUNDLE_DIR/image/seqtrainer-alpine-gpu.sif"
REPO="$BUNDLE_DIR/repository/SeqTrainer"
MODELS="$BUNDLE_DIR/models"
DATA="$BUNDLE_DIR/data"
CONFIG="/repo/notebooks/benchmark_sg/alpine_hpc_benchmark/config/dnabert2_finetune.toml"

mkdir -p "$WORK_ROOT/outputs"
python "$BUNDLE_DIR/tools/validate_offline_bundle.py" --bundle-root "$BUNDLE_DIR"

echo "===== SeqTrainer DNABERT2 Alpine benchmark ====="
date
hostname
echo "Bundle: $BUNDLE_DIR"
echo "Work:   $WORK_ROOT"

apptainer exec --nv \
  --bind "$REPO:/repo:ro" \
  --bind "$MODELS:/models:ro" \
  --bind "$DATA:/data:ro" \
  --bind "$WORK_ROOT:/work" \
  "$IMAGE" \
  python -m seqtrainer.cli.main benchmark run "$CONFIG" \
    --base-dir / \
    --output-dir /work/outputs/dnabert2 \
    --strict

mkdir -p "$BUNDLE_DIR/runs/dnabert2"
cp -a "$WORK_ROOT/outputs/dnabert2/." "$BUNDLE_DIR/runs/dnabert2/"
echo "Completed outputs: $BUNDLE_DIR/runs/dnabert2"
