#!/bin/bash
#SBATCH --job-name=seqtrainer-ipromp
#SBATCH --partition=alpine-gpu
#SBATCH --qos=normal
#SBATCH --account=ACCOUNT_ID
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=seqtrainer-ipromp-%j.out
#SBATCH --error=seqtrainer-ipromp-%j.err

set -euo pipefail

# Edit these two paths for your Alpine project allocation and bundle location.
BUNDLE_DIR="${BUNDLE_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-${TMPDIR:-/scratch/$USER}/seqtrainer_ipromp_${SLURM_JOB_ID}}"
IMAGE="$BUNDLE_DIR/image/seqtrainer-alpine-gpu.sif"
REPO="$BUNDLE_DIR/repository/SeqTrainer"
MODELS="$BUNDLE_DIR/models"
DATA="$BUNDLE_DIR/data"
CONFIG="/repo/notebooks/benchmark_sg/alpine_hpc_benchmark/config/ipromp_external.toml"

mkdir -p "$WORK_ROOT/outputs" "$WORK_ROOT/external_predictions"
python "$BUNDLE_DIR/tools/validate_offline_bundle.py" --bundle-root "$BUNDLE_DIR"

echo "===== SeqTrainer iPro-MP Alpine benchmark ====="
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
  python -m seqtrainer.cli.main benchmark prepare-ipromp "$CONFIG" \
    --base-dir / \
    --output-dir /work/outputs/ipromp

for split in train validation test; do
  apptainer exec --nv \
    --bind "$REPO:/repo:ro" \
    --bind "$MODELS:/models:ro" \
    --bind "$DATA:/data:ro" \
    --bind "$WORK_ROOT:/work" \
    "$IMAGE" \
    python -m seqtrainer.adapters.ipromp_inference \
      --input-fasta "/work/outputs/ipromp/ipromp_fasta/${split}.fasta" \
      --output-csv "/work/external_predictions/${split}_predictions.csv" \
      --split "$split" \
      --dnabert-dir /models/DNABERT-6 \
      --model-dir /models/ipromp_ecoli \
      --species-id 10 \
      --kmer-size 6 \
      --max-length 128 \
      --batch-size 16 \
      --seed 42 \
      --device cuda
done

apptainer exec --nv \
  --bind "$REPO:/repo:ro" \
  --bind "$MODELS:/models:ro" \
  --bind "$DATA:/data:ro" \
  --bind "$WORK_ROOT:/work" \
  "$IMAGE" \
  python -m seqtrainer.cli.main benchmark run "$CONFIG" \
    --base-dir / \
    --output-dir /work/outputs/ipromp \
    --strict

mkdir -p "$BUNDLE_DIR/runs/ipromp"
cp -a "$WORK_ROOT/outputs/ipromp/." "$BUNDLE_DIR/runs/ipromp/"
cp -a "$WORK_ROOT/external_predictions" "$BUNDLE_DIR/runs/ipromp/"
echo "Completed outputs: $BUNDLE_DIR/runs/ipromp"
