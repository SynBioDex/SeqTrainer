#!/bin/bash
#SBATCH --job-name=seqtrainer-dnabert2-ft
#SBATCH --partition=aa100
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=seqtrainer-dnabert2-ft-%j.out
#SBATCH --error=seqtrainer-dnabert2-ft-%j.err
#
# Replace this with your Alpine allocation/account before submitting:
##SBATCH --account=<YOUR_ALPINE_ACCOUNT>

set -euo pipefail

echo "===== SeqTrainer DNABERT2 fine-tuning on Alpine ====="
date
hostname

module purge || true
module load anaconda || true
module load cuda || true

BUNDLE_DIR="${BUNDLE_DIR:-$SLURM_SUBMIT_DIR}"
WORK_ROOT="${WORK_ROOT:-/scratch/alpine/$USER/seqtrainer_dnabert2_work}"
REPO_URL="${REPO_URL:-https://github.com/simplyshree/SeqTrainer.git}"
BRANCH="${BRANCH:-issue-3-all-model-baselines}"

REPO_DIR="$WORK_ROOT/SeqTrainer"
ENV_DIR="$WORK_ROOT/envs/seqtrainer-dnabert2-ft"
HF_HOME_DIR="$WORK_ROOT/huggingface"
RUN_ROOT="$WORK_ROOT/runs/dnabert2_finetune_seed42"
DATA_DIR="$REPO_DIR/data/promoter_classification"
CONFIG_PATH="$REPO_DIR/config-examples/benchmarks/dnabert2_finetune.toml"

mkdir -p "$WORK_ROOT" "$WORK_ROOT/envs" "$HF_HOME_DIR" "$RUN_ROOT"

echo "Bundle dir: $BUNDLE_DIR"
echo "Work root:  $WORK_ROOT"
echo "Repo dir:   $REPO_DIR"
echo "Run root:   $RUN_ROOT"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

cd "$REPO_DIR"
echo "Repo commit:"
git rev-parse HEAD

echo "Applying DNABERT2 benchmark patch/config from bundle..."
cp "$BUNDLE_DIR/src_patch/dnabert2_benchmark.py" "$REPO_DIR/src/seqtrainer/torch/dnabert2_benchmark.py"
cp "$BUNDLE_DIR/config/dnabert2_finetune.toml" "$CONFIG_PATH"

mkdir -p "$DATA_DIR"
if [ -f "$REPO_DIR/data/data_DNABERT/promoter_classification_DNABERT.zip" ]; then
  echo "Extracting bundled promoter split zip..."
  python3 -m zipfile -e "$REPO_DIR/data/data_DNABERT/promoter_classification_DNABERT.zip" "$DATA_DIR"
elif [ -d "$BUNDLE_DIR/data/promoter_classification" ]; then
  echo "Copying promoter split CSVs from bundle data directory..."
  cp "$BUNDLE_DIR"/data/promoter_classification/*.csv "$DATA_DIR"/ 2>/dev/null || true
fi

required_files=(
  "train_EP_DNA_BERT2_genomic_order.csv"
  "eval_EP_DNA_BERT2_genomic_order.csv"
  "test_EP_DNA_BERT2_genomic_order.csv"
)

missing=0
for name in "${required_files[@]}"; do
  if [ ! -f "$DATA_DIR/$name" ]; then
    echo "Missing required split: $DATA_DIR/$name"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  echo ""
  echo "Data setup failed. Add the three shared split CSVs to:"
  echo "  $DATA_DIR"
  echo "or add them before upload under:"
  echo "  $BUNDLE_DIR/data/promoter_classification"
  exit 2
fi

echo "Split files:"
ls -lh "$DATA_DIR"/*.csv

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "Installing runtime dependencies..."
python -m pip install \
  "numpy==1.24.4" \
  "pandas==2.0.3" \
  "scikit-learn==1.3.2" \
  "torch==2.2.2" \
  "transformers==4.29.2" \
  "einops==0.6.1" \
  "packaging"

# DNABERT2's public remote code tries a FlashAttention/Triton path that caused
# Colab and local compatibility failures. This benchmark patch disables flash
# attention and uses the stable PyTorch attention path.
python -m pip uninstall -y triton flash-attn flash_attn || true

python -m pip install -e "$REPO_DIR"

export HF_HOME="$HF_HOME_DIR"
export TRANSFORMERS_CACHE="$HF_HOME_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment check:"
python - <<'PY'
import json
import sys
import torch
import transformers
import pandas
import sklearn
print(json.dumps({
    "python": sys.version,
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "pandas": pandas.__version__,
    "sklearn": sklearn.__version__,
}, indent=2))
PY

echo "Running SeqTrainer benchmark..."
seqtrainer benchmark run "$CONFIG_PATH" \
  --base-dir "$REPO_DIR" \
  --output-dir "$RUN_ROOT" \
  --strict

echo "Run finished. Artifact files:"
find "$RUN_ROOT" -maxdepth 3 -type f | sort

echo "Key metric preview:"
python - <<PY
from pathlib import Path
import pandas as pd
metrics = Path("$RUN_ROOT") / "metrics.csv"
if metrics.exists():
    print(pd.read_csv(metrics).to_string(index=False))
else:
    print("metrics.csv was not found:", metrics)
PY

date
echo "===== Done ====="
