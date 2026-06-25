#!/bin/bash
#SBATCH --job-name=dnabert2-v2
#SBATCH --account=<YOUR_ALPINE_ACCOUNT>
#SBATCH --partition=aa100
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=dnabert2-v2-%j.out
#SBATCH --error=dnabert2-v2-%j.err

set -euo pipefail

module purge
module load slurm/alpine

REPO_DIR="${REPO_DIR:-$PWD}"
# Keep a stable output path so --resume can reuse embedding caches after a
# timeout or preemption. Slurm log filenames still include the job id.
RUN_ROOT="${RUN_ROOT:-${SCRATCH:-$HOME}/seqtrainer/dnabert2_v2_hpc}"
ENV_DIR="${ENV_DIR:-${SCRATCH:-$HOME}/envs/seqtrainer-dnabert2}"

mkdir -p "$RUN_ROOT"
cd "$REPO_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv "$ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV_DIR/bin/python" -m pip install \
    torch==2.2.2 transformers==4.29.2 einops==0.6.1 \
    numpy==1.24.4 pandas==2.0.3 scikit-learn==1.3.2
  "$ENV_DIR/bin/python" -m pip install -e .
  "$ENV_DIR/bin/python" -m pip uninstall -y triton || true
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${SCRATCH:-$HOME}/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

nvidia-smi
"$ENV_DIR/bin/python" \
  notebooks/benchmarks_sg/dnabert_benchmark/dnabert2_v2_experiment.py \
  --profile hpc \
  --repo-dir "$REPO_DIR" \
  --output-dir "$RUN_ROOT" \
  --resume

echo "Run artifacts: $RUN_ROOT"
