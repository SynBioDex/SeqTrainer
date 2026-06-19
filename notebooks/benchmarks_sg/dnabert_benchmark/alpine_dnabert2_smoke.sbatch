#!/bin/bash
#SBATCH --job-name=seqtrainer-dnabert2-smoke
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load CURC-recommended modules if required for your environment.
# module purge
# module load python
# module load cuda

source .venv-dnabert2/bin/activate
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch, transformers, seqtrainer
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('cuda_device_name', torch.cuda.get_device_name(0))
print('transformers', transformers.__version__)
print('seqtrainer', seqtrainer.__file__)
PY

python -m seqtrainer.cli.main benchmark prepare-dnabert2   config-examples/benchmarks/dnabert2_smoke.toml   --base-dir .   --output-dir outputs/benchmarks/dnabert2_alpine_tokenization_smoke
