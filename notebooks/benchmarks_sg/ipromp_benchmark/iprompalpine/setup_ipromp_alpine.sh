#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQTRAINER_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/${USER}/seqtrainer_ipromp}"
ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/env}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
DNABERT_DIR="${DNABERT_DIR:-${MODEL_ROOT}/DNABERT-6}"
IPROMP_MODEL_DIR="${IPROMP_MODEL_DIR:-${MODEL_ROOT}/ipromp_ecoli}"

module purge
module load anaconda
eval "$(conda shell.bash hook)"

mkdir -p "${PROJECT_ROOT}" "${MODEL_ROOT}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_DIR}" python=3.10 pip
fi
conda activate "${ENV_DIR}"

conda install --yes --prefix "${ENV_DIR}" pytorch=2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
python -m pip install --upgrade "pip<25" setuptools wheel
python -m pip install \
  "transformers==4.29.2" \
  "huggingface_hub<1.0" \
  "numpy<2" \
  "pandas>=1.5,<3" \
  "scikit-learn>=1.3,<2" \
  remotezip requests rdflib sbol2 tomli
python -m pip install --no-deps -e "${SEQTRAINER_ROOT}"

python "${SCRIPT_DIR}/download_ecoli_weights.py" --output-dir "${IPROMP_MODEL_DIR}"
python - "${DNABERT_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhihan1996/DNA_bert_6",
    local_dir=sys.argv[1],
    allow_patterns=[
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
PY

python - <<'PY'
import torch
import transformers
import seqtrainer

print("SeqTrainer:", seqtrainer.__file__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Environment ready: ${ENV_DIR}"
echo "DNABERT-6 ready: ${DNABERT_DIR}"
echo "E. coli folds ready: ${IPROMP_MODEL_DIR}"
