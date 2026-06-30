#!/usr/bin/env bash
#SBATCH --job-name=seqtrainer-ipromp
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ipromp-%j.out
#SBATCH --error=logs/ipromp-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQTRAINER_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config/ipromp_external.toml"
PROJECT_ROOT="${PROJECT_ROOT:-/projects/${USER}/seqtrainer_ipromp}"
ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/env}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
DNABERT_DIR="${DNABERT_DIR:-${MODEL_ROOT}/DNABERT-6}"
IPROMP_MODEL_DIR="${IPROMP_MODEL_DIR:-${MODEL_ROOT}/ipromp_ecoli}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the folder containing the three shared benchmark CSV files}"
WORK_ROOT="${WORK_ROOT:-/scratch/alpine/${USER}/seqtrainer_ipromp/${SLURM_JOB_ID}}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-/projects/${USER}/seqtrainer_ipromp/results/${SLURM_JOB_ID}}"
RUN_NAME="ipromp_external_ep_genomic_order"
RUN_DIR="${WORK_ROOT}/outputs/benchmarks/${RUN_NAME}"

module purge
module load anaconda
eval "$(conda shell.bash hook)"
conda activate "${ENV_DIR}"

mkdir -p "${WORK_ROOT}/data/promoter_classification" "${FINAL_OUTPUT_DIR}"
for filename in \
  train_EP_DNA_BERT2_genomic_order.csv \
  eval_EP_DNA_BERT2_genomic_order.csv \
  test_EP_DNA_BERT2_genomic_order.csv; do
  test -f "${DATA_DIR}/${filename}" || { echo "Missing ${DATA_DIR}/${filename}" >&2; exit 2; }
  cp "${DATA_DIR}/${filename}" "${WORK_ROOT}/data/promoter_classification/${filename}"
done

for fold in 1 2 3 4 5; do
  test -f "${IPROMP_MODEL_DIR}/10_fold_${fold}.pth" || { echo "Missing E. coli fold ${fold}" >&2; exit 2; }
done
test -f "${DNABERT_DIR}/pytorch_model.bin" || { echo "Missing DNABERT-6 model" >&2; exit 2; }

echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
nvidia-smi
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"

cd "${WORK_ROOT}"
seqtrainer benchmark prepare-ipromp "${CONFIG_PATH}" --base-dir "${WORK_ROOT}"

for split in train validation test; do
  python -m seqtrainer.adapters.ipromp_inference \
    --input-fasta "${RUN_DIR}/ipromp_fasta/${split}.fasta" \
    --output-csv "${RUN_DIR}/external_predictions/${split}_predictions.csv" \
    --split "${split}" \
    --dnabert-dir "${DNABERT_DIR}" \
    --model-dir "${IPROMP_MODEL_DIR}" \
    --species-id 10 \
    --kmer-size 6 \
    --max-length 300 \
    --batch-size "${BATCH_SIZE:-16}" \
    --seed 42 \
    --device cuda
done

seqtrainer benchmark run "${CONFIG_PATH}" --base-dir "${WORK_ROOT}" --strict
cp -a "${RUN_DIR}/." "${FINAL_OUTPUT_DIR}/"

echo "Completed iPro-MP benchmark"
echo "Persistent results: ${FINAL_OUTPUT_DIR}"
