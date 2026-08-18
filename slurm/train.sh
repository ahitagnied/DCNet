#!/bin/bash
# Train DCNet on Roboflow COCO under GOLLUM_DATA_ROOT (default /scratch/ad158/gollum).
#
#   cd ~/DCNet
#   sbatch slurm/train.sh
#   GOLLUM_DATA_ROOT=/scratch/ad158/gollum-v2 sbatch --export=ALL --job-name=dcnet-v2 slurm/train.sh
#
#SBATCH --partition=commons
#SBATCH --account=av21
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=dcnet-ft
# Override GPU type at submit time, e.g.:
#   sbatch --gres=gpu:h100:1 slurm/train.sh
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

module load GCCcore/13.3.0
module load CUDA/12.4.0 2>/dev/null || module load CUDA/12.6.0

VENV="${VENV:-/scratch/ad158/gollum/dcnet-venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export GOLLUM_DATA_ROOT="${GOLLUM_DATA_ROOT:-/scratch/ad158/gollum}"
DATA_TAG="$(basename "$GOLLUM_DATA_ROOT")"
OUT_DIR="${OUT_DIR:-${GOLLUM_DATA_ROOT}/output/${DATA_TAG}_r50-${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

export TORCH_HOME="${TORCH_HOME:-/scratch/ad158/gollum/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/ad158/gollum/.cache}"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

# Weights & Biases -> https://wandb.ai/ahitagnied-rice-university/gollum-dcnet
export WANDB_ENTITY="${WANDB_ENTITY:-ahitagnied-rice-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-gollum-dcnet}"
export WANDB_NAME="${WANDB_NAME:-${DATA_TAG}-r50-${SLURM_JOB_ID}}"
export WANDB_DIR="${WANDB_DIR:-/scratch/ad158/gollum/.cache/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

NUM_GPUS=1

echo "host=$(hostname) job=${SLURM_JOB_ID} gpus=${CUDA_VISIBLE_DEVICES:-} n=${NUM_GPUS}"
echo "data=${GOLLUM_DATA_ROOT}"
echo "out=${OUT_DIR}"
nvidia-smi -L

if ! python -c "import MultiScaleDeformableAttention" >/dev/null 2>&1; then
  echo "Building PCD CUDA ops..."
  (
    cd "$ROOT/dcnet/modeling/PCD/ops"
    rm -rf build dist *.egg-info
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" python setup.py build install
  )
fi

python -c "import detectron2, torch, MultiScaleDeformableAttention; print('ok', torch.__version__, detectron2.__version__, 'cuda', torch.cuda.is_available())"

WEIGHTS="${WEIGHTS:-detectron2://ImageNetPretrained/torchvision/R-50.pkl}"

python train_net.py \
  --config-file configs/CIS-R50.yaml \
  --num-gpus "$NUM_GPUS" \
  OUTPUT_DIR "$OUT_DIR" \
  MODEL.WEIGHTS "$WEIGHTS" \
  "$@"
