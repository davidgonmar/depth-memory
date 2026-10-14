#!/usr/bin/env bash
# Trains one model of the paper with torchrun on every GPU of this node.
#
#   launch.sh [--gpu b200|h100] [--microbatch N] [--backend fast|flexible] SCALE MODEL LEARNING_RATE [RESUME_RUN_DIR]
#
# SCALE is 400m or 1p5b and MODEL names a config in pretrain/configs/SCALE/ (preln, attnres_block,
# attnres_full, mhc, ldm, ldm_k4). LEARNING_RATE is the peak learning rate, for example 0.001.
# The global batch is fixed; the GPU count and the microbatch only change gradient accumulation.
# --gpu sets the default microbatch (the B200 one used in the paper, or half of it on H100).
# --backend selects LDM's training backend (fast by default). With RESUME_RUN_DIR the run continues
# from its latest checkpoint.
#
# Environment: DATA_DIR (tokenized FineWeb-Edu, required), WANDB_ENTITY (required), WANDB_PROJECT
# (default ldm-pretraining-SCALE), CHECKPOINT_DIR (default ./checkpoints), SCRATCH_DIR for compiler
# caches (default $TMPDIR or /tmp), RUN_ID to end the run name (default the launch time).
set -Eeuo pipefail
usage="usage: $0 [--gpu b200|h100] [--microbatch N] [--backend fast|flexible] SCALE MODEL LEARNING_RATE [RESUME_RUN_DIR]"
gpu=b200 microbatch= backend=fast
while [[ "${1:-}" == --* ]]; do
  [[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
  case $1 in
    --gpu) gpu=$2 ;;
    --microbatch) microbatch=$2 ;;
    --backend) backend=$2 ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
  shift 2
done
[[ $# -ge 3 && $# -le 4 ]] || { echo "$usage" >&2; exit 2; }
scale=$1 model=$2 learning_rate=$3
[[ $gpu == b200 || $gpu == h100 ]] || { echo "--gpu must be b200 or h100" >&2; exit 2; }
[[ $backend == fast || $backend == flexible ]] || { echo "--backend must be fast or flexible" >&2; exit 2; }
: "${DATA_DIR:?set DATA_DIR to the tokenized FineWeb-Edu directory}"
: "${WANDB_ENTITY:?set WANDB_ENTITY to your W&B entity}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
model_config="$repo_root/paper-experiments/pretrain/configs/$scale/$model.json"
[[ -f $model_config ]] || { echo "no model config $model_config" >&2; exit 2; }

execution_args=()
if [[ $model == ldm* ]]; then
  execution_args=(--ldm-execution-config "$repo_root/paper-experiments/pretrain/configs/execution/ldm_$backend.json")
fi
if [[ -z $microbatch ]]; then
  case $scale/$model in
    400m/ldm*) microbatch=32 ;;
    400m/*) microbatch=64 ;;
    1p5b/ldm*) microbatch=16 ;;
    1p5b/*) microbatch=32 ;;
    *) echo "unknown scale $scale" >&2; exit 2 ;;
  esac
  [[ $gpu == b200 ]] || microbatch=$((microbatch / 2))
fi
[[ $microbatch =~ ^[1-9][0-9]*$ ]] || { echo "the microbatch must be a positive integer" >&2; exit 2; }

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
# The learning rate appears in the run name as 0p001 for 1e-3.
lr_tag=$(LC_ALL=C printf '%.10f' "$learning_rate" | sed 's/0*$//; s/\./p/')
variant=$model
[[ $backend == fast || $model != ldm* ]] || variant="$model-$backend"
run_name="$variant-lr$lr_tag-s42-$run_id"
save_folder="${CHECKPOINT_DIR:-$repo_root/checkpoints}/ldm-pretraining-$scale-$run_name"
resume_args=()
if [[ -n ${4:-} ]]; then
  [[ -d $4 ]] || { echo "resume directory does not exist: $4" >&2; exit 3; }
  save_folder=$(cd "$4" && pwd)
  resume_args=(--resume-run-dir "$save_folder")
else
  [[ ! -e $save_folder ]] || { echo "output already exists: $save_folder" >&2; exit 3; }
fi

runtime_root="${SCRATCH_DIR:-${TMPDIR:-/tmp}}/$run_name"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor"
export TRITON_CACHE_DIR="$runtime_root/triton"
export CUTE_DSL_CACHE_DIR="$runtime_root/cutedsl"
export TILELANG_CACHE_DIR="$runtime_root/tilelang"
export WANDB_DIR="$runtime_root/wandb"
export PYTHONUNBUFFERED=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUTE_DSL_CACHE_DIR" "$TILELANG_CACHE_DIR" "$WANDB_DIR"
cd "$repo_root"

exec torchrun --standalone --nproc-per-node=gpu paper-experiments/pretrain/train.py \
  --model-config "$model_config" --train-config "$repo_root/paper-experiments/pretrain/configs/$scale/train.json" \
  "${execution_args[@]}" \
  --learning-rate "$learning_rate" --device-microbatch-size "$microbatch" \
  --data-dir "$DATA_DIR" --run-name "$run_name" --save-folder "$save_folder" --work-dir "$runtime_root" \
  "${resume_args[@]}" \
  --wandb-project "${WANDB_PROJECT:-ldm-pretraining-$scale}" --wandb-entity "$WANDB_ENTITY"
