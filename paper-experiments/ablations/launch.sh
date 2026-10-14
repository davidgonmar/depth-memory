#!/usr/bin/env bash
# Trains one 400M ablation with torchrun on every GPU of this node: 15B tokens (14,305 steps) at
# learning rate 1e-3, LDM on the flexible backend.
#
#   launch.sh VARIANT [b200|h100] [RESUME_RUN_DIR]
#
# VARIANT is a config in ablations/configs/400m/ (written by generate_configs.py), or preln for the
# PreLN row. The microbatch is 32 on B200 and 16 on H100, which gives four microbatches per step on
# 4 B200 or 8 H100 GPUs. Environment: as for pretrain/launch.sh; WANDB_PROJECT defaults to
# ldm-ablations-400m.
set -Eeuo pipefail
[[ $# -ge 1 && $# -le 3 ]] || { echo "usage: $0 VARIANT [b200|h100] [RESUME_RUN_DIR]" >&2; exit 2; }
variant=$1 gpu=${2:-b200}
learning_rate=0.001
case $gpu in
  b200) microbatch=32 ;;
  h100) microbatch=16 ;;
  *) echo "the GPU must be b200 or h100, got $gpu" >&2; exit 2 ;;
esac
: "${DATA_DIR:?set DATA_DIR to the tokenized FineWeb-Edu directory}"
: "${WANDB_ENTITY:?set WANDB_ENTITY to your W&B entity}"

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $variant == preln ]]; then
  model_config="$repo_root/paper-experiments/pretrain/configs/400m/preln.json"
  execution_args=()
else
  model_config="$repo_root/paper-experiments/ablations/configs/400m/$variant.json"
  execution_args=(--ldm-execution-config "$repo_root/paper-experiments/pretrain/configs/execution/ldm_flexible.json")
fi
[[ -f $model_config ]] || { echo "unknown ablation variant: $variant ($model_config missing)" >&2; exit 2; }

run_id=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
run_name="abl-$variant-15b-lr${learning_rate//./p}-s42-$run_id"
save_folder="${CHECKPOINT_DIR:-$repo_root/checkpoints}/ldm-ablations-400m-$run_name"
resume_args=()
if [[ -n ${3:-} ]]; then
  [[ -d $3 ]] || { echo "resume directory does not exist: $3" >&2; exit 3; }
  save_folder=$(cd "$3" && pwd)
  resume_args=(--resume-run-dir "$save_folder")
else
  [[ ! -e $save_folder ]] || { echo "output already exists: $save_folder" >&2; exit 3; }
fi

runtime_root="${SCRATCH_DIR:-${TMPDIR:-/tmp}}/$run_name"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/torchinductor"
export TRITON_CACHE_DIR="$runtime_root/triton"
export WANDB_DIR="$runtime_root/wandb"
export PYTHONUNBUFFERED=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR"
cd "$repo_root"

exec torchrun --standalone --nproc-per-node=gpu paper-experiments/pretrain/train.py \
  --model-config "$model_config" --train-config "$repo_root/paper-experiments/ablations/configs/train.json" \
  "${execution_args[@]}" \
  --learning-rate "$learning_rate" --device-microbatch-size "$microbatch" \
  --data-dir "$DATA_DIR" --run-name "$run_name" --save-folder "$save_folder" --work-dir "$runtime_root" \
  "${resume_args[@]}" \
  --wandb-project "${WANDB_PROJECT:-ldm-ablations-400m}" --wandb-entity "$WANDB_ENTITY"
