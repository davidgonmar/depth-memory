#!/usr/bin/env bash
# One interleaved measurement job: every arm of MODE at SIZE, in ROUNDS rounds, each arm a fresh
# process, all on this node (train, every GPU) or on one GPU (prefill, decode). The arm order is
# reversed every other round, so a drift of the GPU's speed over the job does not favour the arms
# that run late.
#
#   rounds.sh MODE SIZE GPU ROUNDS [--length L] [--output FILE]
#
# MODE is train, prefill or decode; SIZE is 1p7b, 4b, 8b or 14b; GPU is b200 or h100 (it selects
# the microbatch and batch tables of measure.py). --length L is the training or prefill sequence
# length, or the decode context. Without it, training runs the recipe's length, prefill every length
# of measure.PREFILL_BATCHES and decode every context of decode.py. Results are appended to
# results/GPU/MODE.jsonl, or to FILE. Every line records the commit, so the job refuses to start on a
# tree with uncommitted changes to tracked files. A failing arm does not stop the others; the job
# exits nonzero if any arm failed. The mHC kernels are compiled by TileLang and need nvcc on PATH.
set -Euo pipefail
(( $# >= 4 )) || { echo "usage: $0 MODE SIZE GPU ROUNDS [--length L] [--output FILE]" >&2; exit 2; }
mode=$1 size=$2 gpu=$3 rounds=$4
shift 4
length="" output=()
while (( $# )); do
  case $1 in
    --length) length=$2; shift 2 ;;
    --output) output=(--output "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# The compile caches are shared by the rounds of a job.
scratch="${SCRATCH_DIR:-${TMPDIR:-/tmp}}/ldm-throughput-$mode-$size"
mkdir -p "$scratch"
export TRITON_CACHE_DIR="$scratch/triton" TORCHINDUCTOR_CACHE_DIR="$scratch/inductor"
export TILELANG_CACHE_DIR="$scratch/tilelang" CUTE_DSL_CACHE_DIR="$scratch/cute" CUDA_CACHE_PATH="$scratch/cuda"
export PYTHONPATH="$repo_root/pretrain/OLMo-core/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export THROUGHPUT_JOB="${THROUGHPUT_JOB:-$(date +%Y%m%d-%H%M%S)-$mode-$size}"
cd "$repo_root"
[[ -z "$(git status --porcelain --untracked-files=no -- . ':!paper-experiments/throughput/results')" ]] \
  || { echo "the tree has uncommitted changes to tracked files" >&2; exit 2; }
python=${PYTHON:-python}
here=paper-experiments/throughput
measure() { $python -c "import sys; sys.path.insert(0, '$here'); import measure; $1"; }
arms=$(measure "print(*measure.ARMS['$mode'])") || exit 2
lengths=("$length")   # train and decode: one process per arm
if [[ $mode == prefill && -z $length ]]; then
  lengths=($(measure "print(*measure.PREFILL_BATCHES['$gpu']['$size'])")) || exit 2
fi
echo "[rounds $(date '+%F %T')] $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l): $mode $size $gpu${length:+ at $length}, commit $(git rev-parse --short HEAD), job $THROUGHPUT_JOB, $rounds rounds of: $arms"
failed=()
reversed=$(printf '%s\n' $arms | tac | tr '\n' ' ')
for round in $(seq 1 "$rounds"); do
  order=$arms
  (( round % 2 == 0 )) && order=$reversed
  for length in "${lengths[@]}"; do
    for arm in $order; do
      echo "[rounds $(date '+%T')] round $round $arm${length:+ at $length}"
      case $mode in
        train) command=($python -m torch.distributed.run --standalone --nproc-per-node=gpu $here/measure.py --mode train
                        ${length:+--sequence-length $length}) ;;
        prefill) command=($python $here/measure.py --mode prefill --sequence-length "$length") ;;
        decode) command=($python $here/decode.py ${length:+--context $length}) ;;
        *) echo "unknown mode $mode" >&2; exit 2 ;;
      esac
      "${command[@]}" --size "$size" --arm "$arm" --gpu "$gpu" --round "$round" "${output[@]}" || failed+=("round $round $arm${length:+ $length}")
    done
  done
done
if (( ${#failed[@]} )); then
  printf '[rounds] FAILED: %s\n' "${failed[@]}"
  exit 1
fi
echo "[rounds $(date '+%T')] done"
