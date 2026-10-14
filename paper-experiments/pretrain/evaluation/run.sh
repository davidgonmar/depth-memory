#!/usr/bin/env bash
# Evaluates one native checkpoint zero-shot with the LM Evaluation Harness on one GPU: ARC-Easy,
# ARC-Challenge, HellaSwag, PIQA, WinoGrande, OpenBookQA, LAMBADA and WikiText.
#
#   run.sh CHECKPOINT OUTPUT_JSON
#
# CHECKPOINT is a step directory of a training run (for example .../step47684). The architecture
# is read from the run's config.json. Environment: EVAL_CACHE_DIR for the dataset cache (default
# .cache/eval in the repository), SCRATCH_DIR for compiler caches (default $TMPDIR or /tmp),
# EVAL_TOKENIZER to override the tokenizer saved with the run.
set -Eeuo pipefail
[[ $# == 2 ]] || { echo "usage: $0 CHECKPOINT OUTPUT_JSON" >&2; exit 2; }
checkpoint=$1 output=$2
for marker in config.json .metadata.json model_and_optim/.metadata; do
  [[ -f $checkpoint/$marker ]] || { echo "incomplete checkpoint: $checkpoint/$marker" >&2; exit 3; }
done
[[ ! -e $output ]] || { echo "results already exist: $output" >&2; exit 3; }
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
runtime_root="${SCRATCH_DIR:-${TMPDIR:-/tmp}}/ldm-eval-$$"
cache_root="${EVAL_CACHE_DIR:-$repo_root/.cache/eval}"
mkdir -p "$runtime_root" "$(dirname "$output")" "$cache_root"
export PYTHONPATH="$repo_root/pretrain/OLMo-core/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$cache_root/huggingface" HF_DATASETS_CACHE="$cache_root/datasets"
export TRITON_CACHE_DIR="$runtime_root/triton" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TILELANG_CACHE_DIR="$runtime_root/tilelang"
cd "$repo_root"
exec python eval/run_zero_shot.py "$checkpoint" \
  --tasks arc_easy arc_challenge hellaswag piqa winogrande openbookqa lambada_openai wikitext \
  --num-fewshot 0 --batch-size 32 --max-length 2048 --device cuda:0 --dtype bfloat16 \
  ${EVAL_TOKENIZER:+--tokenizer "$EVAL_TOKENIZER"} \
  --output-path "$output"
