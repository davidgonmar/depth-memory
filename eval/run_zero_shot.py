from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lm_eval

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# The zero-shot suite of the paper.
PAPER_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "openbookqa",
    "lambada_openai",
    "wikitext",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot lm-eval-harness tasks on a native OLMo-core checkpoint."
    )
    parser.add_argument("checkpoint_dir", help="Checkpoint step directory or model_and_optim dir")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=PAPER_TASKS,
        help="lm-eval task names (default: the suite of the paper).",
    )
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path or HF id")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda, cuda:0, cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional maximum context length override",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Optional model dtype override at load time",
    )
    parser.add_argument("--limit", type=float, default=None, help="Optional lm-eval sample limit")
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Few-shot count. By default, honor each lm-eval task's configured value.",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Include prompts, generations, and per-sample scores in the output JSON.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save the raw lm-eval JSON result",
    )
    return parser.parse_args()


def _print_summary(results: dict[str, Any]) -> None:
    task_results = results.get("results", {})
    print(json.dumps(task_results, indent=2, ensure_ascii=False))


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in obj]
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return str(obj)


def main() -> None:
    args = parse_args()

    from olmo_lm import OLMoNativeLM

    lm = OLMoNativeLM.build(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_to_jsonable(results), indent=2, ensure_ascii=False)
        )

    _print_summary(results)


if __name__ == "__main__":
    main()
