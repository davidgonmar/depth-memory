"""Training-step and prefill throughput of one arm at one Qwen3 size (protocol in README.md).

Training: torchrun on every GPU of one node, the recipe's train-step semantics (FSDP, FP32
master parameters and optimizer state, BF16 compute, FP32 reduction, compiled model, gradient
clipping, AdamW with the recipe's weight-decay groups), at the size's common microbatch
(TRAIN_MICROBATCH). The recipe's sequence length by default; at a longer one the microbatch and the
global batch shrink by the same factor, so the tokens per microbatch and per step stay the recipe's.
Prefill: one GPU, BF16 weights prepared for inference once, compiled forward under inference mode,
the last position's logits, at one sequence length (the recipe's by default) and that length's
batches (PREFILL_BATCHES). Fixed synthetic tokens. One process measures one arm in one round;
rounds.sh interleaves the arms of a size in rounds on one node or one GPU. Every measurement appends
one JSON line, and a measurement that fails a validity check (validity()) raises instead of writing
a number.

  torchrun --standalone --nproc-per-node=8 measure.py --mode train --size 1p7b --arm ldm --gpu b200
  torchrun --standalone --nproc-per-node=8 measure.py --mode train --size 1p7b --arm ldm --gpu b200 --sequence-length 4096
  python measure.py --mode prefill --size 1p7b --arm ldm --gpu b200 --sequence-length 4096
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent / "configs"
RESULTS = Path(__file__).resolve().parent / "results"
SIZES = ("1p7b", "4b", "8b", "14b")
# LDM trains on the fast backend and is evaluated on the fast-inference backend at every size.
LDM_EXECUTION = ROOT / "paper-experiments/pretrain/configs/execution/ldm_fast.json"
METHOD = {"preln": "preln", "ldm": "ldm", "mhc": "mhc", "mhc_upstream": "mhc", "mhc_no_recompute": "mhc",
          "attnres_block": "attnres", "attnres_full": "attnres"}
ARMS = {
    "train": ("preln", "ldm", "mhc", "mhc_upstream", "mhc_no_recompute", "attnres_block", "attnres_full"),
    "prefill": ("preln", "ldm", "mhc", "mhc_upstream", "attnres_block", "attnres_full"),
    "decode": ("preln", "ldm", "mhc", "mhc_upstream", "attnres_block", "attnres_full"),
}
IMPLEMENTATION = {
    "preln": "OLMo-core pre-norm transformer",
    "ldm": "LDM: fast backend (training), fast-inference backend (prefill, decode), tuned launch tables",
    "mhc": "mHC: DeepSeek TileKernels, cuBLAS projection backward, streams recomputed per 6 layers, "
           "projection split per width and token count from the tuned table",
    "mhc_upstream": "mHC: DeepSeek TileKernels as released (TileLang projection backward, one-split decode "
                    "projection), streams recomputed per 6 layers",
    "mhc_no_recompute": "mHC: DeepSeek TileKernels, cuBLAS projection backward, every stream kept",
    "attnres_block": "AttnRes, blocks of 8: FLA Triton kernels",
    "attnres_full": "AttnRes, full: FLA Triton kernels",
}
# Device microbatch of the training measurement, per GPU type and size: the largest power of two at
# which every training arm of the size fits at the recipe's sequence length.
# Every arm of a size runs at this microbatch; a longer sequence divides it by the length's factor.
TRAIN_MICROBATCH = {"b200": {"1p7b": 16, "4b": 8, "8b": 8, "14b": 4}}
# Prefill: at every length, powers of two from batch 1 up to PREFILL_SMALL_TOKENS tokens per forward,
# then the largest batch at which every prefill arm of the size fits. PREFILL_LARGEST holds that batch at
# the first length; memory follows the tokens per forward, so a longer length takes
# the same tokens in fewer sequences, and each batch of the ladder keeps its tokens across lengths.
PREFILL_LENGTHS = (2048, 4096, 8192)
PREFILL_SMALL_TOKENS = 32_768
PREFILL_LARGEST = {"b200": {"1p7b": 256, "4b": 128, "8b": 128, "14b": 64},
                   "h100": {"1p7b": 64, "4b": 32, "8b": 32, "14b": 16}}


def prefill_batches(largest: int, length: int) -> tuple[int, ...]:
    small = {2**i for i in range((PREFILL_SMALL_TOKENS // length).bit_length())}
    return tuple(sorted(small | {largest * PREFILL_LENGTHS[0] // length}))


PREFILL_BATCHES = {gpu: {size: {length: prefill_batches(largest, length) for length in PREFILL_LENGTHS}
                         for size, largest in sizes.items()}
                   for gpu, sizes in PREFILL_LARGEST.items()}
for path in (ROOT, ROOT / "pretrain/OLMo-core/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def recipe():
    spec = importlib.util.spec_from_file_location("paper_train", ROOT / "paper-experiments/pretrain/train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_config(train, size, arm, gpu, *, inference):
    path = CONFIGS / size / f"{arm}.json"
    raw = train.read_json(path)
    method = METHOD[arm]
    config = train.build_model(path, raw, method=method, tokenizer_vocab_size=int(raw["vocab_size"]),
                               ldm_execution_path=LDM_EXECUTION if method == "ldm" else None)
    if method == "ldm" and inference:
        config.block.ldm.execution.backend.name = "fast-inference"
    train.configure_compiler(config)
    return config, raw


def inference_config(train, size, arm, gpu):
    """The prefill and decode model: BF16 weights the way the generation module loads a checkpoint
    (every config with a dtype); FP32 control parameters (mHC controllers, LDM forgetting logits)
    keep their own dtype."""
    config, raw = model_config(train, size, arm, gpu, inference=True)
    config.apply(lambda c: setattr(c, "dtype", train.DType.bfloat16) if hasattr(c, "dtype") else None)
    return config, raw


def prepare_inference(model, arm):
    """What a loaded checkpoint goes through once before inference: LDM turns its parameter modules
    into the tensors the fast-inference kernels read, mHC folds each controller's RMS weight into
    its projection, rounded to TF32. The other arms have nothing to prepare."""
    if METHOD[arm] == "ldm":
        from olmo_core.nn.ldm.backends.fast_inference.backend import prepare_inference as prepare

        prepare(model)
    elif METHOD[arm] == "mhc":
        from olmo_core.nn.hyper_connections import prepare_inference as prepare

        prepare(model)


def parameter_count(config):
    return sum(p.numel() for p in config.build(init_device="meta").parameters())


def dynamo_counters():
    counters = torch._dynamo.utils.counters
    return {"graph_breaks": int(sum(counters["graph_break"].values())),
            "graph_break_reasons": {reason[:300]: int(n) for reason, n in counters["graph_break"].items()},
            "recompiles": int(sum(counters["recompiles"].values())),
            "unique_graphs": int(counters["stats"]["unique_graphs"]),
            "frames_ok": int(counters["frames"]["ok"]), "frames_total": int(counters["frames"]["total"])}


def autotuned_configs():
    """Configurations held by every Triton autotuner in the process (FLA's kernels autotune at the
    first call of a shape); a change across the measured steps means tuning ran inside them."""
    from triton.runtime.autotuner import Autotuner

    # type() rather than isinstance: the heap holds dead weak proxies, which isinstance dereferences.
    return sum(len(obj.cache) for obj in gc.get_objects() if issubclass(type(obj), Autotuner))


def expected_graph_breaks(arm, mode):
    """The functions behind the graph breaks an arm has by design; any other break fails the
    measurement. Decode, every arm: OLMo-core calls FlashAttention 4 (the only KV-cache attention)
    through `dispatch_flash_attn_4`, which compiled graphs call as an opaque function
    (torch.compiler.disable). Prefill and decode, LDM: the fast-inference block calls its edge
    kernels through `Segment.read`, an opaque function as well, twice per block. Inside a CUDA graph
    neither costs a launch."""
    expected = ("dispatch_flash_attn_4",) if mode == "decode" else ()
    if METHOD[arm] == "ldm" and mode in ("prefill", "decode"):
        expected += ("Segment.read",)
    return expected


def validity(times, before, after, tuned_before, tuned_after, *, label, expected=(), step_limit=1.5):
    """The checks a measurement passes before its number is written (README, "Validity"). `expected`
    holds the functions of the arm's graph breaks by design (expected_graph_breaks); a measured step
    over `step_limit` times the median fails (None: no limit)."""
    problems = []
    unexpected = {reason: n for reason, n in after["graph_break_reasons"].items()
                  if not any(part in reason for part in expected)}
    if unexpected:
        problems.append(f"{sum(unexpected.values())} unexpected Dynamo graph breaks: {unexpected}")
    if after["unique_graphs"] != before["unique_graphs"] or after["recompiles"] != before["recompiles"]:
        problems.append(f"compilation inside the measured steps: {before} then {after}")
    if tuned_after != tuned_before:
        problems.append(f"Triton autotuning inside the measured steps ({tuned_before}, then {tuned_after} "
                        "configurations)")
    median = statistics.median(times)
    if step_limit is not None and max(times) > step_limit * median:
        problems.append(f"a measured step took {max(times) / median:.2f} times the median: {times}")
    if problems:
        raise RuntimeError(f"{label}: invalid measurement\n  " + "\n  ".join(problems))


def base_record(args, mode, parameters, config):
    return {"mode": mode, "size": args.size, "arm": args.arm, "implementation": IMPLEMENTATION[args.arm],
            "gpu": args.gpu, "round": args.round,
            "job": os.environ.get("THROUGHPUT_JOB"),
            "parameters": parameters, "attention": str(config.block.sequence_mixer.backend or "torch"),
            "vocab_size": config.vocab_size}


def write_result(args, mode, record):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True)
    record["commit"] = commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    record["device"] = torch.cuda.get_device_name(0)
    output = args.output or RESULTS / args.gpu / f"{mode}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print("RESULT " + json.dumps(record), flush=True)


def clip_gradients(model, max_norm):
    """The recipe's gradient clipping (TransformerTrainModule._clip_grad_norm under FSDP)."""
    from torch.distributed.tensor import DTensor

    parameters = list(model.parameters())
    total = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None], norm_type=2.0,
                                          error_if_nonfinite=False)
    if isinstance(total, DTensor):
        total = total.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total)


def build_train(args, accumulation=None):
    """The recipe's training model and step on this rank. `accumulation` overrides the number of
    microbatches per step."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    train = recipe()
    config, raw = model_config(train, args.size, args.arm, args.gpu, inference=False)
    sequence_length = args.sequence_length or train.SEQUENCE_LENGTH
    fixed = TRAIN_MICROBATCH.get(args.gpu, {}).get(args.size)
    if fixed is not None:
        # A longer sequence shrinks the microbatch by the same factor, so every length trains on the
        # recipe's tokens per microbatch and per step, and only attention's share of the work grows.
        if fixed * train.SEQUENCE_LENGTH % sequence_length:
            raise ValueError(f"{sequence_length}-token sequences do not divide the {args.size} microbatch of "
                             f"{fixed} x {train.SEQUENCE_LENGTH} tokens")
        fixed = fixed * train.SEQUENCE_LENGTH // sequence_length
    microbatch = args.microbatch or fixed
    if microbatch is None:
        raise ValueError(f"no default microbatch for {args.size} on {args.gpu}: pass --microbatch")
    global_sequences = train.GLOBAL_BATCH_TOKENS // sequence_length
    if train.GLOBAL_BATCH_TOKENS % sequence_length or global_sequences % (world * microbatch):
        raise ValueError("the global batch must split evenly into rank microbatches")
    accumulation = accumulation or global_sequences // (world * microbatch)
    parameters = parameter_count(config)
    train.seed_all(train.SEED)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    model = config.build(init_device="meta")
    optim_config = train.AdamWConfig(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                                     group_overrides=train.mhc_weight_decay_override(METHOD[args.arm]))
    train.configure_weight_decay(model, optim_config)
    model.apply_compile()
    model.apply_fsdp(dp_mesh=mesh, param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                     wrapping_strategy=train.TransformerDataParallelWrappingStrategy.full, prefetch_factor=1)
    model.init_weights(max_seq_len=sequence_length, max_local_microbatch_size=microbatch * sequence_length,
                       device=device, world_mesh=mesh)
    model.train()
    optimizer = optim_config.build(model)
    generator = torch.Generator(device=device).manual_seed(train.SEED + local_rank)
    tokens = torch.randint(0, int(raw["vocab_size"]), (microbatch, sequence_length), device=device,
                           generator=generator)
    global_tokens = train.GLOBAL_BATCH_TOKENS

    def step():
        for index in range(accumulation):
            # As the recipe's train module: FSDP waits for every gradient reduction only after the
            # last microbatch's backward.
            model.set_is_last_backward(index == accumulation - 1)
            out = model(tokens, labels=tokens, loss_reduction="sum", loss_div_factor=global_tokens, return_logits=False)
            out.loss.backward()
        clip_gradients(model, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    fields = {"gpus": world, "microbatch": microbatch, "microbatch_rule": microbatch == fixed,
              "accumulation": accumulation, "sequence_length": sequence_length,
              "global_tokens": world * accumulation * microbatch * sequence_length}
    return model, step, parameters, config, fields


def train_mode(args):
    import torch.distributed as dist

    model, step, parameters, config, fields = build_train(args)
    # As the recipe's trainer (OLMo-core's default GarbageCollectorCallback): no automatic garbage
    # collection while training; otherwise a collection lands inside some measured steps.
    gc.collect()
    gc.disable()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before, tuned_before = dynamo_counters(), autotuned_configs()
    times = []
    for _ in range(args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    after, tuned_after = dynamo_counters(), autotuned_configs()
    # The step time is the slowest rank's (the barrier aligns the starts); every rank checks its own counters.
    slowest = torch.tensor(times, device="cuda")
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    times = slowest.tolist()
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device="cuda")
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    validity(times, before, after, tuned_before, tuned_after, label=f"train {args.size} {args.arm}",
             expected=expected_graph_breaks(args.arm, "train"))
    median = statistics.median(times)
    if dist.get_rank() == 0:
        record = base_record(args, "train", parameters, config)
        record.update(fields)
        record.update({
            "step_ms": median * 1000, "step_ms_all": [t * 1000 for t in times],
            "tokens_per_s_per_gpu": fields["global_tokens"] / median / fields["gpus"],
            "peak_memory_gib": peak.item(), "steps": args.steps, "warmup": args.warmup, "dynamo": after,
            "autotuned_configs": tuned_after})
        write_result(args, "train", record)
    dist.destroy_process_group()


def build_prefill(args):
    """The prefill model on one GPU, prepared for inference, for sequences of args.sequence_length."""
    device = torch.device("cuda", 0)
    train = recipe()
    config, raw = inference_config(train, args.size, args.arm, args.gpu)
    parameters = parameter_count(config)
    train.seed_all(train.SEED)
    model = config.build(init_device="meta")
    # Every batch of the ladder is a distinct shape, compiled once during its warm-up forwards.
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 8192
    model.apply_compile()
    model.init_weights(max_seq_len=args.sequence_length, device=device)
    model.eval()
    prepare_inference(model, args.arm)
    return model, parameters, config, int(raw["vocab_size"])


def prefill_mode(args):
    model, parameters, config, vocab = build_prefill(args)
    device = torch.device("cuda", 0)
    for batch in args.batch:
        generator = torch.Generator(device=device).manual_seed(0)
        tokens = torch.randint(0, vocab, (batch, args.sequence_length), device=device, generator=generator)
        record = base_record(args, "prefill", parameters, config)
        record.update({"gpus": 1, "batch": batch, "sequence_length": args.sequence_length})
        with torch.inference_mode():
            # A server's prefill needs only the last position's logits, which start the decode.
            for _ in range(args.warmup):
                model(tokens, logits_to_keep=1)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before, tuned_before = dynamo_counters(), autotuned_configs()
            times = []
            for _ in range(args.steps):
                start = time.perf_counter()
                model(tokens, logits_to_keep=1)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            after, tuned_after = dynamo_counters(), autotuned_configs()
        validity(times, before, after, tuned_before, tuned_after,
                 label=f"prefill {args.size} {args.arm} batch {batch}",
                 expected=expected_graph_breaks(args.arm, "prefill"))
        median = statistics.median(times)
        record.update({
            "forward_ms": median * 1000, "forward_ms_all": [t * 1000 for t in times],
            "tokens_per_s": tokens.numel() / median, "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            "steps": args.steps, "warmup": args.warmup, "dynamo": after, "autotuned_configs": tuned_after})
        write_result(args, "prefill", record)
        del tokens
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "prefill"), required=True)
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--gpu", choices=("b200", "h100"), required=True)
    parser.add_argument("--microbatch", type=int, help="train: device microbatch (default: TRAIN_MICROBATCH)")
    parser.add_argument("--sequence-length", type=int,
                        help="tokens per sequence (default: the recipe's); training keeps the recipe's tokens per "
                             "microbatch and per step")
    parser.add_argument("--batch", type=int, nargs="+",
                        help="prefill: sequences per forward (default: the size's PREFILL_BATCHES at the length)")
    parser.add_argument("--steps", type=int, default=5, help="measured steps (train) or forwards (prefill)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--round", type=int, default=0, help="round of the interleaved job (rounds.sh)")
    parser.add_argument("--output", type=Path, help="append the JSON lines here (default results/<gpu>/<mode>.jsonl)")
    args = parser.parse_args()
    if args.arm not in ARMS[args.mode]:
        parser.error(f"--arm must be one of {ARMS[args.mode]} in {args.mode} mode")
    args.sequence_length = args.sequence_length or recipe().SEQUENCE_LENGTH
    if args.mode == "prefill":
        if args.batch is None:
            batches = PREFILL_BATCHES.get(args.gpu, {}).get(args.size, {}).get(args.sequence_length)
            if batches is None:
                parser.error(f"no default prefill batches for {args.size} at {args.sequence_length} tokens on "
                             f"{args.gpu}: pass --batch")
            args.batch = list(batches)
    # A compiled arm that exceeds Dynamo's recompile budget or fails to compile raises instead of
    # running eager.
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.config.suppress_errors = False
    if args.mode == "train":
        train_mode(args)
    else:
        prefill_mode(args)


if __name__ == "__main__":
    main()
