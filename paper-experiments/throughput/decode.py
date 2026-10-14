"""KV-cached greedy decode throughput of one arm at one Qwen3 size (paper-experiments/throughput/README.md).

One GPU, BF16 weights prepared for inference once, every transformer block compiled with
``torch.compile`` as in the prefill harness, forward under inference mode. Every arm attends with
FlashAttention 4 over a preallocated KV cache; the models are the prefill harness's (`measure.py`),
LDM on the forward-only ``fast-inference`` backend. For each context length and batch size, one
untimed repetition compiles and warms every kernel; then a prompt of ``context`` random tokens is
prefilled in chunks of PREFILL_CHUNK positions, as a server does (time to first token), ``warmup``
greedy steps run eagerly, one whole decode step (forward, greedy argmax, and the write of the next
token into the step's input) is captured in a CUDA graph and replayed once untimed, and ``steps``
graph replays are timed with CUDA events. An arm whose step cannot be captured fails.
Reported per configuration: the median time per output token, decode tokens/s, the bytes a decode
step must read (every weight except the input embedding, plus the KV cache), the fraction of the
GPU's memory bandwidth that implies, peak memory, and with ``--check`` the agreement of the last
decoded logits with a cache-free forward over the same tokens.

  python decode.py --size 1p7b --arm ldm --gpu b200 --batch 1 2 4 8 16 32 64 --context 2048 4096 8192
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# Prompt positions per prefill forward (every arm; the whole batch at once).
PREFILL_CHUNK = 512
# Peak HBM bandwidth by GPU name, bytes/s.
PEAK_BANDWIDTH = {"NVIDIA B200": 8.0e12, "NVIDIA H100 80GB HBM3": 3.35e12, "NVIDIA H100": 3.9e12}  # "NVIDIA H100": the 94 GB NVL


def harness():
    spec = importlib.util.spec_from_file_location("throughput_measure", HERE / "measure.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(measure, args, max_seq_len, device):
    from olmo_core.nn.attention import AttentionBackendName, AttentionConfig

    train = measure.recipe()
    config, raw = measure.inference_config(train, args.size, args.arm, args.gpu)
    config.apply(lambda c: setattr(c, "backend", AttentionBackendName.flash_4)
                 if isinstance(c, AttentionConfig) else None)
    parameters = measure.parameter_count(config)
    train.seed_all(train.SEED)
    model = config.build(init_device="meta")
    # Blocks compile with static shapes: one graph per block and distinct (batch, length) shape.
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 8192
    model.apply_compile()
    model.init_weights(max_seq_len=max_seq_len, device=device)
    model.eval()
    measure.prepare_inference(model, args.arm)
    return model, int(raw["vocab_size"]), parameters, config


def attention_layers(model):
    return [block.attention for block in model.blocks.values()]


def reset_cache(model, batch, max_seq_len):
    for attention in attention_layers(model):
        attention.kv_cache_manager = None
    torch.cuda.empty_cache()
    for attention in attention_layers(model):
        attention.init_kv_cache_manager(batch, max_seq_len)


def free_cache(model):
    for attention in attention_layers(model):
        attention.kv_cache_manager = None
    torch.cuda.empty_cache()


def step_bytes(model, batch, context):
    """Bytes one decode step reads at `context` cached positions: every parameter but the input
    embedding (only `batch` rows of it are gathered) and the keys and values of every layer."""
    embedding = model.embeddings.weight
    weights = sum(p.numel() * p.element_size() for p in model.parameters() if p is not embedding)
    attention = attention_layers(model)[0]
    kv = 2 * len(attention_layers(model)) * batch * context * attention.n_kv_heads * attention.head_dim * 2
    return weights + kv


def run(model, vocab, batch, context, steps, warmup, generator, timed):
    device = next(model.parameters()).device
    reset_cache(model, batch, context + warmup + 1 + steps)
    prompt = torch.randint(0, vocab, (batch, context), device=device, generator=generator)
    leftpad = torch.zeros(batch, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for position in range(0, context, PREFILL_CHUNK):
        logits = model(prompt[:, position:position + PREFILL_CHUNK], logits_to_keep=1,
                       cache_leftpad=leftpad if position == 0 else None)
    torch.cuda.synchronize()
    first_token = time.perf_counter() - start
    tokens = [prompt, logits.argmax(-1)]
    for _ in range(warmup):
        logits = model(tokens[-1], logits_to_keep=1)
        tokens.append(logits.argmax(-1))
    torch.cuda.synchronize()
    # Capture records the step without executing it: the cache position and the input advance
    # only when the graph is replayed.
    step_input = tokens[-1].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = model(step_input, logits_to_keep=1)
        step_input.copy_(logits.argmax(-1))
    # One untimed replay first: the first launch of a captured graph pays one-off costs.
    graph.replay()
    tokens.append(step_input.clone())
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps if timed else 1)]
    for begin, end in events:
        begin.record()
        graph.replay()
        end.record()
        tokens.append(step_input.clone())
    torch.cuda.synchronize()
    times = [begin.elapsed_time(end) / 1000 for begin, end in events]
    return first_token, times, torch.cat(tokens, dim=1), logits


def check(model, sequence, decoded_logits):
    """The last decoded logits against a cache-free forward over the same tokens (one forward over
    the whole sequence, which can exceed the memory the chunked decode needs)."""
    free_cache(model)
    try:
        expected = model(sequence[:, :-1], logits_to_keep=1).float()
    except torch.OutOfMemoryError:
        return {"check_out_of_memory": True}
    actual = decoded_logits.float()
    return {"check_max_abs_logit_difference": (actual - expected).abs().max().item(),
            "check_top1_agreement": (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--gpu", choices=("b200", "h100"), required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--context", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--round", type=int, default=0, help="round of the interleaved job (rounds.sh)")
    parser.add_argument("--check", action="store_true", help="compare the last step with a cache-free forward")
    parser.add_argument("--output", type=Path, help="append the JSON lines here (default results/<gpu>/decode.jsonl)")
    args = parser.parse_args()
    measure = harness()
    if args.arm not in measure.ARMS["decode"]:
        parser.error(f"--arm must be one of {measure.ARMS['decode']}")
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch._dynamo.config.suppress_errors = False
    device = torch.device("cuda", 0)
    max_seq_len = max(args.context) + args.warmup + 1 + args.steps
    model, vocab, parameters, config = build(measure, args, max_seq_len, device)
    peak = PEAK_BANDWIDTH.get(torch.cuda.get_device_name(0))
    with torch.inference_mode():
        for context in args.context:
            for batch in args.batch:
                record = measure.base_record(args, "decode", parameters, config)
                record.update({"attention": "flash_4", "cuda_graphs": True, "compiled": True, "batch": batch,
                               "context": context, "steps": args.steps, "warmup": args.warmup})
                try:
                    generator = torch.Generator(device=device).manual_seed(0)
                    run(model, vocab, batch, context, args.steps, args.warmup, generator, timed=False)
                    before, tuned_before = measure.dynamo_counters(), measure.autotuned_configs()
                    generator = torch.Generator(device=device).manual_seed(1)
                    first_token, times, sequence, logits = run(model, vocab, batch, context, args.steps,
                                                               args.warmup, generator, timed=True)
                    after, tuned_after = measure.dynamo_counters(), measure.autotuned_configs()
                except torch.OutOfMemoryError:
                    # The cell does not fit this arm on one GPU: recorded as such, and the arm goes on
                    # with the next cell.
                    record.update({"out_of_memory": True, "dynamo": measure.dynamo_counters()})
                if record.get("out_of_memory"):
                    free_cache(model)
                    measure.write_result(args, "decode", record)
                    continue
                measure.validity(times, before, after, tuned_before, tuned_after,
                                 label=f"decode {args.size} {args.arm} batch {batch} context {context}",
                                 expected=measure.expected_graph_breaks(args.arm, "decode"),
                                 # A graph replay cannot compile or tune (the counters cover that),
                                 # and a single slow replay does not move the median of 128.
                                 step_limit=None)
                per_token = statistics.median(times)
                read = step_bytes(model, batch, context + args.warmup + 1 + args.steps // 2)
                record.update({
                    "time_to_first_token_ms": first_token * 1000, "per_token_ms": per_token * 1000,
                    "per_token_ms_min": min(times) * 1000, "per_token_ms_max": max(times) * 1000,
                    "decode_tokens_per_s": batch / per_token, "step_bytes": read,
                    "bandwidth_fraction": None if peak is None else read / per_token / peak,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "dynamo": after, "autotuned_configs": tuned_after,
                })
                if args.check:
                    record.update(check(model, sequence, logits))
                measure.write_result(args, "decode", record)
                free_cache(model)


if __name__ == "__main__":
    main()
