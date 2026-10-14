"""Tune the split count of the mHC pre projection on this GPU and write it to
tuned/sm<arch>.json, the table `pre_big_fuse.projection_splits` reads.

For each width and token count (decode batches up to the largest prefill of
paper-experiments/throughput), every split count the kernel accepts runs the no-grad pre path as a model prepared for
inference calls it (`mhc_pre_big_fuse` with the projection rounded at load: the projection kernel,
then the fused kernel). The calls are timed in a CUDA graph that cycles through distinct
projections, enough (with the streams) to exceed twice the L2 cache, so each call reads its
projection from memory as each layer of a decode step does; the time per call is the median over
replays. Each token count gets the fewest splits within 1 % of the fastest. The table lists the
token counts at which the choice changes; above the largest measured count its choice applies.

    python -m olmo_core.kernels.tile_kernels.modeling.mhc.ops.tune_projection --hidden 2048 [--streams 4]
        [--records FILE.jsonl] [--table FILE.json]
With --records and no --hidden, the table is written from the measurements in that file (for
widths measured in parallel on several GPUs).
"""
import argparse
import json
import os
import statistics

import torch

from olmo_core.kernels.tile_kernels.mhc.norm_fn_kernel import round_to_tf32
from olmo_core.kernels.tile_kernels.modeling.mhc.ops.pre_big_fuse import (
    clear_table_cache,
    mhc_pre_big_fuse,
    table_path,
    valid_splits,
)

TOKENS = (1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
COLD_BYTES = 256 << 20
REPLAYS = 30
TIE = 0.01


def pre(residual, fn, scale, base, splits):
    return mhc_pre_big_fuse(residual, fn, scale, base, rms_eps=1e-6, mhc_pre_eps=1e-6, mhc_sinkhorn_eps=1e-6,
                            mhc_post_mult_value=2.0, sinkhorn_repeat=20, n_splits=splits, fn_rounded=True)


def time_per_call(residual, projections, scale, base, splits):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for fn in projections[:3]:
            pre(residual, fn, scale, base, splits)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in projections:
            pre(residual, fn, scale, base, splits)
    graph.replay()
    graph.replay()
    times = []
    for _ in range(REPLAYS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / len(projections))
    del graph
    return statistics.median(times)


def measure(streams, hidden):
    device = torch.device("cuda")
    mult3 = streams * (streams + 2)
    reduction = streams * hidden
    generator = torch.Generator(device=device).manual_seed(0)
    projections = [round_to_tf32(torch.randn(mult3, reduction, device=device, generator=generator) * 0.02)
                   for _ in range(max(4, COLD_BYTES // (mult3 * reduction * 4)))]
    scale = torch.full((3,), 0.8, device=device)
    base = torch.randn(mult3, device=device, generator=generator) * 0.2
    splits_to_try = valid_splits(reduction)
    records = []
    for tokens in TOKENS:
        residual = torch.randn(tokens, streams, hidden, device=device, generator=generator, dtype=torch.bfloat16)
        count = max(4, COLD_BYTES // (mult3 * reduction * 4 + residual.numel() * 2))
        one_split = pre(residual, projections[0], scale, base, 1)
        times = {}
        for splits in splits_to_try:
            outputs = pre(residual, projections[0], scale, base, splits)
            difference = max((a.float() - b.float()).abs().max().item() for a, b in zip(outputs, one_split))
            times[splits] = time_per_call(residual, projections[:count], scale, base, splits)
            record = dict(gpu=torch.cuda.get_device_name(device), streams=streams, hidden=hidden, tokens=tokens,
                          splits=splits, us=round(times[splits], 3), projections=count,
                          max_difference_to_one_split=difference)
            print(json.dumps(record), flush=True)
            records.append(record)
        print(f"hidden {hidden} tokens {tokens}: " + " ".join(f"{s}:{t:.2f}" for s, t in times.items()), flush=True)
        torch.cuda.empty_cache()
    return records


def table_entries(records):
    """Per width, [token count, splits] pairs: at each token count the fewest splits within TIE of the
    fastest, keeping the largest count of each run of equal choices (a forward takes the entry of the
    smallest listed count at or above its token count)."""
    times = {}
    for record in records:
        times.setdefault((record["streams"], record["hidden"]), {}).setdefault(record["tokens"], {})[
            record["splits"]] = record["us"]
    entries = {}
    for (streams, hidden), per_count in sorted(times.items()):
        best = {}
        for tokens, per_split in per_count.items():
            fastest = min(per_split.values())
            best[tokens] = min(s for s, t in per_split.items() if t <= fastest * (1 + TIE))
        counts = sorted(best)
        entries[f"{streams}x{hidden}"] = {"splits": [
            [tokens, best[tokens]] for i, tokens in enumerate(counts)
            if i == len(counts) - 1 or best[counts[i + 1]] != best[tokens]]}
    return entries


def write_table(path, records):
    gpus = sorted({record["gpu"] for record in records})
    if len(gpus) != 1:
        raise ValueError(f"records from more than one GPU type: {gpus}")
    table = {"__meta__": {"gpu": gpus[0]}}
    if os.path.exists(path):
        with open(path) as handle:
            table = json.load(handle)
        if table["__meta__"]["gpu"] != gpus[0]:
            raise ValueError(f"{path} was tuned on {table['__meta__']['gpu']}, the records on {gpus[0]}")
    entries = table_entries(records)
    table.update(entries)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(dict(sorted(table.items())), handle, indent=1)
        handle.write("\n")
    clear_table_cache()
    for key, entry in entries.items():
        print(f"{path}: {key} {entry['splits']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", type=int, nargs="+", help="widths to measure")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--records", help="append every measurement to this JSON-lines file")
    parser.add_argument("--table", help="table to update (default: this GPU's table next to the op)")
    args = parser.parse_args()
    if args.hidden:
        records = [record for hidden in args.hidden for record in measure(args.streams, hidden)]
        if args.records:
            with open(args.records, "a") as handle:
                handle.writelines(json.dumps(record) + "\n" for record in records)
    elif args.records:
        with open(args.records) as handle:
            records = [json.loads(line) for line in handle]
    else:
        parser.error("give --hidden to measure, or --records to write the table from earlier measurements")
    write_table(args.table or table_path(torch.device("cuda")), records)


if __name__ == "__main__":
    main()
