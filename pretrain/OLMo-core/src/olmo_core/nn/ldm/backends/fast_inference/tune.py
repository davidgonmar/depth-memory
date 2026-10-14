"""Tune the launches of one model shape on this GPU and write them to its table (`tuning`).

Every measurement times one segment of `EDGES` edges of random weights (the branch as the written value,
the RMSNorm after every read but the last) by CUDA-graph replay, through `InferenceState` with explicit
launches, and checks the output against the prefill kernels at the first tiling, within a few BF16 ulps.

The prefill search runs at `PREFILL_TOKENS` tokens. It tries every tokens-per-cluster and cluster size
with two key lanes, then every key-lane count for the best. The decode search runs at every token count
of `TOKEN_COUNTS`. It tries tokens per cluster, cluster size and first-layer layout (resident, or streamed
in chunks) with the default lane splits, then every lane split for the best. The decode classes run up to
the largest token count where the decode kernel beats the prefill kernels, and consecutive token counts
with the same tiling merge into one class.

    python -m olmo_core.nn.ldm.backends.fast_inference.tune \
        --groups 16 --key-dim 8 --value-dim 128 --hidden 64
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
from pathlib import Path

import torch
from torch import nn

from . import state
from .edges import InferenceEdge
from .kernels.edge import MAX_CLUSTER, THREADS, WARPS
from .tuning import DecodeClass, DecodeTiling, Launches, PrefillTiling, shape_key, table_path, write

PREFILL_TOKENS = 16384
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
EDGES = 11
SPAN = 4
CLUSTER_TOKENS = (1, 2, 4, 8, 16)
LANES = (1, 2, 4, 8)
WEIGHT_CHUNKS = (512, 1024)           # streamed first-layer chunks, columns
MAX_ACCUMULATORS = 64                 # tokens x first-layer rows per warp, in registers
TOLERANCE_ULPS = 8                    # a candidate's output against the reference tiling's, BF16 ulps
TIE = 0.01                            # a candidate must beat the incumbent by 1 % to replace it


class _Projection(nn.Module):
    """The layout of an address network's trained parameters (`InferenceEdge.from_sites` reads it)."""

    def __init__(self, d, hidden, groups, key_dim, device):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, device=device, dtype=torch.bfloat16)
        self.w2 = nn.Linear(hidden, groups * key_dim, device=device, dtype=torch.bfloat16)
        self.gate = nn.Linear(hidden, groups, device=device, dtype=torch.bfloat16)
        self.context_norm = nn.RMSNorm(d, eps=1e-6, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            self.w2.weight.mul_(4.0)


class _Norm(nn.Module):
    def __init__(self, d, device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=torch.bfloat16))
        self.eps = 1e-6


class Bench:
    """One segment of random weights at one token count."""

    def __init__(self, groups, key_dim, value_dim, hidden, tokens):
        d = groups * value_dim
        self.sizes = groups, key_dim, value_dim, hidden
        self.edge = InferenceEdge.from_sites(
            _Projection(d, hidden, groups, key_dim, "cuda"), torch.randn(groups, key_dim, device="cuda"),
            torch.randn(groups, device="cuda") + 3.5, _Projection(d, hidden, groups, key_dim, "cuda"),
            torch.randn(groups, key_dim, device="cuda"), torch.randn(groups, device="cuda") * 0.3 + 1.0,
            groups=groups, key_dim=key_dim, eps=1e-6)
        self.norm = _Norm(d, "cuda")
        self.stream = torch.randn(tokens, d, device="cuda", dtype=torch.bfloat16)
        self.deltas = [torch.randn(tokens, d, device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(EDGES)]

    def segment(self, launches: Launches):
        groups, key_dim, value_dim, hidden = self.sizes
        segment = state.InferenceState(total_edges=EDGES, span=SPAN, groups=groups, key_dim=key_dim,
                                       value_dim=value_dim, hidden_dim=hidden, covariance_scale=0.7, eps=1e-6,
                                       context_eps=1e-6, launches=launches)
        stream = self.stream.clone()            # each step adds its branch into the stream in place
        segment.begin(stream)
        output = stream
        for step in range(EDGES):
            stream, output = segment.step(self.edge, stream, self.deltas[step],
                                          None if step == EDGES - 1 else self.norm, delta=self.deltas[step])
        return output

    def time(self, launches: Launches, replays=20, trials=5):
        """Median microseconds per edge, and the segment's output."""
        with torch.no_grad():
            self.segment(launches)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.segment(launches)
            graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(trials):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    graph.replay()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / replays / EDGES)
        return statistics.median(samples), output.clone()


class Search:
    """The fastest candidate of one bench, every candidate checked against the reference output."""

    def __init__(self, bench, reference, log, **context):
        self.bench, self.reference, self.log, self.context = bench, reference, log, context
        self.scale = reference.float().abs().max().clamp_min(1.0).item()
        self.best = None
        self.results = []

    def measure(self, launches, tiling, stage):
        try:
            us, output = self.bench.time(launches)
        except ValueError as error:               # the tiling does not fit shared memory
            self.log({**self.context, "stage": stage, "tiling": tiling._asdict(), "error": str(error)})
            return None
        error = (output.float() - self.reference.float()).abs().max().item()
        record = {**self.context, "stage": stage, "tiling": tiling._asdict(), "us_per_edge": round(us, 3),
                  "max_abs_vs_reference": error, "scale": self.scale}
        self.log(record)
        if error > TOLERANCE_ULPS * 2.0 ** -7 * self.scale:
            raise AssertionError(f"{tiling} disagrees with the reference tiling: {record}")
        self.results.append((us, tiling))
        if self.best is None or us < self.best[0] * (1 - TIE):
            self.best = (us, tiling)
        return us


def _divisors(groups):
    return [c for c in range(1, min(groups, MAX_CLUSTER) + 1) if groups % c == 0]


def _prefill_shapes(groups, key_dim):
    for tokens in CLUSTER_TOKENS:
        for cluster in _divisors(groups):
            if tokens * (groups // cluster) * key_dim <= THREADS:
                yield PrefillTiling(tokens, cluster, min(2, key_dim))


def _decode_shapes(tokens, groups, key_dim, value_dim, hidden):
    model_dim = groups * value_dim
    layouts = [(model_dim, 1)] + [(chunk, 2) for chunk in WEIGHT_CHUNKS if chunk < model_dim and model_dim % chunk == 0]
    for cluster_tokens in CLUSTER_TOKENS:
        if cluster_tokens > tokens:
            continue
        for cluster in _divisors(groups):
            rows = -(-2 * hidden // cluster)
            if (cluster_tokens * (groups // cluster) * key_dim > THREADS
                    or cluster_tokens * -(-rows // WARPS) > MAX_ACCUMULATORS):
                continue
            for chunk, stages in layouts:
                yield DecodeTiling(cluster_tokens, cluster, min(2, key_dim), 4, chunk, stages)


def tune(groups, key_dim, value_dim, hidden, log) -> Launches:
    torch.manual_seed(0)
    bench = Bench(groups, key_dim, value_dim, hidden, PREFILL_TOKENS)
    first = next(_prefill_shapes(groups, key_dim))
    _, reference = bench.time(Launches(first, ()))
    search = Search(bench, reference, log, tokens=PREFILL_TOKENS)
    for tiling in _prefill_shapes(groups, key_dim):
        search.measure(Launches(tiling, ()), tiling, "prefill")
    finalist = search.best[1]
    for key_lanes in LANES:
        if key_lanes <= key_dim and key_lanes != finalist.key_lanes:
            tiling = finalist._replace(key_lanes=key_lanes)
            search.measure(Launches(tiling, ()), tiling, "prefill lanes")
    prefill = search.best[1]
    log({"tokens": PREFILL_TOKENS, "best prefill": prefill._asdict(), "us_per_edge": round(search.best[0], 3)})

    classes = []
    for tokens in TOKEN_COUNTS:
        torch.manual_seed(tokens)
        bench = Bench(groups, key_dim, value_dim, hidden, tokens)
        prefill_us, reference = bench.time(Launches(prefill, ()))
        search = Search(bench, reference, log, tokens=tokens, prefill_us_per_edge=round(prefill_us, 3))
        for tiling in _decode_shapes(tokens, groups, key_dim, value_dim, hidden):
            search.measure(Launches(prefill, (DecodeClass(tokens, tiling),)), tiling, "shape")
        if search.best is None:
            raise ValueError(f"no decode tiling fits {shape_key(groups, key_dim, value_dim, hidden)} "
                             f"at {tokens} tokens")
        finalist = search.best[1]
        for key_lanes in LANES:
            for slices in LANES:
                tiling = finalist._replace(key_lanes=key_lanes, slices=slices)
                if tiling != finalist and key_lanes <= key_dim:
                    search.measure(Launches(prefill, (DecodeClass(tokens, tiling),)), tiling, "lanes")
        log({"tokens": tokens, "best decode": search.best[1]._asdict(), "us_per_edge": round(search.best[0], 3),
             "prefill_us_per_edge": round(prefill_us, 3)})
        if search.best[0] >= prefill_us:
            break
        classes.append(DecodeClass(tokens, search.best[1]))
    merged = []
    for entry in classes:
        if merged and merged[-1].tiling == entry.tiling:
            merged[-1] = entry
        else:
            merged.append(entry)
    return Launches(prefill, tuple(merged))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--key-dim", type=int, default=8)
    parser.add_argument("--value-dim", type=int, required=True)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--output", type=Path, help="the table file (default: tuning.table_path)")
    parser.add_argument("--log", type=Path, help="one JSON line per measurement")
    args = parser.parse_args()
    key = shape_key(args.groups, args.key_dim, args.value_dim, args.hidden)
    output = args.output or table_path(torch.cuda.get_device_capability())
    handle = open(args.log, "a") if args.log else None

    def log(record):
        line = json.dumps({"device": torch.cuda.get_device_name(), "shape": key, **record})
        print(line, flush=True)
        if handle:
            handle.write(line + "\n")
            handle.flush()

    # The rounds revisit the same tilings at every token count, so every compiled program is kept.
    state._program = functools.lru_cache(maxsize=None)(state._program.__wrapped__)
    tuned = tune(args.groups, args.key_dim, args.value_dim, args.hidden, log)
    write(output, torch.cuda.get_device_name(), key, tuned)
    log({"table": str(output), "prefill": tuned.prefill._asdict(),
         "decode": [{"max_tokens": c.max_tokens, "tiling": c.tiling._asdict()} for c in tuned.decode]})


if __name__ == "__main__":
    main()
