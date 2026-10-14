"""Offline tuning of the state kernels' launch schedules at one training shape on this GPU.

    python -m olmo_core.nn.ldm.backends.fast.tune --tokens 65536 --groups 16 \\
        --key-dim 8 --value-dim 128 --edges 57

``--tokens`` is the per-GPU micro-batch in tokens (sequences x length) and ``--edges`` the model's
state edges (2 * layers + 1). ``--value-dim`` is the group width; the kernels run a width that is
not a multiple of 8 at its padded width, which is the width the table's keys name. One forward
and backward of a SegmentState of that shape runs on synthetic inputs. At every launch shape it
measures each candidate schedule of schedules.candidates: a schedule the kernel would not run as named is skipped, every other one
must reproduce the default schedule's outputs bit for bit, and the fastest (median of CUDA-event
timings, operands restored before each run) is written to tuned/<arch>.json. ``--kernels`` limits
the sweep to some of the three kernels, for instance after only one of them changed; the others
keep their table entries. ``--running-covariance-dtype float32`` tunes the forward of a model that
sets that execution setting; its launches have their own table keys. The backward kernels do not
depend on it, so ``--kernels forward`` is enough when the shape's backward entries exist.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import statistics
import time

import torch

from olmo_core.nn.ldm import state_step_op

from . import schedules, state


def _inputs(args, generator):
    shape = (args.tokens, args.groups)
    per_edge = []
    for _ in range(args.edges):
        key = torch.nn.functional.normalize(torch.randn(*shape, args.key_dim, device="cuda", generator=generator), dim=-1)
        query = torch.nn.functional.normalize(torch.randn(*shape, args.key_dim, device="cuda", generator=generator), dim=-1)
        value = (0.5 * torch.randn(*shape, args.value_dim, device="cuda", generator=generator)).bfloat16()
        scale = 0.8 + 0.4 * torch.rand(*shape, device="cuda", generator=generator)
        forgetting = 0.98 + 0.015 * torch.rand(args.groups, device="cuda", generator=generator)
        per_edge.append([t.requires_grad_() for t in (key, query, value, scale, forgetting)])
    cotangent = (0.1 * torch.randn(*shape, args.value_dim, device="cuda", generator=generator)).bfloat16()
    return per_edge, cotangent


def _one_pass(args, per_edge, cotangent):
    segment_state = state.SegmentState(total_edges=args.edges, span=args.span, groups=args.groups,
                                       key_dim=args.key_dim, value_dim=args.value_dim,
                                       running_covariance_dtype=args.running_covariance_dtype)
    state_step_op.set_active_state(segment_state)
    token = segment_state.new_token(per_edge[0][2])
    reads = []
    for tensors in per_edge:
        read, token = segment_state.step(token, *tensors)
        reads.append(read)
    torch.autograd.backward(reads, [cotangent] * len(reads))
    torch.cuda.synchronize()


def _resident_ctas(program, threads, shared_bytes):
    """CTAs per SM the driver can keep resident for a compiled program (registers included)."""
    import cuda.bindings.driver as driver

    def checked(result):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"occupancy query failed: {error}")
        return values[0]

    # The program's kernels live in one CUDA library (cudaLibrary_t and CUlibrary are the same handle).
    executor = program._default_executor or program.to(None)
    library = driver.CUlibrary(int(executor.jit_module.cuda_library[0]))
    (kernel,) = checked(driver.cuLibraryEnumerateKernels(1, library))
    function = checked(driver.cuKernelGetFunction(kernel))
    return int(checked(driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, threads, shared_bytes)))


def _distinct(tensor):
    """``tensor`` without its broadcast (zero-stride) dimensions, so it can be copied into."""
    kept = [(size, stride) for size, stride in zip(tensor.shape, tensor.stride()) if stride != 0]
    return tensor.as_strided([size for size, _ in kept], [stride for _, stride in kept])


def _compile(item):
    kernel, options, sizes, metadata, capability, schedule, edge = item
    try:
        state._program(kernel, options, sizes, metadata, capability, 0, schedule, edge)
        return None
    except schedules.ScheduleNotHonoured:
        return None
    except Exception as error:  # noqa: BLE001  (reported; the sweep compiles it again in-process)
        return f"{kernel} {schedule}: {type(error).__name__}: {str(error)[:200]}"


def _worker_init():
    torch.cuda.set_device(0)


class Sweeper:
    """Stands in for state._launch: records launch shapes, then measures every candidate."""

    def __init__(self, reps, kernels):
        self.reps = reps
        self.kernels = kernels
        self.properties = torch.cuda.get_device_properties(0)
        self.capability = torch.cuda.get_device_capability(0)
        self.shapes = {}
        self.winners = {}
        self.records = {}
        self.measuring = False

    def __call__(self, kernel, sizes, tensors, edge=None, *, options=(), training):
        _, metadata = state._metadata(tensors)
        key = schedules.launch_key(kernel, sizes, options, edge)
        if kernel not in self.kernels:
            schedule = schedules.schedule(kernel, sizes, options, edge, tensors[0].device, training=False)
            state._run(kernel, sizes, tensors, metadata, edge, options, schedule)
            return
        if not self.measuring:
            self.shapes.setdefault(key, (kernel, tuple(options), sizes, metadata, self.capability, edge))
            state._run(kernel, sizes, tensors, metadata, edge, options, schedules.DEFAULTS[kernel])
            return
        if key not in self.winners:
            self.winners[key] = self.sweep(key, kernel, sizes, tensors, metadata, edge, options)
        state._run(kernel, sizes, tensors, metadata, edge, options, self.winners[key])

    def sweep(self, key, kernel, sizes, tensors, metadata, edge, options):
        operands, seen = [], set()
        for tensor in tensors:
            if tensor is not None:
                view = _distinct(tensor.detach())
                identity = (view.data_ptr(), tuple(view.shape), tuple(view.stride()))
                if identity not in seen:
                    seen.add(identity)
                    operands.append(view)
        snapshot = [t.clone() for t in operands]

        def restore():
            for target, source in zip(operands, snapshot, strict=True):
                target.copy_(source)

        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reference, results = None, []
        for candidate in schedules.candidates(kernel):
            try:
                operation = schedules.build_operation(kernel, sizes, options, self.properties, candidate)
            except schedules.ScheduleNotHonoured as error:
                results.append({"schedule": candidate, "skipped": str(error)})
                continue
            program = state._program(kernel, options, sizes, metadata, self.capability, 0, candidate, edge)
            blocks_per_sm = candidate[schedules.AXES[kernel].index("blocks_per_sm")]
            if candidate != schedules.DEFAULTS[kernel]:
                resident = _resident_ctas(program, 32 * operation.warps, operation.shared_bytes)
                if resident < blocks_per_sm:
                    results.append({"schedule": candidate,
                                    "skipped": f"{resident} CTAs resident, {blocks_per_sm} named"})
                    continue
            restore()
            state._run(kernel, sizes, tensors, metadata, edge, options, candidate)
            torch.cuda.synchronize()
            if reference is None:
                reference = [t.clone() for t in operands]
            elif not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                         for a, b in zip(reference, operands, strict=True)):
                raise RuntimeError(f"{key}: schedule {candidate} changes the result")
            times = []
            for _ in range(self.reps):
                restore()
                start.record()
                state._run(kernel, sizes, tensors, metadata, edge, options, candidate)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            results.append({"schedule": candidate, "ms": statistics.median(times)})
        restore()
        measured = [r for r in results if "ms" in r]
        best = min(measured, key=lambda r: r["ms"])
        self.records[key] = {"schedule": dict(zip(schedules.AXES[kernel], best["schedule"])),
                             "default_ms": round(measured[0]["ms"], 4), "best_ms": round(best["ms"], 4),
                             "measurements": results}
        print(f"TUNED {key}: {measured[0]['ms']:.4f} -> {best['ms']:.4f} ms {best['schedule']} "
              f"({len(measured)} measured, {len(results) - len(measured)} skipped)", flush=True)
        del snapshot, reference
        torch.cuda.empty_cache()
        return best["schedule"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--key-dim", type=int, required=True)
    parser.add_argument("--value-dim", type=int, required=True)
    parser.add_argument("--edges", type=int, required=True)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--running-covariance-dtype", choices=tuple(state.RUNNING_COVARIANCE_DTYPES),
                        default="bfloat16", help="the fast backend's running_covariance_dtype setting")
    parser.add_argument("--reps", type=int, default=7)
    parser.add_argument("--kernels", nargs="+", choices=tuple(schedules.AXES), default=tuple(schedules.AXES),
                        help="the kernels to tune (default: all three)")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="processes compiling the candidates ahead of the measurements")
    parser.add_argument("--record", help="also write every measurement to this JSON file")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    per_edge, cotangent = _inputs(args, torch.Generator(device="cuda").manual_seed(99))
    sweeper = Sweeper(args.reps, set(args.kernels))
    launch = state._launch
    state._launch = sweeper
    try:
        _one_pass(args, per_edge, cotangent)
        items = [(kernel, options, sizes, metadata, capability, candidate, edge)
                 for kernel, options, sizes, metadata, capability, edge in sweeper.shapes.values()
                 for candidate in schedules.candidates(kernel)]
        begin = time.perf_counter()
        with multiprocessing.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
            errors = [error for error in pool.map(_compile, items, chunksize=1) if error]
        print(f"COMPILED {len(items)} candidates of {len(sweeper.shapes)} launch shapes in "
              f"{time.perf_counter() - begin:.0f}s, {len(errors)} errors", flush=True)
        sweeper.measuring = True
        _one_pass(args, per_edge, cotangent)
    finally:
        state._launch = launch

    path = schedules.table_path(torch.device("cuda", 0))
    table = {}
    if os.path.exists(path):
        with open(path) as handle:
            table = json.load(handle)
    table["__meta__"] = {"gpu": torch.cuda.get_device_name(0)}
    for key, record in sweeper.records.items():
        table[key] = {name: record[name] for name in ("schedule", "default_ms", "best_ms")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(table, handle, indent=1, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    schedules.clear_table_cache()
    print(f"wrote {len(sweeper.records)} entries to {path}", flush=True)
    if args.record:
        with open(args.record, "w") as handle:
            json.dump({"arguments": vars(args), "gpu": torch.cuda.get_device_name(0),
                       "records": sweeper.records}, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
