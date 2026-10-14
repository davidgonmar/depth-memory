"""Launch schedules of the three state kernels and the per-GPU tables that pick them.

A schedule assigns groups to warps, CTAs and lanes. The forward and the window end take a batch
(groups per shared stage), consumer warps and resident CTAs per SM, and the forward also a lane
width for its D loop; the interior takes warps per CTA, CTAs per SM and a lane width. Under every
schedule each group is computed by one warp with the same FP32 expressions and the same order of
every sum, so all schedules give bitwise the same result. The window end has no lane-width axis
because its lane width decides which D columns a lane adds up before the butterfly, which is part
of the summation order.

tuned/<arch>.json holds the fastest schedule of every launch shape the models train at, measured
by ``python -m olmo_core.nn.ldm.backends.fast.tune``. A training launch whose shape is
not in the table raises; launches without gradients (evaluation) whose shape is not in the table
take the default schedule.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import lru_cache

import torch

AXES = {
    "forward": ("batch", "consumers", "blocks_per_sm", "lane_width"),
    "interior": ("warps", "blocks_per_sm", "lane_width"),
    "window_end": ("batch", "consumers", "blocks_per_sm"),
}
# Interior lane width 0 is default_lane_width(D).
DEFAULTS = {"forward": (32, 8, 1, 8), "interior": (2, 6, 0), "window_end": (4, 4, 2)}
_SIZE_NAMES = {"forward": "ENGKD", "interior": "ENGKD", "window_end": "EFNGKD"}
_TUNE_COMMAND = "python -m olmo_core.nn.ldm.backends.fast.tune"


def candidates(kernel: str) -> tuple[tuple[int, ...], ...]:
    """Every schedule the tuner measures, the default first."""
    if kernel == "forward":
        schedules = [(batch, consumers, blocks_per_sm, lane_width)
                     for batch in (8, 16, 32, 64) for consumers in (2, 4, 8, 16) if consumers <= batch
                     for blocks_per_sm in (1, 2) for lane_width in (2, 4, 8)]
    elif kernel == "interior":
        schedules = [(warps, blocks_per_sm, lane_width) for warps in (1, 2, 4, 8)
                     for blocks_per_sm in range(1, 9) for lane_width in (2, 4, 8)]
    elif kernel == "window_end":
        schedules = [(batch, consumers, blocks_per_sm) for batch in (4, 8, 16)
                     for consumers in (1, 2, 4, 8, 16) if batch % consumers == 0
                     for blocks_per_sm in (1, 2, 3, 4)]
    else:
        raise ValueError(f"unknown state kernel {kernel!r}")
    default = DEFAULTS[kernel]
    return (default,) + tuple(schedule for schedule in schedules if schedule != default)


class ScheduleNotHonoured(ValueError):
    """A candidate schedule the kernel would not run as named at this shape."""


def _resident_limit(properties, threads: int, shared_bytes: int) -> int:
    """CTAs per SM that the shared memory and thread footprint allow (registers are checked by the
    tuner against the compiled kernel)."""
    per_sm = getattr(properties, "shared_memory_per_multiprocessor", None) or 233472
    by_smem = per_sm // (shared_bytes + 1024) if shared_bytes else 32
    by_threads = getattr(properties, "max_threads_per_multi_processor", 2048) // threads
    return max(0, min(by_smem, by_threads, 32))


def build_operation(kernel: str, sizes: tuple[int, ...], options: tuple, properties,
                    schedule: tuple[int, ...]):
    """The kernel object for one schedule, checked to run as named (see ScheduleNotHonoured). Only
    the default schedule may be fitted down to a smaller shared-memory budget."""
    settings = dict(options, sms=properties.multi_processor_count)
    default = schedule == DEFAULTS[kernel]
    if kernel == "forward":
        from .kernels.forward import make_forward as factory
        batch, consumers, blocks_per_sm, lane_width = schedule
        per_sm = getattr(properties, "shared_memory_per_multiprocessor", None) or 233472
        # One CTA per SM keeps the whole opt-in budget; more CTAs share the SM.
        budget = min(properties.shared_memory_per_block_optin, per_sm // blocks_per_sm - 1024) - 2048
        settings.update(batch=batch, consumers=consumers, blocks_per_sm=blocks_per_sm,
                        lane_width=lane_width, smem_budget=budget)
    elif kernel == "interior":
        from .kernels.interior import make_interior as factory
        warps, blocks_per_sm, lane_width = schedule
        settings.update(warps=warps, blocks_per_sm=blocks_per_sm, lane_width=lane_width or None)
    elif kernel == "window_end":
        from .kernels.window_end import make_window_end as factory
        batch, consumers, blocks_per_sm = schedule
        settings.update(batch=batch, consumers=consumers, blocks_per_sm=blocks_per_sm)
    else:
        raise ValueError(f"unknown state kernel {kernel!r}")
    try:
        operation = factory(*sizes, **settings)
    except ValueError as error:
        if default:
            raise
        raise ScheduleNotHonoured(f"{kernel} schedule {schedule}: {error}") from error
    if not default:
        if kernel == "forward" and (operation.batch, operation.consumers) != (batch, consumers):
            raise ScheduleNotHonoured(
                f"forward schedule {schedule} fits only as batch={operation.batch} "
                f"consumers={operation.consumers}")
        limit = _resident_limit(properties, 32 * operation.warps, operation.shared_bytes)
        if blocks_per_sm > limit:
            raise ScheduleNotHonoured(f"{kernel} schedule {schedule}: {blocks_per_sm} CTAs per SM "
                                      f"requested, {limit} fit ({operation.shared_bytes} B shared per CTA)")
    return operation


def _float32_covariance(options: tuple) -> bool:
    """Whether a forward launch keeps the running covariance in FP32."""
    return dict(options).get("running_covariance_dtype") == "float32"


def launch_key(kernel: str, sizes: tuple[int, ...], options: tuple, edge: int | None) -> str:
    """The table key of one launch shape, e.g. ``window_end E4 F4 N65536 G16 K8 D128``. A forward
    with an FP32 running covariance has its own schedules, under keys with ``float32_covariance``."""
    words = [kernel] + [f"{name}{size}" for name, size in zip(_SIZE_NAMES[kernel], sizes, strict=True)]
    if dict(options).get("first_window"):
        words.append("first_window")
    if _float32_covariance(options):
        words.append("float32_covariance")
    if edge is not None:
        words.append(f"edge{edge}")
    return " ".join(words)


def _arch(device: torch.device) -> str:
    """The table name of a GPU: its SM version."""
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def table_path(device: torch.device) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned", f"{_arch(device)}.json")


@lru_cache(maxsize=None)
def _table(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


_untuned_shapes = False


@contextmanager
def untuned_shapes():
    """Run training launches whose shape is not in the table with the default schedule."""
    global _untuned_shapes
    previous, _untuned_shapes = _untuned_shapes, True
    try:
        yield
    finally:
        _untuned_shapes = previous


def schedule(kernel: str, sizes: tuple[int, ...], options: tuple, edge: int | None,
             device: torch.device, *, training: bool) -> tuple[int, ...]:
    """The table's schedule of one launch."""
    key = launch_key(kernel, sizes, options, edge)
    path = table_path(device)
    entry = _table(path).get(key)
    if entry is not None:
        return tuple(entry["schedule"][axis] for axis in AXES[kernel])
    if training and not _untuned_shapes:
        n, g, k, d = sizes[-4:]
        flag = " --running-covariance-dtype float32" if _float32_covariance(options) else ""
        raise RuntimeError(
            f"no tuned schedule for '{key}' on {torch.cuda.get_device_name(device)} ({path}). "
            f"Tune this shape with: {_TUNE_COMMAND} --tokens {n} --groups {g} --key-dim {k} "
            f"--value-dim {d}{flag} --edges <the model's state edges, 2 * layers + 1>")
    return DEFAULTS[kernel]


def clear_table_cache():
    """Forget loaded tables (after the tuner writes one)."""
    _table.cache_clear()
