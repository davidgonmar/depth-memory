import json
import os
from functools import lru_cache
from typing import Optional

import torch

from olmo_core.kernels.tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from olmo_core.kernels.tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse

# Local: the projection kernel walks its reduction in blocks of this many columns.
PROJECTION_BLOCK = 256
TUNE_COMMAND = "python -m olmo_core.kernels.tile_kernels.modeling.mhc.ops.tune_projection"


@lru_cache(maxsize=None)
def valid_splits(reduction: int) -> tuple[int, ...]:
    """Local: the split counts the projection kernel accepts for an n*D reduction, each part a whole
    number of its blocks."""
    if reduction % PROJECTION_BLOCK:
        raise ValueError(f"the projection's reduction ({reduction}) is not a multiple of {PROJECTION_BLOCK}")
    blocks = reduction // PROJECTION_BLOCK
    return tuple(splits for splits in range(1, blocks + 1) if blocks % splits == 0)


@lru_cache(maxsize=None)
def table_path(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned", f"sm{major}{minor}.json")


@lru_cache(maxsize=None)
def _table(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def clear_table_cache() -> None:
    """Forget loaded tables (after the tuner writes one)."""
    _table.cache_clear()
    projection_splits.cache_clear()


@lru_cache(maxsize=None)
def projection_splits(num_tokens: int, mhc_mult: int, hidden_size: int, device: torch.device) -> int:
    """Local: how many ways the projection splits its n*D reduction, from the table tuned on this GPU
    type (tuned/sm<arch>.json, written by tune_projection.py). Upstream grids the projection over
    tokens only, one block walking the whole reduction, so a forward of a few tokens (a decode step)
    runs a handful of blocks; splitting the reduction runs more blocks at once, and the partial sums
    are added in split order, so only the FP32 summation order changes. A table entry lists token
    counts with the split count chosen at each (the fewest within 1 % of the fastest); a forward uses the entry of the smallest listed
    count at or above its token count, and above the largest listed count (where one split, upstream's
    schedule, is fastest) that count's entry. A width missing from the table raises, unless its
    reduction is a single block (one split is then the only schedule)."""
    reduction = mhc_mult * hidden_size
    if valid_splits(reduction) == (1,):
        return 1
    path = table_path(device)
    key = f"{mhc_mult}x{hidden_size}"
    entry = _table(path).get(key)
    if entry is None:
        raise RuntimeError(
            f"no tuned projection split for {mhc_mult} streams of width {hidden_size} on "
            f"{torch.cuda.get_device_name(device)} ({path}). Tune it with: "
            f"{TUNE_COMMAND} --streams {mhc_mult} --hidden {hidden_size}")
    for tokens, splits in entry["splits"]:
        if num_tokens <= tokens:
            return splits
    return entry["splits"][-1][1]


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: Optional[int] = None,
    fn_rounded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Local: n_splits=None takes the split count tuned for this token count (projection_splits); a
    # number is used as given (1 is upstream's schedule). fn_rounded=True when fn already holds
    # round_to_tf32 of the projection, rounded once at load
    # (olmo_core.nn.hyper_connections.prepare_inference); the rounding is not idempotent, so it is not
    # applied twice.
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)

    if n_splits is None:
        n_splits = projection_splits(num_tokens, mhc_mult, hidden_size, residual.device)
    elif n_splits not in valid_splits(mhc_hidden_size):
        raise ValueError(f"the projection cannot split its reduction of {mhc_hidden_size} {n_splits} ways "
                         f"(valid: {valid_splits(mhc_hidden_size)})")
    gemm_out_mul = torch.empty(
        num_tokens, n_splits, mhc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(num_tokens, n_splits, dtype=torch.float32, device=residual.device)

    if not fn_rounded:
        fn = round_to_tf32(fn)

    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, n_splits, mhc_hidden_size // n_splits)
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
    )
    # END of TileLang implementation of pre-norm-fn forward matmul

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
