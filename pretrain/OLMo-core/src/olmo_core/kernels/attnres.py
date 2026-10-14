# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# That license is reproduced in LICENSE-flash-linear-attention next to this file.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

# Source-copied from fla-org/flash-linear-attention commit
# 02e270330b4fc5bb2fd738281b5d594bc89e6d5f (2026-07-25).
# The Triton kernels are copied here so AttnRes does not go through FLA's op
# dispatch; FLA is still imported for its `exp` helper and autotune-cache
# settings. OLMo-core owns the compile/autograd registration below.
from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp
from fla.utils import autotune_cache_kwargs
from torch.library import custom_op, register_autograd

# This boundary is intentionally opaque. Torch 2.12's transparent Triton
# capture cannot functionalize the tuple of separately allocated source
# tensors used by FLA's no-stack pointer ABI. A custom op keeps the kernel in
# a full compiled graph without a graph break while preserving that ABI.

try:
    from fla.ops.utils.cache import fla_cache_autotune
except ImportError:
    fla_cache_autotune = lambda **kwargs: triton.autotune(**kwargs)


@fla_cache_autotune(
    configs=[
        triton.Config({"BL": BL}, num_warps=num_warps, num_stages=num_stages)
        for BL in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["L2", "D", "HAS_ONORM"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["L"])
def attnres_fwd_kernel(
    q,
    res,
    w,
    ow,
    o,
    rstd,
    logit,
    lse,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    key_eps: tl.constexpr,
    output_eps: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BD]
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    # [BD] q * w, reused across all residual-source tiles
    b_qw = tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32) * tl.load(
        w + o_d, mask=m_d, other=0.0
    ).to(tl.float32)

    # online softmax over L; b_o accumulates in registers so each v tile is read once (logit + weighted sum)
    b_m = tl.full([], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([], dtype=tl.float32)
    b_o = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        # [BL]
        o_l = (i_l * BL).to(tl.int64) + tl.arange(0, BL)
        m_l = o_l < L
        # per-tile base pointers from the length-L2 padded tuple; OOB rows keep res[0] and are masked by m_l
        p_v = res[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
        p_v = tl.multiple_of(p_v, 16)

        # [BL, BD] gather: row l from source l at offset i_n*D + o_d
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        # [BL]
        b_rstd = tl.rsqrt(tl.sum(b_v * b_v, axis=1) / D + key_eps)
        b_logit = tl.sum(b_v * b_qw[None, :], axis=1) * b_rstd
        b_s = tl.where(m_l, b_logit, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, axis=0)), b_m
        b_r = exp(b_mp - b_m)
        # [BL]
        b_p = exp(b_s - b_m)
        b_acc = b_acc * b_r + tl.sum(b_p, axis=0)
        # [BD]
        b_o = b_o * b_r + tl.sum(b_p[:, None] * b_v, axis=0)

        # rstd and logit saved for bwd_dv
        p_rstd = rstd + i_n + o_l * N
        p_logit = logit + i_n + o_l * N
        tl.store(p_rstd, b_rstd.to(rstd.dtype.element_ty), mask=m_l)
        tl.store(p_logit, b_logit.to(logit.dtype.element_ty), mask=m_l)

    tl.store(lse + i_n, b_m + tl.log(b_acc))

    # [BD] pre-norm mixed residual sum_l p_l * v_l
    b_o = b_o / b_acc
    # Fold the optional output RMSNorm into the returned output. Backward
    # recomputes this pre-norm mixture from the sources.
    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / D + output_eps)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_o = b_o * b_o_rstd * b_ow
    p_o = o + i_n * D + o_d
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_d)


@fla_cache_autotune(
    configs=[
        triton.Config({"BL": BL}, num_warps=num_warps, num_stages=num_stages)
        for BL in [1, 2, 4, 8]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3]
    ],
    key=["L2", "D", "HAS_ONORM"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["L"])
def attnres_bwd_kernel_dv(
    q,
    res,
    w,
    ow,
    rstd,
    logit,
    lse,
    do,
    dres,
    dqw,
    dow_partial,
    N,
    L,
    L2: tl.constexpr,
    D: tl.constexpr,
    output_eps: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)

    # [BD]
    o_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    m_d = o_d < D
    b_qw = tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32) * tl.load(
        w + o_d, mask=m_d, other=0.0
    ).to(tl.float32)
    b_lse = tl.load(lse + i_n).to(tl.float32)
    p_do = do + i_n * D + o_d
    b_do = tl.load(p_do, mask=m_d, other=0.0).to(tl.float32)
    # Recompute the pre-output-norm mixture instead of saving a full activation.
    b_o_pre = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        o_l = (i_l * BL).to(tl.int64) + tl.arange(0, BL)
        m_l = o_l < L
        p_v = res[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
        p_v = tl.multiple_of(p_v, 16)
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_l[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        p_logit = logit + i_n + o_l * N
        b_logit = tl.load(p_logit, mask=m_l, other=0.0).to(tl.float32)
        b_p = tl.where(m_l, exp(b_logit - b_lse), 0.0)
        b_o_pre += tl.sum(b_p[:, None] * b_v, axis=0)

    # output RMSNorm bwd: turn b_do into the gradient w.r.t. the pre-norm output and stage dow_partial
    if HAS_ONORM:
        b_o_rstd = tl.rsqrt(tl.sum(tl.where(m_d, b_o_pre * b_o_pre, 0.0), axis=0) / D + output_eps)
        b_ow = tl.load(ow + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_xhat = b_o_pre * b_o_rstd
        b_c1 = tl.sum(tl.where(m_d, b_xhat * b_ow * b_do, 0.0), axis=0) / D
        p_dow = dow_partial + i_n * D + o_d
        tl.store(p_dow, (b_xhat * b_do).to(p_dow.dtype.element_ty), mask=m_d)
        b_do = (b_ow * b_do - b_xhat * b_c1) * b_o_rstd
    # delta = sum_l p*dp = <do_pre, o_pre>
    b_delta = tl.sum(tl.where(m_d, b_do * b_o_pre, 0.0), axis=0)

    # [BD] dqw accumulates over the L tiles
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    for i_l in range(tl.cdiv(L, BL)):
        # [BL]
        o_l = (i_l * BL).to(tl.int64) + tl.arange(0, BL)
        m_l = o_l < L
        m_v = m_l[:, None] & m_d[None, :]
        # per-tile source / dv base pointers from the length-L2 padded tuple
        p_v = res[0] + o_l * 0
        p_dv = dres[0] + o_l * 0
        for i in tl.static_range(1, L2):
            p_v = tl.where(o_l == i, res[i], p_v)
            p_dv = tl.where(o_l == i, dres[i], p_dv)
        p_v = tl.multiple_of(p_v, 16)
        p_dv = tl.multiple_of(p_dv, 16)
        # [BL, BD] v tile, read once and reused for dp / dv / dqw
        b_v = tl.load(
            tl.multiple_of(p_v[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            mask=m_v,
            other=0.0,
        ).to(tl.float32)

        p_rstd = rstd + i_n + o_l * N
        p_logit = logit + i_n + o_l * N
        # [BL]; recompute probs from logit + lse, OOB rows masked to 0
        b_rstd = tl.load(p_rstd, mask=m_l, other=0.0).to(tl.float32)
        b_logit = tl.load(p_logit, mask=m_l, other=0.0).to(tl.float32)
        b_p = tl.where(m_l, exp(b_logit - b_lse), 0.0)

        # softmax bwd with delta already known
        b_dp = tl.sum(b_v * b_do[None, :], axis=1)
        b_ds = b_p * (b_dp - b_delta)
        # [BL, BD]
        b_k = b_v * b_rstd[:, None]
        b_dv = b_p[:, None] * b_do[None, :] + (b_ds * b_rstd)[:, None] * (
            b_qw[None, :] - b_k * (b_logit / D)[:, None]
        )
        tl.store(
            tl.multiple_of(p_dv[:, None] + (i_n * D + o_d[None, :]), (1, 16)),
            b_dv.to(dres[0].dtype.element_ty),
            mask=m_v,
        )
        # [BD]
        b_dqw += tl.sum(b_ds[:, None] * b_k, axis=0)

    p_dqw = dqw + i_n * D + o_d
    tl.store(p_dqw, b_dqw, mask=m_d)


@fla_cache_autotune(
    configs=[
        triton.Config({"BN": BN, "BD": BD}, num_warps=num_warps, num_stages=num_stages)
        for BN, BD, num_warps in [
            (1024, 16, 4),
            (2048, 32, 4),
            (2048, 32, 8),
            (4096, 32, 8),
            (4096, 64, 8),
        ]
        for num_stages in [3, 4]
    ],
    key=["N", "D", "HAS_ONORM"],
    **autotune_cache_kwargs,
)
@triton.jit
def attnres_bwd_kernel_dqdw(
    q,
    w,
    dqw,
    dow_partial,
    dq,
    dw,
    dow,
    N,
    D: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int32)

    # [BD]
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D

    # column-sum dqw (and dow_partial) over the N axis
    # [BD]
    b_dqw = tl.zeros([BD], dtype=tl.float32)
    b_dow = tl.zeros([BD], dtype=tl.float32)
    for i_n in range(0, N, BN):
        o_n = i_n.to(tl.int64) + tl.arange(0, BN)
        m_nd = (o_n[:, None] < N) & m_d[None, :]
        p_dqw = dqw + o_n[:, None] * D + o_d[None, :]
        b_dqw += tl.sum(tl.load(p_dqw, mask=m_nd, other=0.0).to(tl.float32), axis=0)
        if HAS_ONORM:
            p_dow = dow_partial + o_n[:, None] * D + o_d[None, :]
            b_dow += tl.sum(tl.load(p_dow, mask=m_nd, other=0.0).to(tl.float32), axis=0)

    # the logit uses the q * w product, so dq = (sum_n dqw) * w and dw = (sum_n dqw) * q
    # [BD]
    b_q = tl.load(q + o_d, mask=m_d, other=0.0).to(tl.float32)
    b_w = tl.load(w + o_d, mask=m_d, other=0.0).to(tl.float32)
    tl.store(dq + o_d, b_dqw * b_w, mask=m_d)
    tl.store(dw + o_d, b_dqw * b_q, mask=m_d)
    if HAS_ONORM:
        tl.store(dow + o_d, b_dow, mask=m_d)


def _pad_sources(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Pad a source tuple to one static Triton specialization bucket."""
    source_bucket = max(8, triton.next_power_of_2(len(tensors)))
    return tuple(tensors) + (tensors[0],) * (source_bucket - len(tensors))


@custom_op("olmo_core::attnres_forward", mutates_args={})
def _attnres_forward(
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    key_rms_eps: float,
    output_rms_eps: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile-visible forward operator backed by the pinned FLA Triton kernel."""
    for source_idx, residual in enumerate(residuals):
        if residual.data_ptr() % 16 != 0:
            raise ValueError(
                "AttnRes requires every flattened residual source to be "
                f"16-byte aligned; source {source_idx} is not aligned"
            )
    output_shape = residuals[0].shape
    source_count = len(residuals)
    hidden_size = output_shape[-1]
    token_count = residuals[0].numel() // hidden_size
    padded_residuals = _pad_sources(residuals)

    output = torch.empty_like(residuals[0])
    stats_shape = (source_count, *output_shape[:-1])
    rstd = torch.empty(stats_shape, device=residuals[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    lse = torch.empty(output_shape[:-1], device=residuals[0].device, dtype=torch.float32)

    attnres_fwd_kernel[(token_count,)](
        q=query,
        res=padded_residuals,
        w=rms_weight,
        ow=output_rms_weight,
        o=output,
        rstd=rstd,
        logit=logit,
        lse=lse,
        N=token_count,
        L=source_count,
        L2=len(padded_residuals),
        D=hidden_size,
        key_eps=key_rms_eps,
        output_eps=output_rms_eps,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=has_output_norm,
    )
    return output, rstd, logit, lse


@_attnres_forward.register_fake
def _attnres_forward_fake(
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    key_rms_eps: float,
    output_rms_eps: float,
    has_output_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del query, rms_weight, output_rms_weight, key_rms_eps, output_rms_eps, has_output_norm
    output_shape = residuals[0].shape
    stats_shape = (len(residuals), *output_shape[:-1])
    output = torch.empty_like(residuals[0])
    rstd = torch.empty(stats_shape, device=residuals[0].device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    lse = torch.empty(output_shape[:-1], device=residuals[0].device, dtype=torch.float32)
    return output, rstd, logit, lse


@custom_op("olmo_core::attnres_backward", mutates_args={})
def _attnres_backward(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    output_rms_eps: float,
    has_output_norm: bool,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile-visible backward operator backed by the pinned FLA Triton kernels."""
    source_count = len(residuals)
    hidden_size = grad_output.shape[-1]
    token_count = grad_output.numel() // hidden_size
    padded_residuals = _pad_sources(residuals)

    grad_residuals = [torch.empty_like(residual) for residual in residuals]
    padded_grad_residuals = _pad_sources(grad_residuals)
    grad_query_weight = torch.empty_like(grad_output, dtype=torch.float32)
    grad_query = torch.empty_like(query)
    grad_rms_weight = torch.empty_like(rms_weight)
    grad_output_rms_weight = torch.empty_like(output_rms_weight)
    grad_output_rms_weight_partial = (
        torch.empty_like(grad_output, dtype=torch.float32)
        if has_output_norm
        else torch.empty(0, device=grad_output.device, dtype=torch.float32)
    )

    attnres_bwd_kernel_dv[(token_count,)](
        q=query,
        res=padded_residuals,
        w=rms_weight,
        ow=output_rms_weight,
        rstd=rstd,
        logit=logit,
        lse=lse,
        do=grad_output,
        dres=padded_grad_residuals,
        dqw=grad_query_weight,
        dow_partial=grad_output_rms_weight_partial,
        N=token_count,
        L=source_count,
        L2=len(padded_residuals),
        D=hidden_size,
        output_eps=output_rms_eps,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=has_output_norm,
    )

    def grid(meta):
        return (triton.cdiv(hidden_size, meta["BD"]),)

    attnres_bwd_kernel_dqdw[grid](
        q=query,
        w=rms_weight,
        dqw=grad_query_weight,
        dow_partial=grad_output_rms_weight_partial,
        dq=grad_query,
        dw=grad_rms_weight,
        dow=grad_output_rms_weight,
        N=token_count,
        D=hidden_size,
        HAS_ONORM=has_output_norm,
    )
    return grad_residuals, grad_query, grad_rms_weight, grad_output_rms_weight


@_attnres_backward.register_fake
def _attnres_backward_fake(
    grad_output: torch.Tensor,
    query: torch.Tensor,
    residuals: list[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rstd: torch.Tensor,
    logit: torch.Tensor,
    lse: torch.Tensor,
    output_rms_eps: float,
    has_output_norm: bool,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    del rstd, logit, lse, output_rms_eps, has_output_norm
    return (
        [torch.empty_like(residual) for residual in residuals],
        torch.empty_like(query),
        torch.empty_like(rms_weight),
        torch.empty_like(output_rms_weight),
    )


def _attnres_setup_context(ctx, inputs, output) -> None:
    (
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        _key_rms_eps,
        output_rms_eps,
        has_output_norm,
    ) = inputs
    _result, rstd, logit, lse = output
    ctx.save_for_backward(
        query,
        rms_weight,
        output_rms_weight,
        rstd,
        logit,
        lse,
        *residuals,
    )
    ctx.source_count = len(residuals)
    ctx.output_rms_eps = output_rms_eps
    ctx.has_output_norm = has_output_norm
    ctx.mark_non_differentiable(rstd, logit, lse)


def _attnres_autograd_backward(ctx, grad_output, _grad_rstd, _grad_logit, _grad_lse):
    (
        query,
        rms_weight,
        output_rms_weight,
        rstd,
        logit,
        lse,
        *residuals,
    ) = ctx.saved_tensors
    if len(residuals) != ctx.source_count:
        raise RuntimeError("AttnRes autograd context has an invalid residual-source count")
    grad_residuals, grad_query, grad_rms_weight, grad_output_rms_weight = _attnres_backward(
        grad_output.contiguous(),
        query,
        list(residuals),
        rms_weight,
        output_rms_weight,
        rstd,
        logit,
        lse,
        ctx.output_rms_eps,
        ctx.has_output_norm,
    )
    return (
        grad_query,
        grad_residuals,
        grad_rms_weight,
        grad_output_rms_weight,
        None,
        None,
        None,
    )


register_autograd(
    _attnres_forward,
    _attnres_autograd_backward,
    setup_context=_attnres_setup_context,
)


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    key_rms_eps: float = 1e-6,
    output_rms_eps: float | None = None,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""
    Apply AttnRes residual aggregation.

    AttnRes normalizes each residual source with RMSNorm, scores it against `query`, applies softmax over the
    residual-source dimension, and returns the weighted sum of residual sources.
    See `Attention Residuals <https://arxiv.org/abs/2603.15031>`_.

    Args:
        query (torch.Tensor):
            Per-layer pseudo-query with `D` elements (e.g. `[D]` or `[1, D]`), where `D` is the hidden size.
        residuals (Sequence[torch.Tensor]):
            Non-empty sequence of same-dtype, same-`D` residual sources, each of shape `[..., D]`.
        rms_weight (torch.Tensor):
            RMSNorm scale for key normalization of shape `[D]`.
        output_rms_weight (torch.Tensor, optional):
            If set, an extra RMSNorm with this weight is applied to the mixed residual before returning, fusing the
            prenorm that would otherwise follow the AttnRes call (e.g. `attn_norm` / `mlp_norm`). Default: `None`.
        key_rms_eps (float):
            Epsilon for the learned source/key RMSNorm. Default: `1e-6`.
        output_rms_eps (float, optional):
            Epsilon for the fused downstream RMSNorm. It defaults to
            `key_rms_eps` when an output norm is present.
        return_weights (bool):
            Whether to return depth softmax probabilities. Default: `False`.

    Returns:
        o (torch.Tensor):
            Mixed residual of shape `[..., D]`.
        p (torch.Tensor):
            Depth softmax probabilities of shape `[L, ...]` if `return_weights=True`, otherwise not returned.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    for residual in residuals:
        if residual.shape != output_shape:
            raise ValueError("all AttnRes residual sources must have the same shape")
        if residual.dtype != residuals[0].dtype or residual.device != residuals[0].device:
            raise ValueError("all AttnRes residual sources must share dtype and device")

    flat_residuals = [residual.reshape(-1, hidden_size).contiguous() for residual in residuals]
    flat_query = query.reshape(-1).contiguous()
    flat_rms_weight = rms_weight.reshape(-1).contiguous()
    if flat_query.numel() != hidden_size or flat_rms_weight.numel() != hidden_size:
        raise ValueError("AttnRes query and source RMS weight must match hidden size")

    has_output_norm = output_rms_weight is not None
    if output_rms_weight is None:
        flat_output_rms_weight = flat_rms_weight.new_empty(0)
    else:
        flat_output_rms_weight = output_rms_weight.reshape(-1).contiguous()
        if flat_output_rms_weight.numel() != hidden_size:
            raise ValueError("AttnRes output RMS weight must match hidden size")
    resolved_output_eps = key_rms_eps if output_rms_eps is None else output_rms_eps

    output, _rstd, logit, lse = _attnres_forward(
        flat_query,
        flat_residuals,
        flat_rms_weight,
        flat_output_rms_weight,
        key_rms_eps,
        resolved_output_eps,
        has_output_norm,
    )
    output = output.view(output_shape)

    if return_weights:
        probabilities = (logit - lse).exp().view(len(residuals), *output_shape[:-1])
        return output, probabilities
    return output


__all__ = ["fused_attnres"]
