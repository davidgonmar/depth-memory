"""Traceable ``torch.library`` custom ops over the vendored DeepSeek TileKernels mHC kernels.

Upstream calls its kernels through ``torch.autograd.Function`` classes, which Dynamo cannot trace,
and some of its backward kernels reject the non-packed cotangents the compiler produces. Each op
here is one opaque node for Dynamo and Inductor, with a fake implementation for shapes and an
autograd rule that is itself a custom op, so an mHC block compiles forward and backward. Each op
reproduces one upstream composition with the same raw kernels, order and TF32 rounding points:

* ``olmo_mhc::pre``  = ``mhc_pre_norm_fn`` -> ``mhc_pre_split_mixes`` -> ``sinkhorn_normalize``
  -> ``mhc_pre_apply_mix`` (upstream ``functional.mhc_pre``, training branch).
* ``olmo_mhc::post`` = ``mhc_post``.
* ``olmo_mhc::head`` = upstream ``functional.mhc_head`` (padded projection, head mix, apply).
* ``olmo_mhc::pre_inference`` = upstream ``functional.mhc_pre`` without gradients (RMS-weight merge,
  then the fused ``mhc_pre_big_fuse`` kernel), for prefill and decode.

The kernels are upstream's except for the local edits listed in ``kernels/tile_kernels/VENDORED.md``.

In the backward, ``pre`` accumulates its two stream-gradient contributions (apply-mix and
normalize-project) into one zero-initialised buffer through the kernels' own accumulate semantics,
and autograd adds ``post``'s residual gradient to it (upstream parks that buffer on the input
tensor's storage, ``grad_from_mhc_post``). This costs one bf16 zero fill and one bf16 add of the
stream tensor per connector and changes the bf16 rounding order of the stream gradient. Cotangents
are made contiguous before the kernels, and the ops return gradients and allocate their outputs.

Recompute (paper section 4.3.2, Table 3) lives at the end of this file: ``olmo_mhc::pre_r`` /
``post_r`` are ``pre`` / ``post`` whose autograd rule saves the group's entry residual
``x_{l0}`` plus the earlier layers' mixes and branch outputs instead of the layer's own streams,
and rebuilds ``x_l`` in backward through upstream's ``mhc_multilayer_recompute`` (one replay
per group, cached across the group's consumers). ``mhc_branch_input`` does the same for the
branch side: ``H^pre x_l`` and ``RMSNorm(H^pre x_l)`` are rebuilt from the replay, so the
branch's first projections do not store their input either.
"""
from typing import List, Optional

import torch
import torch.nn.functional as F

__all__ = ["mhc_pre", "mhc_pre_inference", "mhc_post", "mhc_head", "mhc_pre_recompute",
           "mhc_post_recompute", "mhc_branch_input"]

_LIB = "olmo_mhc"


def _grads(grads):
    """Backward cotangents for a list-returning op arrive as one sequence; unused outputs are None."""
    if len(grads) == 1 and isinstance(grads[0], (list, tuple)):
        grads = grads[0]
    return list(grads)


def _tf32_exact(t: torch.Tensor) -> torch.Tensor:
    """Clear the 13 low mantissa bits (what TF32 tensor cores discard); t is already half-ulp
    rounded by upstream's ``round_to_tf32``."""
    return (t.view(torch.int32) & -0x2000).view(torch.float32)


def _kernels():
    from olmo_core.kernels.tile_kernels.config import get_num_sms
    from olmo_core.kernels.tile_kernels.mhc import (
        head_compute_mix_kernel as head_k,
        norm_fn_kernel as norm_k,
        post_kernel as post_k,
        pre_apply_mix_kernel as apply_k,
        pre_split_mixes_kernel as split_k,
        sinkhorn_kernel as sink_k,
    )

    return get_num_sms, head_k, norm_k, post_k, apply_k, split_k, sink_k


# --------------------------------------------------------------------------------------------
# Shared pieces of the projection (normw merge -> TF32 -> project+RMS), forward and backward.
# --------------------------------------------------------------------------------------------
def _project_forward(x_flat, fn, normw, norm_eps, norm_k):
    """x_flat [N, nD] bf16, fn [M, nD] fp32 -> (mixes [N, M] fp32, out_mul [N,1,M], sqrsum [N,1], fn_t [M, nD])."""
    m, hidden = fn.shape
    n = x_flat.shape[0]
    fn_m = torch.empty_like(fn)
    norm_k._mhc_fn_normw_merge_fwd(*fn.shape)(fn, normw, fn_m)
    fn_t = norm_k.round_to_tf32(fn_m)
    out_mul = torch.empty(n, 1, m, dtype=torch.float32, device=x_flat.device)
    sqrsum = torch.empty(n, 1, dtype=torch.float32, device=x_flat.device)
    norm_k._mhc_pre_norm_fn_fwd_mul(m, 1, hidden)(x_flat, fn_t, out_mul, sqrsum)
    out_mul_n = torch.empty_like(out_mul)
    sqrsum_n = torch.empty_like(sqrsum)
    mixes = torch.empty(n, m, dtype=torch.float32, device=x_flat.device)
    norm_k._mhc_pre_norm_fn_fwd_norm(m, 1, hidden, norm_eps, 1)(
        out_mul.view(1, n, 1, m), sqrsum.view(1, n, 1), out_mul_n, sqrsum_n, mixes
    )
    return mixes, out_mul_n, sqrsum_n, fn_t


def _proj_bwd_math(g, x, fn_t, sqrsum_grad, x_grad):
    """d_fn = d_mixes^T x ;  d_x += d_mixes fn_t + 2 x d_sqrsum. Compiled by Inductor, which
    may fuse the elementwise epilogue into the second matmul's tile loop so the fp32 product
    never round-trips memory."""
    x32 = x.float()
    d_fn_t = g.t() @ x32
    x_grad = x_grad + (g @ fn_t + 2.0 * x32 * sqrsum_grad).to(torch.bfloat16)
    return d_fn_t, x_grad


_PROJ_BWD_COMPILED = None
# Inductor benchmarks cuBLAS (ATEN) and its Triton GEMM templates per matmul shape at compile
# time and keeps the fastest; the elementwise epilogue is fused into the second GEMM when a
# Triton template wins.
_PROJ_BWD_OPTIONS = {"max_autotune_gemm": True, "max_autotune_gemm_backends": "ATEN,TRITON",
                     "epilogue_fusion": True}


def _proj_bwd_compiled():
    global _PROJ_BWD_COMPILED
    if _PROJ_BWD_COMPILED is None:
        _PROJ_BWD_COMPILED = torch.compile(_proj_bwd_math, dynamic=True, options=_PROJ_BWD_OPTIONS)
    return _PROJ_BWD_COMPILED


def _project_backward(d_mixes, x_flat, fn, normw, fn_t, out_mul, sqrsum, norm_eps, x_grad_flat, norm_k,
                      upstream_bwd_mul):
    """Returns (x_grad_flat, d_fn, d_normw): the stream gradient accumulated onto x_grad_flat
    (bf16, same shape as x_flat), and the fp32 parameter gradients.

    ``upstream_bwd_mul`` selects upstream's exact `_mhc_pre_norm_fn_bwd_mul` kernel for the two
    projection matmuls; otherwise they run as TF32 GEMMs chosen by Inductor's autotuner (see
    ``_proj_bwd_compiled``)."""
    m, hidden = fn.shape
    out_mul_grad = torch.empty_like(out_mul)
    sqrsum_grad = torch.empty_like(sqrsum)
    norm_k._mhc_pre_norm_fn_bwd_norm(m, 1, hidden, norm_eps)(
        d_mixes, out_mul, sqrsum, out_mul_grad, sqrsum_grad
    )
    out_mul_grad = norm_k.round_to_tf32(out_mul_grad)
    if upstream_bwd_mul:
        d_fn_t = torch.empty_like(fn)
        norm_k._mhc_pre_norm_fn_bwd_mul(m, 1, hidden)(
            out_mul_grad, sqrsum_grad, x_flat, fn_t, x_grad_flat, d_fn_t
        )
    else:
        # Deviation from upstream, for speed only. Upstream's `_mhc_pre_norm_fn_bwd_mul`
        # computes these two matmuls in one kernel whose grid is (hidden / 128) CTAs, each walking
        # every token block serially (upstream notes that the TileLang version has no split-K).
        # Here the same products run as TF32 GEMMs (cuBLAS or Inductor's Triton templates, whichever
        # the autotuner measures faster per shape): TF32 is what the TileLang kernel's
        # tensor-core gemm uses, the operands are the same TF32 values (out_mul_grad rounded by
        # upstream's `round_to_tf32`, whose 13 leftover low bits the tensor cores drop and we
        # drop explicitly; x is bf16, exact in TF32), so only the fp32 summation order differs.
        compiled = _proj_bwd_compiled()
        g = _tf32_exact(out_mul_grad).view(-1, m)
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")   # TF32 tensor cores for the two GEMMs
        try:
            d_fn_t, x_grad_flat = compiled(g, x_flat, _tf32_exact(fn_t), sqrsum_grad.view(-1, 1), x_grad_flat)
        finally:
            torch.set_float32_matmul_precision(precision)
    d_fn = torch.zeros_like(fn)
    d_normw = torch.zeros_like(normw)
    norm_k._mhc_fn_normw_merge_bwd(*fn.shape)(fn, normw, d_fn_t, d_fn, d_normw)
    return x_grad_flat, d_fn, d_normw


# --------------------------------------------------------------------------------------------
# pre: streams [B, T, n, D] -> layer_input [B, T, D], pre_mix, post_mix, comb (+ saved intermediates)
# --------------------------------------------------------------------------------------------
@torch.library.custom_op(f"{_LIB}::pre", mutates_args=())
def _pre(
    streams: torch.Tensor, fn: torch.Tensor, normw: torch.Tensor, alpha: torch.Tensor,
    b: torch.Tensor, mhc_mult: int, post_mult: float, pre_eps: float, sinkhorn_eps: float,
    sinkhorn_repeat: int, norm_eps: float, upstream_bwd_mul: bool,
) -> List[torch.Tensor]:
    _, _, norm_k, _, apply_k, split_k, sink_k = _kernels()
    streams = streams.contiguous()
    outer = streams.shape[:-2]
    n, d = streams.shape[-2], streams.shape[-1]
    n2, n3 = n * n, n * n + 2 * n
    x_flat = streams.view(-1, n * d)
    tokens = x_flat.shape[0]
    mixes, out_mul, sqrsum, fn_t = _project_forward(x_flat, fn, normw, norm_eps, norm_k)
    pre_mix = mixes.new_empty(tokens, n)
    post_mix = mixes.new_empty(tokens, n)
    comb_logits = mixes.new_empty(tokens, n2)
    split_k._mhc_pre_split_mixes_fwd(mhc_mult, post_mult, pre_eps, token_block_size=32)(
        mixes, alpha, b, pre_mix, post_mix, comb_logits
    )
    comb_logits = comb_logits.view(tokens, n, n)
    comb = torch.empty_like(comb_logits)
    sink_k._mhc_sinkhorn_fwd(n, 1, sinkhorn_repeat, sinkhorn_eps)(comb_logits, comb)
    layer_input = torch.empty(*outer, d, dtype=torch.bfloat16, device=streams.device)
    apply_k._mhc_pre_apply_mix_fwd(n, d)(streams.view(-1, n, d), pre_mix, layer_input.view(-1, d))
    return [
        layer_input,
        pre_mix.view(*outer, n, 1),
        post_mix.view(*outer, n, 1),
        comb.view(*outer, n, n),
        # saved for backward only
        mixes.view(*outer, n3),
        out_mul,
        sqrsum,
        comb_logits.view(*outer, n, n),
        fn_t,
    ]


@_pre.register_fake
def _(streams, fn, normw, alpha, b, mhc_mult, post_mult, pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps,
      upstream_bwd_mul):
    outer = streams.shape[:-2]
    n, d = streams.shape[-2], streams.shape[-1]
    f32 = dict(dtype=torch.float32, device=streams.device)
    return [
        torch.empty(*outer, d, dtype=torch.bfloat16, device=streams.device),
        torch.empty(*outer, n, 1, **f32),
        torch.empty(*outer, n, 1, **f32),
        torch.empty(*outer, n, n, **f32),
        torch.empty(*outer, fn.shape[0], **f32),
        torch.empty(_prod(outer), 1, fn.shape[0], **f32),
        torch.empty(_prod(outer), 1, **f32),
        torch.empty(*outer, n, n, **f32),
        torch.empty_like(fn),
    ]


def _prod(shape):
    out = 1
    for s in shape:
        out = out * s
    return out


@torch.library.custom_op(f"{_LIB}::pre_backward", mutates_args=())
def _pre_backward_op(
    d_layer_input: torch.Tensor, d_pre_mix: torch.Tensor, d_post_mix: torch.Tensor,
    d_comb: torch.Tensor, streams: torch.Tensor, fn: torch.Tensor, normw: torch.Tensor,
    alpha: torch.Tensor, b: torch.Tensor, pre_mix: torch.Tensor, post_mix: torch.Tensor,
    mixes: torch.Tensor, out_mul: torch.Tensor, sqrsum: torch.Tensor, comb_logits: torch.Tensor,
    fn_t: torch.Tensor, mhc_mult: int, post_mult: float, sinkhorn_eps: float,
    sinkhorn_repeat: int, norm_eps: float, upstream_bwd_mul: bool,
) -> List[torch.Tensor]:
    get_num_sms, _, norm_k, _, apply_k, split_k, sink_k = _kernels()
    n, d = streams.shape[-2], streams.shape[-1]
    n3 = fn.shape[0]
    x = streams.view(-1, n, d)
    tokens = x.shape[0]
    # One accumulator for both stream-gradient contributions (kernels add into it).
    x_grad = torch.zeros_like(x)
    d_pre = apply_k._mhc_pre_apply_mix_bwd(n, d)(
        d_layer_input.contiguous().view(tokens, d), x, pre_mix.view(tokens, n), x_grad
    )
    if d_pre_mix is not None and d_pre_mix.numel():
        d_pre = d_pre + d_pre_mix.reshape(tokens, n)
    d_comb_logits = torch.empty_like(comb_logits.view(tokens, n, n))
    sink_k._mhc_sinkhorn_bwd(n, 32, sinkhorn_repeat, sinkhorn_eps)(
        d_comb.contiguous().view(tokens, n, n), comb_logits.view(tokens, n, n), d_comb_logits
    )
    num_sms = get_num_sms()
    d_mixes = torch.empty(tokens, n3, dtype=torch.float32, device=streams.device)
    d_alpha_part = torch.empty(num_sms, *alpha.shape, dtype=alpha.dtype, device=alpha.device)
    d_b_part = torch.empty(num_sms, *b.shape, dtype=b.dtype, device=b.device)
    split_k._mhc_pre_split_mixes_bwd(mhc_mult, post_mult, token_block_size=32, num_sms=num_sms)(
        d_pre.contiguous().view(tokens, n), d_post_mix.contiguous().view(tokens, n),
        d_comb_logits.view(tokens, n * n), mixes.view(tokens, n3), post_mix.view(tokens, n),
        alpha, b, d_mixes, d_alpha_part, d_b_part,
    )
    x_grad_flat, d_fn, d_normw = _project_backward(
        d_mixes, x.view(tokens, n * d), fn, normw, fn_t, out_mul, sqrsum, norm_eps,
        x_grad.view(tokens, n * d), norm_k, upstream_bwd_mul,
    )
    return [x_grad_flat.view_as(streams), d_fn, d_normw, d_alpha_part.sum(0), d_b_part.sum(0)]


@_pre_backward_op.register_fake
def _(d_layer_input, d_pre_mix, d_post_mix, d_comb, streams, fn, normw, alpha, b, pre_mix, post_mix,
      mixes, out_mul, sqrsum, comb_logits, fn_t, mhc_mult, post_mult, sinkhorn_eps, sinkhorn_repeat, norm_eps,
      upstream_bwd_mul):
    return [torch.empty_like(streams), torch.empty_like(fn), torch.empty_like(normw),
            torch.empty_like(alpha), torch.empty_like(b)]


def _pre_setup(ctx, inputs, output):
    (streams, fn, normw, alpha, b, mhc_mult, post_mult, pre_eps, sinkhorn_eps, sinkhorn_repeat,
     norm_eps, upstream_bwd_mul) = inputs
    _, pre_mix, post_mix, _, mixes, out_mul, sqrsum, comb_logits, fn_t = output
    ctx.save_for_backward(streams, fn, normw, alpha, b, pre_mix, post_mix, mixes, out_mul, sqrsum,
                          comb_logits, fn_t)
    ctx.consts = (mhc_mult, post_mult, sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul)


def _pre_backward(ctx, *grads):
    d_layer_input, d_pre_mix, d_post_mix, d_comb, *_ = _grads(grads)
    saved = ctx.saved_tensors
    streams = saved[0]
    if d_layer_input is None:
        d_layer_input = streams.new_zeros(*streams.shape[:-2], streams.shape[-1])
    if d_pre_mix is None:
        d_pre_mix = streams.new_zeros(0, dtype=torch.float32)
    if d_post_mix is None:
        d_post_mix = torch.zeros_like(saved[6])
    if d_comb is None:
        d_comb = torch.zeros_like(saved[10])
    out = torch.ops.olmo_mhc.pre_backward(d_layer_input, d_pre_mix, d_post_mix, d_comb, *saved, *ctx.consts)
    return (*out, None, None, None, None, None, None, None)


torch.library.register_autograd(f"{_LIB}::pre", _pre_backward, setup_context=_pre_setup)


def mhc_pre(streams, fn, alpha, b, *, norm_weight, norm_eps, mhc_mult, post_mult_value, pre_eps,
            sinkhorn_eps, sinkhorn_repeat, upstream_bwd_mul=False):
    """Training-mode ``functional.mhc_pre`` as one traceable op.

    Returns ``(layer_input, pre_mix, post_mix, comb)``; the last three are the mixes upstream's
    ``mhc_post`` and ``mhc_multilayer_recompute`` consume. ``upstream_bwd_mul=True`` runs the
    projection backward through upstream's exact kernel instead of the autotuned GEMMs.
    """
    out = torch.ops.olmo_mhc.pre(streams, fn, norm_weight, alpha, b, mhc_mult, post_mult_value,
                                 pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps, bool(upstream_bwd_mul))
    return out[0], out[1], out[2], out[3]


# --------------------------------------------------------------------------------------------
# pre without gradients: upstream's fused inference composition, one op so a compiled block has no
# graph break around the TileLang calls.
# --------------------------------------------------------------------------------------------
@torch.library.custom_op(f"{_LIB}::pre_inference", mutates_args=())
def _pre_inference(
    streams: torch.Tensor, fn: torch.Tensor, normw: Optional[torch.Tensor], alpha: torch.Tensor,
    b: torch.Tensor, mhc_mult: int, post_mult: float, pre_eps: float, sinkhorn_eps: float,
    sinkhorn_repeat: int, norm_eps: float, n_splits: Optional[int],
) -> List[torch.Tensor]:
    from olmo_core.kernels.tile_kernels.modeling.mhc.ops.norm_fn import _MHCFnNormwMerge
    from olmo_core.kernels.tile_kernels.modeling.mhc.ops.pre_big_fuse import mhc_pre_big_fuse

    with torch.no_grad():
        # Upstream's no-grad mhc_pre: merge the RMS weight into the projection, then the fused kernel.
        # Without a norm weight the projection was merged and rounded to TF32 once at load.
        if normw is not None:
            fn = _MHCFnNormwMerge.apply(fn, normw)
        post_mix, comb, layer_input = mhc_pre_big_fuse(
            streams.contiguous(), fn, alpha, b, rms_eps=norm_eps, mhc_pre_eps=pre_eps,
            mhc_sinkhorn_eps=sinkhorn_eps, mhc_post_mult_value=post_mult, sinkhorn_repeat=sinkhorn_repeat,
            n_splits=n_splits, fn_rounded=normw is None,
        )
    return [layer_input, post_mix, comb]


@_pre_inference.register_fake
def _(streams, fn, normw, alpha, b, mhc_mult, post_mult, pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps,
      n_splits):
    outer = streams.shape[:-2]
    n, d = streams.shape[-2], streams.shape[-1]
    f32 = dict(dtype=torch.float32, device=streams.device)
    return [torch.empty(*outer, d, dtype=torch.bfloat16, device=streams.device),
            torch.empty(*outer, n, 1, **f32), torch.empty(*outer, n, n, **f32)]


def mhc_pre_inference(streams, fn, alpha, b, *, norm_weight, norm_eps, mhc_mult, post_mult_value, pre_eps,
                      sinkhorn_eps, sinkhorn_repeat, n_splits):
    """No-grad ``functional.mhc_pre`` as one traceable op; returns ``(layer_input, post_mix, comb)``.
    ``norm_weight`` None means ``fn`` already holds the merged projection rounded to TF32
    (hyper_connections.prepare_inference). ``n_splits`` None takes the projection split count tuned for
    the token count on this GPU type; 1 is upstream's single split."""
    out = torch.ops.olmo_mhc.pre_inference(streams, fn, norm_weight, alpha, b, mhc_mult, post_mult_value, pre_eps,
                                           sinkhorn_eps, sinkhorn_repeat, norm_eps, n_splits)
    return out[0], out[1], out[2]


# --------------------------------------------------------------------------------------------
# post: F_post,res application (residual merge), the same kernel call as upstream mhc_post.
# --------------------------------------------------------------------------------------------
@torch.library.custom_op(f"{_LIB}::post", mutates_args=())
def _post(x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor,
          comb: torch.Tensor) -> torch.Tensor:
    _, _, _, post_k, _, _, _ = _kernels()
    return post_k.mhc_post_fwd(x, residual, post_mix, comb, None)


@_post.register_fake
def _(x, residual, post_mix, comb):
    return torch.empty_like(residual)


@torch.library.custom_op(f"{_LIB}::post_backward", mutates_args=())
def _post_backward_op(d_out: torch.Tensor, x: torch.Tensor, residual: torch.Tensor,
                      post_mix: torch.Tensor, comb: torch.Tensor) -> List[torch.Tensor]:
    _, _, _, post_k, _, _, _ = _kernels()
    d_x, d_res, d_post, d_comb = post_k.mhc_post_bwd(
        x, residual, post_mix, comb, d_out.contiguous(), fuse_grad_acc=False
    )
    return [d_x, d_res, d_post, d_comb]


@_post_backward_op.register_fake
def _(d_out, x, residual, post_mix, comb):
    return [torch.empty_like(x), torch.empty_like(residual), torch.empty_like(post_mix),
            torch.empty_like(comb)]


def _post_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _post_backward(ctx, d_out):
    return tuple(torch.ops.olmo_mhc.post_backward(d_out, *ctx.saved_tensors))


torch.library.register_autograd(f"{_LIB}::post", _post_backward, setup_context=_post_setup)


def mhc_post(x, residual, post_mix, comb, out=None):
    if out is not None:
        raise NotImplementedError("the traceable mhc_post allocates its own output")
    return torch.ops.olmo_mhc.post(x, residual, post_mix, comb)


# --------------------------------------------------------------------------------------------
# head: upstream functional.mhc_head (pad fn to n^2+2n rows, project, head mix, apply).
# --------------------------------------------------------------------------------------------
@torch.library.custom_op(f"{_LIB}::head", mutates_args=())
def _head(streams: torch.Tensor, fn: torch.Tensor, normw: torch.Tensor, scale: torch.Tensor,
          base: torch.Tensor, mhc_mult: int, pre_eps: float, norm_eps: float,
          upstream_bwd_mul: bool) -> List[torch.Tensor]:
    _, head_k, norm_k, _, apply_k, _, _ = _kernels()
    streams = streams.contiguous()
    outer = streams.shape[:-2]
    n, d = streams.shape[-2], streams.shape[-1]
    n3 = mhc_mult * (2 + mhc_mult)
    fn_pad = F.pad(fn, (0, 0, 0, n3 - fn.shape[0])) if fn.shape[0] < n3 else fn
    x_flat = streams.view(-1, n * d)
    tokens = x_flat.shape[0]
    mixes, out_mul, sqrsum, fn_t = _project_forward(x_flat, fn_pad, normw, norm_eps, norm_k)
    mixes_h = mixes[:, :mhc_mult].contiguous()
    mix = torch.empty_like(mixes_h)
    head_k._mhc_head_compute_mix_fwd(mhc_mult, pre_eps, token_block_size=32)(
        mixes_h, scale.reshape(1), base, mix
    )
    out = torch.empty(*outer, d, dtype=torch.bfloat16, device=streams.device)
    apply_k._mhc_pre_apply_mix_fwd(n, d)(streams.view(-1, n, d), mix, out.view(-1, d))
    return [out, mix.view(*outer, n), mixes_h.view(*outer, n), out_mul, sqrsum, fn_t]


@_head.register_fake
def _(streams, fn, normw, scale, base, mhc_mult, pre_eps, norm_eps, upstream_bwd_mul):
    outer = streams.shape[:-2]
    n, d = streams.shape[-2], streams.shape[-1]
    n3 = mhc_mult * (2 + mhc_mult)
    f32 = dict(dtype=torch.float32, device=streams.device)
    return [torch.empty(*outer, d, dtype=torch.bfloat16, device=streams.device),
            torch.empty(*outer, n, **f32), torch.empty(*outer, n, **f32),
            torch.empty(_prod(outer), 1, n3, **f32), torch.empty(_prod(outer), 1, **f32),
            torch.empty(n3, fn.shape[1], **f32)]


@torch.library.custom_op(f"{_LIB}::head_backward", mutates_args=())
def _head_backward_op(
    d_out: torch.Tensor, streams: torch.Tensor, fn: torch.Tensor, normw: torch.Tensor,
    scale: torch.Tensor, base: torch.Tensor, mix: torch.Tensor, mixes_h: torch.Tensor,
    out_mul: torch.Tensor, sqrsum: torch.Tensor, fn_t: torch.Tensor, mhc_mult: int,
    norm_eps: float, upstream_bwd_mul: bool,
) -> List[torch.Tensor]:
    get_num_sms, head_k, norm_k, _, apply_k, _, _ = _kernels()
    n, d = streams.shape[-2], streams.shape[-1]
    n3 = mhc_mult * (2 + mhc_mult)
    x = streams.view(-1, n, d)
    tokens = x.shape[0]
    x_grad = torch.zeros_like(x)
    d_mix = apply_k._mhc_pre_apply_mix_bwd(n, d)(
        d_out.contiguous().view(tokens, d), x, mix.view(tokens, n), x_grad
    )
    num_sms = get_num_sms()
    d_mixes_h = torch.empty(tokens, mhc_mult, dtype=torch.float32, device=streams.device)
    d_scale_part = torch.empty(num_sms, 1, dtype=scale.dtype, device=scale.device)
    d_base_part = torch.empty(num_sms, *base.shape, dtype=base.dtype, device=base.device)
    head_k._mhc_head_compute_mix_bwd(mhc_mult, token_block_size=32, num_sms=num_sms)(
        d_mix.contiguous().view(tokens, n), mixes_h.view(tokens, n), scale.reshape(1), base,
        d_mixes_h, d_scale_part, d_base_part,
    )
    d_mixes = torch.zeros(tokens, n3, dtype=torch.float32, device=streams.device)
    d_mixes[:, :mhc_mult] = d_mixes_h
    fn_pad = F.pad(fn, (0, 0, 0, n3 - fn.shape[0])) if fn.shape[0] < n3 else fn
    x_grad_flat, d_fn_pad, d_normw = _project_backward(
        d_mixes, x.view(tokens, n * d), fn_pad, normw, fn_t, out_mul, sqrsum, norm_eps,
        x_grad.view(tokens, n * d), norm_k, upstream_bwd_mul,
    )
    return [x_grad_flat.view_as(streams), d_fn_pad[: fn.shape[0]].contiguous(), d_normw,
            d_scale_part.sum(0).reshape(scale.shape), d_base_part.sum(0)]


@_head_backward_op.register_fake
def _(d_out, streams, fn, normw, scale, base, mix, mixes_h, out_mul, sqrsum, fn_t, mhc_mult, norm_eps,
      upstream_bwd_mul):
    return [torch.empty_like(streams), torch.empty_like(fn), torch.empty_like(normw),
            torch.empty_like(scale), torch.empty_like(base)]


def _head_setup(ctx, inputs, output):
    streams, fn, normw, scale, base, mhc_mult, pre_eps, norm_eps, upstream_bwd_mul = inputs
    _, mix, mixes_h, out_mul, sqrsum, fn_t = output
    ctx.save_for_backward(streams, fn, normw, scale, base, mix, mixes_h, out_mul, sqrsum, fn_t)
    ctx.consts = (mhc_mult, norm_eps, upstream_bwd_mul)


def _head_backward(ctx, *grads):
    d_out, *_ = _grads(grads)
    out = torch.ops.olmo_mhc.head_backward(d_out, *ctx.saved_tensors, *ctx.consts)
    return (*out, None, None, None, None)


torch.library.register_autograd(f"{_LIB}::head", _head_backward, setup_context=_head_setup)


def mhc_head(residual, fn, scale, base, *, norm_weight, norm_eps=1e-6, mhc_mult=4, pre_eps=1e-6,
             n_splits=16, fuse_grad_acc=False, upstream_bwd_mul=False):
    """Upstream ``functional.mhc_head`` signature, one traceable op. ``n_splits`` is accepted
    for signature parity (the projection kernel has one split); ``fuse_grad_acc`` must be off."""
    del n_splits
    if fuse_grad_acc:
        raise NotImplementedError("the traceable mhc_head returns gradients; main_grad fusion is off")
    return torch.ops.olmo_mhc.head(residual, fn, norm_weight, scale, base, mhc_mult, pre_eps, norm_eps,
                                   bool(upstream_bwd_mul))[0]


# ============================================================================================
# Recompute: mHC paper section 4.3.2 / Table 3.
#
# Per group of L_r consecutive connectors only x_{l0} (the group's entry residual, n*C per
# token), every layer's mixes (n^2 + 2n scalars per token) and every layer's branch output
# F(H^pre x_l) (C per token) are stored. x_l (n*C), H^pre x_l (C) and RMSNorm(H^pre x_l) (C)
# are transient: rebuilt in backward by upstream's ``mhc_multilayer_recompute`` kernel (the
# residual updates of the group without F) plus one RMSNorm per layer. Autograd visits the
# group's layers in reverse, so the first consumer (the last layer's ``post_r`` backward) holds
# the longest prefix and replays the whole group once; the cache serves every other consumer
# and is dropped when the last one has read it. Peak transient memory is the paper's
# (n + 2) C L_r per token.
#
# Everything here is traceable: three custom ops (``group_replay``, ``pre_r``, ``post_r``) and
# one ``autograd.Function`` whose forward and backward are plain tensor code, so a block that
# uses them compiles as one graph and the recompute lives in the compiled backward.
# ============================================================================================
_REPLAY: dict = {}   # storage ptr of x_{l0} -> {"inputs": [H^pre x_l], "streams": [x_1..], "pending": n}


def _replay_kernel():
    from olmo_core.kernels.tile_kernels.modeling.mhc.ops.multilayer_recompute import mhc_multilayer_recompute

    return mhc_multilayer_recompute


def _replay_key(x0: torch.Tensor) -> int:
    return x0.untyped_storage().data_ptr()


@torch.library.custom_op(f"{_LIB}::group_replay", mutates_args=())
def _group_replay(
    x0: torch.Tensor, pre_mixes: List[torch.Tensor], post_mixes: List[torch.Tensor],
    combs: List[torch.Tensor], outs: List[torch.Tensor], anchor: torch.Tensor, position: int,
    consumers: int, want_input: bool,
) -> torch.Tensor:
    """Rebuild the group's residual streams from ``x0`` (upstream ``mhc_multilayer_recompute``)
    and return layer ``position``'s stream ``x_l`` (``want_input=False``, ``position >= 1``; the
    callers use ``x0`` directly for position 0) or its branch input ``H^pre x_l``.

    ``pre_mixes`` holds layers ``0..position`` and the other lists ``0..position-1``. The first
    caller of a group replays and caches; the cache is released after ``consumers`` reads or when
    the next group starts replaying (backward handles one group at a time).

    ``anchor`` is unused: it is a cotangent of the calling backward. Every other input exists at
    forward time, and AOTAutograd's partitioner would otherwise schedule this op in the compiled
    *forward* and save its output, which is exactly the storage the scheme removes."""
    del anchor
    if len(pre_mixes) <= position or any(len(lst) < position for lst in (post_mixes, combs, outs)):
        raise RuntimeError(
            f"mHC group replay for position {position} needs pre mixes 0..{position} and the other "
            f"lists 0..{position - 1}; got {len(pre_mixes)}, {len(post_mixes)}, {len(combs)}, {len(outs)}"
        )
    key = _replay_key(x0)
    entry = _REPLAY.get(key)
    if entry is None or len(entry["inputs"]) <= position:
        _REPLAY.clear()
        layers = len(pre_mixes)
        inputs = [torch.empty(*x0.shape[:-2], x0.shape[-1], dtype=x0.dtype, device=x0.device)
                  for _ in range(layers)]
        streams = [torch.empty_like(x0) for _ in range(layers - 1)]
        _replay_kernel()(x0, list(pre_mixes), list(outs[:layers - 1]), list(post_mixes[:layers - 1]),
                         list(combs[:layers - 1]), inputs, streams)
        entry = _REPLAY[key] = {"inputs": inputs, "streams": streams, "pending": consumers}
    entry["pending"] -= 1
    out = entry["inputs"][position] if want_input else entry["streams"][position - 1]
    if entry["pending"] <= 0:
        _REPLAY.pop(key, None)
    return out


@_group_replay.register_fake
def _(x0, pre_mixes, post_mixes, combs, outs, anchor, position, consumers, want_input):
    if want_input:
        return torch.empty(*x0.shape[:-2], x0.shape[-1], dtype=x0.dtype, device=x0.device)
    return torch.empty_like(x0)


def _consumers(n_layers: int) -> int:
    # pre_r, post_r and the branch input of every layer read the cache, except position 0's
    # pre_r / post_r, whose stream is x0 itself.
    return 3 * n_layers - 2


def _stream_for(x0, pre_mixes, post_mixes, combs, outs, anchor, position, n_layers):
    if position == 0:
        return x0
    return torch.ops.olmo_mhc.group_replay(x0, pre_mixes, post_mixes, combs, outs, anchor, position,
                                           _consumers(n_layers), False)


def _first(*tensors):
    for tensor in tensors:
        if tensor is not None:
            return tensor
    raise RuntimeError("backward called without any cotangent")


# ---- pre_r: ``pre`` whose backward rebuilds the streams --------------------------------------
@torch.library.custom_op(f"{_LIB}::pre_r", mutates_args=())
def _pre_r(
    streams: torch.Tensor, fn: torch.Tensor, normw: torch.Tensor, alpha: torch.Tensor,
    b: torch.Tensor, x0: torch.Tensor, g_pre: List[torch.Tensor], g_post: List[torch.Tensor],
    g_comb: List[torch.Tensor], g_out: List[torch.Tensor], position: int, n_layers: int,
    mhc_mult: int, post_mult: float, pre_eps: float, sinkhorn_eps: float, sinkhorn_repeat: int,
    norm_eps: float, upstream_bwd_mul: bool,
) -> List[torch.Tensor]:
    if position == 0:
        _REPLAY.pop(_replay_key(x0), None)   # a new group at this address: drop any stale replay
    return torch.ops.olmo_mhc.pre(streams, fn, normw, alpha, b, mhc_mult, post_mult, pre_eps,
                                  sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul)


@_pre_r.register_fake
def _(streams, fn, normw, alpha, b, x0, g_pre, g_post, g_comb, g_out, position, n_layers, mhc_mult,
      post_mult, pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul):
    return torch.ops.olmo_mhc.pre(streams, fn, normw, alpha, b, mhc_mult, post_mult, pre_eps,
                                  sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul)


def _pre_r_setup(ctx, inputs, output):
    (streams, fn, normw, alpha, b, x0, g_pre, g_post, g_comb, g_out, position, n_layers, mhc_mult,
     post_mult, pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul) = inputs
    _, pre_mix, post_mix, _, mixes, out_mul, sqrsum, comb_logits, fn_t = output
    # ``streams`` is not saved (Table 3: x_l is transient).
    ctx.save_for_backward(fn, normw, alpha, b, x0, pre_mix, post_mix, mixes, out_mul, sqrsum,
                          comb_logits, fn_t, *g_pre, *g_post, *g_comb, *g_out)
    ctx.consts = (mhc_mult, post_mult, sinkhorn_eps, sinkhorn_repeat, norm_eps, upstream_bwd_mul)
    ctx.group = (position, n_layers, len(g_pre), len(g_post), len(g_comb), len(g_out))


def _split_lists(rest, counts):
    lists = []
    for count in counts:
        lists.append(list(rest[:count]))
        rest = rest[count:]
    return lists


def _pre_r_backward(ctx, *grads):
    d_layer_input, d_pre_mix, d_post_mix, d_comb, *_ = _grads(grads)
    saved = ctx.saved_tensors
    fn, normw, alpha, b, x0, pre_mix, post_mix, mixes, out_mul, sqrsum, comb_logits, fn_t = saved[:12]
    position, n_layers, *counts = ctx.group
    g_pre, g_post, g_comb, g_out = _split_lists(saved[12:], counts)
    anchor = _first(d_layer_input, d_pre_mix, d_post_mix, d_comb)
    streams = _stream_for(x0, g_pre + [pre_mix], g_post, g_comb, g_out, anchor, position, n_layers)
    if d_layer_input is None:
        d_layer_input = x0.new_zeros(*x0.shape[:-2], x0.shape[-1])
    if d_pre_mix is None:
        d_pre_mix = x0.new_zeros(0, dtype=torch.float32)
    if d_post_mix is None:
        d_post_mix = torch.zeros_like(post_mix)
    if d_comb is None:
        d_comb = torch.zeros_like(comb_logits)
    out = torch.ops.olmo_mhc.pre_backward(
        d_layer_input, d_pre_mix, d_post_mix, d_comb, streams, fn, normw, alpha, b, pre_mix,
        post_mix, mixes, out_mul, sqrsum, comb_logits, fn_t, *ctx.consts,
    )
    # One gradient slot per input: x0, the four lists (a list of None each), nine scalars.
    n_pre, n_post, n_comb, n_out = counts
    return (*out, None, [None] * n_pre, [None] * n_post, [None] * n_comb, [None] * n_out,
            None, None, None, None, None, None, None, None, None)


torch.library.register_autograd(f"{_LIB}::pre_r", _pre_r_backward, setup_context=_pre_r_setup)


def mhc_pre_recompute(streams, fn, alpha, b, *, group, norm_weight, norm_eps, mhc_mult, post_mult_value,
                      pre_eps, sinkhorn_eps, sinkhorn_repeat, upstream_bwd_mul=False):
    """``mhc_pre`` for a connector inside a recompute group (``hyper_connections.MHCGroup``):
    same kernels and outputs; the backward rebuilds this layer's streams from the group's entry
    residual instead of storing them."""
    out = torch.ops.olmo_mhc.pre_r(
        streams, fn, norm_weight, alpha, b, group.x0, list(group.pre_mixes), list(group.post_mixes),
        list(group.combs), list(group.outputs), len(group.pre_mixes), group.n_layers, mhc_mult,
        post_mult_value, pre_eps, sinkhorn_eps, sinkhorn_repeat, norm_eps, bool(upstream_bwd_mul),
    )
    return out[0], out[1], out[2], out[3]


# ---- post_r: ``post`` whose backward rebuilds the residual streams ----------------------------
@torch.library.custom_op(f"{_LIB}::post_r", mutates_args=())
def _post_r(
    x: torch.Tensor, residual: torch.Tensor, post_mix: torch.Tensor, comb: torch.Tensor,
    x0: torch.Tensor, g_pre: List[torch.Tensor], g_post: List[torch.Tensor],
    g_comb: List[torch.Tensor], g_out: List[torch.Tensor], position: int, n_layers: int,
) -> torch.Tensor:
    return torch.ops.olmo_mhc.post(x, residual, post_mix, comb)


@_post_r.register_fake
def _(x, residual, post_mix, comb, x0, g_pre, g_post, g_comb, g_out, position, n_layers):
    return torch.empty_like(residual)


def _post_r_setup(ctx, inputs, output):
    x, residual, post_mix, comb, x0, g_pre, g_post, g_comb, g_out, position, n_layers = inputs
    ctx.save_for_backward(x, post_mix, comb, x0, *g_pre, *g_post, *g_comb, *g_out)   # not ``residual``
    ctx.group = (position, n_layers, len(g_pre), len(g_post), len(g_comb), len(g_out))


def _post_r_backward(ctx, d_out):
    saved = ctx.saved_tensors
    x, post_mix, comb, x0 = saved[:4]
    position, n_layers, *counts = ctx.group
    g_pre, g_post, g_comb, g_out = _split_lists(saved[4:], counts)
    residual = _stream_for(x0, g_pre, g_post, g_comb, g_out, d_out, position, n_layers)
    d_x, d_res, d_post, d_comb = torch.ops.olmo_mhc.post_backward(d_out, x, residual, post_mix, comb)
    n_pre, n_post, n_comb, n_out = counts
    return (d_x, d_res, d_post, d_comb, None, [None] * n_pre, [None] * n_post, [None] * n_comb,
            [None] * n_out, None, None)


torch.library.register_autograd(f"{_LIB}::post_r", _post_r_backward, setup_context=_post_r_setup)


def mhc_post_recompute(x, residual, post_mix, comb, *, group):
    """``mhc_post`` inside a recompute group: the residual streams are rebuilt in backward.
    Call after the layer's ``pre_mix`` has been appended to the group and before its
    ``post_mix`` / ``comb`` / branch output are."""
    return torch.ops.olmo_mhc.post_r(
        x, residual, post_mix, comb, group.x0, list(group.pre_mixes), list(group.post_mixes),
        list(group.combs), list(group.outputs), len(group.post_mixes), group.n_layers,
    )


# ---- branch input: RMSNorm(H^pre x_l) and the branch's first projections ----------------------
def _rms_norm_parts(x: torch.Tensor, eps: float):
    x32 = x.float()
    inv = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return x32 * inv, inv


class _ReplayedBranchInput(torch.autograd.Function):
    """``linear_i(RMSNorm(H^pre x_l))`` for the branch's first projections, storing neither the
    norm's input nor its output: both are rebuilt in backward from the group replay (paper
    Table 3, "accounting for the pre-norm within F"). The norm is OLMo's ``RMSNorm`` in full
    precision (fp32 statistics, weight cast to fp32, output in the input dtype); the projections
    are ordinary ``F.linear`` calls, their weight gradients the same bf16 GEMMs autograd runs.
    """

    @staticmethod
    def forward(ctx, layer_input, norm_weight, norm_bias, eps, position, n_layers, n_lin, n_pre,
                n_post, n_comb, n_out, x0, *tensors):
        weights = tensors[:n_lin]
        biases = tensors[n_lin:2 * n_lin]
        groups = tensors[2 * n_lin:]
        xhat, _ = _rms_norm_parts(layer_input, eps)
        normed = norm_weight.float() * xhat
        if norm_bias is not None:
            normed = normed + norm_bias.float()
        normed = normed.to(layer_input.dtype)
        outputs = tuple(F.linear(normed, w, bias) for w, bias in zip(weights, biases))
        ctx.save_for_backward(norm_weight, x0, *weights, *groups)
        ctx.meta = (eps, position, n_layers, n_lin, n_pre, n_post, n_comb, n_out,
                    norm_bias is not None, tuple(bias is not None for bias in biases))
        return outputs

    @staticmethod
    def backward(ctx, *d_outputs):
        eps, position, n_layers, n_lin, n_pre, n_post, n_comb, n_out, has_norm_bias, has_bias = ctx.meta
        saved = ctx.saved_tensors
        norm_weight, x0 = saved[:2]
        weights = saved[2:2 + n_lin]
        g_pre, g_post, g_comb, g_out = _split_lists(saved[2 + n_lin:], (n_pre, n_post, n_comb, n_out))
        layer_input = torch.ops.olmo_mhc.group_replay(
            x0, g_pre, g_post, g_comb, g_out, _first(*d_outputs), position, _consumers(n_layers), True
        )
        xhat, inv = _rms_norm_parts(layer_input, eps)
        w32 = norm_weight.float()
        normed = w32 * xhat
        # (the norm bias does not enter the backward of x or the weight)
        normed = normed.to(layer_input.dtype)
        dim = normed.shape[-1]
        normed_2d = normed.reshape(-1, dim)
        d_weights, d_biases, d_normed = [], [], None
        for d_out, w, bias in zip(d_outputs, weights, has_bias):
            if d_out is None:
                d_weights.append(None)
                d_biases.append(None)
                continue
            d_2d = d_out.reshape(-1, d_out.shape[-1])
            d_weights.append(d_2d.t() @ normed_2d)
            d_biases.append(d_2d.sum(0) if bias else None)
            contribution = d_out @ w
            d_normed = contribution if d_normed is None else d_normed + contribution
        if d_normed is None:
            d_normed = torch.zeros_like(normed)
        d_n32 = d_normed.float()
        d_norm_weight = (d_n32 * xhat).reshape(-1, dim).sum(0).to(norm_weight.dtype)
        d_norm_bias = d_n32.reshape(-1, dim).sum(0).to(norm_weight.dtype) if has_norm_bias else None
        d_xhat = d_n32 * w32
        d_x = inv * (d_xhat - xhat * (d_xhat * xhat).mean(-1, keepdim=True))
        return (d_x.to(layer_input.dtype), d_norm_weight, d_norm_bias, None, None, None, None, None,
                None, None, None, None, *d_weights, *d_biases,
                *([None] * (n_pre + n_post + n_comb + n_out)))


def mhc_branch_input(layer_input, norm, linears, group):
    """Apply the branch's pre-norm and its first linear layers to ``H^pre x_l`` inside a
    recompute group. Returns one tensor per linear. Neither ``layer_input`` nor the normalized
    tensor is stored for backward."""
    if getattr(norm, "weight", None) is None or not getattr(norm, "full_precision", True):
        raise NotImplementedError(
            "mHC recompute rebuilds the branch norm as full-precision RMSNorm with a weight"
        )
    eps = norm.eps if norm.eps is not None else torch.finfo(layer_input.dtype).eps
    weights = [linear.weight for linear in linears]
    biases = [linear.bias for linear in linears]
    return _ReplayedBranchInput.apply(
        layer_input, norm.weight, getattr(norm, "bias", None), float(eps), len(group.post_mixes),
        group.n_layers, len(linears), len(group.pre_mixes), len(group.post_mixes), len(group.combs),
        len(group.outputs), group.x0, *weights, *biases, *group.pre_mixes, *group.post_mixes,
        *group.combs, *group.outputs,
    )
