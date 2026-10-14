"""The address networks of one source as the custom op ``ldm::address_network``.

The forward is an FP32-output GEMM on the raw BF16 source followed by the RMS, bias and SiLU
epilogue. The backward keeps only the source, the working weights, the FP32 GEMM output and the
row scales; it computes the parameter gradients with cuBLAS GEMMs and the input gradient with a
QuACK GEMM epilogue (kernels/address_network.py). Reference forms are at the end of the module.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def _check(x, weight, bias, eps):
    torch._check(x.ndim == 2 and weight.ndim == 2,
                 lambda: "address operands must be matrices")
    torch._check(x.dtype == weight.dtype == torch.bfloat16,
                 lambda: "address working operands must be BF16")
    torch._check(x.device == weight.device, lambda: "address devices must match")
    torch._check(x.shape[1] == weight.shape[1], lambda: "address input widths must match")
    for size in (*x.shape, weight.shape[0]):
        torch._check(size > 0, lambda: "address dimensions must be positive")
    torch._check(math.isfinite(eps) and eps > 0, lambda: "address epsilon must be finite and positive")
    if bias is not None:
        torch._check(bias.ndim == 1 and bias.shape[0] == weight.shape[0],
                     lambda: "address bias width must match")
        torch._check(bias.dtype == x.dtype and bias.device == x.device,
                     lambda: "address bias dtype/device must match")


def _raw_forward(x, weight, bias, eps):
    from .kernels.address_network import post_scale

    # The GEMM runs on the unnormalized source; the epilogue applies the RMS scale, the bias, the
    # BF16 rounding and the SiLU.
    with torch.autocast(x.device.type, enabled=False):
        u = torch.mm(x, weight.T, out_dtype=torch.float32)
        a, r = post_scale(x, u, bias, eps=eps)
    return a, u, r


def _raw_backward(da, x, weight, bias, u, r):
    from .kernels.address_network import backward
    from .kernels.address_network import config_candidates, input_grad

    capability = torch.cuda.get_device_capability(x.device)
    config = next(candidate for candidate in config_candidates(capability)
                  if candidate.tile_m == 128 and candidate.tile_n == 256)
    with torch.autocast(x.device.type, enabled=False):
        return backward(da, x, weight, u, r, bias, input_grad=input_grad, config=config)


@torch.library.custom_op("ldm::address_network", mutates_args=())
def address_network(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                    eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(x, weight, bias, eps)
    return _raw_forward(x, weight, bias, eps)


@address_network.register_fake
def _forward_fake(x, weight, bias, eps):
    _check(x, weight, bias, eps)
    shape = (x.shape[0], weight.shape[0])
    return (x.new_empty(shape), x.new_empty(shape, dtype=torch.float32),
            x.new_empty((x.shape[0],), dtype=torch.float32))


@torch.library.custom_op(
    "ldm::address_network_backward", mutates_args=(),
    schema="(Tensor da, Tensor x, Tensor weight, Tensor? bias, Tensor u, Tensor r) -> (Tensor, Tensor, Tensor?)",
)
def address_network_backward(da: torch.Tensor, x: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor | None, u: torch.Tensor,
                             r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return _raw_backward(da, x, weight, bias, u, r)


@address_network_backward.register_fake
def _backward_fake(da, x, weight, bias, u, r):
    _check(x, weight, bias, 1.0)
    shape = (x.shape[0], weight.shape[0])
    torch._check(da.shape == u.shape == shape, lambda: "address cotangent/saved widths must match")
    torch._check(da.dtype == x.dtype, lambda: "address cotangent must be BF16")
    torch._check(u.dtype == r.dtype == torch.float32, lambda: "address saved scalars must be FP32")
    torch._check(r.shape == (x.shape[0],), lambda: "address saved RMS rows must match")
    torch._check(all(t.device == x.device for t in (da, u, r)),
                 lambda: "address saved tensor devices must match")
    return (x.new_empty(x.shape), weight.new_empty(weight.shape),
            None if bias is None else bias.new_empty(bias.shape))


def _setup_context(ctx, inputs, output):
    x, weight, bias, _eps = inputs
    _a, u, r = output
    ctx.save_for_backward(x, weight, bias, u, r)
    # u and r are the backward's saved context, not differentiable outputs.
    ctx.mark_non_differentiable(u, r)
    ctx.set_materialize_grads(False)


def _backward(ctx, da, _du, _dr):
    x, weight, bias, u, r = ctx.saved_tensors
    return (*address_network_backward(da, x, weight, bias, u, r), None)


address_network.register_autograd(_backward, setup_context=_setup_context)


def _working_parameters(source, layers):
    weights = tuple(layer.weight.to(source.dtype) for layer in layers)
    widths = tuple(weight.shape[0] for weight in weights)
    biases = tuple(getattr(layer, "bias", None) for layer in layers)
    bias = None
    if any(value is not None for value in biases):
        parts = tuple(weight.new_zeros(weight.shape[0]) if value is None else value.to(source.dtype)
                      for weight, value in zip(weights, biases, strict=True))
        bias = parts[0] if len(parts) == 1 else torch.cat(parts)
    weight = weights[0] if len(weights) == 1 else torch.cat(weights)
    return weight, bias, widths


def packed_address_networks(source, layers, *, eps):
    """The hidden rows of first layers of any widths on one source, from one GEMM through the op.
    Gradients reach each layer's parameters through the casts and the concatenation."""
    if not layers:
        return ()
    weight, bias, widths = _working_parameters(source, layers)
    a, _u, _r = address_network(source.reshape(-1, source.shape[-1]), weight, bias, eps)
    return a.reshape(*source.shape[:-1], sum(widths)).split(widths, dim=-1)


def _validate(x, w, b):
    if x.dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("address reference requires BF16, FP32, or FP64 working tensors")
    if x.ndim < 1 or w.ndim != 2 or x.shape[-1] != w.shape[1]:
        raise ValueError("expected x [..., D] and w [H, D]")
    if x.shape[-1] == 0 or w.shape[0] == 0:
        raise ValueError("D and H must be positive")
    if b is not None and b.shape != (w.shape[0],):
        raise ValueError("expected b [H] or None")
    for tensor in (w,) if b is None else (w, b):
        if tensor.dtype != x.dtype or tensor.device != x.device:
            raise ValueError("operands must share the working dtype and device")
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _epsilon(eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")


def functional_forward(x, w, b, eps):
    """The unfused address network: the RMS-normalized context rounded to the working dtype,
    then W1 and the SiLU. For BF16 the linear layer's operands and output are BF16; on CPU they are
    widened only so that the products accumulate in FP32.
    """
    accum = _validate(x, w, b)
    _epsilon(eps)
    with torch.autocast(x.device.type, enabled=False):
        value = x.to(accum)
        r = torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        context = (value * r).to(x.dtype)
        z = context.to(accum) @ w.to(accum).T
        if b is not None:
            z = z + b.to(accum)
        return F.silu(z.to(x.dtype).to(accum)).to(x.dtype)


def reference_forward(x, w, b, eps):
    """The address network as the kernels compute it: a in the working dtype, the raw GEMM output
    u and the RMS scale r in the accumulation dtype."""
    accum = _validate(x, w, b)
    _epsilon(eps)
    with torch.autocast(x.device.type, enabled=False):
        value = x.to(accum)
        r = torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        # Unlike functional_forward, the source is not normalized and rounded before W1: u stays
        # FP32 and the RMS scale is applied before the bias. z and a are rounded to BF16.
        u = value @ w.to(accum).T
        z = r * u
        if b is not None:
            z = z + b.to(accum)
        a = F.silu(z.to(x.dtype).to(accum)).to(x.dtype)
        return a, u, r


def reference_backward(da, x, w, b, u, r):
    """The VJP as the kernels compute it: (dX, dW, dB) in the working dtype.

    The cotangent is rounded to the working dtype first. delta, the SiLU VJP, is rounded once and
    used by dX, the radial dot, dW and dB; only dW uses the rounded r * delta. In FP64 this serves
    finite-difference and autograd checks.
    """
    accum = _validate(x, w, b)
    output_shape = (*x.shape[:-1], w.shape[0])
    if da.shape != output_shape or u.shape != output_shape:
        raise ValueError("da and u must have shape [..., H]")
    if r.shape != (*x.shape[:-1], 1):
        raise ValueError("r must have shape [..., 1]")
    if u.dtype != accum or r.dtype != accum:
        raise ValueError("u and r must retain the accumulation dtype")
    if any(t.device != x.device for t in (da, u, r)):
        raise ValueError("saved tensors and da must share x's device")
    with torch.autocast(x.device.type, enabled=False):
        value = x.to(accum)
        z = r * u
        if b is not None:
            z = z + b.to(accum)
        z = z.to(x.dtype).to(accum)
        sigmoid = torch.sigmoid(z)
        derivative = sigmoid * (1 + z * (1 - sigmoid))
        delta = (da.to(x.dtype).to(accum) * derivative).to(x.dtype)
        wide_delta = delta.to(accum)
        # delta @ W is not rounded to BF16, and the radial term is a dot over the hidden width
        # (of all packed networks together) instead of over the source width.
        direct = wide_delta @ w.to(accum)
        radial = (wide_delta * u).sum(-1, keepdim=True)
        dx = r * direct - value * (r.pow(3) / x.shape[-1]) * radial
        # dW takes BF16(r * delta) where the unfused network would take BF16(r * x); dW and dB
        # accumulate in FP32.
        scaled_delta = (r * wide_delta).to(x.dtype).to(accum)
        dw = scaled_delta.reshape(-1, w.shape[0]).T @ value.reshape(-1, w.shape[1])
        db = None if b is None else wide_delta.reshape(-1, w.shape[0]).sum(0).to(b.dtype)
        return dx.to(x.dtype), dw.to(w.dtype), db
