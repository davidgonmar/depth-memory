"""The address sites: they turn a stream into a write's key, value and scale, or into a read's
query.

These are plain functions; the modules only lend their parameters. The caller chooses each source
(the current or updated stream, the branch delta or the embedding) and runs the state step. Any
number of leading token axes is allowed: values come in as [..., d_model] and leave as unweighted
[..., G, V] views.

Reductions and nonlinearities run in FP32, or FP64 for FP64 inputs. Linear
layers run in ``compute_dtype``, else in the autocast dtype or the source dtype. The normalized key
and the gated query are rounded to BF16 before they are widened to the FP32 kernel inputs.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class PendingWrite(NamedTuple):
    """The write's inputs to the next state step.

    ``keyhat`` is the normalized key [..., G, K] in FP32 (FP64 for FP64 inputs), ``raw_value`` the value
    [..., G, V] in the compute dtype (a view), ``write_scale`` is sqrt(weight) [..., G] in FP32
    and ``forgetting`` has the logit's broadcastable shape, usually [G]. Key and value are not
    weighted here: the state applies the scale to both and returns one gradient for it::

        read, token = state.step(token, pending.keyhat, prepared.query,
                                 pending.raw_value, pending.write_scale, pending.forgetting)
        output = finish_read(read, prepared, add_to=stream, grouped=True)
    """

    keyhat: torch.Tensor
    raw_value: torch.Tensor
    write_scale: torch.Tensor
    forgetting: torch.Tensor


class PreparedRead(NamedTuple):
    """The query, FP32 [..., G, K]."""

    query: torch.Tensor


def _accumulation_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _working_dtype(source: torch.Tensor, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        return dtype
    if source.dtype != torch.float64 and torch.is_autocast_enabled(source.device.type):
        return torch.get_autocast_dtype(source.device.type)
    return source.dtype


def _scaled_weight(layer: nn.Module, dtype: torch.dtype, scale: torch.Tensor | None) -> torch.Tensor:
    """The weight with an input-channel scale folded in, W diag(s), so that x @ W'.T is
    (x * s) @ W.T. The reference backend rounds x * s to the input dtype before the GEMM; this
    rounds W * s instead."""
    weight = layer.weight.to(dtype)
    if scale is None:
        return weight
    return weight * scale.to(dtype).unsqueeze(0)


def _linear(x: torch.Tensor, layer: nn.Module, scale: torch.Tensor | None = None) -> torch.Tensor:
    bias = getattr(layer, "bias", None)
    return F.linear(x, _scaled_weight(layer, x.dtype, scale), None if bias is None else bias.to(x.dtype))


def context_gain(projection: nn.Module | None) -> torch.Tensor | None:
    """The learnable RMS gain of a projection's address context, if it has one."""
    norm = None if projection is None else getattr(projection, "context_norm", None)
    return None if norm is None else norm.weight


def linear_group(x: torch.Tensor, layers: tuple[nn.Module, ...],
                 scales: tuple[torch.Tensor | None, ...] | None = None) -> tuple[torch.Tensor, ...]:
    """Linear layers of any widths on the same input, as one GEMM on the concatenated weights.

    Each weight is cast to the compute dtype before the concatenation. The input gradient is then
    one GEMM, rounded once, instead of a sum of separately rounded per-layer gradients. Each
    layer still receives its own parameter gradients.
    """
    if not layers:
        return ()
    if scales is None:
        scales = (None,) * len(layers)
    if len(layers) == 1:
        return (_linear(x, layers[0], scales[0]),)
    weights = tuple(_scaled_weight(layer, x.dtype, scale) for layer, scale in zip(layers, scales, strict=True))
    biases = tuple(getattr(layer, "bias", None) for layer in layers)
    widths = tuple(weight.shape[0] for weight in weights)
    bias = None
    if any(value is not None for value in biases):
        bias = torch.cat(tuple(
            weight.new_zeros(weight.shape[0]) if value is None else value.to(x.dtype)
            for weight, value in zip(weights, biases, strict=True)
        ))
    return F.linear(x, torch.cat(weights), bias).split(widths, dim=-1)


def address_networks(source, layers, *, eps, gains=None):
    """The first-layer hidden rows of the address networks of one source, from one
    ``ldm::address_network`` call. ``gains`` holds each network's learnable context RMS gain, or
    None; it is folded into that network's W1."""
    from .address_network import packed_address_networks

    return packed_address_networks(source, fold_context_gains(layers, gains), eps=eps)


class _FoldedLayer(NamedTuple):
    """W1 with the context gain folded into its input channels."""
    weight: torch.Tensor
    bias: torch.Tensor | None


def fold_context_gains(layers, gains):
    """The layers with their context gains folded into the weights; a layer without a gain is
    returned unchanged."""
    if gains is None or all(g is None for g in gains):
        return tuple(layers)
    return tuple(
        layer if g is None else _FoldedLayer(_scaled_weight(layer, layer.weight.dtype, g),
                                             getattr(layer, "bias", None))
        for layer, g in zip(layers, gains, strict=True)
    )


class _AddressFeatures(NamedTuple):
    address: torch.Tensor
    hidden: torch.Tensor | None
    address_dtype: torch.dtype
    gate_logit: torch.Tensor | None


def _packed_address(source, projection, static_address, *, rms_eps, eps, normalize,
                    gate, hidden=None):
    if projection is None:
        address = _static_address(source, static_address, eps=eps, normalize=normalize)
        return _AddressFeatures(address, None, static_address.dtype, None)
    if hidden is None:
        hidden, = address_networks(source, (projection.w1,), eps=rms_eps,
                                 gains=(context_gain(projection),))
    return _address_heads(hidden, projection, static_address, eps=eps,
                          normalize=normalize, gate=gate)


def _address_heads(hidden, projection, static_address, *, eps, normalize, gate):
    """The address from the hidden rows: W2, the optional gate and the grouped normalization."""
    layers = (projection.w2,)
    if gate:
        layers += (projection.gate,)
    outputs = linear_group(hidden, layers)
    delta = outputs[0].reshape(*hidden.shape[:-1], *static_address.shape)
    address = grouped_address_norm(delta, static_address, eps=eps, normalize=normalize)
    return _AddressFeatures(address, hidden, delta.dtype,
                            outputs[1] if gate else None)


def _write_result(features, raw_value, static_address, lambda_logit, *, dtype, eps, lambda_init):
    if features.gate_logit is not None:
        logit = features.gate_logit
        weight = (2 * torch.sigmoid(logit.to(_accumulation_dtype(logit)))).to(features.address_dtype)
        scale = sqrt_write_weight(weight, eps=eps)
    else:
        scale = torch.ones_like(features.address[..., 0], dtype=_accumulation_dtype(features.address))
    return PendingWrite(
        features.address.to(_accumulation_dtype(features.address)),
        raw_value.to(dtype).unflatten(-1, (static_address.shape[0], -1)),
        scale, forgetting_value(lambda_logit, lambda_init=lambda_init, eps=eps),
    )


def _read_result(features, projection, static_address, *, gain_logit, dtype, linear_gain=False):
    query = features.address
    if features.gate_logit is not None:
        query = query * _read_weight_from_logit(features.gate_logit, projection, static_address)
    if gain_logit is not None:
        # The parameter is the log of the gain unless the gain is linear.
        gain = gain_logit.to(_accumulation_dtype(gain_logit))
        if not linear_gain:
            gain = gain.exp()
        gain = gain.reshape(()) if gain.numel() == 1 else gain.unsqueeze(-1)
        query = query * gain.to(query.dtype)
    return PreparedRead(query.to(_accumulation_dtype(query)))


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    eps: float,
) -> torch.Tensor:
    """RMS normalization with an optional weight and bias, in FP32 (FP64 for FP64 inputs) and
    rounded once to the input dtype. Without weight and bias it is the address context."""
    with torch.autocast(x.device.type, enabled=False):
        value = x.to(_accumulation_dtype(x))
        out = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        if weight is not None:
            out = out * weight.to(out.dtype)
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.to(x.dtype)


def grouped_address_norm(
    address: torch.Tensor,
    static_address: torch.Tensor,
    *,
    eps: float,
    normalize: bool = True,
) -> torch.Tensor:
    """The address plus the static address, normalized per group. With operands of different
    dtypes this is raw / max(||raw||, eps) in the promoted dtype; with equal dtypes it is
    raw * rsqrt(sum(raw^2) + eps), rounded back to that dtype."""
    with torch.autocast(address.device.type, enabled=False):
        accumulation = _accumulation_dtype(address, static_address)
        raw = address.to(accumulation) + static_address.to(accumulation)
        if normalize:
            if address.dtype != static_address.dtype:
                raw = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps)
            else:
                raw = raw * torch.rsqrt(raw.square().sum(-1, keepdim=True) + eps)
        return raw.to(torch.promote_types(address.dtype, static_address.dtype))


def _static_address(source, static_address, *, eps, normalize):
    raw = static_address.to(_accumulation_dtype(static_address))
    if normalize:
        raw = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(eps)
    return raw.to(static_address.dtype).expand(*source.shape[:-1], *static_address.shape)


def forgetting_value(logit: torch.Tensor, *, lambda_init: float, eps: float) -> torch.Tensor:
    """The forgetting factor sigmoid(logit), floored at eps. With lambda_init >= 1 it is exactly 1."""
    with torch.autocast(logit.device.type, enabled=False):
        value = logit.to(_accumulation_dtype(logit))
        if lambda_init >= 1.0:
            return torch.ones_like(value)
        return torch.sigmoid(value).clamp_min(eps)


def sqrt_write_weight(weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    """The write scale sqrt(max(weight, eps)), applied to both key and value."""
    return weight.to(_accumulation_dtype(weight)).clamp_min(eps).sqrt()


def scale_write_value(raw_value: torch.Tensor, write_scale: torch.Tensor) -> torch.Tensor:
    """FP32(raw_value) * write_scale as the kernel computes it, [..., G, V] with no BF16
    rounding."""
    dtype = _accumulation_dtype(raw_value, write_scale)
    return raw_value.to(dtype) * write_scale.to(dtype).unsqueeze(-1)


def _gate_enabled(projection: nn.Module | None, name: str, enabled: bool | None) -> bool:
    present = projection is not None and getattr(projection, name, None) is not None
    if enabled is True and not present:
        raise ValueError(f"enabled {name} requires an existing projection.{name}")
    return present if enabled is None else enabled


def prepare_write(
    source: torch.Tensor,
    raw_value: torch.Tensor,
    projection: nn.Module | None,
    static_address: torch.Tensor,
    lambda_logit: torch.Tensor,
    *,
    rms_eps: float,
    eps: float,
    lambda_init: float,
    write_gate: bool | None = None,
    normalize: bool = True,
    compute_dtype: torch.dtype | None = None,
) -> PendingWrite:
    """The write of the embedding or of a branch output.

    ``projection`` is the key projection (input_k_proj, attn_k_proj or mlp_k_proj), or None for a
    static address. ``write_gate=None`` uses the projection's gate if it has one; False turns a
    registered gate off. ``source`` and ``raw_value`` are [..., d_model]; the returned value is a
    grouped [..., G, V] view.
    """
    dtype = _working_dtype(source, compute_dtype)
    with torch.autocast(source.device.type, enabled=False):
        features = _packed_address(
            source.to(dtype), projection, static_address, rms_eps=rms_eps, eps=eps,
            normalize=normalize, gate=_gate_enabled(projection, "gate", write_gate),
        )
        return _write_result(
            features, raw_value, static_address, lambda_logit, dtype=dtype, eps=eps, lambda_init=lambda_init,
        )


def _read_weight_from_logit(logit, projection, static_address):
    wide = logit.to(_accumulation_dtype(logit))
    kind = getattr(projection, "gate_parametrization", "unit_sigmoid")
    if kind == "per_coord_sigmoid":
        return (2 * torch.sigmoid(wide)).to(logit.dtype).reshape(*logit.shape[:-1], *static_address.shape)
    if kind == "group_softmax":
        scale = projection.gate_log_scale.to(wide.dtype).exp()
        weight = static_address.shape[0] * scale * torch.softmax(wide, dim=-1)
    else:
        weight = 2 * torch.sigmoid(wide)
        if kind in ("log_scaled_sigmoid", "static_scale"):
            weight = weight * projection.gate_log_scale.to(wide.dtype).exp()
        elif kind != "unit_sigmoid":
            raise ValueError(f"unknown read-gate parameterization {kind!r}")
    return weight.to(logit.dtype).unsqueeze(-1)


def prepare_read(
    source: torch.Tensor,
    projection: nn.Module | None,
    static_address: torch.Tensor,
    *,
    rms_eps: float,
    eps: float,
    gain_logit: torch.Tensor | None = None,
    linear_gain: bool = False,
    read_gate: bool | None = None,
    normalize: bool = True,
    compute_dtype: torch.dtype | None = None,
) -> PreparedRead:
    """The attention, MLP or final query from the given source (the stream, or the branch delta
    for delta queries)."""
    dtype = _working_dtype(source, compute_dtype)
    with torch.autocast(source.device.type, enabled=False):
        features = _packed_address(
            source.to(dtype), projection, static_address, rms_eps=rms_eps, eps=eps,
            normalize=normalize, gate=_gate_enabled(projection, "gate", read_gate),
        )
        return _read_result(features, projection, static_address, gain_logit=gain_logit, dtype=dtype,
                            linear_gain=linear_gain)


def can_pair_address_networks(write_source, read_source, write_projection, read_projection,
                            *, write_rms_eps, read_rms_eps,
                            write_compute_dtype=None, read_compute_dtype=None):
    """Whether a write and a read can share one address-network GEMM: the same source tensor,
    both addresses dynamic, and the same RMS epsilon and working dtype. The caller also needs both
    sites under the same checkpoint policy."""
    return (
        write_source is read_source
        and write_projection is not None and read_projection is not None
        and write_rms_eps == read_rms_eps
        and _working_dtype(write_source, write_compute_dtype)
        == _working_dtype(read_source, read_compute_dtype)
    )


def prepare_write_read(
    write_source, raw_value, write_projection, write_static_address, lambda_logit,
    read_source, read_projection, read_static_address, *, write_rms_eps, read_rms_eps,
    eps, lambda_init, gain_logit=None, write_gate=None, read_gate=None, linear_gain=False,
    normalize_key=True, normalize_query=True,
    write_compute_dtype=None, read_compute_dtype=None,
):
    """A write and the following read on the same source, with one address-network GEMM for
    both; sites that cannot share it are prepared separately. Sharing the GEMM rounds the input
    gradient once instead of per site (see linear_group)."""
    if not can_pair_address_networks(
        write_source, read_source, write_projection, read_projection,
        write_rms_eps=write_rms_eps, read_rms_eps=read_rms_eps,
        write_compute_dtype=write_compute_dtype, read_compute_dtype=read_compute_dtype,
    ):
        return (
            prepare_write(write_source, raw_value, write_projection, write_static_address, lambda_logit,
                          rms_eps=write_rms_eps, eps=eps, lambda_init=lambda_init,
                          write_gate=write_gate, normalize=normalize_key, compute_dtype=write_compute_dtype),
            prepare_read(read_source, read_projection, read_static_address, rms_eps=read_rms_eps, eps=eps,
                         gain_logit=gain_logit, read_gate=read_gate, linear_gain=linear_gain,
                         normalize=normalize_query, compute_dtype=read_compute_dtype),
        )
    dtype = _working_dtype(write_source, write_compute_dtype)
    with torch.autocast(write_source.device.type, enabled=False):
        source = write_source.to(dtype)
        write_hidden, read_hidden = address_networks(
            source, (write_projection.w1, read_projection.w1), eps=write_rms_eps,
            gains=(context_gain(write_projection), context_gain(read_projection)))
        write_features = _packed_address(
            source, write_projection, write_static_address, rms_eps=write_rms_eps, eps=eps,
            normalize=normalize_key, gate=_gate_enabled(write_projection, "gate", write_gate),
            hidden=write_hidden,
        )
        read_features = _packed_address(
            source, read_projection, read_static_address, rms_eps=read_rms_eps, eps=eps,
            normalize=normalize_query, gate=_gate_enabled(read_projection, "gate", read_gate),
            hidden=read_hidden,
        )
        return (
            _write_result(write_features, raw_value, write_static_address, lambda_logit,
                          dtype=dtype, eps=eps, lambda_init=lambda_init),
            _read_result(read_features, read_projection, read_static_address, gain_logit=gain_logit, dtype=dtype,
                         linear_gain=linear_gain),
        )


def finish_read(
    raw_read: torch.Tensor,
    prepared: PreparedRead,
    *,
    add_to: torch.Tensor | None = None,
    grouped: bool = False,
) -> torch.Tensor:
    """The read added to ``add_to`` as [..., d_model], or the read alone when ``add_to`` is None.
    ``grouped`` flattens the state's [..., G, V] read first."""
    result = raw_read.flatten(-2) if grouped else raw_read
    return result if add_to is None else add_to + result.to(add_to.dtype)


__all__ = [
    "PendingWrite",
    "PreparedRead",
    "can_pair_address_networks",
    "finish_read",
    "grouped_address_norm",
    "linear_group",
    "prepare_read",
    "prepare_write",
    "prepare_write_read",
    "forgetting_value",
    "rms_norm",
    "scale_write_value",
    "sqrt_write_weight",
]
