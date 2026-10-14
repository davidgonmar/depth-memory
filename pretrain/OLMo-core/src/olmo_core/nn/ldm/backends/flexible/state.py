"""Segment state of the flexible backend and its first-order autograd.

The state lives in mutable tensors outside the compiled graphs. The two kernels are wrapped as
custom ops that declare their mutations; the model-facing ``step`` is a second custom-op pair that
reaches the registered segment through this module, so a block traces as one graph. A lineage token
chains the steps so backward consumes a segment in strict reverse order, unwinding the memory in
place and restoring the covariance from its tape; the segment registry checks that order.
"""

import os
from typing import NamedTuple

import torch

from . import kernels


class SegmentState(NamedTuple):
    """State of one segment; every field is a tensor so the tuple is a valid graph input."""

    memory: torch.Tensor  # [..., G, K, D/G] memory M, BF16
    covariance: torch.Tensor  # [..., G, K, K] covariance P in its storage dtype (empty without RLS)
    covariance_tape: torch.Tensor  # [S, ..., G, K, K] P before each of the S writes (empty without RLS)
    error_tape: torch.Tensor  # [S, ..., G, D/G] residuals e of the S writes, working dtype
    gain_tape: torch.Tensor  # [S, ..., G, K] FP32
    den_tape: torch.Tensor  # [S, ..., G] FP32 denominators
    lam_tape: torch.Tensor  # [S, ..., G] FP32 forgetting factors
    grad_memory: torch.Tensor  # cotangent of M, BF16
    grad_covariance: torch.Tensor  # cotangent of P, FP32
    cursor: torch.Tensor  # [1] int32, number of writes currently applied
    slot_ids: torch.Tensor  # [S] int32 slot indices kept on the device


class Slot(NamedTuple):
    """Tape rows of one write, built by the eager segment loop so no slot index enters a graph."""

    covariance: torch.Tensor  # P before the write (empty without RLS)
    error: torch.Tensor
    gain: torch.Tensor
    den: torch.Tensor
    lam: torch.Tensor
    index: torch.Tensor  # [1] int32 on the device


def allocate(
    hidden,
    *,
    groups,
    key_dim,
    alpha,
    slots,
    covariance_dtype=torch.bfloat16,
    rule=kernels.RLS,
):
    """Allocate the state for ``hidden`` and return ``(SegmentState, token)``.

    ``alpha`` is the diagonal of the initial covariance P_{-1}: a float for ``alpha * I`` or a
    ``[groups, key_dim]`` tensor.
    """
    if not hidden.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        raise ValueError("flexible requires CUDA; select reference on CPU")
    if hidden.shape[-1] % groups:
        raise ValueError("model width must be divisible by LDM groups")
    shape = (*hidden.shape[:-1], groups)
    value_dim = hidden.shape[-1] // groups
    device = hidden.device
    memory = torch.zeros(*shape, key_dim, value_dim, device=device, dtype=torch.bfloat16)
    if rule == kernels.RLS:
        diagonal = _alpha_diagonal(alpha, groups, key_dim, device)
        covariance = torch.diag_embed(diagonal).expand(*shape, key_dim, key_dim).clone().to(covariance_dtype)
        covariance_tape = torch.empty(slots, *shape, key_dim, key_dim, device=device, dtype=covariance_dtype)
    else:
        covariance = torch.empty(0, device=device, dtype=covariance_dtype)
        covariance_tape = torch.empty(slots, 0, device=device, dtype=covariance_dtype)
    state = SegmentState(
        memory=memory,
        covariance=covariance,
        covariance_tape=covariance_tape,
        error_tape=torch.empty(slots, *shape, value_dim, device=device, dtype=hidden.dtype),
        gain_tape=torch.empty(slots, *shape, key_dim, device=device, dtype=torch.float32),
        den_tape=torch.empty(slots, *shape, device=device, dtype=torch.float32),
        lam_tape=torch.empty(slots, *shape, device=device, dtype=torch.float32),
        grad_memory=torch.zeros_like(memory),
        grad_covariance=torch.zeros_like(covariance, dtype=torch.float32),
        cursor=torch.zeros(1, device=device, dtype=torch.int32),
        slot_ids=torch.arange(slots, device=device, dtype=torch.int32),
    )
    return state, hidden.new_empty(0)


def _alpha_diagonal(alpha, groups, key_dim, device):
    if isinstance(alpha, torch.Tensor):
        if tuple(alpha.shape) != (groups, key_dim):
            raise ValueError("alpha tensor must have shape [groups, key_dim]")
        return alpha.detach().to(device=device, dtype=torch.float32)
    return torch.full((groups, key_dim), float(alpha), device=device, dtype=torch.float32)


def slot_views(state, index):
    """The tape rows of write ``index``."""
    return Slot(
        state.covariance_tape[index],
        state.error_tape[index],
        state.gain_tape[index],
        state.den_tape[index],
        state.lam_tape[index],
        state.slot_ids[index : index + 1],
    )


def _rows(memory, key, query, value=None):
    """Operands expanded to one row per token/group and made contiguous for the kernels."""
    shape = memory.shape[:-2]
    key_dim, value_dim = memory.shape[-2:]
    key = key.expand(*shape, key_dim).contiguous()
    query = query.expand(*shape, key_dim).contiguous()
    if value is None:
        return key, query
    return key, query, value.expand(*shape, value_dim).contiguous()


@torch.library.custom_op(
    "ldm_flexible::step_forward",
    mutates_args=("memory", "covariance", "cov", "error", "gain", "den", "lam", "cursor"),
)
def step_forward(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    cov: torch.Tensor,
    error: torch.Tensor,
    gain: torch.Tensor,
    den: torch.Tensor,
    lam: torch.Tensor,
    cursor: torch.Tensor,
    slot: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    forgetting: torch.Tensor,
    rule: int,
) -> torch.Tensor:
    if value.dtype != error.dtype:
        raise TypeError(f"LDM write value dtype {value.dtype} must match the working dtype {error.dtype}")
    key, query, value = _rows(memory, key, query, value)
    forgetting = forgetting.float().expand(memory.shape[:-2]).contiguous()
    out = torch.empty_like(value)
    kernels.launch_forward(memory, covariance, cov, key, value, query, forgetting, out, error, gain,
                           den, lam, cursor, slot, rule)
    return out


@step_forward.register_fake
def _fake_step_forward(memory, covariance, cov, error, gain, den, lam, cursor, slot, key, value,
                       query, forgetting, rule):
    return value.new_empty((*memory.shape[:-2], memory.shape[-1]))


@torch.library.custom_op(
    "ldm_flexible::step_reverse",
    mutates_args=("memory", "covariance", "grad_memory", "grad_covariance", "cursor"),
)
def step_reverse(
    memory: torch.Tensor,
    covariance: torch.Tensor,
    cov: torch.Tensor,
    grad_memory: torch.Tensor,
    grad_covariance: torch.Tensor,
    error: torch.Tensor,
    gain: torch.Tensor,
    den: torch.Tensor,
    lam: torch.Tensor,
    cursor: torch.Tensor,
    slot: torch.Tensor,
    key: torch.Tensor,
    query: torch.Tensor,
    grad_out: torch.Tensor,
    rule: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key, query = _rows(memory, key, query)
    return kernels.launch_reverse(memory, covariance, cov, grad_memory, grad_covariance, key, query,
                                  lam, error, gain, den, grad_out.contiguous(), cursor, slot, rule)


@step_reverse.register_fake
def _fake_step_reverse(memory, covariance, cov, grad_memory, grad_covariance, error, gain, den, lam,
                       cursor, slot, key, query, grad_out, rule):
    shape = memory.shape[:-2]
    key_dim, value_dim = memory.shape[-2:]
    return (
        key.new_empty((*shape, key_dim)),
        error.new_empty((*shape, value_dim)),
        query.new_empty((*shape, key_dim)),
        torch.empty(shape, device=memory.device, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------------------------
# The model-facing step: a custom op pair Dynamo traces as opaque nodes, so a block compiles as one
# graph. The segment state is not a graph input: run_segment registers it here and the ops reach it
# through this registry, advancing a host-side position along the slots. The token chain orders the
# calls (each backward consumes the next step's token gradient), and the forward returns its segment
# id and slot index as a CPU tensor that the backward reads without a device synchronization.
# ---------------------------------------------------------------------------------------------
_ACTIVE = {"segment": None}
_SEGMENTS: dict = {}   # id -> live segment record, until its backward has unwound it
_NEXT_ID = [0]


def activate(state, views, rule=kernels.RLS):
    """Make ``state`` the segment the step ops write into; ``views`` are its slots in write order.
    With gradients enabled the segment stays reachable by id until its backward has unwound it,
    so several models' forwards and backwards may interleave."""
    record = {"state": state, "views": tuple(views), "position": 0, "rule": rule, "id": _NEXT_ID[0]}
    _NEXT_ID[0] += 1
    _ACTIVE["segment"] = record
    if torch.is_grad_enabled():
        _SEGMENTS[record["id"]] = record


def _active():
    record = _ACTIVE["segment"]
    if record is None:
        raise RuntimeError("no active flexible segment; run_segment registers one")
    return record


@torch.library.custom_op("ldm_flexible::step", mutates_args=())
def _step(token: torch.Tensor, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor,
          forgetting: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del alpha  # only its gradient is produced, by the last unwind
    record = _active()
    state, position = record["state"], record["position"]
    slot = record["views"][position]
    out = step_forward(state.memory, state.covariance, slot.covariance, slot.error, slot.gain, slot.den,
                       slot.lam, state.cursor, slot.index, key, value, query, forgetting, record["rule"])
    record["position"] = position + 1
    return out, torch.empty_like(token), torch.tensor([record["id"], position], dtype=torch.int64)


@_step.register_fake
def _fake_step(token, key, value, query, forgetting, alpha):
    state = _active()["state"]
    return (value.new_empty((*state.memory.shape[:-2], state.memory.shape[-1])), torch.empty_like(token),
            torch.empty((2,), dtype=torch.int64, device="cpu"))


@torch.library.custom_op("ldm_flexible::step_backward", mutates_args=())
def _step_backward(handle: torch.Tensor, grad_out: torch.Tensor, token_grad: torch.Tensor | None,
                   key: torch.Tensor, query: torch.Tensor,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del token_grad  # ordering dependency only
    segment_id, index = (int(v) for v in handle.tolist())
    record = _SEGMENTS.get(segment_id)
    if record is None:
        raise RuntimeError("LDM segment already unwound; run a new forward before another backward")
    if record["position"] != index + 1:
        raise RuntimeError("LDM segment unwound out of order; backward must consume it from its last write")
    state, slot = record["state"], record["views"][index]
    dk, dv, dq, dl = step_reverse(state.memory, state.covariance, slot.covariance, state.grad_memory,
                                  state.grad_covariance, slot.error, slot.gain, slot.den, slot.lam,
                                  state.cursor, slot.index, key, query, grad_out, record["rule"])
    record["position"] = index
    # alpha is the diagonal of P_{-1}: its gradient is the diagonal of the P cotangent left after the
    # first write is unwound, summed over tokens; every other write contributes zero.
    groups, key_dim = state.memory.shape[-3], state.memory.shape[-2]
    if index == 0:
        _SEGMENTS.pop(segment_id, None)
        if state.covariance.numel():
            diagonal = torch.diagonal(state.grad_covariance, dim1=-2, dim2=-1)
            dalpha = diagonal.reshape(-1, groups, key_dim).sum(0).float()
        else:
            dalpha = torch.zeros(groups, key_dim, device=key.device, dtype=torch.float32)
    else:
        dalpha = torch.zeros(groups, key_dim, device=key.device, dtype=torch.float32)
    return dk, dv, dq, dl, dalpha


@_step_backward.register_fake
def _fake_step_backward(handle, grad_out, token_grad, key, query):
    state = _active()["state"]
    shape = state.memory.shape[:-2]
    key_dim, value_dim = state.memory.shape[-2:]
    f32 = dict(device=key.device, dtype=torch.float32)
    return (key.new_empty((*shape, key_dim)), grad_out.new_empty((*shape, value_dim)),
            query.new_empty((*shape, key_dim)), torch.empty(shape, **f32),
            torch.empty((shape[-1], key_dim), **f32))


def _setup(ctx, inputs, output):
    token, key, value, query, forgetting, alpha = inputs
    ctx.save_for_backward(output[2], key, query)
    ctx.shapes = (key.shape, value.shape, query.shape, forgetting.shape, alpha.shape)
    ctx.token_dtype, ctx.out = token.dtype, (output[0].shape, output[0].dtype, output[0].device)


def _backward(ctx, grad_out, token_grad, position_grad):
    del position_grad
    handle, key, query = ctx.saved_tensors
    if grad_out is None:
        shape, dtype, device = ctx.out
        grad_out = torch.zeros(shape, dtype=dtype, device=device)
    dk, dv, dq, dl, dalpha = torch.ops.ldm_flexible.step_backward(handle, grad_out.contiguous(), token_grad, key, query)
    kshape, vshape, qshape, lshape, ashape = ctx.shapes
    return (torch.empty(0, device=key.device, dtype=ctx.token_dtype), dk.sum_to_size(kshape),
            dv.sum_to_size(vshape), dq.sum_to_size(qshape), dl.sum_to_size(lshape), dalpha.reshape(ashape))


_step.register_autograd(_backward, setup_context=_setup)


def step(key, value, query, forgetting, token, alpha):
    """One write and read of the active segment; returns ``(read, token)``.

    ``alpha`` is the ``[groups, key_dim]`` diagonal of P_{-1} (a float is broadcast) and receives
    a gradient from the first write's unwinding.
    """
    state = _active()["state"]
    key, query, value = _rows(state.memory, key, query, value)
    forgetting = forgetting.float().expand(state.memory.shape[:-2]).contiguous()
    if not isinstance(alpha, torch.Tensor):
        groups, key_dim = state.memory.shape[-3], state.memory.shape[-2]
        alpha = torch.full((groups, key_dim), float(alpha), device=key.device, dtype=torch.float32)
    out, token, _ = torch.ops.ldm_flexible.step(token, key, value, query, forgetting, alpha)
    return out, token
