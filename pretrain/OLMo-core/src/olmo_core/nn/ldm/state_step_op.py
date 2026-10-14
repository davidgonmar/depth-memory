"""The fast backend's state step as a pair of custom ops, so Dynamo traces a whole block as one graph.

The ops call the active ``SegmentState``'s ``_forward`` and ``_backward``. With the block in one
graph, Inductor fuses the site code around the two state launches and the residual-gradient sums.

Exactly one segment state is active per model forward and backward (``run_segment`` registers it).
A token tensor chains the steps, so the backward consumes the edges in strict reverse order (each
backward op takes the next edge's token gradient), and the forward op returns its edge index as a
CPU int64 scalar that the backward op reads without a device synchronization.
"""
from __future__ import annotations

import torch

_ACTIVE = {"segment_state": None}


def set_active_state(segment_state) -> None:
    _ACTIVE["segment_state"] = segment_state


def active_state():
    segment_state = _ACTIVE["segment_state"]
    if segment_state is None:
        raise RuntimeError("no active fast-backend segment state; run_segment registers one")
    return segment_state


@torch.library.custom_op("ldm::state_step", mutates_args=())
def state_step(token: torch.Tensor, raw_key: torch.Tensor, query: torch.Tensor, raw_value: torch.Tensor,
               scale: torch.Tensor, forgetting: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_state = active_state()
    if token.device != raw_value.device or token.numel() != 0:
        raise ValueError("the dependency token must be empty and on the value device")
    read, edge = segment_state._forward(raw_key, query, raw_value, scale, forgetting)
    return read.contiguous(), torch.empty_like(token), torch.full((), edge, dtype=torch.int64)


@state_step.register_fake
def _state_step_fake(token, raw_key, query, raw_value, scale, forgetting):
    return (torch.empty(raw_value.shape, dtype=raw_value.dtype, device=raw_value.device),
            torch.empty_like(token), torch.empty((), dtype=torch.int64, device="cpu"))


_GRAD_DTYPES = (torch.float32, torch.float32, torch.bfloat16, torch.float32, torch.float32)


def _expanded_shapes(segment_state, token_shape):
    g, k, v = segment_state.groups, segment_state.key_dim, segment_state.value_dim
    return ((*token_shape, g, k), (*token_shape, g, k), (*token_shape, g, v), (*token_shape, g), (*token_shape, g))


@torch.library.custom_op("ldm::state_step_backward", mutates_args=())
def state_step_backward(edge: torch.Tensor, read_gradient: torch.Tensor, token_gradient: torch.Tensor | None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del token_gradient  # ordering dependency only
    segment_state = active_state()
    gradients = segment_state._backward(int(edge), read_gradient)
    shapes = _expanded_shapes(segment_state, tuple(read_gradient.shape[:-2]))
    return tuple(g.reshape(s).to(d).contiguous()
                 for g, s, d in zip(gradients, shapes, _GRAD_DTYPES, strict=True))


@state_step_backward.register_fake
def _state_step_backward_fake(edge, read_gradient, token_gradient):
    shapes = _expanded_shapes(active_state(), tuple(read_gradient.shape[:-2]))
    return tuple(torch.empty(s, dtype=d, device=read_gradient.device) for s, d in zip(shapes, _GRAD_DTYPES, strict=True))


def _setup_context(ctx, inputs, output):
    token, *operands = inputs
    ctx.save_for_backward(output[2])
    ctx.shapes = tuple(t.shape for t in operands)
    ctx.dtypes = tuple(t.dtype for t in operands)
    ctx.token_dtype = token.dtype
    ctx.read_shape, ctx.read_dtype, ctx.device = output[0].shape, output[0].dtype, output[0].device


def _backward(ctx, read_gradient, token_gradient, edge_gradient):
    del edge_gradient
    (edge,) = ctx.saved_tensors
    if read_gradient is None:
        # A loss may ignore this read; the segment_state still reverses the edge.
        read_gradient = torch.zeros(ctx.read_shape, dtype=ctx.read_dtype, device=ctx.device)
    gradients = torch.ops.ldm.state_step_backward(edge, read_gradient, token_gradient)
    outputs = [g.sum_to_size(s).to(d) for g, s, d in zip(gradients, ctx.shapes, ctx.dtypes, strict=True)]
    return torch.empty(0, device=ctx.device, dtype=ctx.token_dtype), *outputs


state_step.register_autograd(_backward, setup_context=_setup_context)
