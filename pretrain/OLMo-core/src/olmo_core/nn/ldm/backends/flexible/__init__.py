"""Flexible LDM backend: PyTorch controllers around a fused state step.

The controllers (address projections, gates, residual joins) are ordinary modules that compile with
the transformer block; only the state write/read step and its reverse are kernels (see state.py and
kernels.py). M is stored in BF16; the storage precision of P and the recurrence rule are
configuration. The block, boundaries and forgetting parameters are built in parameters.py.
"""

from dataclasses import dataclass, replace
from typing import Any, NamedTuple, Optional, cast

import torch
from torch.distributed import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FSDPModule, fully_shard

from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy

from ...components import apply_read_gain, complete_read
from . import kernels, state as _state
from .parameters import (
    block_num_params,
    block_parameters,
    boundary_num_params,
    build_boundaries,
    build_forgetting_parameters,
    forgetting_num_params,
)
from .state import SegmentState


@dataclass(frozen=True)
class Settings:
    covariance_dtype: str = "bfloat16"

    @classmethod
    def from_mapping(cls, values):
        unknown = set(values) - {"covariance_dtype"}
        if unknown:
            raise ValueError(f"Unknown flexible settings: {sorted(unknown)}")
        result = cls(**values)
        if result.covariance_dtype not in {"float32", "bfloat16"}:
            raise ValueError("covariance_dtype must be float32 or bfloat16")
        return result


def validate_quality(quality):
    quality.validate()


def validate_execution(common):
    common.validate()


RULES = {"rls": kernels.RLS, "delta_rule": kernels.DELTA_RULE, "outer_product": kernels.OUTER_PRODUCT}


class Pending(NamedTuple):
    """The segment state between a write and the read that consumes it."""

    state: SegmentState
    token: torch.Tensor
    alpha: torch.Tensor  # [G, K] FP32 diagonal of P_{-1}, differentiable when learned
    key: torch.Tensor | None
    value: torch.Tensor | None
    forgetting: torch.Tensor | None
    last_delta: torch.Tensor
    rule: int = kernels.RLS


def _project_write(projector, source, value, groups, eps):
    """Weighted key and value of a write, rounded to the working dtype for the kernel."""
    key, weight = projector.project(source)
    value_dtype = value.dtype
    value = value.unflatten(-1, (groups, -1)).float()
    key = key.float()
    if weight is not None:
        scale = weight.float().clamp_min(eps).sqrt().unsqueeze(-1)
        key, value = key * scale, value * scale
    return key.to(source.dtype), value.to(value_dtype)


def _project_read(parameters, source):
    return parameters.project(source)


def _static_address(anchor, source, groups, key_dim, normalize, eps, gain):
    address = anchor.view(*([1] * (source.ndim - 1)), groups, key_dim).expand(
        *source.shape[:-1], groups, key_dim
    )
    if normalize:
        address = torch.nn.functional.normalize(address.float(), dim=-1, eps=eps).to(source.dtype)
    return apply_read_gain(address, gain)


def _finish_read(value, stream, add):
    return complete_read(value, stream if add else None).to(stream.dtype)


# The boundary sites run outside the blocks and compile on their own.
_compiled_write = torch.compile(_project_write, fullgraph=True, dynamic=False)
_compiled_read = torch.compile(_project_read, fullgraph=True, dynamic=False)
_compiled_finish = torch.compile(_finish_read, fullgraph=True, dynamic=False)
_compiled_static = torch.compile(_static_address, fullgraph=True, dynamic=False)


def _write(pending, site, stream, delta, logit, config, *, embedding=False, compiled=False):
    q = config.quality
    source = delta if q.addresses.key.source == "branch_delta" else stream
    value = delta if embedding or q.value_source == "branch_delta" else stream
    projector = site.write.write if embedding else site.write
    if projector.projection is None:
        # Working tensors are passed explicitly: FSDP swaps registered parameters for working views.
        project = _compiled_static if compiled else _static_address
        key = project(
            projector.anchor, source, q.groups, q.key_dim, projector.normalize, projector.eps, None
        )
        value = value.unflatten(-1, (q.groups, -1))
    else:
        project = _compiled_write if compiled else _project_write
        key, value = project(projector, source, value, q.groups, config.quality.eps)
    return pending._replace(key=key, value=value, forgetting=site.forgetting_factor(logit), last_delta=delta)


def _read(pending, parameters, stream, *, add, compiled=False):
    source = pending.last_delta if parameters.source == "branch_delta" else stream
    if parameters.projection is None:
        project = _compiled_static if compiled else _static_address
        query = project(
            parameters.anchor, source, parameters.groups, parameters.key_dim, parameters.normalize,
            parameters.eps, parameters.read_gain,
        )
    else:
        query = (_compiled_read if compiled else _project_read)(parameters, source)
    if pending.key is None:
        # Reading the empty initial memory keeps a zero gradient to the query controller.
        value = (
            (query.sum(-1, keepdim=True) * 0)
            .to(stream.dtype)
            .expand(*query.shape[:-1], stream.shape[-1] // parameters.groups)
        )
    else:
        value, token = _state.step(pending.key, pending.value, query, pending.forgetting, pending.token,
                                   pending.alpha)
        pending = pending._replace(token=token)
    finish = _compiled_finish if compiled else _finish_read
    out = finish(value.flatten(-2), stream, add)
    return out, pending._replace(key=None, value=None, forgetting=None)


class Block(TransformerBlock):
    """Pre-norm block plus its LDM controllers; the model runs it only through ``run_segment``."""

    def __init__(self, *, d_model, block_idx, n_layers, ldm, init_device="cpu",
                 init_std=0.02, **kwargs):
        super().__init__(d_model=d_model, block_idx=block_idx, n_layers=n_layers,
                         init_device=init_device, **kwargs)
        ldm.validate()
        self.ldm = ldm
        self.ldm_rule = block_parameters(
            d_model, ldm, context_eps=self.attention_norm.eps, init_device=init_device,
            dtype=next(self.attention.parameters()).dtype, init_std=init_std,
        )

    def apply_compile(self):
        self.compile(fullgraph=False, dynamic=False)

    def forward(self, x, *, ldm_state, ldm_lambda, loss_div_factor=None, **kwargs):
        q = self.ldm.quality
        parameters = self.ldm_rule
        read, state = _read(ldm_state, parameters.attn.read, x, add=not q.replace_read)
        delta = self.attention_residual_stream.dropout(
            self.attention(self.attention_norm(read), **kwargs)
        )
        x = delta if q.no_post_function_residual else x + delta
        state = _write(state, parameters.attn.write, x, delta, ldm_lambda[0], self.ldm)
        read, state = _read(state, parameters.mlp.read, x, add=not q.replace_read)
        delta = self.feed_forward_residual_stream.dropout(
            self.feed_forward(self.feed_forward_norm(read))
        )
        x = delta if q.no_post_function_residual else x + delta
        state = _write(state, parameters.mlp.write, x, delta, ldm_lambda[1], self.ldm)
        return x, state

    def apply_fsdp(self, dp_mesh: Optional[DeviceMesh] = None, prefetch_factor: int = 0,
                   wrapping_strategy=TransformerDataParallelWrappingStrategy.full, **fsdp_kwargs: Any):
        # The FP32 tapes and forgetting factors travel inside ldm_state and must not be cast.
        if (policy := fsdp_kwargs.get("mp_policy")) is not None:
            fsdp_kwargs["mp_policy"] = replace(policy, cast_forward_inputs=False)
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            attention = cast(FSDPModule, fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs))
            feed_forward = cast(FSDPModule, fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs))
            root = cast(FSDPModule, fully_shard(self, mesh=dp_mesh, **{**fsdp_kwargs, "reshard_after_forward": False}))
            if prefetch_factor > 0:
                root.set_modules_to_forward_prefetch([attention])
                attention.set_modules_to_forward_prefetch([feed_forward])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


def build_block(*, ldm, **kwargs):
    return Block(ldm=ldm, **kwargs)


def run_segment(
    hidden,
    blocks,
    *,
    boundary,
    output_boundary,
    forgetting,
    config,
    all_block_kwargs,
    per_block_kwargs,
    compile_enabled,
):
    if any(isinstance(block, CheckpointWrapper) for block in blocks.values()):
        raise ValueError(
            "flexible cannot recompute whole-block checkpoints (a recomputed block would repeat its "
            "state writes); checkpoint the address networks instead"
        )
    settings = Settings.from_mapping(config.execution.backend.settings)
    q = config.quality
    rule = RULES[q.recurrence]
    embedding_logit, logits, ridge = forgetting(with_ridge=True)
    # Diagonal of P_{-1}: the learned ridge inverse, or the constant alpha.
    if ridge is None:
        alpha = torch.full((q.groups, q.key_dim), float(q.rls.alpha_init), device=hidden.device)
    else:
        alpha = ridge.float().reciprocal()
    slots = 1 + 2 * len(blocks)
    segment, token = _state.allocate(
        hidden,
        groups=q.groups,
        key_dim=q.key_dim,
        alpha=alpha.detach(),
        slots=slots,
        covariance_dtype=getattr(torch, settings.covariance_dtype),
        rule=rule,
    )
    _state.activate(segment, [_state.slot_views(segment, s) for s in range(slots)], rule)
    state = Pending(segment, token, alpha, None, None, None, hidden, rule)
    state = _write(state, boundary, hidden, hidden, embedding_logit, config, embedding=True,
                   compiled=compile_enabled)
    for index, block in blocks.items():
        kwargs = dict(all_block_kwargs)
        kwargs.update(per_block_kwargs.get(int(index), {}))
        hidden, state = block(hidden, ldm_state=state, ldm_lambda=logits[int(index)], **kwargs)
    hidden, _ = _read(state, output_boundary.output, hidden, add=not q.replace_read,
                      compiled=compile_enabled)
    return hidden
