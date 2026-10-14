"""Reference PyTorch implementation of LDM: memory M and covariance P, written and read in order.

Conventions
-----------
``memory`` has shape ``[..., K, D]`` and maps an address to a value through
``memory.transpose(-1, -2) @ address``. ``covariance`` has shape
``[..., K, K]``. Keys and queries have shape ``[..., K]`` and values have
shape ``[..., D]``.

This module intentionally materializes every intermediate. It is the oracle
for mathematical and gradient parity, not a performance implementation.

Precision: the address networks run in the model's working dtype (the
shared ``components`` modules). The recurrence state, weighted observations,
gains and forgetting factors are FP32 (FP64 for an FP64 model), and every read
is returned in the working dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.ldm.components import (
    LDMBlockParameters,
    ForgettingParameters,
    LDMInputParameters,
    LDMOutputParameters,
    ReadParameters,
    RLSWriteSite,
    complete_read,
)
from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy
from olmo_core.utils import mark_dynamic


@dataclass(frozen=True)
class Settings:
    """The reference has no execution tuning settings."""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Settings":
        if values:
            raise ValueError(f"reference does not support settings {sorted(values)}")
        return cls()


def build_forgetting_parameters(
    *, config: Any, n_layers: int, init_device: str
) -> ForgettingParameters:
    quality = config.quality
    return ForgettingParameters(
        groups=quality.groups,
        n_layers=n_layers,
        lambda_init=quality.rls.lambda_init,
        learn_lambda=quality.rls.learn_lambda,
        init_device=init_device,
        key_dim=quality.key_dim,
        learn_alpha=quality.rls.learn_alpha,
        alpha_init=quality.rls.alpha_init,
        ridge_min=quality.rls.ridge_min,
    )


def forgetting_num_params(*, config: Any, n_layers: int) -> int:
    sites = 2 * n_layers + 1
    ridge = config.quality.groups * config.quality.key_dim if config.quality.rls.learn_alpha else 0
    return sites * config.quality.groups + ridge


@dataclass(frozen=True)
class MState:
    """M-form state with explicit memory and covariance tensors."""

    memory: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class SegmentState:
    """Full-segment reference state, including the last branch delta."""

    recurrence: MState
    last_delta: torch.Tensor


def _validate(state: MState, key: torch.Tensor, value: torch.Tensor) -> None:
    if state.memory.shape[-2] != key.shape[-1]:
        raise ValueError("memory K dimension must match key K dimension")
    if state.memory.shape[-1] != value.shape[-1]:
        raise ValueError("memory D dimension must match value D dimension")
    if state.covariance.shape[-2:] != (key.shape[-1], key.shape[-1]):
        raise ValueError("covariance must have shape [..., K, K]")


_RECURRENCES = ("rls", "delta_rule", "outer_product")


def write(
    state: MState,
    key: torch.Tensor,
    value: torch.Tensor,
    forgetting_factor: torch.Tensor,
    recurrence: str = "rls",
) -> MState:
    """Apply one write under the selected recurrence.

    The three rules differ only in the gain and in whether the current
    prediction is subtracted, matching the flexible backend's kernel
    (``backends/flexible/kernels.py``, ``RULE`` branches)::

        rls            gain = Pk / (lambda + k'Pk)   error = v - M'k          P updated
        delta_rule     gain = k                      error = v - (lambda M)'k  M decayed by lambda
        outer_product  gain = k                      error = v                M decayed by lambda

    RLS forgets through the covariance; the two plain rules have no covariance,
    so the forgetting factor decays the memory itself before the write (the
    depth analogue of a gated delta rule / gated linear attention). The
    covariance is carried unchanged by those rules so :class:`MState` keeps one
    shape across rules.
    """

    _validate(state, key, value)
    if recurrence not in _RECURRENCES:
        raise ValueError(f"unknown recurrence {recurrence!r}; expected one of {_RECURRENCES}")

    if recurrence == "rls":
        covariance_key = torch.einsum("...ij,...j->...i", state.covariance, key)
        denominator = forgetting_factor + torch.sum(key * covariance_key, dim=-1)
        gain = covariance_key / denominator.unsqueeze(-1)
        key_covariance = torch.einsum("...i,...ij->...j", key, state.covariance)
        covariance = (
            state.covariance - gain.unsqueeze(-1) * key_covariance.unsqueeze(-2)
        ) / forgetting_factor[..., None, None]
        memory = state.memory
    else:
        gain = key
        covariance = state.covariance
        memory = state.memory * forgetting_factor[..., None, None]

    if recurrence == "outer_product":
        residual = value
    else:
        prediction = torch.einsum("...kd,...k->...d", memory, key)
        residual = value - prediction
    memory = memory + gain.unsqueeze(-1) * residual.unsqueeze(-2)
    return MState(memory=memory, covariance=covariance)


def read(state: MState, query: torch.Tensor) -> torch.Tensor:
    """Read a value from the current M-form state."""

    if state.memory.shape[-2] != query.shape[-1]:
        raise ValueError("memory K dimension must match query K dimension")
    return torch.einsum("...kd,...k->...d", state.memory, query)


def write_then_read(
    state: MState,
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    forgetting_factor: torch.Tensor,
    recurrence: str = "rls",
) -> tuple[MState, torch.Tensor]:
    """Apply one write and read the post-write state."""

    next_state = write(state, key, value, forgetting_factor, recurrence)
    return next_state, read(next_state, query)


def validate_quality(quality: Any) -> None:
    """The reference implements every model choice of the configuration."""

    del quality


def validate_execution(common: Any) -> None:
    """The reference accepts every common execution setting."""

    del common


def apply_compile(block: Any) -> None:
    """Compile attention, the feed-forward and the address networks; the recurrence stays eager."""

    block.attention.compile(fullgraph=False)
    block.feed_forward.compile(fullgraph=False)
    for site in (block.ldm_rule.attn, block.ldm_rule.mlp):
        for projector in (site.read.projection, site.write.write.projection):
            if projector is not None:
                projector.compile(fullgraph=False)


class _ReferenceBlock(TransformerBlock):
    """A pre-norm transformer block with its LDM read and write sites."""

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        ldm: Any,
        init_device: str = "cpu",
        init_std: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            init_device=init_device,
            **kwargs,
        )
        ldm.validate()
        self.ldm = ldm
        parameter_dtype = next(self.attention.parameters()).dtype
        self.ldm_rule = LDMBlockParameters(
            d_model,
            ldm,
            context_eps=self.attention_norm.eps,
            init_device=init_device,
            dtype=parameter_dtype,
            init_std=init_std,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del x, kwargs
        raise RuntimeError("LDM blocks are executed only through backend.run_segment()")

    def apply_compile(self) -> None:
        apply_compile(self)

    def apply_fsdp(
        self,
        dp_mesh: Optional[DeviceMesh] = None,
        prefetch_factor: int = 0,
        wrapping_strategy: TransformerDataParallelWrappingStrategy = TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs: Any,
    ) -> None:
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            fsdp_att = cast(
                FSDPModule,
                fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs),
            )
            fsdp_mlp = cast(
                FSDPModule,
                fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs),
            )
            root_kwargs = dict(fsdp_kwargs)
            root_kwargs["reshard_after_forward"] = False
            fsdp_root = cast(
                FSDPModule,
                fully_shard(self, mesh=dp_mesh, **root_kwargs),
            )
            if prefetch_factor > 0:
                fsdp_root.set_modules_to_forward_prefetch([fsdp_att])
                fsdp_att.set_modules_to_forward_prefetch([fsdp_mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **fsdp_kwargs)


def block_num_params(
    *,
    config: Any,
    d_model: int,
    block_idx: int,
    n_layers: int,
) -> int:
    """Count the parameters of one block's LDM sites."""

    del block_idx, n_layers
    module = LDMBlockParameters(
        d_model,
        config,
        # Epsilon does not affect the affine norm's parameter count.
        context_eps=config.quality.eps,
        init_device="meta",
        dtype=torch.float32,
        init_std=0.02,
    )
    return sum(parameter.numel() for parameter in module.parameters())


def boundary_num_params(*, config: Any, d_model: int) -> int:
    """Count the parameters of the embedding write and the final read."""

    common = dict(context_eps=config.quality.eps, init_device="meta", dtype=torch.float32,
                  init_std=0.02)
    modules = (LDMInputParameters(d_model, config, **common),
               LDMOutputParameters(d_model, config, **common))
    return sum(parameter.numel() for module in modules for parameter in module.parameters())


def build_block(*, ldm: Any, **kwargs: Any) -> Any:
    """Build one transformer block with its LDM sites."""

    return _ReferenceBlock(ldm=ldm, **kwargs)


def build_boundaries(
    *,
    d_model: int,
    first_block: Any,
    init_device: str,
    dtype: torch.dtype,
    init_std: float,
) -> tuple[nn.Module, nn.Module]:
    """Build the embedding-write and final-read parameters."""

    config = first_block.ldm
    common = dict(context_eps=first_block.attention_norm.eps, init_device=init_device,
                  dtype=dtype, init_std=init_std)
    return (LDMInputParameters(d_model, config, **common),
            LDMOutputParameters(d_model, config, **common))


def _state_dtype(stream: torch.Tensor) -> torch.dtype:
    """Recurrence precision: FP32, or FP64 when the model itself runs in FP64.

    The state never follows a low-precision working dtype; a float64 model
    yields a float64 oracle for measuring the FP32 state's own rounding.
    """

    return torch.promote_types(torch.float32, stream.dtype)


def _weighted_observation(
    key: torch.Tensor,
    value: torch.Tensor,
    gate: Optional[torch.Tensor],
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate is None:
        return key, value
    scale = gate.to(key.dtype).clamp_min(eps).sqrt()
    return key * scale.unsqueeze(-1), value * scale.unsqueeze(-1)


def _segment_write(
    state: SegmentState,
    site: RLSWriteSite,
    updated_stream: torch.Tensor,
    branch_delta: torch.Tensor,
    *,
    key_source: str,
    value_source: str,
    lambda_logit: torch.Tensor,
    groups: int,
    eps: float,
    recurrence: str = "rls",
) -> SegmentState:
    source = branch_delta if key_source == "branch_delta" else updated_stream
    value = branch_delta if value_source == "branch_delta" else updated_stream
    key, gate = site.write.project(source)
    dtype = _state_dtype(updated_stream)
    grouped_value = value.view(*value.shape[:-1], groups, value.shape[-1] // groups)
    key, grouped_value = _weighted_observation(
        key.to(dtype), grouped_value.to(dtype), gate, eps=eps
    )
    next_recurrence = write(
        state.recurrence,
        key,
        grouped_value,
        site.forgetting_factor(lambda_logit).to(dtype),
        recurrence,
    )
    return SegmentState(next_recurrence, branch_delta)


def _segment_read(
    state: SegmentState,
    parameters: ReadParameters,
    current_stream: torch.Tensor,
    *,
    branch_delta: Optional[torch.Tensor],
    add_to: Optional[torch.Tensor],
) -> torch.Tensor:
    source = (
        (state.last_delta if branch_delta is None else branch_delta)
        if parameters.source == "branch_delta"
        else current_stream
    )
    query = parameters.project(source).to(state.recurrence.memory.dtype)
    value = read(state.recurrence, query).reshape(*source.shape[:-1], -1)
    out = complete_read(value, add_to)
    # The recurrence state is FP32 and the stream stays in the working dtype. Without ``add_to``
    # (``replace_read``) the FP32 read would otherwise flow into BF16 branches.
    return out.to(current_stream.dtype)


def initial_covariance(
    prefix: torch.Size,
    groups: int,
    key_dim: int,
    *,
    alpha_init: float,
    ridge: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """``P_{-1}``: ``alpha_init * I`` per group, or ``diag(1 / ridge)`` when learned."""

    if ridge is None:
        eye = torch.eye(key_dim, device=device, dtype=dtype) * alpha_init
        eye = eye.view(*([1] * len(prefix)), 1, key_dim, key_dim)
    else:
        if tuple(ridge.shape) != (groups, key_dim):
            raise ValueError("ridge must have shape [groups, key_dim]")
        eye = torch.diag_embed(ridge.to(dtype).reciprocal())
        eye = eye.view(*([1] * len(prefix)), groups, key_dim, key_dim)
    return eye.expand(*prefix, groups, key_dim, key_dim).clone()


def _initial_state(
    hidden: torch.Tensor,
    boundary: nn.Module,
    config: Any,
    embedding_logit: torch.Tensor,
    ridge: Optional[torch.Tensor] = None,
) -> SegmentState:
    groups, key_dim = config.quality.groups, config.quality.key_dim
    group_dim = hidden.shape[-1] // groups
    prefix = hidden.shape[:-1]
    dtype = _state_dtype(hidden)
    covariance = initial_covariance(
        prefix,
        groups,
        key_dim,
        alpha_init=config.quality.rls.alpha_init,
        ridge=ridge,
        device=hidden.device,
        dtype=dtype,
    )
    memory = torch.zeros(
        *prefix,
        groups,
        key_dim,
        group_dim,
        device=hidden.device,
        dtype=dtype,
    )
    key, gate = boundary.project(hidden)
    grouped_value = hidden.view(*prefix, groups, group_dim).to(dtype)
    key, grouped_value = _weighted_observation(
        key.to(dtype),
        grouped_value,
        gate,
        eps=config.quality.eps,
    )
    recurrence = write(
        MState(memory, covariance),
        key,
        grouped_value,
        boundary.forgetting_factor(embedding_logit).to(dtype),
        config.quality.recurrence,
    )
    return SegmentState(recurrence, hidden)


def run_block(
    block: Any,
    hidden: torch.Tensor,
    state: SegmentState,
    *,
    config: Any,
    lambda_logits: torch.Tensor,
    loss_div_factor: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, SegmentState]:
    """Run one block: read, attention, write, read, feed-forward, write."""

    del loss_div_factor   # dense feed-forward; must not reach Attention's kwargs
    parameters = block.ldm_rule
    read_add_enabled = not config.quality.replace_read
    post_function_residual_enabled = (
        not config.quality.no_post_function_residual
    )
    attention_input = _segment_read(
        state,
        parameters.attn.read,
        hidden,
        branch_delta=None,
        add_to=hidden if read_add_enabled else None,
    )
    attention_update = block.attention_residual_stream.dropout(
        block.attention(block.attention_norm(attention_input), **kwargs)
    )
    attention_stream = (
        hidden + attention_update
        if post_function_residual_enabled
        else attention_update
    )
    state = _segment_write(
        state,
        parameters.attn.write,
        attention_stream,
        attention_update,
        key_source=config.quality.addresses.key.source,
        value_source=config.quality.value_source,
        lambda_logit=lambda_logits[0],
        groups=config.quality.groups,
        eps=config.quality.eps,
        recurrence=config.quality.recurrence,
    )
    mlp_input = _segment_read(
        state,
        parameters.mlp.read,
        attention_stream,
        branch_delta=attention_update,
        add_to=attention_stream if read_add_enabled else None,
    )
    mlp_update = block.feed_forward_residual_stream.dropout(
        block.feed_forward(block.feed_forward_norm(mlp_input))
    )
    output = (
        attention_stream + mlp_update
        if post_function_residual_enabled
        else mlp_update
    )
    state = _segment_write(
        state,
        parameters.mlp.write,
        output,
        mlp_update,
        key_source=config.quality.addresses.key.source,
        value_source=config.quality.value_source,
        lambda_logit=lambda_logits[1],
        groups=config.quality.groups,
        eps=config.quality.eps,
        recurrence=config.quality.recurrence,
    )
    return output, state


def run_segment(
    hidden: torch.Tensor,
    blocks: Mapping[str, Any],
    *,
    boundary: nn.Module,
    output_boundary: nn.Module,
    forgetting: ForgettingParameters,
    config: Any,
    all_block_kwargs: Mapping[str, Any],
    per_block_kwargs: Mapping[int, Mapping[str, Any]],
    compile_enabled: bool,
) -> torch.Tensor:
    """Run the embedding write, every block and the final read on an explicit FP32 state."""

    Settings.from_mapping(config.execution.backend.settings)
    embedding_logit, block_logits, ridge = forgetting(with_ridge=True)
    state = _initial_state(hidden, boundary, config, embedding_logit, ridge)
    for block_key, block in blocks.items():
        block_idx = int(block_key)
        if compile_enabled:
            mark_dynamic(hidden, (0, 1), strict=False)
        hidden, state = run_block(
            block,
            hidden,
            state,
            config=config,
            lambda_logits=block_logits[block_idx],
            **all_block_kwargs,
            **per_block_kwargs.get(block_idx, {}),
        )
    return _segment_read(
        state,
        output_boundary.output,
        hidden,
        branch_delta=state.last_delta,
        add_to=None if config.quality.replace_read else hidden,
    )


__all__ = [
    "MState",
    "SegmentState",
    "Settings",
    "ForgettingParameters",
    "apply_compile",
    "block_num_params",
    "boundary_num_params",
    "build_block",
    "build_boundaries",
    "build_forgetting_parameters",
    "initial_covariance",
    "read",
    "run_block",
    "run_segment",
    "forgetting_num_params",
    "validate_execution",
    "validate_quality",
    "write",
    "write_then_read",
]
