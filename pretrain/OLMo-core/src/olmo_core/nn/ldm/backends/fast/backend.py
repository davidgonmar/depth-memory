"""The fast backend, registered as ``fast``: the transformer block, the input and output
boundaries and the segment runner around the windowed state.

Parameters come from ``ldm.parameters`` and the backbone modules from OLMo-core. Only
the state (``state.SegmentState``) holds the recurrence, the write scaling and the tapes; no module
keeps a state, a token or a dense memory tensor. The address sites (``sites``) are plain PyTorch
functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import fully_shard
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.ldm import state_step_op as _state_step_op
from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.config import TransformerDataParallelWrappingStrategy
from olmo_core.utils import mark_dynamic

from olmo_core.nn.ldm.parameters import (
    ForgettingParameters,
    LDMBoundary,
    LDMParameterConfig as BlockConfig,
    LDMOutputBoundary,
    LDMRule,
    block_config,
)
from .state import RUNNING_COVARIANCE_DTYPES, SegmentState
from .sites import (
    PendingWrite,
    PreparedRead,
    can_pair_address_networks,
    finish_read,
    prepare_read,
    prepare_write,
    prepare_write_read,
)


@dataclass(frozen=True)
class Settings:
    """``span``: the window size E. ``read_checkpoint``: recompute each read's residual add and
    norm in the backward instead of keeping their activations; only the norm's row scale is
    kept. ``running_covariance_dtype``: "bfloat16" or "float32", the dtype of the covariance P
    that the forward carries from edge to edge (the saved copies the backward reads stay BF16)."""

    span: int = 4
    read_checkpoint: bool = True
    running_covariance_dtype: str = "bfloat16"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Settings:
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"the fast backend has no settings {sorted(unknown)}")
        settings = cls(**values)
        if type(settings.span) is not int or settings.span <= 0:
            raise ValueError("span must be a positive integer")
        if type(settings.read_checkpoint) is not bool:
            raise ValueError("read_checkpoint must be a bool")
        if (type(settings.running_covariance_dtype) is not str
                or settings.running_covariance_dtype not in RUNNING_COVARIANCE_DTYPES):
            raise ValueError(f"running_covariance_dtype must be one of {sorted(RUNNING_COVARIANCE_DTYPES)}")
        return settings


def validate_quality(quality):
    quality.validate()
    if quality.key_dim not in (4, 8, 16):
        raise ValueError("SegmentState supports key_dim 4, 8, or 16")


def validate_execution(common):
    common.validate()


def parameter_config(config) -> BlockConfig:
    """The constructor config of this backend's parameter modules, after validating the model config."""
    validate_quality(config.quality)
    validate_execution(config.execution.common)
    Settings.from_mapping(config.execution.backend.settings)
    return block_config(config)


class PendingState(NamedTuple):
    """The pending write carried from one block to the next state step, with the dependency
    token. Values are raw, grouped and unweighted; ``previous_delta`` serves branch-delta
    queries and is None otherwise.
    """

    segment_state: SegmentState
    token: torch.Tensor
    keyhat: torch.Tensor
    raw_value: torch.Tensor
    scale: torch.Tensor
    forgetting: torch.Tensor
    previous_delta: torch.Tensor | None


def _pending(segment_state, token, write: PendingWrite, previous_delta=None):
    return PendingState(segment_state, token, *write, previous_delta)


COMPUTE_DTYPE = torch.bfloat16      # the dtype the sites hand the state


def _site_checkpointed(config, projection):
    # The sites use the projection modules only for their parameters, so a checkpoint wrapper on
    # any child of a projection checkpoints the whole site.
    wrapped = projection is not None and any(
        isinstance(module, CheckpointWrapper) for module in projection.modules()
    )
    return torch.is_grad_enabled() and (config.checkpoint_address_projections or wrapped)


def _site_checkpoint_policy(ctx, op, *args, **kwargs):
    # Save the address network's outputs, which are narrow, rather than another d_model-wide input.
    if op.name() == "ldm::address_network":
        return CheckpointPolicy.MUST_SAVE
    # Save the FP32 normalizers (rsqrt for dynamic addresses, vector_norm for static ones):
    # recomputing them inside a fused kernel changes the BF16 cotangents.
    if op in (torch.ops.aten.rsqrt.default, torch.ops.aten.linalg_vector_norm.default):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _site_checkpoint_contexts():
    return create_selective_checkpoint_contexts(_site_checkpoint_policy)


def _site_call(function, config, projection, *args, **kwargs):
    if _site_checkpointed(config, projection):
        return checkpoint(
            function, *args, use_reentrant=False, preserve_rng_state=False,
            context_fn=_site_checkpoint_contexts, **kwargs,
        )
    return function(*args, **kwargs)


def _write_site(module, source, raw_value, projection, static_address, logit):
    c = module.config
    # The branch value stays out of the checkpoint's inputs: the site gets the source as a
    # placeholder value and the real value is attached afterwards.
    prepared = _site_call(
        prepare_write,
        c,
        projection,
        source,
        source,
        projection,
        static_address,
        logit,
        rms_eps=module.layer_norm_eps,
        eps=c.eps,
        lambda_init=c.lambda_init,
        write_gate=c.write_gate,
        normalize=c.normalize_key,
        compute_dtype=COMPUTE_DTYPE,
    )
    return prepared._replace(
        raw_value=raw_value.to(COMPUTE_DTYPE).unflatten(-1, (static_address.shape[0], -1))
    )


def _read_site(module, source, projection, static_address, gain):
    c = module.config
    return _site_call(
        prepare_read,
        c,
        projection,
        source,
        projection,
        static_address,
        rms_eps=module.layer_norm_eps,
        eps=c.eps,
        gain_logit=gain,
        linear_gain=True,
        read_gate=c.read_gate,
        normalize=c.normalize_query,
        compute_dtype=COMPUTE_DTYPE,
    )


def _prepare_shared_write_read(source, write_projection, write_static_address, logit,
                               read_projection, read_static_address, **kwargs):
    # A checkpoint can hand one tensor back as distinct objects, which would fail the pairing
    # test when the site is recomputed. Take the source once and pass it for all three roles.
    return prepare_write_read(
        source, source, write_projection, write_static_address, logit,
        source, read_projection, read_static_address, **kwargs,
    )


def _write_read_site(module, write_source, raw_value, write_projection, write_static_address,
                     logit, read_source, read_projection, read_static_address, gain):
    c = module.config
    dtype_write = dtype_read = COMPUTE_DTYPE
    if (
        _site_checkpointed(c, write_projection) != _site_checkpointed(c, read_projection)
        or not can_pair_address_networks(
            write_source, read_source, write_projection, read_projection,
            write_rms_eps=module.layer_norm_eps, read_rms_eps=module.layer_norm_eps,
            write_compute_dtype=dtype_write, read_compute_dtype=dtype_read,
        )
    ):
        return (
            _write_site(module, write_source, raw_value, write_projection, write_static_address, logit),
            _read_site(module, read_source, read_projection, read_static_address, gain),
        )
    # Both sites run under one checkpoint; the branch value stays out of its inputs.
    write, read = _site_call(
        _prepare_shared_write_read, c, write_projection,
        write_source, write_projection, write_static_address, logit,
        read_projection, read_static_address,
        write_rms_eps=module.layer_norm_eps, read_rms_eps=module.layer_norm_eps,
        eps=c.eps, lambda_init=c.lambda_init, gain_logit=gain, linear_gain=True,
        write_gate=c.write_gate, read_gate=c.read_gate,
        normalize_key=c.normalize_key, normalize_query=c.normalize_query,
        write_compute_dtype=dtype_write, read_compute_dtype=dtype_read,
    )
    return write._replace(raw_value=raw_value.to(dtype_write).unflatten(-1, (write_static_address.shape[0], -1))), read


class FastRule(LDMRule):
    """The block's parameters with the functional sites as its forward: one site by role
    (``attn_write``, ``mlp_read``, ...) or a paired write and read on one source."""

    def forward(self, role, source, raw_value=None, lambda_logit=None):
        if isinstance(role, tuple):
            write_role, read_role = role
            write_branch, write_op = write_role.split("_")
            read_branch, read_op = read_role.split("_")
            if (write_branch not in ("attn", "mlp") or read_branch not in ("attn", "mlp")
                    or write_op != "write" or read_op != "read"):
                raise ValueError(f"unknown site pair {role!r}")
            write_source, read_source = source
            return _write_read_site(
                self, write_source, raw_value, getattr(self, f"{write_branch}_k_proj"),
                getattr(self, f"{write_branch}_key"), lambda_logit,
                read_source, getattr(self, f"{read_branch}_q_proj"),
                getattr(self, f"{read_branch}_query"),
                getattr(self, f"{read_branch}_read_gain"),
            )
        branch, operation = role.split("_")
        if branch not in ("attn", "mlp") or operation not in ("read", "write"):
            raise ValueError(f"unknown site {role!r}")
        if operation == "write":
            return _write_site(
                self,
                source,
                raw_value,
                getattr(self, f"{branch}_k_proj"),
                getattr(self, f"{branch}_key"),
                lambda_logit,
            )
        return _read_site(
            self,
            source,
            getattr(self, f"{branch}_q_proj"),
            getattr(self, f"{branch}_query"),
            getattr(self, f"{branch}_read_gain"),
        )


class FastBoundary(LDMBoundary):
    def forward(self, hidden, lambda_logit):
        return _write_site(self, hidden, hidden, self.input_k_proj, self.input_key, lambda_logit)


class FastOutputBoundary(LDMOutputBoundary):
    def forward(self, source):
        return _read_site(
            self, source, self.output_q_proj, self.output_query, self.output_read_gain
        )


def _finish_read_normed(raw_read, stream, norm):
    """The read added to the stream, then the backbone norm."""
    return norm(stream + raw_read.flatten(-2).to(stream.dtype))


def _read_checkpoint_policy(ctx, op, *args, **kwargs):
    if op == torch.ops.aten.rsqrt.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _read_checkpoint_contexts():
    return create_selective_checkpoint_contexts(_read_checkpoint_policy)


def consume_pending_normed(state, prepared, stream, norm, *, read_checkpoint=True):
    """Consume a pending read and normalize it; with ``read_checkpoint`` the read's add and norm
    are recomputed in the backward instead of keeping their activations."""
    if not (read_checkpoint and torch.is_grad_enabled()):
        read, token = consume_pending(state, prepared, stream)
        return norm(read), token
    raw_read, token = state.segment_state.step(
        state.token, state.keyhat, prepared.query, state.raw_value, state.scale, state.forgetting
    )
    # Only the read's add and the norm are recomputed, never a backbone projection. The norm's
    # FP32 row scale is kept, so the backward does not reduce each row again before its own sums.
    normed = checkpoint(_finish_read_normed, raw_read, stream, norm,
                        use_reentrant=False, preserve_rng_state=False,
                        context_fn=_read_checkpoint_contexts)
    return normed, token


def consume_pending(state: PendingState, prepared: PreparedRead, stream):
    """The pending write and the next read, added to the stream."""
    raw_read, token = state.segment_state.step(
        state.token, state.keyhat, prepared.query, state.raw_value, state.scale, state.forgetting
    )
    return finish_read(raw_read, prepared, add_to=stream, grouped=True), token


def _query_source(quality, stream, previous_delta):
    if quality.addresses.query.source == "branch_delta" and previous_delta is not None:
        return previous_delta
    return stream


def attention_write_read(module, x, delta, logit):
    """The post-attention residual add, then the attention write and the MLP read as one pair."""
    q = module.ldm.quality
    updated = torch.add(x, delta, alpha=module.attention_residual_stream.alpha)
    write, prepared = module.ldm_rule(
        ("attn_write", "mlp_read"), (updated, _query_source(q, updated, delta)), delta, logit,
    )
    return updated, write, prepared


class FastTransformerBlock(TransformerBlock):
    def __init__(
        self,
        *,
        d_model,
        block_idx,
        n_layers,
        ldm,
        layer_norm,
        init_device="cpu",
        init_std,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            layer_norm=layer_norm,
            init_device=init_device,
            **kwargs,
        )
        self._init_rule(ldm, d_model, block_idx, n_layers, init_device, init_std)

    def _init_rule(self, config, d_model, block_idx, n_layers, init_device, init_std):
        self.ldm = config
        self._settings = Settings.from_mapping(config.execution.backend.settings)
        self.ldm_config = parameter_config(config)
        self.ldm_rule = FastRule(
            d_model=d_model,
            block_idx=block_idx,
            n_layers=n_layers,
            layer_norm_eps=self.attention_norm.eps,
            config=self.ldm_config,
            init_device=init_device,
            dtype=next(self.attention.parameters()).dtype,
            init_std=init_std,
        )

    def _write(self, branch, stream, delta, lambda_logit):
        """The branch write: keys from the stream, values from the branch output."""
        return self.ldm_rule(f"{branch}_write", stream, delta, lambda_logit)

    def forward(
        self,
        x,
        *,
        ldm_state: PendingState,
        ldm_lambda_logits,
        loss_div_factor=None,
        **kwargs,
    ):
        q = self.ldm.quality
        state = ldm_state
        prepared = self.ldm_rule("attn_read", _query_source(q, x, state.previous_delta))
        normed, token = consume_pending_normed(
            state, prepared, x, self.attention_norm, read_checkpoint=self._settings.read_checkpoint)
        delta = self.attention_residual_stream.dropout(self.attention(normed, **kwargs))
        updated, write, prepared = attention_write_read(self, x, delta, ldm_lambda_logits[0])
        state = _pending(state.segment_state, token, write)
        normed, token = consume_pending_normed(
            state, prepared, updated, self.feed_forward_norm, read_checkpoint=self._settings.read_checkpoint)
        delta = self.feed_forward_residual_stream.dropout(self.feed_forward(normed))
        output = torch.add(updated, delta, alpha=self.feed_forward_residual_stream.alpha)
        write = self._write("mlp", output, delta, ldm_lambda_logits[1])
        previous_delta = delta if q.addresses.query.source == "branch_delta" else None
        return output, _pending(state.segment_state, token, write, previous_delta)

    def apply_fsdp(
        self,
        dp_mesh=None,
        prefetch_factor=0,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        **fsdp_kwargs,
    ):
        root_kwargs = dict(fsdp_kwargs)
        if (policy := root_kwargs.get("mp_policy")) is not None:
            root_kwargs["mp_policy"] = replace(policy, cast_forward_inputs=False)
        if wrapping_strategy == TransformerDataParallelWrappingStrategy.fine_grained:
            attention = fully_shard(self.attention, mesh=dp_mesh, **fsdp_kwargs)
            mlp = fully_shard(self.feed_forward, mesh=dp_mesh, **fsdp_kwargs)
            root_kwargs["reshard_after_forward"] = False
            root = fully_shard(self, mesh=dp_mesh, **root_kwargs)
            if prefetch_factor > 0:
                root.set_modules_to_forward_prefetch([attention])
                attention.set_modules_to_forward_prefetch([mlp])
        else:
            fully_shard(self, mesh=dp_mesh, **root_kwargs)

    def apply_tp(self, *args, **kwargs):
        raise OLMoConfigurationError(
            "the fast backend has no DTensor state sharding for tensor parallelism"
        )

    def apply_pp(self, *args, **kwargs):
        raise OLMoConfigurationError("an LDM segment must remain in one pipeline stage")


def build_block(*, ldm, **kwargs):
    if kwargs.pop("feed_forward_moe", None) is not None or kwargs.get("feed_forward") is None:
        raise ValueError("the fast backend requires a dense feed-forward block")
    Settings.from_mapping(ldm.execution.backend.settings)
    return FastTransformerBlock(ldm=ldm, **kwargs)


def build_boundaries(*, d_model, first_block, init_device, dtype, init_std):
    kwargs = {
        "d_model": d_model,
        "layer_norm_eps": first_block.attention_norm.eps,
        "config": first_block.ldm_config,
        "init_device": init_device,
        "dtype": dtype,
        "init_std": init_std,
    }
    return FastBoundary(**kwargs), FastOutputBoundary(**kwargs)


def build_forgetting_parameters(*, config, n_layers, init_device):
    q = config.quality
    return ForgettingParameters(
        groups=q.groups,
        n_layers=n_layers,
        lambda_init=q.rls.lambda_init,
        learn_lambda=q.rls.learn_lambda,
        init_device=init_device,
    )


def forgetting_num_params(*, config, n_layers):
    return (2 * n_layers + 1) * config.quality.groups


def block_num_params(*, config, d_model, block_idx, n_layers):
    rule = FastRule(
        d_model=d_model,
        block_idx=block_idx,
        n_layers=n_layers,
        layer_norm_eps=config.quality.eps,
        config=parameter_config(config),
        init_device="meta",
        dtype=torch.float32,
    )
    return sum(p.numel() for p in rule.parameters())


def boundary_num_params(*, config, d_model):
    kwargs = {
        "d_model": d_model,
        "layer_norm_eps": config.quality.eps,
        "config": parameter_config(config),
        "init_device": "meta",
        "dtype": torch.float32,
    }
    return sum(
        p.numel()
        for cls in (FastBoundary, FastOutputBoundary)
        for p in cls(**kwargs).parameters()
    )


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
    execution_factory=SegmentState,
):
    """The blocks of one model forward around a fresh state; every block runs through
    Module.__call__.

    A whole-block checkpoint cannot be recomputed against the mutable state, so only child-module
    and site checkpoints are allowed. ``execution_factory`` builds the state.
    """
    settings = Settings.from_mapping(config.execution.backend.settings)
    if not blocks:
        raise ValueError("an LDM segment requires at least one block")
    if any(isinstance(block, CheckpointWrapper) for block in blocks.values()):
        raise RuntimeError(
            "the fast backend cannot recompute a whole-block checkpoint: "
            "the state is mutable; checkpoint child modules or sites instead"
        )
    q = config.quality
    segment_state = execution_factory(
        total_edges=2 * len(blocks) + 1,
        span=settings.span,
        groups=q.groups,
        key_dim=q.key_dim,
        value_dim=hidden.shape[-1] // q.groups,
        covariance_scale=q.rls.alpha_init,
        running_covariance_dtype=settings.running_covariance_dtype,
    )
    _state_step_op.set_active_state(segment_state)
    token = segment_state.new_token(hidden)
    root_logit, block_logits = forgetting()
    state = _pending(segment_state, token, boundary(hidden, root_logit))
    for block_idx, block in blocks.items():
        if compile_enabled:
            mark_dynamic(hidden, (0, 1), strict=False)
        kwargs = dict(all_block_kwargs)
        kwargs.update(per_block_kwargs.get(int(block_idx), {}))
        hidden, state = block(
            hidden,
            ldm_state=state,
            ldm_lambda_logits=block_logits[int(block_idx)],
            **kwargs,
        )
    prepared = output_boundary(_query_source(q, hidden, state.previous_delta))
    hidden, _ = consume_pending(state, prepared, hidden)
    return hidden
