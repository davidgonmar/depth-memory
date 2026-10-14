"""Forward-only LDM execution, registered as ``fast-inference``.

A model is built with the ``fast`` backend's parameter modules, so its checkpoints load unchanged. After
loading, `prepare_inference` replaces those modules once with the tensors the kernels read (`edges`). Per
edge, a forward of many tokens (prefill) runs the projection kernel, the head/gate GEMM and the prefill
edge kernel, and a forward of few tokens (decode) runs the decode edge kernel alone. The token count is a
runtime dimension of every kernel. Nothing is kept for a backward: training uses the ``fast`` backend.
See DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.ldm.parameters import (
    ForgettingParameters,
    LDMBoundary,
    LDMOutputBoundary,
    LDMRule,
    block_config,
)
from olmo_core.nn.transformer.block import TransformerBlock

from .edges import InferenceEdge, InferenceRule
from .state import InferenceState
from .tuning import launches

HIDDEN_WIDTHS = (32, 64)


# A fast checkpoint also stores read_checkpoint, which only affects the backward, so inference ignores it.
FAST_TRAINING_SETTINGS = frozenset({"read_checkpoint"})


@dataclass(frozen=True)
class Settings:
    span: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Settings:
        values = {name: value for name, value in values.items() if name not in FAST_TRAINING_SETTINGS}
        # The edge kernels keep the running covariance in BF16, the fast backend's default. A model
        # trained with an FP32 running covariance would read differently here, so it is refused.
        covariance_dtype = values.pop("running_covariance_dtype", "bfloat16")
        if covariance_dtype != "bfloat16":
            raise ValueError("fast-inference keeps the running covariance in BF16 and does not run "
                             f"running_covariance_dtype={covariance_dtype!r}; the fast backend runs this "
                             "model forward-only under torch.no_grad")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"fast-inference has no settings {sorted(unknown)}")
        settings = cls(**values)
        if type(settings.span) is not int or not 1 <= settings.span <= 8:
            raise ValueError("fast-inference span must be an integer between 1 and 8")
        return settings


def validate_quality(quality):
    """Accept only the released models' depth mixer: RLS with key_dim 4 or 8, address hidden width 32 or
    64, normalized dynamic addresses on the stream, write and read gates, the branch output as the written
    value, a learned read gain, a learned forgetting factor, and both residual additions. Anything else
    raises."""
    quality.validate()
    a, rls, gain = quality.addresses, quality.rls, quality.learned_read_gain
    required = {
        "the RLS recurrence": quality.recurrence == "rls",
        "key_dim 4 or 8": quality.key_dim in (4, 8),
        f"address hidden widths {HIDDEN_WIDTHS}": a.hidden_dim in HIDDEN_WIDTHS,
        "dynamic addresses on the stream": (a.query.mode == a.key.mode == "dynamic"
                                            and "branch_delta" not in (a.query.source, a.key.source)),
        "normalized keys and queries": a.normalize_key and a.normalize_query,
        "write and read gates": quality.write_gate.enabled and quality.read_gate.enabled,
        "the branch output as the written value": quality.value_source == "branch_delta",
        "a learned read gain": gain.enabled,
        "a learned forgetting factor": rls.learn_lambda and rls.lambda_init < 1.0,
        "both residual additions": not (quality.replace_read or quality.no_post_function_residual),
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise ValueError(f"fast-inference runs the released models' depth mixer; this config lacks {missing}")


def validate_execution(common):
    common.validate()


def parameter_config(config):
    validate_quality(config.quality)
    validate_execution(config.execution.common)
    Settings.from_mapping(config.execution.backend.settings)
    return block_config(config)


class Stream(NamedTuple):
    """A residual stream whose last branch is not added yet: ``base + alpha * delta``."""

    base: torch.Tensor
    delta: torch.Tensor
    alpha: float

    def resolve(self) -> torch.Tensor:
        return torch.add(self.base, self.delta, alpha=self.alpha)


class Segment:
    """Holds the written value of an edge until the block asks for the edge's read."""

    def __init__(self, state: InferenceState):
        self.state = state
        self.pending = False
        self.value = None

    def write(self, value: torch.Tensor):
        """`value` is the written value: the embedding at the first write, the branch output after."""
        if self.pending:
            raise RuntimeError("a pending write was not read before the next write")
        self.pending, self.value = True, value

    @torch.compiler.disable
    def read(self, edge: InferenceEdge, stream, norm):
        """Run `edge` and return the stream and the next branch input, ``norm(stream + read)``. When
        `stream` is a `Stream`, its pending residual add happens here. The final read passes no norm and
        gets ``stream + read``. Compiled graphs call this as an opaque function because it launches
        kernels."""
        if not self.pending:
            raise RuntimeError("no pending write to read from")
        self.pending = False
        base, delta, alpha = stream if isinstance(stream, Stream) else (stream, None, 1.0)
        return self.state.step(edge, base, self.value, norm, delta=delta, alpha=alpha)


def _check_norm(norm) -> None:
    """The edge kernels apply the block's RMSNorm themselves, which must be full precision with a gain
    and no bias."""
    if getattr(norm, "bias", None) is not None or not getattr(norm, "full_precision", True):
        raise OLMoConfigurationError("fast-inference applies a bias-free full-precision RMSNorm")
    if getattr(norm, "weight", None) is None:
        raise OLMoConfigurationError("fast-inference applies an RMSNorm with a learned gain")


class FastInferenceTransformerBlock(TransformerBlock):
    def __init__(self, *, d_model, block_idx, n_layers, ldm, layer_norm, init_device="cpu",
                 init_std, **kwargs):
        super().__init__(d_model=d_model, block_idx=block_idx, n_layers=n_layers, layer_norm=layer_norm,
                         init_device=init_device, **kwargs)
        self.ldm = ldm
        self.ldm_config = parameter_config(ldm)
        self.ldm_rule = LDMRule(
            d_model=d_model, block_idx=block_idx, n_layers=n_layers,
            layer_norm_eps=self.attention_norm.eps,
            config=self.ldm_config, init_device=init_device,
            dtype=next(self.attention.parameters()).dtype, init_std=init_std,
        )
        _check_norm(self.attention_norm)
        _check_norm(self.feed_forward_norm)

    def forward(self, x, *, ldm_state: Segment, loss_div_factor=None, **kwargs):
        del loss_div_factor
        segment, edges = ldm_state, self.ldm_rule
        x, normed = segment.read(edges.attention, x, self.attention_norm)
        delta = self.attention_residual_stream.dropout(self.attention(normed, **kwargs))
        segment.write(delta)
        updated, normed = segment.read(edges.feed_forward, Stream(x, delta, self.attention_residual_stream.alpha),
                                       self.feed_forward_norm)
        delta = self.feed_forward_residual_stream.dropout(self.feed_forward(normed))
        segment.write(delta)
        # The residual add of this branch is left to the next read, which fuses it with its addresses.
        return Stream(updated, delta, self.feed_forward_residual_stream.alpha)

    def apply_tp(self, *args, **kwargs):
        raise OLMoConfigurationError("fast-inference has no tensor parallelism")

    def apply_pp(self, *args, **kwargs):
        raise OLMoConfigurationError("fast-inference keeps a segment in one pipeline stage")


def build_block(*, ldm, **kwargs):
    if kwargs.pop("feed_forward_moe", None) is not None or kwargs.get("feed_forward") is None:
        raise ValueError("fast-inference builds dense feed-forward blocks")
    return FastInferenceTransformerBlock(ldm=ldm, **kwargs)


def build_boundaries(*, d_model, first_block, init_device, dtype, init_std):
    kwargs = {
        "d_model": d_model,
        "layer_norm_eps": first_block.attention_norm.eps,
        "config": first_block.ldm_config,
        "init_device": init_device,
        "dtype": dtype,
        "init_std": init_std,
    }
    return LDMBoundary(**kwargs), LDMOutputBoundary(**kwargs)


def build_forgetting_parameters(*, config, n_layers, init_device):
    q = config.quality
    return ForgettingParameters(groups=q.groups, n_layers=n_layers, lambda_init=q.rls.lambda_init,
                                learn_lambda=q.rls.learn_lambda, init_device=init_device)


def forgetting_num_params(*, config, n_layers):
    return (2 * n_layers + 1) * config.quality.groups


def block_num_params(*, config, d_model, block_idx, n_layers):
    rule = LDMRule(d_model=d_model, block_idx=block_idx, n_layers=n_layers,
                   layer_norm_eps=config.quality.eps, config=parameter_config(config),
                   init_device="meta", dtype=torch.float32)
    return sum(p.numel() for p in rule.parameters())


def boundary_num_params(*, config, d_model):
    kwargs = {"d_model": d_model, "layer_norm_eps": config.quality.eps, "config": parameter_config(config),
              "init_device": "meta", "dtype": torch.float32}
    return sum(p.numel() for cls in (LDMBoundary, LDMOutputBoundary)
               for p in cls(**kwargs).parameters())


def prepare_inference(model) -> None:
    """Turn a loaded ``fast-inference`` model into its inference form, in place, once.

    Every edge's trained parameter modules (both address networks, static addresses, forgetting logit
    and read gain) become one `InferenceEdge` holding exactly the tensors the kernels read. A block's
    ``ldm_rule`` becomes an `InferenceRule` with the edges that end at its attention and
    feed-forward inputs, the output boundary becomes the final edge, and the input boundary and the
    forgetting parameters, now folded into the edges, are removed. The model is inference only
    afterwards, and loading training parameters into it raises.
    """
    blocks = list(model.blocks.values())
    if not getattr(model, "_ldm_enabled", False) or not blocks:
        raise ValueError("prepare_inference needs a model with depth-mixer blocks")
    if not all(isinstance(block, FastInferenceTransformerBlock) for block in blocks):
        raise ValueError("prepare_inference needs a model built with the fast-inference backend")
    if isinstance(model.ldm_output_boundary, InferenceEdge):
        raise ValueError("the model is already in its inference form")
    q = blocks[0].ldm.quality
    boundary, output = model.ldm_boundary, model.ldm_output_boundary
    embedding_logit, block_logits = model.ldm_retention()
    writes = [(boundary.input_k_proj, boundary.input_key, embedding_logit)]
    reads = []
    for index, block in enumerate(blocks):
        sites = block.ldm_rule
        writes += [(sites.attn_k_proj, sites.attn_key, block_logits[index][0]),
                   (sites.mlp_k_proj, sites.mlp_key, block_logits[index][1])]
        reads += [(sites.attn_q_proj, sites.attn_query, sites.attn_read_gain),
                  (sites.mlp_q_proj, sites.mlp_query, sites.mlp_read_gain)]
    reads.append((output.output_q_proj, output.output_query, output.output_read_gain))
    edges = [InferenceEdge.from_sites(*write, *read, groups=q.groups, key_dim=q.key_dim, eps=q.eps)
             for write, read in zip(writes, reads, strict=True)]
    for index, block in enumerate(blocks):
        block.ldm_rule = InferenceRule(edges[2 * index], edges[2 * index + 1])
    model.ldm_output_boundary = edges[-1]
    model.ldm_boundary = None
    model.ldm_retention = None


def run_segment(hidden, blocks, *, boundary, output_boundary, forgetting, config, all_block_kwargs,
                per_block_kwargs, compile_enabled):
    """One forward over the whole depth: the embedding write, every block, the final read."""
    del boundary, forgetting, compile_enabled
    if torch.is_grad_enabled():
        raise RuntimeError("fast-inference is forward-only: run it under torch.no_grad or inference_mode; "
                           "training uses the fast backend")
    if not blocks:
        raise ValueError("fast-inference requires at least one block")
    if not isinstance(output_boundary, InferenceEdge):
        raise RuntimeError("call fast_inference.backend.prepare_inference(model) after loading the checkpoint")
    settings = Settings.from_mapping(config.execution.backend.settings)
    q = config.quality
    value_dim = hidden.shape[-1] // q.groups
    state = InferenceState(
        total_edges=2 * len(blocks) + 1, span=settings.span, groups=q.groups, key_dim=q.key_dim,
        value_dim=value_dim, hidden_dim=q.addresses.hidden_dim,
        covariance_scale=q.rls.alpha_init, eps=q.eps,
        context_eps=next(iter(blocks.values())).attention_norm.eps,
        launches=launches(torch.cuda.get_device_capability(hidden.device), q.groups, q.key_dim, value_dim,
                          q.addresses.hidden_dim),
    )
    state.begin(hidden)
    segment = Segment(state)
    segment.write(hidden)
    for block_idx, block in blocks.items():
        kwargs = dict(all_block_kwargs)
        kwargs.update(per_block_kwargs.get(int(block_idx), {}))
        hidden = block(hidden, ldm_state=segment, **kwargs)
    return segment.read(output_boundary, hidden, None)[1]
