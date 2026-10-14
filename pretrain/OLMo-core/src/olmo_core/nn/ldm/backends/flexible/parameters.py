"""Parameter containers of the flexible backend: block sites, embedding write and final read, forgetting logits."""

from typing import Any

import torch
import torch.nn as nn

from olmo_core.nn.ldm.components import (
    ForgettingParameters,
    LDMBlockParameters,
    LDMInputParameters,
    LDMOutputParameters,
)


def block_parameters(d_model: int, config: Any, *, context_eps: float, init_device: str,
                     dtype: torch.dtype, init_std: float) -> LDMBlockParameters:
    return LDMBlockParameters(
        d_model, config, context_eps=context_eps, init_device=init_device, dtype=dtype, init_std=init_std
    )


def build_boundaries(*, d_model: int, first_block: Any, init_device: str, dtype: torch.dtype,
                     init_std: float) -> tuple[nn.Module, nn.Module]:
    config = first_block.ldm
    common = dict(context_eps=first_block.attention_norm.eps, init_device=init_device, dtype=dtype,
                  init_std=init_std)
    return LDMInputParameters(d_model, config, **common), LDMOutputParameters(d_model, config, **common)


def build_forgetting_parameters(*, config: Any, n_layers: int, init_device: str) -> ForgettingParameters:
    q = config.quality
    return ForgettingParameters(
        groups=q.groups, n_layers=n_layers, lambda_init=q.rls.lambda_init, learn_lambda=q.rls.learn_lambda,
        init_device=init_device, key_dim=q.key_dim, learn_alpha=q.rls.learn_alpha,
        alpha_init=q.rls.alpha_init, ridge_min=q.rls.ridge_min,
    )


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def block_num_params(*, config: Any, d_model: int, block_idx: int, n_layers: int) -> int:
    del block_idx, n_layers
    return _count(block_parameters(d_model, config, context_eps=config.quality.eps,
                                   init_device="meta", dtype=torch.float32, init_std=0.02))


def boundary_num_params(*, config: Any, d_model: int) -> int:
    common = dict(context_eps=config.quality.eps, init_device="meta", dtype=torch.float32, init_std=0.02)
    return _count(LDMInputParameters(d_model, config, **common)) + _count(
        LDMOutputParameters(d_model, config, **common)
    )


def forgetting_num_params(*, config: Any, n_layers: int) -> int:
    q = config.quality
    ridge = q.groups * q.key_dim if q.rls.learn_alpha else 0
    return (2 * n_layers + 1) * q.groups + ridge
