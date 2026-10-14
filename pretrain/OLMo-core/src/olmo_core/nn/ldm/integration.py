"""Backend selection. A run names one backend; that module parses its own settings (raising on
any it does not know), builds the blocks and boundaries, and runs the whole segment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping


_BACKEND_MODULES = {
    "reference": "olmo_core.nn.ldm.backends.reference",
    "flexible": "olmo_core.nn.ldm.backends.flexible",
    "fast": "olmo_core.nn.ldm.backends.fast.backend",
    # Forward-only execution of fast-backend checkpoints, for prefill and decode.
    "fast-inference": "olmo_core.nn.ldm.backends.fast_inference.backend",
}


def _load_backend_module(name: str) -> Any:
    try:
        module_name = _BACKEND_MODULES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown LDM backend {name!r}; "
            f"expected one of {sorted(_BACKEND_MODULES)}"
        ) from error
    return import_module(module_name)


def load_backend_settings(name: str, settings: Mapping[str, Any]) -> Any:
    """Load settings through the named backend's local parser."""

    backend_module = _load_backend_module(name)
    return backend_module.Settings.from_mapping(settings)


def _validate_fast_shapes(config: Any, d_model: int) -> None:
    """The fast backends need K in {4, 8, 16} and d_model divisible by the number of groups and by
    8 (the address networks' input-gradient GEMM copies BF16 rows of d_model with TMA). The fast
    backend pads any per-group value width D to a multiple of 8 inside its state; the
    fast-inference kernels need D divisible by 64."""
    name = config.execution.backend.name
    if name in {"reference", "flexible"}:
        return
    quality = config.quality
    if quality.key_dim not in (4, 8, 16):
        raise ValueError("the fast backends require key_dim K in {4, 8, 16}")
    if quality.groups <= 0 or d_model <= 0 or d_model % quality.groups:
        raise ValueError("the fast backends require d_model divisible by positive groups")
    if d_model % 8:
        raise ValueError(f"the fast backends require d_model divisible by 8; got d_model={d_model}")
    value_dim = d_model // quality.groups
    if name == "fast-inference" and value_dim % 64:
        raise ValueError(
            "the fast-inference backend requires a per-group value width D=d_model/groups "
            f"divisible by 64; got D={value_dim} (d_model={d_model}, groups={quality.groups})"
        )


def validate_backend_selection(
    name: str,
    settings: Mapping[str, Any],
    quality: Any,
    common: Any,
) -> Any:
    """Validate settings and model math in the selected backend module."""

    parsed = load_backend_settings(name, settings)
    backend_module = _load_backend_module(name)
    backend_module.validate_quality(quality)
    backend_module.validate_execution(common)
    return parsed


def block_num_params(
    *,
    config: Any,
    d_model: int,
    block_idx: int,
    n_layers: int,
) -> int:
    """Count the parameters of one block's LDM sites on the selected backend."""

    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.block_num_params(
        config=config,
        d_model=d_model,
        block_idx=block_idx,
        n_layers=n_layers,
    )


def boundary_num_params(*, config: Any, d_model: int) -> int:
    """Count the parameters of the embedding write and the final read on the selected backend."""

    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.boundary_num_params(config=config, d_model=d_model)


def forgetting_num_params(*, config: Any, n_layers: int) -> int:
    """Count the selected backend's forgetting parameters."""

    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.forgetting_num_params(config=config, n_layers=n_layers)


def build_block(*, ldm: Any, **kwargs: Any) -> Any:
    """Build the selected backend's transformer block."""

    ldm.validate()
    _validate_fast_shapes(ldm, kwargs["d_model"])
    name = ldm.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.build_block(ldm=ldm, **kwargs)


def build_boundaries(*, config: Any, **kwargs: Any) -> tuple[Any, Any]:
    """Build the selected backend's embedding-write and final-read parameters."""

    _validate_fast_shapes(config, kwargs["d_model"])
    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.build_boundaries(**kwargs)


def build_forgetting_parameters(*, config: Any, **kwargs: Any) -> Any:
    """Build the selected backend's forgetting parameters."""

    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.build_forgetting_parameters(config=config, **kwargs)


def run_segment(*args: Any, config: Any, **kwargs: Any) -> Any:
    """Run the whole segment (every block, the embedding write and the final read) on the selected backend."""

    hidden = args[0] if args else kwargs["hidden"]
    _validate_fast_shapes(config, hidden.shape[-1])
    name = config.execution.backend.name
    backend_module = _load_backend_module(name)
    return backend_module.run_segment(*args, config=config, **kwargs)
