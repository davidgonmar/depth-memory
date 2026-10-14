"""Parameter modules shared by the fast and fast-inference backends.

The rule, the two boundaries and the forgetting logits only build and initialize
parameters; each backend computes with them in its own way. ``block_config`` maps the
model's quality config to the constructor config.
"""

from .config import LDMParameterConfig
from .forgetting import ForgettingParameters
from .mixer import LDMBoundary, LDMOutputBoundary, LDMRule


def block_config(config) -> LDMParameterConfig:
    """The constructor config of the rule and boundaries for one model config."""
    q, common = config.quality, config.execution.common
    return LDMParameterConfig(
        parameter_bank=common.parameter_bank,
        key_dim=q.key_dim,
        groups=q.groups,
        hidden_dim=q.addresses.hidden_dim,
        normalize_query=q.addresses.normalize_query,
        normalize_key=q.addresses.normalize_key,
        eps=q.eps,
        lambda_init=q.rls.lambda_init,
        dynamic_query=q.addresses.query.mode == "dynamic",
        dynamic_key=q.addresses.key.mode == "dynamic",
        read_gate=q.read_gate.enabled,
        write_gate=q.write_gate.enabled,
        # A disabled gate has no head to initialize, whatever initial_gain its config holds.
        write_gate_init=q.write_gate.initial_gain if q.write_gate.enabled else 1.0,
        read_gate_init=q.read_gate.initial_gain if q.read_gate.enabled else 1.0,
        read_gain=q.learned_read_gain.enabled,
        read_gain_init=q.learned_read_gain.initial_gain,
        checkpoint_address_projections=common.checkpoint_address_projections,
    )


__all__ = [
    "LDMBoundary",
    "LDMParameterConfig",
    "LDMOutputBoundary",
    "LDMRule",
    "ForgettingParameters",
    "block_config",
]
