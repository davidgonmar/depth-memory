"""The LDM configuration: a ``quality`` section that defines the model and an ``execution`` section
with the numerics shared by every backend and the selected backend with its own settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from olmo_core.config import Config


@dataclass
class LDMAddressEndpointConfig(Config):
    """How one address (the read query or the write key) is obtained.

    ``dynamic`` addresses are produced from ``source`` by the SiLU address network. ``static``
    addresses are learned per site and have no source. A static query requires the read gate to be
    disabled, and a static key requires the write gate to be disabled, because the gates are heads
    of the address network.
    """

    mode: str = "dynamic"
    source: Optional[str] = "current_stream"


def _default_query_address() -> LDMAddressEndpointConfig:
    return LDMAddressEndpointConfig(mode="dynamic", source="current_stream")


def _default_key_address() -> LDMAddressEndpointConfig:
    return LDMAddressEndpointConfig(mode="dynamic", source="updated_stream")


@dataclass
class LDMAddressesConfig(Config):
    """How read queries and write keys are produced."""

    hidden_dim: int = 256
    query: LDMAddressEndpointConfig = field(default_factory=_default_query_address)
    key: LDMAddressEndpointConfig = field(default_factory=_default_key_address)
    normalize_query: bool = True
    normalize_key: bool = True


@dataclass
class LDMRLSConfig(Config):
    """Initial covariance and forgetting factor of the recurrence."""

    alpha_init: float = 1.0
    lambda_init: float = 0.99
    learn_lambda: bool = True
    # A learned ridge: the initial covariance is P_{-1} = diag(1 / ridge) with one value per group
    # and key dimension, ridge = ridge_min + softplus(raw), initialised at 1 / alpha_init. The floor
    # bounds alpha by 1 / ridge_min. Only the reference and flexible backends implement it.
    learn_alpha: bool = False
    ridge_min: float = 0.25


@dataclass
class LDMGateConfig(Config):
    """Per-group token gate ``2 * sigmoid(logit)`` on the read query or the write observation."""

    enabled: bool = True
    initial_gain: float = 1.0


@dataclass
class LDMLearnedReadGainConfig(Config):
    """Learned per-group gain of the read query; the identity at ``initial_gain = 1``."""

    enabled: bool = True
    initial_gain: float = 1.0


@dataclass
class LDMQualityConfig(Config):
    """The model: everything that defines the function a checkpoint computes."""

    key_dim: int = 8
    recurrence: str = "rls"
    groups: int = 16
    addresses: LDMAddressesConfig = field(default_factory=LDMAddressesConfig)
    value_source: str = "branch_delta"
    rls: LDMRLSConfig = field(default_factory=LDMRLSConfig)
    read_gate: LDMGateConfig = field(default_factory=LDMGateConfig)
    write_gate: LDMGateConfig = field(default_factory=LDMGateConfig)
    learned_read_gain: LDMLearnedReadGainConfig = field(
        default_factory=LDMLearnedReadGainConfig
    )
    # Residual-path ablations. ``replace_read`` feeds each branch only the memory read, including
    # at the final read; ``no_post_function_residual`` replaces the stream by the branch output.
    # Only the reference and flexible backends implement them.
    replace_read: bool = False
    no_post_function_residual: bool = False
    # Numerical floor of the address norms, the write weights and the forgetting factors.
    eps: float = 1e-6

    def validate(self) -> None:
        if not 0 < self.eps < 1:
            raise ValueError("quality.eps must be in (0, 1)")
        if self.key_dim <= 0:
            raise ValueError("quality.key_dim must be positive")
        if self.recurrence not in {"rls", "delta_rule", "outer_product"}:
            raise ValueError("quality.recurrence must be rls, delta_rule or outer_product")
        if self.groups <= 0:
            raise ValueError("quality.groups must be positive")
        if self.addresses.hidden_dim <= 0:
            raise ValueError("quality.addresses.hidden_dim must be positive")

        query = self.addresses.query
        key = self.addresses.key
        for label, endpoint in (("query", query), ("key", key)):
            if endpoint.mode not in {"dynamic", "static"}:
                raise ValueError(f"quality.addresses.{label}.mode must be dynamic or static")
            if endpoint.mode == "static" and endpoint.source is not None:
                raise ValueError(
                    f"quality.addresses.{label}.source must be null for a static address"
                )
        if query.mode == "dynamic" and query.source not in {"current_stream", "branch_delta"}:
            raise ValueError("a dynamic query source must be current_stream or branch_delta")
        if key.mode == "dynamic" and key.source not in {"updated_stream", "branch_delta"}:
            raise ValueError("a dynamic key source must be updated_stream or branch_delta")
        if query.mode == "static" and self.read_gate.enabled:
            raise ValueError("a static query address requires quality.read_gate.enabled=false")
        if key.mode == "static" and self.write_gate.enabled:
            raise ValueError("a static key address requires quality.write_gate.enabled=false")
        if self.value_source not in {"branch_delta", "updated_stream"}:
            raise ValueError("quality.value_source must be branch_delta or updated_stream")

        if self.rls.alpha_init <= 0:
            raise ValueError("quality.rls.alpha_init must be positive")
        if self.rls.ridge_min < 0:
            raise ValueError("quality.rls.ridge_min must be non-negative")
        if self.rls.learn_alpha:
            if self.recurrence != "rls":
                raise ValueError("quality.rls.learn_alpha requires the rls recurrence")
            if 1.0 / self.rls.alpha_init <= self.rls.ridge_min:
                raise ValueError(
                    "quality.rls.learn_alpha requires 1/alpha_init > ridge_min so the "
                    "initial ridge is representable"
                )
        if not 0 < self.rls.lambda_init <= 1:
            raise ValueError("quality.rls.lambda_init must be in (0, 1]")
        if self.rls.learn_lambda and self.rls.lambda_init == 1:
            raise ValueError(
                "quality.rls.learn_lambda=true requires quality.rls.lambda_init < 1, "
                "since lambda=1 has no finite logit"
            )
        for name, gate in (("read_gate", self.read_gate), ("write_gate", self.write_gate)):
            if not 0 < gate.initial_gain < 2:
                raise ValueError(f"quality.{name}.initial_gain must be in (0, 2)")
        if self.learned_read_gain.initial_gain <= 0:
            raise ValueError("quality.learned_read_gain.initial_gain must be positive")


@dataclass
class LDMCommonExecutionConfig(Config):
    """Execution choices that every backend understands. Neither changes the model's function."""

    # Recompute the address networks in the backward instead of saving their activations.
    checkpoint_address_projections: bool = False
    # Hold the learned site parameters of every block and boundary in flat tensors, one per
    # weight-decay class, instead of ~30 small parameters each. This removes FSDP's per-tensor host
    # cost. The checkpoint layout differs from the unbanked one, so the two cannot be mixed.
    parameter_bank: bool = False

    def validate(self) -> None:
        for name in ("checkpoint_address_projections", "parameter_bank"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"execution.common.{name} must be a boolean")


@dataclass
class LDMBackendConfig(Config):
    """The selected backend and the settings that only that backend parses."""

    name: str = "reference"
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LDMExecutionConfig(Config):
    """Common numerical policy plus the backend-local execution definition."""

    common: LDMCommonExecutionConfig = field(default_factory=LDMCommonExecutionConfig)
    backend: LDMBackendConfig = field(default_factory=LDMBackendConfig)


@dataclass
class LDMConfig(Config):
    """One complete LDM configuration."""

    quality: LDMQualityConfig = field(default_factory=LDMQualityConfig)
    execution: LDMExecutionConfig = field(default_factory=LDMExecutionConfig)

    def validate(self) -> None:
        from .integration import validate_backend_selection

        self.quality.validate()
        self.execution.common.validate()
        quality = self.quality
        reference_only = [
            name for name in ("replace_read", "no_post_function_residual") if getattr(quality, name)
        ]
        if quality.recurrence != "rls":
            reference_only.append("recurrence")
        if quality.value_source != "branch_delta":
            reference_only.append("value_source")
        if quality.rls.learn_alpha:
            reference_only.append("learn_alpha")
        if quality.addresses.key.mode == "dynamic" and quality.addresses.key.source != "updated_stream":
            reference_only.append("key_source")
        if reference_only and self.execution.backend.name not in {"reference", "flexible"}:
            raise ValueError(
                f"quality option(s) {reference_only} require "
                "execution.backend.name='reference' or 'flexible'"
            )
        validate_backend_selection(
            self.execution.backend.name,
            self.execution.backend.settings,
            self.quality,
            self.execution.common,
        )

    def block_num_params(self, d_model: int, *, block_idx: int, n_layers: int) -> int:
        """Count the selected backend's per-block parameters."""

        from .integration import block_num_params

        return block_num_params(
            config=self, d_model=d_model, block_idx=block_idx, n_layers=n_layers
        )

    def boundary_num_params(self, d_model: int) -> int:
        """Count the selected backend's embedding-write and final-read parameters."""

        from .integration import boundary_num_params

        return boundary_num_params(config=self, d_model=d_model)

    def forgetting_num_params(self, n_layers: int) -> int:
        """Count the selected backend's forgetting parameters."""

        from .integration import forgetting_num_params

        return forgetting_num_params(config=self, n_layers=n_layers)
