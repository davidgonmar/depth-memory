"""Constructor-level configuration of the fast backends' parameter modules."""

from dataclasses import dataclass

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


@dataclass
class LDMParameterConfig(Config):
    """Model choices the rule and boundary constructors read; ``block_config`` derives it from an
    ``LDMConfig``."""

    key_dim: int = 8
    groups: int = 16
    hidden_dim: int = 256
    normalize_query: bool = True
    normalize_key: bool = True
    eps: float = 1e-5
    lambda_init: float = 0.99
    read_gain: bool = False
    read_gain_init: float = 1.0
    dynamic_query: bool = True
    dynamic_key: bool = True
    read_gate: bool = False
    write_gate: bool = False
    write_gate_init: float = 1.0
    read_gate_init: float = 1.0
    checkpoint_address_projections: bool = False
    # Every learned site parameter of a rule or boundary lives in that container's flat
    # ``ParameterBank``, one tensor per weight-decay class (execution.common.parameter_bank).
    parameter_bank: bool = False

    def __post_init__(self) -> None:
        if self.key_dim not in (4, 8, 16):
            raise OLMoConfigurationError("key_dim must be one of {4, 8, 16}")
        if self.groups <= 0:
            raise OLMoConfigurationError("groups must be positive")
        if self.hidden_dim <= 0:
            raise OLMoConfigurationError("hidden_dim must be positive")
        if self.eps <= 0:
            raise OLMoConfigurationError("eps must be positive")
        if not 0 < self.lambda_init <= 1:
            raise OLMoConfigurationError("lambda_init must be in (0, 1]")
        if self.read_gain_init <= 0:
            raise OLMoConfigurationError("read_gain_init must be positive")
        for name, value in (
            ("write_gate_init", self.write_gate_init),
            ("read_gate_init", self.read_gate_init),
        ):
            if not 0 < value < 2:
                raise OLMoConfigurationError(f"{name} must be in (0, 2)")
        if not self.dynamic_query and self.read_gate:
            raise OLMoConfigurationError("a static query cannot use read_gate")
        if not self.dynamic_key and self.write_gate:
            raise OLMoConfigurationError("a static key cannot use write_gate")
