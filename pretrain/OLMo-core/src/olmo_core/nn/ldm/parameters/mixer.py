"""Parameter modules of the fast and fast-inference backends.

``LDMRule`` holds one block's read and write address projections
and static addresses, ``LDMBoundary`` the embedding write's key and
``LDMOutputBoundary`` the final read's query. They build and
initialize parameters only.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.ldm.components import _rms_norm as _context_rms_norm
from olmo_core.nn.ldm.components import (
    _BANK_DECAY,
    BankedLinear,
    ParameterBank,
    _BankedControlParameters,
    _bank_initializing,
    _linear as _bank_linear,
)
from olmo_core.nn.transformer.init import init_linear

from .config import LDMParameterConfig


class _LDMAddressProj(nn.Module):
    """Address network of one site: hidden layer ``w1``, address head ``w2`` and optional gate head.

    Under ``execution.common.parameter_bank`` every head is a ``BankedLinear`` whose
    ``weight``/``bias`` are views of the container's current bank tensor, taken on
    every access. Weights and address-network biases go to the decay bank and the
    gate bias to the no-decay bank, the paper recipe's partition of the unbanked
    names, so turning the bank on moves no parameter between optimizer groups.
    """

    def __init__(
        self,
        d_model: int,
        address_dim: int,
        hidden_dim: int,
        *,
        gate_dim: int = 0,
        bias: bool,
        init_device: str,
        dtype: torch.dtype,
        bank: Optional[ParameterBank] = None,
        context_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_dim = gate_dim
        # Learned per-channel gain of the RMS-normalized address-network input.
        self.context_norm = _context_rms_norm(
            d_model, eps=context_eps, bank=bank, init_device=init_device, dtype=dtype
        )
        # Logit written into the gate head bias at init. The effective gate is
        # 2*sigmoid(logit), so 0.0 means an exactly identity gate.
        self.gate_bias_init = 0.0
        self.w1 = _bank_linear(
            d_model, hidden_dim, bias=bias, bank=bank, init_device=init_device, dtype=dtype,
            bias_group=_BANK_DECAY,
        )
        self.w2 = _bank_linear(
            hidden_dim, address_dim, bias=bias, bank=bank, init_device=init_device, dtype=dtype,
            bias_group=_BANK_DECAY,
        )
        self.gate = (
            _bank_linear(
                hidden_dim,
                gate_dim,
                bias=True,
                bank=bank,
                init_device=init_device,
                dtype=dtype,
            )
            if gate_dim
            else None
        )

    def _parameter_bank(self) -> Optional[ParameterBank]:
        return self.w1.weight_slot.bank if isinstance(self.w1, BankedLinear) else None

    def reset_parameters(
        self,
        *,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        bank = self._parameter_bank()
        if bank is not None and not bank.writable():
            # Sharded bank outside ParameterBank.initializing(): the heads are
            # DTensor slices with no write-back. Transformer.init_weights'
            # per-module sweep lands here after apply_fsdp; the enclosing
            # rule's or boundary's reset_parameters() (which runs these same lines
            # inside the bank's initializing() context) is the one that counts.
            return
        self.context_norm.reset_parameters()
        init_linear(self.w1, std=std, generator=generator)
        init_linear(self.w2, std=std, generator=generator)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, self.gate_bias_init)


def _gate_logit(gain: float) -> float:
    """Gate head bias whose 2*sigmoid equals ``gain``."""
    return math.log(gain / (2.0 - gain))


class LDMBoundary(_BankedControlParameters):
    """Embedding-state initializer for grouped RLS LDM."""

    def __init__(
        self,
        *,
        d_model: int,
        layer_norm_eps: float,
        config: LDMParameterConfig,
        init_device: str,
        dtype: torch.dtype,
        init_std: float = 0.02,
    ):
        super().__init__()
        if d_model % config.groups != 0:
            raise OLMoConfigurationError("LDM groups must divide d_model")
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.config = config
        self.init_std = init_std
        self.key_dim = config.key_dim
        self.groups = config.groups
        self.state_dim = d_model
        self.group_dim = self.state_dim // self.groups
        self.address_dim = self.groups * self.key_dim
        # execution.common.parameter_bank: one flat bank per boundary, registered
        # first so its tensors are ``<boundary>.parameter_bank.{decay,no_decay}``.
        bank = (
            ParameterBank(init_device=init_device, dtype=dtype)
            if config.parameter_bank
            else None
        )
        if bank is not None:
            self.parameter_bank = bank
        input_key = torch.empty(
            self.groups,
            self.key_dim,
            device=init_device,
            dtype=dtype,
        )
        if config.dynamic_key:
            self.register_buffer("input_key", input_key, persistent=True)
        elif bank is None:
            self.input_key = nn.Parameter(input_key)
        else:
            # A learned static key decays like a weight.
            self._bank_control(
                "input_key", bank.reserve(self.groups, self.key_dim, group=_BANK_DECAY)
            )
        self.input_k_proj = (
            _LDMAddressProj(
                d_model,
                self.address_dim,
                config.hidden_dim,
                gate_dim=self.groups if config.write_gate else 0,
                bias=True,
                context_eps=config.eps,
                init_device=init_device,
                dtype=dtype,
                bank=bank,
            )
            if config.dynamic_key
            else None
        )
        if config.write_gate and config.write_gate_init != 1.0:
            assert self.input_k_proj is not None
            self.input_k_proj.gate_bias_init = _gate_logit(config.write_gate_init)
        if bank is not None:
            bank.seal()

    def reset_parameters(
        self,
        *,
        init_std: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        std = self.init_std if init_std is None else init_std
        # Same init functions in the same order whether or not the parameters
        # are banked; ``initializing()`` only supplies writable, unsharded
        # storage for the views (a no-op context without a bank).
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            if self.config.dynamic_key:
                nn.init.zeros_(self.input_key)
            else:
                nn.init.normal_(self.input_key, mean=0.0, std=1.0, generator=generator)
            if self.input_k_proj is not None:
                self.input_k_proj.reset_parameters(std=std, generator=generator)


class LDMOutputBoundary(_BankedControlParameters):
    """Final LM-head read from grouped RLS LDM state."""

    def __init__(
        self,
        *,
        d_model: int,
        layer_norm_eps: float,
        config: LDMParameterConfig,
        init_device: str,
        dtype: torch.dtype,
        init_std: float = 0.02,
    ):
        super().__init__()
        if d_model % config.groups != 0:
            raise OLMoConfigurationError("LDM groups must divide d_model")
        self.d_model = d_model
        self.layer_norm_eps = layer_norm_eps
        self.config = config
        self.init_std = init_std
        self.key_dim = config.key_dim
        self.groups = config.groups
        self.state_dim = d_model
        self.group_dim = self.state_dim // config.groups
        self.address_dim = self.groups * self.key_dim
        # execution.common.parameter_bank: one flat bank per boundary, registered
        # first so its tensors are ``<boundary>.parameter_bank.{decay,no_decay}``.
        bank = (
            ParameterBank(init_device=init_device, dtype=dtype)
            if config.parameter_bank
            else None
        )
        if bank is not None:
            self.parameter_bank = bank
        output_query = torch.empty(
            self.groups, self.key_dim, device=init_device, dtype=dtype
        )
        if config.dynamic_query:
            self.register_buffer("output_query", output_query, persistent=True)
        elif bank is None:
            self.output_query = nn.Parameter(output_query)
        else:
            # A learned static query decays like a weight.
            self._bank_control(
                "output_query", bank.reserve(self.groups, self.key_dim, group=_BANK_DECAY)
            )
        gains = self.groups
        if not config.read_gain:
            self.output_read_gain = None
        elif bank is None:
            self.output_read_gain = nn.Parameter(
                torch.empty((gains,), device=init_device, dtype=dtype)
            )
        else:
            # The read gain stays in the module dtype, so
            # ``control_group(dtype)`` is the ordinary no-decay bank and the
            # ``no_decay_fp32`` group stays empty (never allocated).
            self._bank_control(
                "output_read_gain",
                bank.reserve(gains, group=bank.control_group(dtype)),
            )
        self.output_q_proj = (
            _LDMAddressProj(
                d_model,
                self.address_dim,
                config.hidden_dim,
                gate_dim=self.groups if config.read_gate else 0,
                bias=True,
                context_eps=config.eps,
                init_device=init_device,
                dtype=dtype,
                bank=bank,
            )
            if config.dynamic_query
            else None
        )
        if config.read_gate and self.output_q_proj is None:
            raise OLMoConfigurationError(
                "read_gate is enabled at the output boundary but no read-gate "
                "projection exists; refusing to silently omit the configured gate"
            )
        if config.read_gate_init != 1.0:
            if self.output_q_proj is None:
                raise OLMoConfigurationError(
                    "'read_gate_init' is not supported by the output-boundary "
                    "read projection"
                )
            self.output_q_proj.gate_bias_init = _gate_logit(config.read_gate_init)
        if bank is not None:
            bank.seal()

    def reset_parameters(
        self,
        *,
        init_std: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        std = self.init_std if init_std is None else init_std
        # Same init functions in the same order whether or not the parameters
        # are banked; ``initializing()`` only supplies writable, unsharded
        # storage for the views (a no-op context without a bank).
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            if self.config.dynamic_query:
                nn.init.zeros_(self.output_query)
            else:
                nn.init.normal_(self.output_query, mean=0.0, std=1.0, generator=generator)
            if self.output_q_proj is not None:
                self.output_q_proj.reset_parameters(std=std, generator=generator)
            if self.output_read_gain is not None:
                nn.init.constant_(self.output_read_gain, self.config.read_gain_init)


class LDMRule(_BankedControlParameters):
    """Per-block read/write parameters for grouped RLS depth mixing."""

    def __init__(
        self,
        *,
        d_model: int,
        block_idx: int,
        n_layers: int,
        layer_norm_eps: float,
        config: LDMParameterConfig,
        init_device: str,
        dtype: torch.dtype,
        init_std: float = 0.02,
    ):
        super().__init__()
        if d_model % config.groups != 0:
            raise OLMoConfigurationError("LDM groups must divide d_model")
        self.d_model = d_model
        self.block_idx = block_idx
        self.n_layers = n_layers
        self.layer_norm_eps = layer_norm_eps
        self.config = config
        self.init_std = init_std
        self.key_dim = config.key_dim
        self.groups = config.groups
        self.state_dim = d_model
        self.group_dim = self.state_dim // self.groups
        self.address_dim = self.groups * self.key_dim
        # execution.common.parameter_bank: one flat bank per block, registered
        # first so its tensors are ``<rule>.parameter_bank.{decay,no_decay}``.
        # Linear weights, address-network biases and learned static addresses go to
        # ``decay``; gate biases and read gains to ``no_decay``. That is
        # the paper recipe's partition of the unbanked names, so the bank is an
        # optimizer-equivalent swap (see _LDMAddressProj).
        bank = (
            ParameterBank(init_device=init_device, dtype=dtype)
            if config.parameter_bank
            else None
        )
        if bank is not None:
            self.parameter_bank = bank
        attn_query = torch.empty(
            self.groups, self.key_dim, device=init_device, dtype=dtype
        )
        mlp_query = torch.empty_like(attn_query)
        if config.dynamic_query:
            self.register_buffer("attn_query", attn_query, persistent=True)
            self.register_buffer("mlp_query", mlp_query, persistent=True)
        elif bank is None:
            self.attn_query = nn.Parameter(attn_query)
            self.mlp_query = nn.Parameter(mlp_query)
        # Banked static queries are reserved with the keys below, after every
        # projection weight, so a G*K static address never shifts an
        # address-network weight off its natural alignment inside the decay bank.

        def address_projection(gate_dim: int) -> _LDMAddressProj:
            return _LDMAddressProj(
                d_model,
                self.address_dim,
                config.hidden_dim,
                gate_dim=gate_dim,
                bias=True,
                context_eps=config.eps,
                init_device=init_device,
                dtype=dtype,
                bank=bank,
            )

        read_gate_dim = self.groups if config.read_gate else 0
        self.attn_q_proj = address_projection(read_gate_dim) if config.dynamic_query else None
        self.mlp_q_proj = address_projection(read_gate_dim) if config.dynamic_query else None
        gains = self.groups
        if not config.read_gain:
            self.attn_read_gain = None
            self.mlp_read_gain = None
        elif bank is None:
            self.attn_read_gain = nn.Parameter(
                torch.empty((gains,), device=init_device, dtype=dtype)
            )
            self.mlp_read_gain = nn.Parameter(
                torch.empty((gains,), device=init_device, dtype=dtype)
            )
        else:
            # The read gains stay in the module dtype, so
            # ``control_group(dtype)`` is the ordinary no-decay bank and the
            # ``no_decay_fp32`` group stays empty (never allocated).
            gain_group = bank.control_group(dtype)
            self._bank_control("attn_read_gain", bank.reserve(gains, group=gain_group))
            self._bank_control("mlp_read_gain", bank.reserve(gains, group=gain_group))

        write_gate_dim = self.groups if config.write_gate else 0
        self.attn_k_proj = address_projection(write_gate_dim) if config.dynamic_key else None
        self.mlp_k_proj = address_projection(write_gate_dim) if config.dynamic_key else None
        # Set the gate head biases so the gates start at config.*_gate_init
        # instead of the identity 1.0.
        for gain, projections, kind in (
            (config.write_gate_init, (self.attn_k_proj, self.mlp_k_proj), "write"),
            (config.read_gate_init, (self.attn_q_proj, self.mlp_q_proj), "read"),
        ):
            if gain == 1.0:
                continue
            present = [proj for proj in projections if proj is not None]
            if not present:
                raise OLMoConfigurationError(
                    f"'{kind}_gate_init' was set but this block has no {kind}-address "
                    "projection to set"
                )
            for proj in present:
                proj.gate_bias_init = _gate_logit(gain)
        attn_key = torch.empty(
            self.groups, self.key_dim, device=init_device, dtype=dtype
        )
        mlp_key = torch.empty_like(attn_key)
        if config.dynamic_key:
            self.register_buffer("attn_key", attn_key, persistent=True)
            self.register_buffer("mlp_key", mlp_key, persistent=True)
        elif bank is None:
            self.attn_key = nn.Parameter(attn_key)
            self.mlp_key = nn.Parameter(mlp_key)
        if bank is not None:
            # Learned static addresses are the last
            # decay-bank reservations; see the note at the queries above.
            for name, is_buffer in (
                ("attn_query", config.dynamic_query),
                ("mlp_query", config.dynamic_query),
                ("attn_key", config.dynamic_key),
                ("mlp_key", config.dynamic_key),
            ):
                if not is_buffer:
                    self._bank_control(
                        name, bank.reserve(self.groups, self.key_dim, group=_BANK_DECAY)
                    )
            bank.seal()

    def reset_parameters(
        self,
        *,
        init_std: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        std = self.init_std if init_std is None else init_std
        # Same init functions in the same order whether or not the parameters
        # are banked; ``initializing()`` only supplies writable, unsharded
        # storage for the views (a no-op context without a bank).
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            for query in (self.attn_query, self.mlp_query):
                if self.config.dynamic_query:
                    nn.init.zeros_(query)
                else:
                    nn.init.normal_(query, mean=0.0, std=1.0, generator=generator)
            for key in (self.attn_key, self.mlp_key):
                if self.config.dynamic_key:
                    nn.init.zeros_(key)
                else:
                    nn.init.normal_(key, mean=0.0, std=1.0, generator=generator)
            for proj in (self.attn_k_proj, self.mlp_k_proj):
                if proj is not None:
                    proj.reset_parameters(std=std, generator=generator)
            for proj in (self.attn_q_proj, self.mlp_q_proj):
                if proj is not None:
                    proj.reset_parameters(std=std, generator=generator)
            for gain in (self.attn_read_gain, self.mlp_read_gain):
                if gain is not None:
                    nn.init.constant_(gain, self.config.read_gain_init)
