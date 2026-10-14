"""Learned LDM parameter modules and the math around them, shared by the reference and flexible
backends: address networks, gates, read gains, the parameter bank and the forgetting logits. The
state and its recurrence belong to the backends.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from olmo_core.exceptions import OLMoConfigurationError


def identity_centered_gate(
    gate_logit: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """The identity-centred token gate ``2 * sigmoid(logit)``."""

    compute = (
        gate_logit if gate_logit.dtype in (torch.float32, torch.float64) else gate_logit.float()
    )
    return (2.0 * torch.sigmoid(compute)).to(dtype)


def apply_read_gain(query: torch.Tensor, gain: Optional[torch.Tensor]) -> torch.Tensor:
    """Scale each group of the ``[..., G, K]`` query by its learned read gain."""

    if gain is None:
        return query
    return query * gain.to(query.dtype).unsqueeze(-1)


def complete_read(read: torch.Tensor, add_to: Optional[torch.Tensor]) -> torch.Tensor:
    """Add the memory read to the residual stream, or return it alone when there is no stream."""

    if add_to is None:
        return read
    return add_to + read.to(add_to.dtype)


def _truncated_normal_(
    tensor: torch.Tensor,
    *,
    std: float,
    generator: Optional[torch.Generator],
) -> None:
    nn.init.trunc_normal_(
        tensor,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )


def forgetting_from_logit(logit: torch.Tensor, *, lambda_init: float, eps: float) -> torch.Tensor:
    """The forgetting factor ``sigmoid(logit)``, floored at ``eps``; exactly one without forgetting."""
    if lambda_init >= 1.0:
        return torch.ones_like(logit, dtype=torch.float32)
    return torch.sigmoid(logit.float()).clamp_min(eps)


def forgetting_logit_init(lambda_init: float) -> float:
    """The logit whose forgetting factor is ``lambda_init``."""
    if lambda_init >= 1.0:
        return 0.0
    return math.log(lambda_init / (1.0 - lambda_init))


def _gate_logit(initial_gain: float) -> float:
    probability = initial_gain / 2.0
    return math.log(probability / (1.0 - probability))


_BANK_DECAY = "decay"
_BANK_NO_DECAY = "no_decay"
_BANK_NO_DECAY_FP32 = "no_decay_fp32"


class _BankSlot:
    """One reserved region of a flat LDM parameter bank."""

    __slots__ = ("bank", "group", "numel", "offset", "shape")

    def __init__(
        self,
        bank: ParameterBank,
        group: str,
        offset: int,
        shape: tuple[int, ...],
    ) -> None:
        self.bank = bank
        self.group = group
        self.offset = offset
        self.shape = shape
        numel = 1
        for size in shape:
            numel *= size
        self.numel = numel

    def tensor(self) -> torch.Tensor:
        """View this slot inside whichever bank tensor is current *right now*.

        FSDP2 replaces the registered parameter with an unsharded working copy
        for the duration of a forward, so a view may never be cached at init.
        """

        flat = self.bank.storage(self.group)
        return flat[self.offset : self.offset + self.numel].view(self.shape)


class ParameterBank(nn.Module):
    """Flat storage for the learned site parameters of one LDM container.

    Every depth site otherwise contributes ten to twenty individually tiny
    parameters, and FSDP2 pays per-tensor host cost for each of them in the
    pre-forward cast, the post-backward reduce, and the optimizer step. A bank
    keeps one flat tensor per weight-decay class instead; the site modules hold
    offset/shape descriptors that are resolved on every access.

    The split follows the weight-decay classes: ``decay`` holds the linear
    weights and learned static addresses, ``no_decay`` the biases, norm gains
    and read gains. A bias can be reserved in ``decay`` instead (see
    :class:`BankedLinear`). ``no_decay_fp32`` exists only when the module dtype is
    not float32, so that an FP32 control parameter is never silently downcast.
    """

    def __init__(self, *, init_device: str, dtype: torch.dtype, slot_alignment: int = 1) -> None:
        super().__init__()
        # Slot starts are padded to slot_alignment bytes so that raw-pointer (CuTe) kernels can read
        # the views; 1 packs the slots tightly.
        self.slot_alignment = int(slot_alignment)
        self.init_device = init_device
        self.group_dtypes: dict[str, torch.dtype] = {
            _BANK_DECAY: dtype,
            _BANK_NO_DECAY: dtype,
            _BANK_NO_DECAY_FP32: torch.float32,
        }
        self.group_sizes: dict[str, int] = {name: 0 for name in self.group_dtypes}
        self.sealed = False
        self._override: dict[str, torch.Tensor] = {}
        for name in self.group_dtypes:
            self.register_parameter(name, None)

    def control_group(self, dtype: torch.dtype) -> str:
        """The no-decay group that stores ``dtype`` without changing it."""

        if dtype == self.group_dtypes[_BANK_NO_DECAY]:
            return _BANK_NO_DECAY
        if dtype == torch.float32:
            return _BANK_NO_DECAY_FP32
        raise ValueError(f"an LDM parameter bank cannot store a {dtype} control parameter")

    def reserve(self, *shape: int, group: str) -> _BankSlot:
        """Reserve one parameter-shaped region of a bank group."""

        if self.sealed:
            raise RuntimeError("this LDM parameter bank has already been allocated")
        # Round the slot start up to slot_alignment bytes; the flat tensor itself is allocator-aligned.
        align = max(1, self.slot_alignment // torch.empty((), dtype=self.group_dtypes[group]).element_size())
        offset = -(-self.group_sizes[group] // align) * align
        slot = _BankSlot(self, group, offset, tuple(int(size) for size in shape))
        self.group_sizes[group] = slot.offset + slot.numel
        return slot

    def seal(self) -> ParameterBank:
        """Allocate one flat parameter for every non-empty group."""

        for name, size in self.group_sizes.items():
            if size == 0:
                continue
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.zeros(
                        size,
                        device=self.init_device,
                        dtype=self.group_dtypes[name],
                    )
                ),
            )
        self.sealed = True
        return self

    def storage(self, group: str) -> torch.Tensor:
        flat = self._override.get(group)
        if flat is not None:
            return flat
        flat = getattr(self, group)
        if flat is None:
            raise RuntimeError("an LDM parameter bank view was taken before allocation")
        return flat

    def writable(self) -> bool:
        """Whether slot views taken right now can be initialized in place.

        True for plain (unsharded) storage and inside :meth:`initializing`.
        False for a sharded bank outside that context: a slot is then a slice
        of a ``DTensor`` with no write-back, so a per-module ``reset_parameters``
        sweep must leave initialization to the enclosing container's reset.
        """

        from torch.distributed.tensor import DTensor

        for name in self.group_dtypes:
            if name in self._override:
                continue
            flat = getattr(self, name)
            if flat is not None and isinstance(flat, DTensor):
                return False
        return True

    @contextlib.contextmanager
    def initializing(self) -> Iterator[None]:
        """Run ``reset_parameters`` against writable, unsharded bank storage."""

        from torch.distributed.tensor import DTensor

        distributed: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            for name in self.group_dtypes:
                flat = getattr(self, name)
                if flat is None:
                    continue
                if isinstance(flat, DTensor):
                    # Build the whole bank locally, identically on every rank,
                    # and copy this rank's shard back: the same contract as
                    # olmo_core.nn.transformer.init for sharded weights.
                    full = torch.zeros(flat.shape, dtype=flat.dtype, device=flat.device)
                    distributed.append((flat, full))
                    self._override[name] = full
                else:
                    # In-place initialization of a view of a leaf parameter is
                    # an autograd error; detached storage aliases the same
                    # memory without it.
                    self._override[name] = flat.detach()
            yield
        finally:
            self._override.clear()
        if distributed:
            from olmo_core.distributed.utils import distribute_like, get_local_tensor

            # ``to_local()`` returns a custom-Function view, so the copy back is
            # only legal with grad mode off; do not rely on the caller for it.
            with torch.no_grad():
                for flat, full in distributed:
                    get_local_tensor(flat).copy_(get_local_tensor(distribute_like(flat, full)))

    def extra_repr(self) -> str:
        return ", ".join(
            f"{name}={size}" for name, size in self.group_sizes.items() if size
        )


class BankedLinear(nn.Module):
    """An ``nn.Linear`` whose weight and bias are views of a parameter bank.

    The weight is always reserved in ``decay``. The bias goes to ``no_decay``
    unless the caller passes ``bias_group="decay"`` for a bias that the
    training recipe decays when unbanked (the fast backends' address-network
    biases under the paper recipe), so that the bank is an optimizer-equivalent
    swap. Only the two module-dtype groups are legal: the bias must share the
    weight's dtype.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        bank: ParameterBank,
        bias_group: str = _BANK_NO_DECAY,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias_group not in (_BANK_DECAY, _BANK_NO_DECAY):
            raise ValueError(
                f"a BankedLinear bias must live in {_BANK_DECAY!r} or "
                f"{_BANK_NO_DECAY!r}, not {bias_group!r}"
            )
        self.weight_slot = bank.reserve(out_features, in_features, group=_BANK_DECAY)
        self.bias_slot = (
            bank.reserve(out_features, group=bias_group) if bias else None
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_slot.tensor()

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return None if self.bias_slot is None else self.bias_slot.tensor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias_slot is not None}"
        )


class BankedRMSNorm(nn.Module):
    """An ``nn.RMSNorm`` whose learned gain is a view of a parameter bank."""

    def __init__(self, normalized_shape: int, *, eps: float, bank: ParameterBank) -> None:
        super().__init__()
        self.normalized_shape = (int(normalized_shape),)
        self.eps = eps
        self.weight_slot = bank.reserve(int(normalized_shape), group=_BANK_NO_DECAY)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_slot.tensor()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape[0]}, eps={self.eps}"


class _BankedControlParameters(nn.Module):
    """Expose bank-backed static addresses and read gains under their ordinary names.

    ``anchor`` (a static address) and ``read_gain`` are plain parameters on the
    unbanked module, so they cannot become properties without changing the
    unbanked state dict. Resolving them through ``__getattr__`` leaves the
    unbanked path untouched.
    """

    def _bank_control(self, name: str, slot: _BankSlot) -> None:
        self.__dict__.setdefault("_bank_control_slots", {})[name] = slot

    def __getattr__(self, name: str) -> Any:
        slots = self.__dict__.get("_bank_control_slots")
        if slots is not None:
            slot = slots.get(name)
            if slot is not None:
                return slot.tensor()
        return super().__getattr__(name)


def _build_parameter_bank(
    config: Any, *, init_device: str, dtype: torch.dtype
) -> Optional[ParameterBank]:
    """One bank per container, only when the run opted into banking."""

    if not getattr(config.execution.common, "parameter_bank", False):
        return None
    return ParameterBank(init_device=init_device, dtype=dtype, slot_alignment=16)


def _bank_initializing(bank: Optional[ParameterBank]) -> Any:
    return contextlib.nullcontext() if bank is None else bank.initializing()


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    bank: Optional[ParameterBank],
    init_device: str,
    dtype: torch.dtype,
    bias_group: str = _BANK_NO_DECAY,
) -> nn.Module:
    if bank is None:
        return nn.Linear(
            in_features, out_features, bias=bias, device=init_device, dtype=dtype
        )
    return BankedLinear(
        in_features, out_features, bias=bias, bank=bank, bias_group=bias_group
    )


def _rms_norm(
    d_model: int,
    *,
    eps: float,
    bank: Optional[ParameterBank],
    init_device: str,
    dtype: torch.dtype,
) -> nn.Module:
    if bank is None:
        return nn.RMSNorm(d_model, eps=eps, device=init_device, dtype=dtype)
    return BankedRMSNorm(d_model, eps=eps, bank=bank)


class AddressProjector(nn.Module):
    """The SiLU address network of one site and its optional token gate.

    A static address still builds the network when its site has a gate; only the address head is
    then omitted.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int,
        address_dim: int,
        emit_address: bool,
        gate_dim: int,
        gate_initial_gain: float,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        bias: bool,
        init_std: float,
        bank: Optional[ParameterBank] = None,
    ) -> None:
        super().__init__()
        self.emit_address = emit_address
        self.init_std = init_std
        self.context_eps = context_eps
        self.context_norm = _rms_norm(
            d_model, eps=context_eps, bank=bank, init_device=init_device, dtype=dtype
        )
        self.w1 = _linear(
            d_model,
            hidden_dim,
            bias=bias,
            bank=bank,
            init_device=init_device,
            dtype=dtype,
        )
        self.address = (
            _linear(
                hidden_dim,
                address_dim,
                bias=bias,
                bank=bank,
                init_device=init_device,
                dtype=dtype,
            )
            if emit_address
            else None
        )
        self.gate = (
            _linear(
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
        self.gate_initial_gain = gate_initial_gain

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.context_norm.reset_parameters()
        for linear in (self.w1, self.address, self.gate):
            if linear is None:
                continue
            _truncated_normal_(linear.weight, std=self.init_std, generator=generator)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, _gate_logit(self.gate_initial_gain))

    def forward(
        self,
        source: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # FP32 normalization with a learned per-channel gain, initially one.
        context = F.rms_norm(
            source.float(), self.context_norm.normalized_shape,
            self.context_norm.weight.float(), self.context_norm.eps,
        )
        hidden = F.silu(self.w1(context.to(source.dtype)))
        address = None if self.address is None else self.address(hidden)
        gate = None if self.gate is None else identity_centered_gate(self.gate(hidden), source.dtype)
        return address, gate


class ReadParameters(_BankedControlParameters):
    """Learned parameters of one read of the depth state."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
        bank: Optional[ParameterBank] = None,
    ) -> None:
        super().__init__()
        quality = config.quality
        self.groups = quality.groups
        self.key_dim = quality.key_dim
        self.eps = config.quality.eps
        self.dynamic = quality.addresses.query.mode == "dynamic"
        self.source = quality.addresses.query.source
        self.normalize = quality.addresses.normalize_query
        if self.dynamic:
            self.anchor = None
        elif bank is None:
            self.anchor = nn.Parameter(
                torch.empty(
                    self.groups,
                    self.key_dim,
                    device=init_device,
                    dtype=dtype,
                )
            )
        else:
            self._bank_control(
                "anchor", bank.reserve(self.groups, self.key_dim, group=_BANK_DECAY)
            )
        needs_projector = self.dynamic or quality.read_gate.enabled
        self.projection = (
            AddressProjector(
                d_model,
                hidden_dim=quality.addresses.hidden_dim,
                address_dim=self.groups * self.key_dim,
                emit_address=self.dynamic,
                gate_dim=self.groups if quality.read_gate.enabled else 0,
                gate_initial_gain=quality.read_gate.initial_gain,
                context_eps=context_eps,
                init_device=init_device,
                dtype=dtype,
                bias=True,
                init_std=init_std,
                bank=bank,
            )
            if needs_projector
            else None
        )
        if not quality.learned_read_gain.enabled:
            self.read_gain = None
        elif bank is None:
            self.read_gain = nn.Parameter(
                torch.empty((self.groups,), device=init_device, dtype=torch.float32)
            )
        else:
            self._bank_control(
                "read_gain", bank.reserve(self.groups, group=bank.control_group(torch.float32))
            )
        self.init_gain = quality.learned_read_gain.initial_gain
        self.checkpoint_projection = config.execution.common.checkpoint_address_projections

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if self.anchor is not None:
            nn.init.normal_(self.anchor, mean=0.0, std=1.0, generator=generator)
        if self.projection is not None:
            self.projection.reset_parameters(generator=generator)
        if self.read_gain is not None:
            nn.init.constant_(self.read_gain, self.init_gain)

    def project(self, source: torch.Tensor) -> torch.Tensor:
        """The gated, gain-scaled read query of shape ``[..., G, K]``."""
        address_delta = gate = None
        if self.projection is not None:
            if self.checkpoint_projection and self.training and source.requires_grad:
                address_delta, gate = checkpoint_utils.checkpoint(
                    self.projection,
                    source,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                address_delta, gate = self.projection(source)
        if address_delta is None:
            if self.anchor is None:
                raise ValueError("dynamic read address projector produced no address")
            query = self.anchor.view(
                *([1] * (source.ndim - 1)), self.groups, self.key_dim
            ).expand(*source.shape[:-1], self.groups, self.key_dim)
        else:
            if self.anchor is not None:
                raise ValueError("dynamic read addresses must not have a separate anchor")
            query = address_delta.view(*source.shape[:-1], self.groups, self.key_dim)
        if self.normalize:
            query = F.normalize(query.float(), dim=-1, eps=self.eps).to(source.dtype)
        if gate is not None:
            query = query * gate.unsqueeze(-1)
        return apply_read_gain(query, self.read_gain)


class WriteParameters(_BankedControlParameters):
    """Learned parameters of one write to the depth state."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
        bank: Optional[ParameterBank] = None,
    ) -> None:
        super().__init__()
        quality = config.quality
        self.groups = quality.groups
        self.key_dim = quality.key_dim
        self.eps = config.quality.eps
        self.dynamic = quality.addresses.key.mode == "dynamic"
        self.normalize = quality.addresses.normalize_key
        if self.dynamic:
            self.anchor = None
        elif bank is None:
            self.anchor = nn.Parameter(
                torch.empty(
                    self.groups,
                    self.key_dim,
                    device=init_device,
                    dtype=dtype,
                )
            )
        else:
            self._bank_control(
                "anchor", bank.reserve(self.groups, self.key_dim, group=_BANK_DECAY)
            )
        needs_projector = self.dynamic or quality.write_gate.enabled
        self.projection = (
            AddressProjector(
                d_model,
                hidden_dim=quality.addresses.hidden_dim,
                address_dim=self.groups * self.key_dim,
                emit_address=self.dynamic,
                gate_dim=self.groups if quality.write_gate.enabled else 0,
                gate_initial_gain=quality.write_gate.initial_gain,
                context_eps=context_eps,
                init_device=init_device,
                dtype=dtype,
                bias=True,
                init_std=init_std,
                bank=bank,
            )
            if needs_projector
            else None
        )
        self.checkpoint_projection = config.execution.common.checkpoint_address_projections

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if self.anchor is not None:
            nn.init.normal_(self.anchor, mean=0.0, std=1.0, generator=generator)
        if self.projection is not None:
            self.projection.reset_parameters(generator=generator)

    def project(
        self,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        address_delta = gate = None
        if self.projection is not None:
            if self.checkpoint_projection and self.training and source.requires_grad:
                address_delta, gate = checkpoint_utils.checkpoint(
                    self.projection,
                    source,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                address_delta, gate = self.projection(source)
        if address_delta is None:
            if self.anchor is None:
                raise ValueError("dynamic write address projector produced no address")
            key = self.anchor.view(
                *([1] * (source.ndim - 1)), self.groups, self.key_dim
            ).expand(*source.shape[:-1], self.groups, self.key_dim)
        else:
            if self.anchor is not None:
                raise ValueError("dynamic write addresses must not have a separate anchor")
            key = address_delta.view(*source.shape[:-1], self.groups, self.key_dim)
        if self.normalize:
            key = F.normalize(key.float(), dim=-1, eps=self.eps).to(source.dtype)
        return key, gate


class RLSWriteSite(nn.Module):
    """One write site; its forgetting logit lives in :class:`ForgettingParameters`."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
        bank: Optional[ParameterBank] = None,
    ) -> None:
        super().__init__()
        self.write = WriteParameters(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        quality = config.quality
        self.lambda_init = quality.rls.lambda_init
        self.eps = config.quality.eps

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.write.reset_parameters(generator=generator)

    def forgetting_factor(self, lambda_logit: torch.Tensor) -> torch.Tensor:
        return forgetting_from_logit(lambda_logit, lambda_init=self.lambda_init, eps=self.eps)


class DepthSiteParameters(nn.Module):
    """The read before a branch and the write after it."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
        bank: Optional[ParameterBank] = None,
    ) -> None:
        super().__init__()
        self.read = ReadParameters(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        self.write = RLSWriteSite(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.read.reset_parameters(generator=generator)
        self.write.reset_parameters(generator=generator)


class LDMBlockParameters(nn.Module):
    """The attention and MLP sites of one transformer block."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
    ) -> None:
        super().__init__()
        bank = _build_parameter_bank(config, init_device=init_device, dtype=dtype)
        if bank is not None:
            self.parameter_bank = bank
        self.attn = DepthSiteParameters(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        self.mlp = DepthSiteParameters(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        if bank is not None:
            bank.seal()

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            self.attn.reset_parameters(generator=generator)
            self.mlp.reset_parameters(generator=generator)


class LDMInputParameters(nn.Module):
    """The embedding write, parameterized like every other write site.

    The embedding is the first branch delta (from a zero stream) and also the
    updated stream, so it is both the key source and the value. Address
    projection, static address, write gate, weighted observation and lambda
    are exactly those of :class:`RLSWriteSite`.
    """

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
    ) -> None:
        super().__init__()
        bank = _build_parameter_bank(config, init_device=init_device, dtype=dtype)
        if bank is not None:
            self.parameter_bank = bank
        self.write = RLSWriteSite(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        if bank is not None:
            bank.seal()

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            self.write.reset_parameters(generator=generator)

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.write.write.project(hidden)

    def forgetting_factor(self, lambda_logit: torch.Tensor) -> torch.Tensor:
        return self.write.forgetting_factor(lambda_logit)


class LDMOutputParameters(nn.Module):
    """Parameters of the final read before the LM head."""

    def __init__(
        self,
        d_model: int,
        config,
        *,
        context_eps: float,
        init_device: str,
        dtype: torch.dtype,
        init_std: float,
    ) -> None:
        super().__init__()
        bank = _build_parameter_bank(config, init_device=init_device, dtype=dtype)
        if bank is not None:
            self.parameter_bank = bank
        self.output = ReadParameters(
            d_model,
            config,
            context_eps=context_eps,
            init_device=init_device,
            dtype=dtype,
            init_std=init_std,
            bank=bank,
        )
        if bank is not None:
            bank.seal()

    def reset_parameters(
        self,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        with _bank_initializing(getattr(self, "parameter_bank", None)):
            self.output.reset_parameters(generator=generator)


class ForgettingParameters(nn.Module):
    """FP32 forgetting logits of every write, and the optional learned ridge of P_{-1}."""

    def __init__(
        self,
        *,
        groups: int,
        n_layers: int,
        lambda_init: float,
        learn_lambda: bool,
        init_device: str,
        key_dim: int = 0,
        learn_alpha: bool = False,
        alpha_init: float = 1.0,
        ridge_min: float = 0.25,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.n_layers = n_layers
        self.lambda_init = lambda_init
        self.learn_alpha = learn_alpha
        self.alpha_init = alpha_init
        self.ridge_min = ridge_min
        requires_grad = learn_lambda and lambda_init < 1.0
        self.root_logit = nn.Parameter(
            torch.empty(groups, device=init_device, dtype=torch.float32),
            requires_grad=requires_grad,
        )
        self.block_logits = nn.Parameter(
            torch.empty(n_layers, 2, groups, device=init_device, dtype=torch.float32),
            requires_grad=requires_grad,
        )
        # ridge = ridge_min + softplus(ridge_raw), one value per group and key dimension;
        # P_{-1} = diag(1 / ridge).
        self.ridge_raw: Optional[nn.Parameter]
        if learn_alpha:
            if key_dim <= 0:
                raise ValueError("learn_alpha requires key_dim")
            if 1.0 / alpha_init <= ridge_min:
                raise ValueError("learn_alpha requires 1/alpha_init > ridge_min")
            self.ridge_raw = nn.Parameter(
                torch.empty(groups, key_dim, device=init_device, dtype=torch.float32)
            )
        else:
            self.register_parameter("ridge_raw", None)

    def _reset_ridge(self) -> None:
        if self.ridge_raw is None:
            return
        # softplus^{-1}(1/alpha_init - ridge_min), so that the initial P_{-1} is alpha_init * I.
        excess = 1.0 / self.alpha_init - self.ridge_min
        nn.init.constant_(self.ridge_raw, math.log(math.expm1(excess)))

    def ridge(self) -> Optional[torch.Tensor]:
        """[groups, key_dim] FP32 ridge, or None when alpha is fixed."""

        if self.ridge_raw is None:
            return None
        return self.ridge_min + F.softplus(self.ridge_raw.float())

    def reset_parameters(
        self, *, generator: Optional[torch.Generator] = None
    ) -> None:
        del generator
        self._reset_ridge()
        value = forgetting_logit_init(self.lambda_init)
        nn.init.constant_(self.root_logit, value)
        nn.init.constant_(self.block_logits, value)

    def forward(self, *, with_ridge: bool = False):
        # FSDP registers hooks on module outputs; fresh zero-copy views keep those hooks off the
        # persistent parameters across iterations. The ridge is derived here, after the FSDP
        # pre-forward hook has unsharded its parameter.
        outputs = (
            self.root_logit.view_as(self.root_logit),
            self.block_logits.view_as(self.block_logits),
        )
        if with_ridge:
            return (*outputs, self.ridge())
        return outputs

    def apply_fsdp(self, dp_mesh: Optional[DeviceMesh]) -> None:
        fully_shard(
            self,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
            reshard_after_forward=False,
        )

    def apply_ddp(self, dp_mesh: Optional[DeviceMesh]) -> None:
        try:
            from torch.distributed._composable.replicate_with_fsdp import replicate
        except ImportError as exc:
            raise OLMoConfigurationError(
                "DDP mixed precision requires PyTorch's replicate_with_fsdp API "
                "to keep the forgetting logits in FP32"
            ) from exc

        replicate(
            self,
            mesh=dp_mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            ),
        )
