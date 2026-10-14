from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from .ldm.components import _BANK_DECAY, _BANK_NO_DECAY, ParameterBank

__all__ = [
    "MHC",
    "MHCParameterBank",
    "MHCGroup",
    "HyperConnectionStreamExpand",
    "MHCLearnedStreamReduce",
    "prepare_inference",
]


class MHCParameterBank(ParameterBank):
    """FP32 bank for every mHC controller and the learned head: one ``decay_<owner>`` and one
    ``no_decay_<owner>`` flat parameter per transformer block (``owner`` = block index) plus the
    head, all inside ONE FSDP unit owned by the model.

    Mirrors the LDM ``ParameterBank``. Why one FSDP unit for all: per-controller units fire FSDP
    hooks inside every compiled block, which breaks the graph and adds host launches.
    Why one parameter pair per block rather than one for the model: a connector slices its
    parameters out of its block's tensor at offsets that are identical for every block, so
    Dynamo compiles one block graph for all blocks (a model-wide flat tensor gives each block
    different offsets, one recompile per block, and the recompile limit falls back to eager).
    The bank is unsharded once per step by calling it at the start of the model forward and
    stays unsharded until the end of backward.
    """

    def __init__(self, *, init_device: str) -> None:
        super().__init__(init_device=init_device, dtype=torch.float32)
        # Drop the base bank's fixed groups; groups are added per owner.
        for name in list(self.group_dtypes):
            del self.group_dtypes[name], self.group_sizes[name]
            self._parameters.pop(name, None)

    def owner_group(self, base: str, owner: str) -> str:
        name = f"{base}_{owner}"
        if name not in self.group_dtypes:
            self.group_dtypes[name] = torch.float32
            self.group_sizes[name] = 0
            self.register_parameter(name, None)
        return name

    def decay_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith(_BANK_DECAY + "_")]

    def no_decay_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith(_BANK_NO_DECAY + "_")]

    def forward(self) -> None:
        """No computation. Calling the module lets FSDP2 / replicate unshard the bank."""
        return None


class _Banked:
    """Mixin: move ``_BANK_SLOTS`` parameters into an :class:`MHCParameterBank`."""

    _BANK_SLOTS: tuple[tuple[str, str], ...] = ()

    def bank_into(self, bank: "MHCParameterBank", owner: str,
                  names: Optional[tuple[str, ...]] = None) -> None:
        """Move parameters into the ``owner``'s bank slots. ``names`` restricts the call to a
        subset so the model can order reservations (the TileLang kernels need 16-byte-aligned
        tensors: reserve the large ones, whose sizes are multiples of 64 elements, first)."""
        slots = dict(self.__dict__.get("_bank_slots") or {})
        for name, group in self._BANK_SLOTS:
            if names is not None and name not in names:
                continue
            if name in slots:
                raise RuntimeError(f"{type(self).__name__}.{name} is already banked")
            parameter = getattr(self, name)
            slots[name] = bank.reserve(*parameter.shape, group=bank.owner_group(group, owner))
            delattr(self, name)
        self.__dict__["_bank_slots"] = slots

    @property
    def banked(self) -> bool:
        return bool(self.__dict__.get("_bank_slots"))

    def _bank_writable(self) -> bool:
        slots = self.__dict__.get("_bank_slots")
        return not slots or next(iter(slots.values())).bank.writable()

    def refresh_bank_views(self) -> None:
        """Called by the model once per forward, after the bank is unsharded: hold the bank's
        group tensors as plain attributes so a compiled block sees ordinary tensor inputs
        (Dynamo cannot trace the FSDP-swapped parameter of a module outside the block)."""
        slots = self.__dict__.get("_bank_slots")
        if slots:
            # Keyed by the base group ("decay" / "no_decay"), never by the per-block group name:
            # Dynamo guards on dict keys, and a per-block key would recompile the block graph
            # once per block (the same reason the slot offsets are identical across blocks).
            self.__dict__["_bank_bases"] = {
                base: slots[name].bank.storage(slots[name].group)
                for name, base in self._BANK_SLOTS if name in slots
            }

    def release_bank_views(self) -> None:
        """Drop the per-forward bank tensors; outside a forward the slots resolve through the
        bank again (a sharded DTensor slice under FSDP, like LDM's bank slots)."""
        self.__dict__.pop("_bank_bases", None)

    def __getattr__(self, name: str):
        slots = self.__dict__.get("_bank_slots")
        if slots and name in slots:
            slot = slots[name]
            bases = self.__dict__.get("_bank_bases")
            if bases is not None:
                base = dict(self._BANK_SLOTS)[name]
                return bases[base][slot.offset : slot.offset + slot.numel].view(slot.shape)
            return slot.tensor()
        return super().__getattr__(name)  # type: ignore[misc]


def _reshape_to_streams(x: torch.Tensor, num_streams: int) -> torch.Tensor:
    """[..., n * C] carrier -> [..., n, C] view (no copy)."""
    if x.ndim < 2:
        raise ValueError(f"Expected rank >= 2 tensor, got shape {tuple(x.shape)}")
    if x.shape[-1] % num_streams != 0:
        raise ValueError(
            f"Model dimension {x.shape[-1]} is not divisible by num_streams={num_streams}"
        )
    return x.view(*x.shape[:-1], num_streams, x.shape[-1] // num_streams)


def _flatten_from_streams(x: torch.Tensor) -> torch.Tensor:
    """[..., n, C] streams -> [..., n * C] carrier (a view when contiguous)."""
    if x.ndim < 3:
        raise ValueError(f"Expected stream tensor rank >= 3, got shape {tuple(x.shape)}")
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


def _validate_mhc_geometry(num_streams: int, dim: int) -> None:
    if num_streams != 4:
        raise ValueError("TileKernels mHC requires exactly 4 residual streams")
    if dim <= 0 or dim % 64:
        raise ValueError("TileKernels mHC requires a positive per-stream width divisible by 64")


class HyperConnectionStreamExpand(nn.Module):
    """Repeat one residual stream into the configured number of streams. The residual
    carrier between blocks is [..., n * C] (the n streams contiguous in the last axis), so
    every connector reads and writes its [..., n, C] streams as views without copies."""

    def __init__(self, num_streams: int):
        super().__init__()
        self.num_streams = num_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_streams == 1:
            return x
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.num_streams, x.shape[-1]).reshape(
            *x.shape[:-1], self.num_streams * x.shape[-1]
        )


class MHCLearnedStreamReduce(_Banked, nn.Module):
    """DeepSeek-V4 learned contraction before the LM-head norm (``hc_head``), computed
    by ``mhc_ops.mhc_head`` (upstream's ``functional.mhc_head`` composition): independent
    sigmoid gates per stream from the RMS-normalized concatenated streams, not a softmax.
    All head parameters are FP32 (the kernel rounds the weight to TF32 for the
    projection, like the block controllers); the output keeps the stream dtype.
    Initialization: Normal(.02) weight, zero bias, scale .01.
    """

    _BANK_SLOTS = (("weight", _BANK_DECAY), ("bias", _BANK_NO_DECAY), ("scale", _BANK_NO_DECAY),
                   ("norm_weight", _BANK_NO_DECAY))

    def __init__(self, num_streams: int, dim: int, *, eps: float, init_device: str,
                 upstream_projection_backward: bool = False):
        super().__init__()
        _validate_mhc_geometry(num_streams, dim)
        self.num_streams = num_streams
        self.dim = dim
        self.norm_eps = eps
        self.hc_eps = 1e-6
        self.upstream_projection_backward = bool(upstream_projection_backward)
        self.norm_weight = nn.Parameter(torch.ones(num_streams * dim,
                                                   device=init_device, dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(num_streams, num_streams * dim,
                                              device=init_device, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(num_streams, device=init_device,
                                            dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(1, device=init_device, dtype=torch.float32))
        self.reset_parameters()
        _warm_tile_kernels()

    @torch.no_grad()
    def reset_parameters(self, *, generator: Optional[torch.Generator] = None) -> None:
        if not self._bank_writable():
            return  # sharded bank: the model resets inside ``bank.initializing()``
        self.weight.normal_(mean=0.0, std=0.02, generator=generator)
        self.norm_weight.fill_(1.0)
        self.bias.zero_()
        self.scale.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p.dtype != torch.float32 for p in (self.weight, self.bias, self.scale, self.norm_weight)):
            raise TypeError("The learned mHC head requires FP32 working parameters")
        if x.shape[-1] != self.num_streams * self.dim:
            raise ValueError("The learned mHC head received an incorrect stream width")
        if x.dtype != torch.bfloat16:
            raise TypeError("TileKernels mHC requires BF16 residual streams")
        kernels = _require_tile_kernels()
        streams = _reshape_to_streams(x.contiguous(), self.num_streams)
        # The final residual has no paired post-op to supply its gradient buffer.
        out = kernels.mhc_head(streams, self.weight, self.scale, self.bias,
                               norm_weight=self.norm_weight,
                               norm_eps=self.norm_eps, mhc_mult=self.num_streams,
                               pre_eps=self.hc_eps, fuse_grad_acc=False,
                               upstream_bwd_mul=self.upstream_projection_backward)
        return out.to(x.dtype)


# mHC recipe (Appendix A of the mHC paper): FP32 controller
# parameters (the projection uses TF32 multiplication with FP32 accumulation inside the
# kernels), BF16 streams, ``t_max`` Sinkhorn-Knopp iterations, and these constants.
MHC_PRE_EPS = 1e-6
MHC_SINKHORN_EPS = 1e-6
MHC_POST_MULT = 2.0


_TILE_KERNELS: Optional[Any] = None


def _require_tile_kernels():
    """DeepSeek's official mHC kernels (TileKernels, TileLang), section 4.3.1 of the mHC paper.

    Imported on first use: the vendored tree needs TileLang and a CUDA toolkit.

    Resolved once and cached: every connector asks for the kernels inside its
    forward, and returning the same namespace object each call keeps Dynamo's
    guards valid.
    """
    if _TILE_KERNELS is not None:
        return _TILE_KERNELS
    return _load_tile_kernels()


def _warm_tile_kernels() -> None:
    """Resolve the kernel namespace at module construction on CUDA machines, so the first
    compiled forward traces the cached fast path (the import path is not traceable)."""
    if _TILE_KERNELS is None and torch.cuda.is_available():
        try:
            _require_tile_kernels()
        except ImportError:
            pass  # forward raises the informative error


@torch.compiler.disable
def _load_tile_kernels():
    global _TILE_KERNELS
    try:
        from types import SimpleNamespace

        from . import mhc_ops
    except ImportError as e:
        raise ImportError(
            "mHC requires DeepSeek TileKernels (`olmo_core.kernels.tile_kernels`) with "
            "TileLang and a CUDA toolkit (nvcc) on PATH"
        ) from e
    # Training runs on the traceable custom ops (``mhc_ops``: upstream's compositions as single
    # ops, so a block compiles without graph breaks; the ``*_recompute`` variants and
    # ``mhc_branch_input`` implement the paper's L_r-group recompute on upstream's replay
    # kernel). Inference runs upstream's fused no-grad ``mhc_pre`` as one op as well.
    _TILE_KERNELS = SimpleNamespace(
        mhc_pre_inference=mhc_ops.mhc_pre_inference, mhc_pre_train=mhc_ops.mhc_pre, mhc_post=mhc_ops.mhc_post,
        mhc_head=mhc_ops.mhc_head, mhc_pre_recompute=mhc_ops.mhc_pre_recompute,
        mhc_post_recompute=mhc_ops.mhc_post_recompute, mhc_branch_input=mhc_ops.mhc_branch_input,
    )
    return _TILE_KERNELS


class MHCController(_Banked, nn.Module):
    """FP32 routing parameters in the TileKernels layout with the ICML mHC version's Appendix A.6
    initialization: ``fn`` [n^2 + 2n, n * dim] (rows: pre (n), post (n), residual (n x n)),
    ``b`` [n^2 + 2n], ``alpha`` [3] = (pre, post, residual) gating factors.

    ``forward`` runs the paper's fused mapping kernels (RMSNorm folded after the
    matmul, TF32 projection, sigmoid/Sinkhorn mixes) and the F_pre application.

    Inside a model the parameters are slots of the model's :class:`MHCParameterBank`
    (``bank_into``); standalone they are ordinary parameters.
    """

    _BANK_SLOTS = (("fn", _BANK_DECAY), ("norm_weight", _BANK_NO_DECAY), ("b", _BANK_NO_DECAY),
                   ("alpha", _BANK_NO_DECAY))

    def __init__(
        self,
        *,
        num_residual_streams: int,
        dim: int,
        gating_factor_init: float,
        init_device: str,
        layer_index: int = 0,
        norm_eps: float = 1e-6,
        upstream_projection_backward: bool = False,
        upstream_inference_projection: bool = False,
    ):
        super().__init__()
        self.gating_factor_init = float(gating_factor_init)
        self.norm_eps = norm_eps
        # True: upstream's exact `_mhc_pre_norm_fn_bwd_mul` kernel for the projection backward.
        # False (default): the same two matmuls as TF32 GEMMs compiled by Inductor (see
        # mhc_ops._project_backward).
        self.upstream_projection_backward = bool(upstream_projection_backward)
        # No-grad forwards: one projection split at every token count (upstream), or None for the split
        # count tuned per token count on this GPU type (modeling/mhc/ops/pre_big_fuse.py).
        self.inference_projection_splits = 1 if upstream_inference_projection else None
        # Set by fold_norm_weight: fn holds fn * norm_weight rounded to TF32, and norm_weight is 1.
        self.norm_weight_folded = False
        self.num_residual_streams = num_residual_streams
        self.selected_stream = layer_index % num_residual_streams
        n = num_residual_streams
        self.fn = nn.Parameter(
            torch.empty(n * n + 2 * n, n * dim, device=init_device, dtype=torch.float32)
        )
        self.norm_weight = nn.Parameter(torch.ones(n * dim, device=init_device, dtype=torch.float32))
        self.b = nn.Parameter(torch.empty(n * n + 2 * n, device=init_device, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.empty(3, device=init_device, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameters(self, *, generator: Optional[torch.Generator] = None) -> None:
        if not self._bank_writable():
            return  # sharded bank: the model resets inside ``bank.initializing()``
        self.fn.zero_()
        self.norm_weight.fill_(1.0)
        self.b.zero_()
        n = self.num_residual_streams
        self.b[:n].fill_(-3.0)
        self.b[self.selected_stream] = 3.0
        residual_bias = self.b[2 * n:].view(n, n)
        residual_bias.fill_(-3.0)
        residual_bias.diagonal().fill_(3.0)
        self.alpha.fill_(self.gating_factor_init)

    @torch.no_grad()
    def fold_norm_weight(self) -> None:
        """Multiply the RMS weight into the projection and round the product to TF32 once, with the
        kernel and rounding upstream's no-grad forward applies at every call
        (``_mhc_fn_normw_merge_fwd``, ``round_to_tf32``), so no-grad forwards read the prepared
        projection instead. Afterwards the controller is inference only: a forward with gradients
        raises."""
        if self.norm_weight_folded:
            raise RuntimeError("the mHC controller's norm weight is already folded into its projection")
        if not self._bank_writable():
            raise RuntimeError("fold the mHC norm weight on an unsharded model (after loading, before inference)")
        from olmo_core.kernels.tile_kernels.mhc.norm_fn_kernel import _mhc_fn_normw_merge_fwd, round_to_tf32

        fn, norm_weight = self.fn, self.norm_weight
        merged = torch.empty_like(fn)
        _mhc_fn_normw_merge_fwd(*fn.shape)(fn.contiguous(), norm_weight.contiguous(), merged)
        fn.copy_(round_to_tf32(merged))
        norm_weight.fill_(1.0)
        self.norm_weight_folded = True

    def forward(self, streams: torch.Tensor, tmax: int, group: "Optional[MHCGroup]" = None):
        # With parameter_bank=False, FSDP/replicate hooks materialize the FP32 parameters when this
        # module is entered through __call__; banked controllers read the bank's views.
        from torch.distributed.tensor import DTensor

        if isinstance(self.fn, DTensor):
            raise RuntimeError(
                "mHC controller parameters were not materialized before forward "
                "(a banked controller needs the model to call its MHCParameterBank first)"
            )
        kernels = _require_tile_kernels()
        n = self.num_residual_streams
        if not torch.is_grad_enabled():
            layer_input, post_mix, comb = kernels.mhc_pre_inference(
                streams, self.fn, self.alpha, self.b,
                norm_weight=None if self.norm_weight_folded else self.norm_weight,
                norm_eps=self.norm_eps, mhc_mult=n, post_mult_value=MHC_POST_MULT,
                pre_eps=MHC_PRE_EPS, sinkhorn_eps=MHC_SINKHORN_EPS, sinkhorn_repeat=tmax,
                n_splits=self.inference_projection_splits,
            )
            return layer_input, None, post_mix, comb
        if self.norm_weight_folded:
            raise RuntimeError("this mHC model was prepared for inference (norm weights folded); "
                               "train from the unfolded checkpoint")
        # norm_fn -> split_mixes -> sinkhorn -> apply_mix, upstream's training composition, as
        # one traceable op (see ``mhc_ops``). Inside a recompute group the same op saves the
        # group's entry residual instead of this layer's streams.
        common = dict(
            norm_weight=self.norm_weight, norm_eps=self.norm_eps, mhc_mult=n,
            post_mult_value=MHC_POST_MULT, pre_eps=MHC_PRE_EPS,
            sinkhorn_eps=MHC_SINKHORN_EPS, sinkhorn_repeat=tmax,
            upstream_bwd_mul=self.upstream_projection_backward,
        )
        if group is not None:
            return kernels.mhc_pre_recompute(streams, self.fn, self.alpha, self.b, group=group, **common)
        return kernels.mhc_pre_train(streams, self.fn, self.alpha, self.b, **common)


class MHCGroup:
    """One recompute group: mHC paper section 4.3.2 / Table 3.

    ``L_r`` consecutive connectors share one entry residual ``x_{l0}``. The group holds that
    tensor and, as the layers run, their mixes and branch outputs (values, no graphs). Every
    connector and branch inside the group hands these to the recompute-aware ops in
    ``mhc_ops``, whose backward rebuilds ``x_l``, ``H^pre x_l`` and ``RMSNorm(H^pre x_l)`` from
    them with upstream's ``mhc_multilayer_recompute`` (one replay per group) instead of storing
    them. No hooks and no eager regions: a block that receives a group compiles as one graph
    (one graph variant per position in the group, since the list lengths differ).

    The model creates one group per ``recompute_layers // 2`` blocks and passes it to the blocks
    as ``mhc_group``; a block forwards it to both connectors, a connector to its branch.
    """

    def __init__(self, entry: torch.Tensor, num_streams: int, n_layers: int):
        self.x0 = _reshape_to_streams(entry.detach().contiguous(), num_streams)
        self.n_layers = int(n_layers)
        self.pre_mixes: list[torch.Tensor] = []
        self.post_mixes: list[torch.Tensor] = []
        self.combs: list[torch.Tensor] = []
        self.outputs: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.outputs)


class MHC(nn.Module):
    """Canonical manifold-constrained Hyper-Connection on DeepSeek's official TileKernels
    implementation (mHC paper section 4.3): fused mapping kernels, F_pre / F_{post,res}
    application kernels, TF32 controller projection with FP32 parameters, BF16 streams,
    and (given an ``MHCGroup``) the L_r-group recomputation of the residual streams.
    """

    def __init__(
        self,
        *,
        num_residual_streams: int,
        dim: int,
        branch: Optional[nn.Module] = None,
        gating_factor_init: float = 0.01,
        sinkhorn_iters: int = 20,
        controller_norm_eps: float = 1e-6,
        init_device: str = "cpu",
        layer_index: int = 0,
        recompute: bool = False,
        upstream_projection_backward: bool = False,
        upstream_inference_projection: bool = False,
    ):
        super().__init__()
        _validate_mhc_geometry(num_residual_streams, dim)
        self.branch = branch
        self.recompute = bool(recompute)
        self.num_residual_streams = num_residual_streams
        self.dim = dim
        self.tmax = int(sinkhorn_iters)
        self.dropout = nn.Identity()
        self.controller = MHCController(
            num_residual_streams=num_residual_streams,
            dim=dim,
            gating_factor_init=gating_factor_init,
            init_device=init_device,
            layer_index=layer_index,
            norm_eps=controller_norm_eps,
            upstream_projection_backward=upstream_projection_backward,
            upstream_inference_projection=upstream_inference_projection,
        )
        self.reset_parameters()
        _warm_tile_kernels()

    @torch.no_grad()
    def reset_parameters(self, *, generator: Optional[torch.Generator] = None) -> None:
        self.controller.reset_parameters(generator=generator)

    def _branch_param_dtype(self, fallback: torch.dtype) -> torch.dtype:
        if self.branch is None:
            return fallback
        for param in self.branch.parameters(recurse=True):
            return param.dtype
        return fallback

    def forward(self, residuals: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if residuals.dtype != torch.bfloat16:
            raise TypeError("TileKernels mHC requires BF16 residual streams")
        streams = _reshape_to_streams(residuals.contiguous(), self.num_residual_streams)
        if streams.shape[-1] != self.dim:
            raise ValueError(
                "Expected mHC streams ending in "
                f"[{self.num_residual_streams}, {self.dim}], got {tuple(streams.shape)}"
            )
        group: Optional[MHCGroup] = kwargs.pop("mhc_group", None)
        if not (self.recompute and torch.is_grad_enabled()):
            group = None
        kernels = _require_tile_kernels()
        branch_input, pre_mix, post_mix, comb = self.controller(streams, self.tmax, group)
        if group is not None:
            group.pre_mixes.append(pre_mix.detach())
            if getattr(self.branch, "mhc_group_aware", False):
                kwargs["mhc_group"] = group
        branch_dtype = self._branch_param_dtype(branch_input.dtype)
        if branch_input.dtype != branch_dtype:
            branch_input = branch_input.to(dtype=branch_dtype)
        if self.branch is None:
            branch_output = branch_input
        else:
            branch_output = self.branch(branch_input, *args, **kwargs)
        (branch_output, *rest), tree_spec = tree_flatten(branch_output)
        branch_output = branch_output.to(dtype=streams.dtype).contiguous()
        if group is not None:
            output = kernels.mhc_post_recompute(branch_output, streams, post_mix, comb, group=group)
            group.post_mixes.append(post_mix.detach())
            group.combs.append(comb.detach())
            group.outputs.append(branch_output.detach())
        else:
            output = kernels.mhc_post(branch_output, streams, post_mix, comb)
        output = self.dropout(_flatten_from_streams(output).to(dtype=residuals.dtype))
        return tree_unflatten((output, *rest), tree_spec)


def prepare_inference(model: nn.Module) -> None:
    """Turn a loaded mHC model into its inference form, in place, once: every controller's RMS
    weight is folded into its projection (``MHCController.fold_norm_weight``). Call it after the
    checkpoint is loaded and before the first no-grad forward; the model is inference only
    afterwards. The learned head keeps its per-call merge (one site per forward)."""
    controllers = [module for module in model.modules() if isinstance(module, MHCController)]
    if not controllers:
        raise ValueError("prepare_inference needs a model with mHC controllers")
    for controller in controllers:
        controller.fold_norm_weight()
