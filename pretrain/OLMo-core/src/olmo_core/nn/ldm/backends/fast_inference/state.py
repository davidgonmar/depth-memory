"""The state of one forward (M, the packed covariance, the window history) and the dispatch of each edge
to its kernels.

A forward of more tokens than the tuned decode classes (prefill) runs the projection kernel (residual add,
row RMS, first layer and activation of both address networks), the head/gate GEMM and the prefill edge
kernel (`kernels.edge_state`). A smaller forward (decode) runs the decode edge kernel
(`kernels.edge_decode`), which does all of it in one launch. The token count is a runtime dimension of
every compiled kernel.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .edges import InferenceEdge
from .kernels.addresses import KEY_WIDTHS, packed_width
from .kernels.projection import head_gate, hidden_rows
from .runtime import check_operands, compile_options, device_key, fake_stream, fake_tensor, on_device, spec
from .tuning import DecodeTiling, Launches, PrefillTiling

ROW_OPTIONS = ("first_window", "commit", "terminal", "norm", "norm_eps", "eps")
STATE_OPERANDS = ("stream", "value", "projected", "head_bias", "gate_bias", "static_address", "read_gain",
                  "forgetting", "covariance", "memory", "residual_history", "gain_history", "output", "norm_weight")
DECODE_OPERANDS = ("source", "delta", "value", "first_layer", "first_bias", "second_layer", "head_bias",
                   "gate_bias", "static_address", "read_gain", "forgetting", "covariance", "memory",
                   "residual_history", "gain_history", "output", "norm_weight")
FLOAT32_OPERANDS = frozenset({"projected", "read_gain", "forgetting", "gain_history"})


def _shapes(span, n, g, k, d, h):
    """The operand shapes of an edge of `n` tokens (None for the runtime token extent)."""
    history = max(span - 1, 1)
    rows = (n, g * d)
    return {"stream": rows, "source": rows, "delta": rows, "value": rows, "output": rows,
            "projected": (2, n, g * (k + 1)), "first_layer": (g * d // 128, 2 * h, 128), "first_bias": (2 * h,),
            "second_layer": (2, g, k + 1, h + 8), "head_bias": (2, g, k), "gate_bias": (2, g),
            "static_address": (2, g, k), "read_gain": (g,), "forgetting": (g,), "covariance": (n, g, packed_width(k)),
            "memory": (n, g, k, d), "residual_history": (history, n, g, d), "gain_history": (history, n, g, k),
            "norm_weight": (g * d,)}


@lru_cache(maxsize=64)
def _specs(names: tuple[str, ...], sizes: tuple[int, ...]) -> tuple:
    """The operand specs of one kernel signature and model shape, the token extent left to run time."""
    span, g, k, d, h = sizes
    shapes = _shapes(span, None, g, k, d, h)
    return tuple(spec(torch.float32 if name in FLOAT32_OPERANDS else torch.bfloat16, shapes[name])
                 for name in names)


@lru_cache(maxsize=256)
def edge_kernel(options: tuple, sizes: tuple[int, ...], tiling: PrefillTiling | DecodeTiling, device_index: int):
    """The edge kernel of one option set, sizes (span, groups, key, value and hidden width) and tiling. A
    `PrefillTiling` gives the prefill kernel and a `DecodeTiling` the decode kernel. A tiling that does not
    fit the device's shared memory raises."""
    from .kernels.edge_decode import EdgeDecode
    from .kernels.edge_state import EdgeState

    settings = dict(options)
    span, g, k, d, h = sizes
    rows = {name: settings[name] for name in ROW_OPTIONS}
    if isinstance(tiling, DecodeTiling):
        kernel = EdgeDecode(span, g, k, d, h, tiling, has_delta=settings["has_delta"],
                            context_eps=settings["context_eps"], **rows)
    else:
        kernel = EdgeState(span, g, k, d, tiling, history=settings["history"], **rows)
    budget = torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    if kernel.shared_bytes > budget:
        raise ValueError(f"{tiling} for groups={g}, value width {d}, hidden width {h} needs {kernel.shared_bytes} "
                         f"bytes of shared memory; the device allows {budget}")
    return kernel


@lru_cache(maxsize=256)
def _program(options: tuple, sizes: tuple[int, ...], tiling, metadata: tuple, capability: tuple[int, int],
             device_index: int):
    """`edge_kernel` compiled. The token count is not part of the key."""
    import cutlass
    from cutlass import cute

    scalars = (cutlass.Float32(1.0), cutlass.Int32(0)) if isinstance(tiling, DecodeTiling) else (cutlass.Int32(0),)
    return cute.compile(edge_kernel(options, sizes, tiling, device_index), *(fake_tensor(item) for item in metadata),
                        *scalars, fake_stream(), options=compile_options(capability))


def _run(names, options, sizes, tiling, tensors, scalars):
    """Launch the edge kernel. `step` has checked the operands that come from outside the state, and the
    state allocates the others."""
    device = tensors["output"].device
    with on_device(device):
        program = _program(options, sizes, tiling, _specs(names, sizes), *device_key(device))
        program(*(tensors[name] for name in names), *scalars)


class InferenceState:
    """The state of one forward over a fixed batch of tokens, created fresh for every model forward.
    `launches` holds the tuned tilings of the model's shape on this GPU (`tuning.launches`)."""

    def __init__(self, *, total_edges: int, span: int, groups: int, key_dim: int, value_dim: int,
                 hidden_dim: int, covariance_scale: float, eps: float, context_eps: float, launches: Launches):
        if any(type(x) is not int or x <= 0 for x in (total_edges, span, groups, key_dim, value_dim, hidden_dim)):
            raise ValueError("edge count, span and sizes must be positive integers")
        if key_dim not in KEY_WIDTHS or hidden_dim not in (32, 64) or value_dim % 64 or groups * value_dim % 128:
            raise ValueError(f"fast-inference runs key widths {KEY_WIDTHS}, hidden width 32 or 64, a value width "
                             "that is a multiple of 64 and a model width that is a multiple of 128")
        if not math.isfinite(covariance_scale) or covariance_scale <= 0:
            raise ValueError("the initial covariance scale must be finite and positive")
        self.total_edges, self.span = total_edges, min(span, total_edges)
        self.groups, self.key_dim, self.value_dim, self.hidden_dim = groups, key_dim, value_dim, hidden_dim
        self.model_dim = groups * value_dim
        self.covariance_scale = float(covariance_scale)
        self.eps, self.context_eps = float(eps), float(context_eps)
        self.launches = launches
        self.edge = 0
        self.tokens = 0

    def begin(self, hidden: torch.Tensor) -> None:
        """Allocate the state for the token batch of `hidden` ([..., d_model] CUDA BF16)."""
        if not hidden.is_cuda or hidden.dtype != torch.bfloat16 or hidden.shape[-1] != self.model_dim:
            raise ValueError("fast-inference runs on CUDA BF16 streams of width groups * value_dim")
        n, g, k, d, e = hidden.numel() // self.model_dim, self.groups, self.key_dim, self.value_dim, self.span
        if n <= 0:
            raise ValueError("a nonempty token batch is required")
        self.tokens = n
        # A commit folds the window's updates into M, so the history holds at most span - 1 of them.
        history = max(e - 1, 1)
        parts = {"memory": (torch.bfloat16, (n, g, k, d)), "covariance": (torch.bfloat16, (n, g, packed_width(k))),
                 "residual_history": (torch.bfloat16, (history, n, g, d)),
                 "gain_history": (torch.float32, (history, n, g, k))}
        # One allocation holds the whole state, so the allocator gets it back as one block after the forward.
        # Separate allocations of these sizes are carved out of cached blocks of other sizes (the logits of an
        # earlier, smaller batch, for instance) and keep them split, and a large batch after a small one could
        # then find no block for its logits. Every part is a multiple of 16 bytes, so every view stays aligned.
        sizes = [math.prod(shape) * dtype.itemsize for dtype, shape in parts.values()]
        storage = torch.empty(sum(sizes), dtype=torch.uint8, device=hidden.device)
        self.memory, self.covariance, self.residual_history, self.gain_history = (
            part.view(dtype).view(shape) for part, (dtype, shape) in zip(storage.split(sizes), parts.values()))
        # The first commit writes M before any edge reads it, and each history row before an edge reads it.
        self.covariance.copy_(_initial_covariance(k, self.covariance_scale, hidden.device).expand_as(self.covariance))
        self.edge = 0

    def step(self, edge: InferenceEdge, source: torch.Tensor, value: torch.Tensor, norm: torch.nn.Module | None,
             *, delta: torch.Tensor | None = None, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Write `value` at the pending site and read at the next. `source` is the stream, or the stream
        before its branch residual when `delta` is given: the step then adds ``alpha * delta`` into `source`
        in place. Returns the stream the read was taken on (`source`) and the branch input
        ``norm(stream + read)``, or ``stream + read`` when `norm` is None (the final read)."""
        if self.edge >= self.total_edges:
            raise RuntimeError("every planned edge of this segment has been executed")
        if not math.isfinite(alpha):
            raise ValueError("a finite residual scale is required")
        n, d, shape = self.tokens, self.model_dim, source.shape
        source = source.reshape(n, d)
        delta = None if delta is None else delta.reshape(n, d)
        value = value.reshape(n, d)
        for tensor in (source, value) if delta is None else (source, delta, value):
            if tensor.dtype != torch.bfloat16 or tensor.device != self.memory.device:
                raise ValueError("the stream, the branch delta and the written value must be BF16 on the state's "
                                 "device")
            check_operands(tensor)
        if ((edge.groups, edge.key_dim, edge.hidden) != (self.groups, self.key_dim, self.hidden_dim)
                or edge.first_layer.shape[0] * 128 != d or edge.first_layer.device != source.device):
            raise ValueError("the edge's sizes and device must be the state's")
        start = self.edge // self.span * self.span
        end = min(start + self.span, self.total_edges)
        span = end - start
        history = max(span - 1, 1)
        terminal = self.edge == self.total_edges - 1
        # The final read has no norm, so its unused weight operand is a view of the stream, not a new tensor.
        norm_weight = source[0] if norm is None else norm.weight.detach().to(torch.bfloat16)
        if norm_weight.shape != (d,):
            raise ValueError("the norm must have a weight of the model width")
        check_operands(norm_weight)
        options = (("first_window", start == 0), ("commit", self.edge == end - 1 and not terminal),
                   ("terminal", terminal), ("eps", self.eps), ("norm", norm is not None),
                   ("norm_eps", 1e-6 if norm is None else float(norm.eps)))
        tensors = {"head_bias": edge.head_bias, "gate_bias": edge.gate_bias, "static_address": edge.static_address,
                   "read_gain": edge.read_gain, "forgetting": edge.forgetting, "covariance": self.covariance,
                   "value": value, "memory": self.memory, "residual_history": self.residual_history[:history],
                   "gain_history": self.gain_history[:history], "norm_weight": norm_weight,
                   "output": torch.empty_like(source)}
        sizes = (span, self.groups, self.key_dim, self.value_dim, self.hidden_dim)
        index = self.edge - start
        tiling = self.launches.decode_tiling(n)
        if tiling is None:
            hidden = hidden_rows(source, delta, alpha, edge, context_eps=self.context_eps)
            tensors.update(stream=source, projected=head_gate(hidden, edge))
            # The prefill kernel keeps only the history rows this edge reads, so more CTAs fit an SM.
            _run(STATE_OPERANDS, (*options, ("history", index)), sizes, self.launches.prefill, tensors, (index,))
        else:
            tensors.update(source=source, delta=source if delta is None else delta, first_layer=edge.first_layer,
                           first_bias=edge.first_bias, second_layer=edge.second_layer)
            _run(DECODE_OPERANDS, (*options, ("has_delta", delta is not None), ("context_eps", self.context_eps)),
                 sizes, tiling, tensors, (float(alpha), index))
        self.edge += 1
        return source.view(shape), tensors["output"].view(shape)


@lru_cache(maxsize=16)
def _initial_covariance(k: int, scale: float, device: torch.device) -> torch.Tensor:
    """The packed ``scale * I`` of one group, built once per device. Every forward clones it on the GPU, so
    a forward makes no host-to-device copy and can be captured in a CUDA graph."""
    packed = torch.zeros((packed_width(k),), dtype=torch.bfloat16)
    for i in range(k):
        # The diagonal slots of the upper triangle, stored row by row.
        packed[i * k - i * (i - 1) // 2] = scale
    return packed.to(device)
