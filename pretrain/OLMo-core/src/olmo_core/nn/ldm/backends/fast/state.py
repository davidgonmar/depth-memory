"""The windowed state of the fast backend: one forward kernel launch per edge, and the window
kernels in the backward.

Only the state step runs in these autograd nodes; attention, the MLPs and the address networks
stay outside. The caller carries the returned dependency token from one step to the next; the
state never stores it or any other tensor of the graph.

M, dM, the residuals and the DY and DV histories are stored in BF16, the covariance as a packed
triangle (BF16 by default, FP32 on request) that is saved in BF16 before every write. Keys, queries,
gains, the covariance gradient, the future value and gain gradients and every reduction are FP32;
the raw value gradient is computed from the unrounded FP32 vi. DESIGN.md lists the rounding points.
"""

from __future__ import annotations

import gc
import math
from functools import lru_cache

import torch

from . import schedules


@lru_cache(maxsize=256)
def _program(kernel: str, options: tuple, sizes: tuple[int, ...], metadata: tuple,
             capability: tuple[int, int], device_index: int, schedule: tuple[int, ...],
             edge: int | None):
    """The compiled ``kernel`` ("forward", "interior" or "window_end") for one option set, window
    size, operand layout and launch schedule on this device. Every kernel takes its tensors, then
    the window-local edge (except the window end), then the stream. Only compiled code and
    metadata are cached, never tensors."""
    # CUDA module handles are context-local, so the device is part of the key; its SM count
    # and shared-memory budget only size the persistent grids.
    properties = torch.cuda.get_device_properties(device_index)
    del device_index
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    types = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    fake = [None if item is None else make_fake_tensor(
        types[item[0]], item[1], item[2], assumed_align=item[3]) for item in metadata]
    if kernel != "window_end":
        fake.append(cutlass.Int32(0))            # the window-local edge
    fake.append(make_fake_stream(use_tvm_ffi_env_stream=False))
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a"}.get(capability, f"sm_{capability[0]}{capability[1]}")
    operation = schedules.build_operation(kernel, sizes, options, properties, schedule)
    if edge is not None:
        # The window-local edge is compiled in; the kernel still takes it as an argument.
        operation.static_edge = edge
    return cute.compile(operation, *fake, options=f"--gpu-arch={arch}")


def _metadata(tensors: list[torch.Tensor | None]):
    """(device, per-operand dtype, shape, strides and 16-byte alignment) of one launch."""
    device = None
    metadata = []
    for tensor in tensors:
        if tensor is None:
            metadata.append(None)
            continue
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError("kernel launch tensors must share one device")
        # Alignment depends on the address (of views in particular), so it is part of every
        # launch's metadata and of the compiled program's key.
        metadata.append((tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()),
                         math.gcd(tensor.data_ptr(), 16)))
    if device is None:
        raise ValueError("a kernel launch requires at least one tensor")
    return device, tuple(metadata)


def _run(kernel, sizes, tensors, metadata, edge, options, schedule):
    """Launch ``kernel`` with an explicit schedule on the current stream."""
    device_index = torch.cuda.current_device()
    program = _program(kernel, options, sizes, metadata, torch.cuda.get_device_capability(device_index),
                       device_index, schedule, edge)
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    # Views can keep requires_grad inside Function.forward, so detach what we hand to the kernel.
    # The autograd graph itself is untouched.
    arguments = [None if tensor is None else from_dlpack(tensor.detach(), assumed_align=item[3])
                 for tensor, item in zip(tensors, metadata, strict=True)]
    if edge is not None:
        arguments.append(cutlass.Int32(edge))
    arguments.append(cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    program(*arguments)


def _launch(kernel: str, sizes: tuple[int, ...], tensors: list[torch.Tensor | None], edge: int | None = None,
            *, options: tuple = (), training: bool):
    """One kernel launch with the schedule its GPU's table names for this shape."""
    device, metadata = _metadata(tensors)
    with torch.cuda.device(device):
        # Compiling a kernel lets the CuTe DSL keep the calling Python frames alive (its
        # decorators store inspect.currentframe()), and with them the launch's tensors. Only the
        # cyclic collector frees them, and the trainer runs it rarely. So after a compile, collect
        # once at the next launch, when those frames are gone.
        misses = _program.cache_info().misses
        if misses != _launch_state["misses"]:
            gc.collect()
            _launch_state["misses"] = misses
        stream = torch.cuda.current_stream(device)
        # Autograd does not know the state's buffers, which are freed when the graph completes, so
        # record the stream that uses them (callers may run nodes on different streams).
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)
        schedule = schedules.schedule(kernel, sizes, options, edge, device, training=training)
        _run(kernel, sizes, tensors, metadata, edge, options, schedule)


_launch_state = {"misses": 0}

# The dtypes the forward may keep the running covariance in.
RUNNING_COVARIANCE_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` if it is contiguous and 16-byte aligned, else a contiguous copy."""
    if tensor.is_contiguous() and tensor.data_ptr() % 16 == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


class SegmentState:
    """The tapes of one forward and the buffers of its backward.

    Values have shape [*tokens, G, D]; keys and queries may broadcast to [*tokens, G, K], scales
    and forgetting factors to [*tokens, G]. Every forward needs a fresh instance, told its number
    of edges so that a short last window commits at its last edge. The initial state is M = 0 and
    P = covariance_scale * I. Only the reads are differentiable outputs, not the end state.

    ``running_covariance_dtype`` is the dtype of the P that the forward carries from edge to edge,
    "bfloat16" or "float32". A BF16 P is rounded at every edge, and those rounding errors
    accumulate along the edges; in FP32 they stay at FP32 level. Each edge's saved copy of P, which
    the backward reads, is BF16 either way.

    The backward runs one window-end kernel per window, which also handles the window's last
    edge, and one interior kernel per other edge.

    The kernels move BF16 rows in 16-byte vectors, so they need a group width that is a multiple
    of 8. A narrower width is padded here, the way FlashAttention pads head dimensions: values and
    read gradients get zero columns up to the padded width, every D-wide buffer is allocated at
    that width, and the reads and value gradients are sliced back to D. Zero columns stay zero
    through the recurrence and add exact zeros to the backward's sums over D, so they change
    nothing. A width that is already a multiple of 8 is used as it is.
    """

    def __init__(self, *, total_edges: int, span: int, groups: int, key_dim: int,
                 value_dim: int, covariance_scale: float = 1.0,
                 running_covariance_dtype: str = "bfloat16"):
        if any(type(x) is not int or x <= 0 for x in (total_edges, span, groups, key_dim, value_dim)):
            raise ValueError("edge count, span and state sizes must be positive integers")
        if key_dim not in (4, 8, 16):
            raise ValueError("M keys must have dimension 4, 8 or 16")
        if not math.isfinite(covariance_scale) or covariance_scale <= 0:
            raise ValueError("the initial covariance scale must be finite and positive")
        if running_covariance_dtype not in RUNNING_COVARIANCE_DTYPES:
            raise ValueError(f"the running covariance dtype must be one of {sorted(RUNNING_COVARIANCE_DTYPES)}")
        self.total_edges, self.span = total_edges, min(span, total_edges)
        self.groups, self.key_dim, self.value_dim = groups, key_dim, value_dim
        # The group width the kernels run at (see the class docstring).
        self.padded_value_dim = -(-value_dim // 8) * 8
        self.covariance_scale = float(covariance_scale)
        self.running_covariance_dtype = running_covariance_dtype
        # A state built with gradients enabled runs training launches, whose shapes must be tuned.
        self.training = torch.is_grad_enabled()
        self.forward_count = 0
        self._initialized = False
        self._graph_task: int | None = None
        self._retained_end_state: tuple[torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def new_token(like: torch.Tensor) -> torch.Tensor:
        return torch.empty(0, device=like.device, dtype=torch.bfloat16,
                           requires_grad=torch.is_grad_enabled())

    def step(self, token: torch.Tensor, raw_key: torch.Tensor, query: torch.Tensor,
             raw_value: torch.Tensor, scale: torch.Tensor, forgetting: torch.Tensor):
        """One write and read of this state through the ``ldm::state_step`` custom op (traceable by
        Dynamo); returns (grouped read, fresh tensor dependency) and never retains ``token``."""
        from olmo_core.nn.ldm import state_step_op

        if not torch.compiler.is_compiling():   # a compiled caller sets the active state before tracing
            state_step_op.set_active_state(self)
        read, token_out, _edge = torch.ops.ldm.state_step(token, raw_key, query, raw_value, scale, forgetting)
        return read, token_out

    def _initialize(self, value: torch.Tensor):
        if value.ndim < 2 or tuple(value.shape[-2:]) != (self.groups, self.value_dim):
            raise ValueError("values must end in [groups, value_dim]")
        self.token_shape = tuple(value.shape[:-2])
        self.n = math.prod(self.token_shape)
        if self.n <= 0:
            raise ValueError("a nonempty token batch is required")
        self.device = value.device
        n, g, k, d, length = self.n, self.groups, self.key_dim, self.padded_value_dim, self.total_edges

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        # Nothing reads M before the first window's commit (first_window kernels).
        self.m = empty((n, g, k, d), torch.bfloat16)
        # P: the packed upper triangle (K(K+1)/2 entries per group, upper-row-major) in the
        # running covariance dtype, padded to 16 bytes; the lower triangle mirrors the upper one.
        # The diagonal slots are indexed on the device: writing them one by one from the host
        # would be a blocking copy each, stalling the stream at the start of every forward.
        dtype = RUNNING_COVARIANCE_DTYPES[self.running_covariance_dtype]
        per_16_bytes = 16 // dtype.itemsize
        triangle = k * (k + 1) // 2
        packed = torch.zeros((-(-triangle // per_16_bytes) * per_16_bytes,), dtype=dtype, device=self.device)
        row = torch.arange(k, device=self.device)
        packed.index_fill_(0, row * k - row * (row - 1) // 2, self.covariance_scale)
        self.p = packed.expand(n, g, packed.numel()).clone()
        self.keys, self.queries, self.gains = [empty((length, n, g, k)) for _ in range(3)]
        self.residuals = empty((length, n, g, d), torch.bfloat16)
        self.denominators, self.forgetting_tape, self.scales = [empty((length, n, g)) for _ in range(3)]
        # The pre-write P of every edge, unpadded rows (the forward writes it).
        self.saved_covariance_tape = empty((length, n, g, k * (k + 1) // 2), torch.bfloat16)
        # Reverse buffers are allocated lazily: inference does not need them.
        self._reverse_buffers = False
        self._initialized = True

    def _padded(self, rows: torch.Tensor) -> torch.Tensor:
        """[N, G, D] rows at the kernels' width with zero columns appended, or ``rows`` itself when
        D needs no padding."""
        padding = self.padded_value_dim - self.value_dim
        return torch.nn.functional.pad(rows, (0, padding)) if padding else rows

    def _unpadded(self, rows: torch.Tensor) -> torch.Tensor:
        """The first D columns of rows at the kernels' width, or ``rows`` itself when D needs no
        padding."""
        return rows[..., :self.value_dim] if self.padded_value_dim != self.value_dim else rows

    def _forward_coordinates(self, raw_key, query, raw_value, scale, forgetting):
        if not self._initialized:
            self._initialize(raw_value)
        if self._graph_task is not None or self.forward_count >= self.total_edges:
            raise RuntimeError("create a fresh SegmentState for each complete forward")
        if tuple(raw_value.shape) != (*self.token_shape, self.groups, self.value_dim):
            raise ValueError("the token shape must remain unchanged within one forward")
        for name, tensor in (("raw_key", raw_key), ("query", query), ("scale", scale), ("forgetting", forgetting)):
            if tensor.device != self.device or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be FP32 on the value device")
        if raw_value.device != self.device or raw_value.dtype != torch.bfloat16:
            raise ValueError("raw values must remain CUDA BF16")
        edge = self.forward_count
        start = edge // self.span * self.span
        end = min(start + self.span, self.total_edges)
        return edge, start, end

    def _forward_operands(self, raw_key, query, scale, forgetting):
        """The per-edge inputs in the forward kernel's layouts: raw keys and queries [N,G,K] and
        scales [N,G], contiguous and 16-byte aligned; forgetting [N,G] likewise, or its broadcast
        over tokens (strides (0, 1)) when it is per group. The kernel forms key = raw_key * scale
        and writes the key, query, scale and forgetting tapes."""
        vector_shape = (*self.token_shape, self.groups, self.key_dim)
        scalar_shape = (*self.token_shape, self.groups)
        raw_key = _aligned(torch.broadcast_to(raw_key, vector_shape).reshape(self.n, self.groups, self.key_dim))
        query = _aligned(torch.broadcast_to(query, vector_shape).reshape(self.n, self.groups, self.key_dim))
        scale = _aligned(torch.broadcast_to(scale, scalar_shape).reshape(self.n, self.groups))
        forgetting = torch.broadcast_to(forgetting, scalar_shape).reshape(self.n, self.groups)
        if forgetting.stride() != (0, 1):
            forgetting = _aligned(forgetting)
        return raw_key, query, scale, forgetting

    def _forward(self, raw_key, query, raw_value, scale, forgetting):
        edge, start, end = self._forward_coordinates(raw_key, query, raw_value, scale, forgetting)
        raw_key, query, scale, forgetting = self._forward_operands(raw_key, query, scale, forgetting)
        value = self._padded(raw_value.contiguous().view(self.n, self.groups, self.value_dim))
        read = torch.empty_like(value)
        saved = self.saved_covariance_tape[start:end]
        tapes = [self.keys[edge], self.queries[edge], self.scales[edge], self.forgetting_tape[edge]]
        sizes = (end - start, self.n, self.groups, self.key_dim, self.padded_value_dim)
        options = (("first_window", start == 0), ("running_covariance_dtype", self.running_covariance_dtype))
        _launch("forward", sizes,
                [self.m, self.p, raw_key, query, value, scale, forgetting, self.gains[start:end],
                 self.residuals[start:end], self.denominators[start:end], read, saved, *tapes],
                edge - start, options=options, training=self.training)
        self.forward_count += 1
        return self._unpadded(read).reshape(raw_value.shape), edge

    def _begin_backward(self):
        if self.forward_count != self.total_edges:
            raise RuntimeError("complete the planned forward before requesting its backward")
        task = torch._C._current_graph_task_id()
        if self._graph_task == task:
            return
        if self._graph_task is not None:
            if self._retained_end_state is None:
                raise RuntimeError("M buffers were consumed; repeated backward requires retain_graph=True")
            self.m.copy_(self._retained_end_state[0])
            self.p.copy_(self._retained_end_state[1])
        # Copies of the end state are kept only when the caller retains the graph for another
        # backward.
        keep_graph = torch._C._autograd._get_current_graph_task_keep_graph()
        if keep_graph and self._retained_end_state is None:
            self._retained_end_state = (self.m.clone(), self.p.clone())
        if not keep_graph:
            # The state is not a saved tensor, so a live loss would keep every tape after the
            # backward. Free the buffers when the graph task completes; edge zero is not a safe
            # point because autograd.grad may skip nodes.
            torch.autograd.Variable._execution_engine.queue_callback(self._release_buffers)
        self._graph_task = task
        self._expected_edge = self.total_edges - 1
        self._active_start = None
        self._future_start = self._future_end = self.total_edges
        self._allocate_reverse()
        self.dm.zero_()
        self.covariance_grad.zero_()

    def _allocate_reverse(self):
        if self._reverse_buffers:
            return
        n, g, k, d, span = self.n, self.groups, self.key_dim, self.padded_value_dim, self.span

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        self.dm = empty((n, g, k, d), torch.bfloat16)       # state gradient G at the window end
        self.covariance_grad = empty((n, g, k * (k + 1) // 2))
        self.future_value_grad = empty((span, n, g, d))                 # S_i = G g_i
        self.dy, self.dv = [empty((span, n, g, d), torch.bfloat16) for _ in range(2)]
        self.future_gain_grad = empty((span, n, g, k))                 # z_i = G r_i
        self._reverse_buffers = True

    def _reverse(self, edge: int, read_gradient: torch.Tensor | None):
        start = edge // self.span * self.span
        end = min(start + self.span, self.total_edges)
        count = end - start
        local = edge - start
        saved = self.saved_covariance_tape[start:end]
        n, g, k, d = self.n, self.groups, self.key_dim, self.padded_value_dim
        # The kernels write one edge of their query and forgetting gradient operands; with a zero
        # edge stride that edge is this edge's own output, handed to autograd.
        query_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        forgetting_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        query_output = query_grad.expand(count, n, g, k)
        forgetting_output = forgetting_grad.expand(count, n, g)
        raw_value_grad = torch.empty((n, g, d), device=self.device, dtype=torch.bfloat16)
        raw_key_grad = torch.empty((n, g, k), device=self.device, dtype=torch.float32)
        scale_grad = torch.empty((n, g), device=self.device, dtype=torch.float32)
        # This edge's read gradient. Both kernels stage it directly and write it into the DY
        # history slot of the edge, where the next window end reads it as a future row.
        current_dy = (torch.zeros_like(raw_value_grad) if read_gradient is None else
                      _aligned(self._padded(read_gradient.reshape(n, g, self.value_dim))))
        if self._active_start != start:
            if edge != end - 1:
                raise RuntimeError("a reverse window must begin at its last causal edge")
            fs, fe = self._future_start, self._future_end
            future = fe - fs
            if future == 0:
                fs, fe = start, start + 1                   # F=0 placeholders the window end never reads
            # DY/DV still hold the completed future window; the kernel consumes each group's
            # future rows before overwriting them with this window's.
            _launch("window_end", (count, future, n, g, k, d),
                    [self.m, self.dm, self.keys[start:end], self.queries[start:end],
                     self.gains[start:end], self.residuals[start:end], self.forgetting_tape[start:end],
                     self.denominators[start:end], self.queries[fs:fe], self.keys[fs:fe],
                     self.dy[:max(1, future)], self.dv[:max(1, future)], self.m, self.dm,
                     self.future_value_grad[:count], self.future_gain_grad[:count], current_dy,
                     self.scales[edge], self.dy[:count], self.dv[:count], query_output,
                     self.covariance_grad, forgetting_output,
                     raw_value_grad, raw_key_grad, scale_grad, saved],
                    options=(("first_window", start == 0),), training=self.training)
            self._active_start = start
            self._future_start, self._future_end = start, end
        else:
            _launch("interior", (count, n, g, k, d),
                    [self.m, self.residuals[start:end], self.dy[:count], self.keys[start:end],
                     self.gains[start:end], self.queries[start:end], self.future_value_grad[:count],
                     self.future_gain_grad[:count],
                     self.dv[:count], query_output, self.covariance_grad,
                     self.forgetting_tape[start:end], self.denominators[start:end], forgetting_output,
                     self.scales[edge], raw_value_grad, raw_key_grad, scale_grad, saved, current_dy], local,
                    training=self.training)
        self._expected_edge -= 1
        return raw_key_grad, query_grad, self._unpadded(raw_value_grad), scale_grad, forgetting_grad

    def _backward(self, edge, read_gradient):
        if torch.is_grad_enabled():
            raise NotImplementedError("this CuTe VJP does not implement higher-order derivatives")
        self._begin_backward()
        # A loss may ignore the last reads. Their edges are reversed with zero read gradients
        # before the first edge that has one.
        while self._expected_edge > edge:
            self._reverse(self._expected_edge, None)
        if self._expected_edge != edge:
            raise RuntimeError("M read nodes were revisited or executed out of reverse causal order")
        return self._reverse(edge, read_gradient)

    def retained_bytes(self) -> dict[str, int]:
        """Bytes of every distinct storage the state holds (not the model's peak memory)."""
        tensors = {name: tensor for name, tensor in vars(self).items()
                   if isinstance(tensor, torch.Tensor)}
        if self._retained_end_state is not None:
            tensors.update(zip(("retained_end_state_m", "retained_end_state_p"),
                               self._retained_end_state, strict=True))
        result, seen = {}, set()
        for name, tensor in tensors.items():
            storage = tensor.untyped_storage()
            if storage._cdata not in seen:
                seen.add(storage._cdata)
                result[name] = storage.nbytes()
        return result

    def _release_buffers(self):
        for name, value in tuple(vars(self).items()):
            if isinstance(value, torch.Tensor):
                delattr(self, name)
        self._retained_end_state = None

