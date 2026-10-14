"""The address-network kernels: the forward's RMS, bias and SiLU epilogue, the backward's prepare
kernel and the input-gradient GEMM (QuACK, with a radial epilogue).

input_grad(x, w, delta, r, s) takes BF16 x [N, D], w [P, D], delta [N, P] and FP32 r [N], s [N],
where delta is the BF16 SiLU cotangent and s = sum_p FP32(delta) * u is the row's radial scalar,
both written by the prepare kernel. It computes dx = BF16(r * (delta @ w) - x * (r^3 / D) * s)
with BF16 GEMM operands, FP32 accumulation and the FP32 epilogue
    coefficient = RN(RN(RN(r * r) * r) * RN(1 / D)),  dx = BF16(RN(RN(r * acc) - RN(RN(x * coefficient) * s))).
"""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import BFloat16, Float32, Int64, cute
from cutlass.utils import SmemAllocator
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.ops import ColVecLoad
from quack.gemm_config import GemmConfig
from quack.rounding import RoundingMode

from .window_end import _sigmoid


LOG = logging.getLogger(__name__)


@gemm_epilogue(mode="element", ops={"r": ColVecLoad("r"), "s": ColVecLoad("s")})
def dx_epilogue(acc, c, r, s, width):
    coefficient = r * r * r * (1.0 / width)
    return {"D": r * acc - c * coefficient * s}


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...] = ()


def config_candidates(capability):
    """The two GEMM schedules of an architecture: 128 or 256 columns per tile."""
    return _config_candidates(tuple(capability))


@lru_cache(maxsize=8)
def _config_candidates(capability):
    # QuACK caches launches by the identity of the config object, not by its value. Returning the
    # same objects every time keeps each call off its slow host path.
    if tuple(capability) not in ((9, 0), (10, 0)):
        raise ValueError("address dX supports only SM90/SM100")
    sm = capability[0]
    return tuple(GemmConfig(tile_m=128, tile_n=columns, cluster_m=1, cluster_n=1,
                            pingpong=sm == 9 and columns == 128, is_dynamic_persistent=False,
                            device_capacity=sm, split_k=1, swap_ab=False)
                 for columns in (128, 256))


def _validate_input_grad(x, w, delta, r, s, out):
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("x/w must have shapes [N,D]/[P,D]")
    n, d = x.shape
    p = w.shape[0]
    if min(n, d, p) <= 0 or w.shape != (p, d) or delta.shape != (n, p):
        raise ValueError("inconsistent or empty N/D/P sizes")
    if r.shape != (n,) or s.shape != (n,):
        raise ValueError("r/s must have shape [N]")
    if any(t.dtype != torch.bfloat16 for t in (x, w, delta)):
        raise TypeError("x/w/delta must be BF16; master-weight casts stay outside")
    if r.dtype != torch.float32 or s.dtype != torch.float32:
        raise TypeError("r/s must be FP32")
    if out is not None:
        if out.shape != x.shape or out.dtype != torch.bfloat16:
            raise ValueError("out must be BF16 [N,D]")
        a, b = out.stride()
        divisor = math.gcd(a, b)
        if (a <= 0 or b <= 0 or
                (b // divisor < n and a // divisor < d)):
            raise ValueError("out must not have overlapping elements")
        if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in (x, w, delta, r, s)):
            raise ValueError("out must not share input storage")
    tensors = (x, w, delta, r, s) + (() if out is None else (out,))
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs/output must share a device")
    return n, d, p


def _tma_reasons(name, tensor):
    reasons = []
    if tensor.data_ptr() % 16:
        reasons.append(f"{name}: base address is not 16-byte aligned")
    strides = tensor.stride()
    major = 1 if strides[1] == 1 else 0 if strides[0] == 1 else None
    if major is None:
        reasons.append(f"{name}: neither matrix axis is contiguous")
    else:
        other = 1 - major
        if tensor.shape[major] % 8:
            reasons.append(f"{name}: contiguous BF16 extent is not divisible by 8")
        if strides[other] < tensor.shape[major] or (2 * strides[other]) % 16:
            reasons.append(f"{name}: outer stride is overlapping or not 16-byte aligned")
    return reasons


def eligibility(x, w, delta, r, s, *, out=None, capability=None):
    """Whether the GEMM can take these operands, from their metadata alone. N need not be a
    multiple of the tile."""
    _, d, _ = _validate_input_grad(x, w, delta, r, s, out)
    if capability is None:
        capability = torch.cuda.get_device_capability(x.device)
    reasons = []
    if tuple(capability) not in ((9, 0), (10, 0)):
        reasons.append("only SM90/SM100 are supported")
    for name, tensor in (("x", x), ("w", w), ("delta", delta)):
        reasons.extend(_tma_reasons(name, tensor))
    if out is not None:
        reasons.extend(_tma_reasons("out", out))
    elif d % 8:
        reasons.append("out: allocated contiguous BF16 extent is not divisible by 8")
    for name, tensor in (("r", r), ("s", s)):
        if tensor.stride() != (1,) or tensor.data_ptr() % 16:
            reasons.append(f"{name}: requires a contiguous 16-byte-aligned FP32 vector")
    return Eligibility(not reasons, tuple(reasons))


def input_grad(x, w, delta, r, s, *, out=None, config=None):
    """dx into ``out`` (a new BF16 [N, D] tensor when absent), outside autograd. ``config`` is one
    of config_candidates(capability), the first by default. Layouts the GEMM cannot take raise
    ValueError."""
    _validate_input_grad(x, w, delta, r, s, out)
    with torch.cuda.device(x.device):
        capability = torch.cuda.get_device_capability(x.device)
        candidates = config_candidates(capability)
        if config is None:
            config = candidates[0]
        if config not in candidates:
            raise ValueError("config must be one of config_candidates(capability)")
        supported = eligibility(x, w, delta, r, s, out=out, capability=capability)
        if not supported.eligible:
            raise ValueError("address dX layout not supported: " + "; ".join(supported.reasons))
        LOG.debug("LDM address dX shape=%s/%s config=%s", x.shape, w.shape, config)
        if out is None:
            out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        stream = torch.cuda.current_stream(x.device)
        for tensor in (x, w, delta, r, s, out):
            tensor.record_stream(stream)
        dx_epilogue(delta, w, C=x, out={"D": out}, r=r, s=s,
                    width=float(x.shape[1]), config=config, tuned=False,
                    rounding_mode=RoundingMode.RN)
    return out


class AddressEpilogue:
    """One CTA per row: the row's FP32 sum of squares, then the epilogue over the P columns."""

    def __init__(self, n, d, p, eps, has_bias, warps):
        if any(type(v) is not int or v <= 0 for v in (n, d, p, warps)):
            raise ValueError("positive dimensions and warp count required")
        if warps > 32:
            raise ValueError("warp count exceeds CUDA block limit")
        if not math.isfinite(eps) or eps <= 0 or type(has_bias) is not bool:
            raise ValueError("positive finite eps and boolean bias policy required")
        self.n, self.d, self.p, self.eps = n, d, p, float(eps)
        self.warps, self.has_bias = warps, has_bias

    @cute.jit
    def __call__(self, x: cute.Tensor, u: cute.Tensor, bias: cute.Tensor,
                 a: cute.Tensor, r: cute.Tensor, stream: cuda.CUstream):
        assert tuple(x.shape) == (self.n, self.d) and x.element_type == BFloat16
        assert tuple(u.shape) == (self.n, self.p) and u.element_type == Float32
        assert tuple(a.shape) == (self.n, self.p) and a.element_type == BFloat16
        assert tuple(r.shape) == (self.n,) and r.element_type == Float32
        if cutlass.const_expr(self.has_bias):
            assert tuple(bias.shape) == (self.p,) and bias.element_type == BFloat16
        self.kernel(x, u, bias, a, r).launch(
            grid=(self.n, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Tensor, u: cute.Tensor, bias: cute.Tensor,
               a: cute.Tensor, r: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        row = Int64(cute.arch.block_idx()[0])
        threads = self.warps * 32
        width = 1
        if cutlass.const_expr(x.stride[1] == 1):
            width = math.gcd(self.d, x.stride[0], min(8, x.iterator.max_alignment // 2))
        values = cute.make_rmem_tensor(width, BFloat16)
        total = Float32(0.0)
        for part in cutlass.range_constexpr(cute.ceil_div(self.d, threads * width)):
            column = (part * threads + tid) * width
            values.fill(0.0)
            if column < self.d:
                if cutlass.const_expr(width > 1):
                    offset = cute.assume(row * x.stride[0] + column, divby=width)
                    source = cute.make_tensor(x.iterator + offset, cute.make_layout(width))
                    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                                               num_bits_per_copy=width * 16)
                    cute.copy(atom, source, values)
                else:
                    values[0] = x[row, column]
            for i in cutlass.range_constexpr(width):
                value = Float32(values[i])
                total += value * value
        for bit in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << bit)
        partials = SmemAllocator().allocate_tensor(Float32, cute.make_layout(self.warps), byte_alignment=16)
        if tid % 32 == 0:
            partials[tid // 32] = total
        cute.arch.sync_threads()
        total = Float32(0.0)
        if tid % 32 < self.warps:
            total = partials[tid % 32]
        for bit in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, 1 << bit)
        inverse = cute.rsqrt(total / Float32(self.d) + Float32(self.eps))
        if tid == 0:
            r[row] = inverse
        for column in cutlass.range(tid, self.p, threads):
            z = (u[row, column] * inverse)
            if cutlass.const_expr(self.has_bias):
                z = (z + Float32(bias[column]))
            z = Float32(BFloat16(z))
            sigmoid = Float32(1.0) / (Float32(1.0) + cute.math.exp(-z))
            a[row, column] = BFloat16(z * sigmoid)


@lru_cache(maxsize=128)
def _compiled_post_scale(shape, eps, has_bias, warps, metadata, device_context):
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
    capability = device_context[-1]
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a"}.get(capability,
                                                           f"sm_{capability[0]}{capability[1]}")
    types = {torch.bfloat16: BFloat16, torch.float32: Float32}
    tensors = [make_fake_tensor(types[dtype], shape, stride, assumed_align=alignment)
               for shape, stride, dtype, alignment in metadata]
    return cute.compile(AddressEpilogue(*shape, eps, has_bias, warps),
                        *tensors, make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch={arch} --enable-tvm-ffi")


def post_scale(x, u, bias=None, *, eps=1e-6, warps=4):
    """BF16 hidden rows ``a[N, P]`` and FP32 row scales ``r[N]`` from the source ``x[N, D]`` and the
    FP32 GEMM output ``u[N, P]``."""
    if x.ndim != 2 or u.ndim != 2 or u.shape[0] != x.shape[0]:
        raise ValueError("x[N,D] and u[N,P] required")
    n, d, p = *x.shape, u.shape[1]
    AddressEpilogue(n, d, p, eps, bias is not None, warps)
    if x.dtype != torch.bfloat16 or u.dtype != torch.float32:
        raise TypeError("BF16 x and FP32 u required")
    if bias is not None and (bias.dtype != torch.bfloat16 or bias.shape != (p,)):
        raise ValueError("the optional bias is BF16 [P]")
    with torch.cuda.device(x.device):
        a = torch.empty((n, p), device=x.device, dtype=torch.bfloat16)
        r = torch.empty((n,), device=x.device, dtype=torch.float32)
        tensors = (x, u, x if bias is None else bias, a, r)
        error, context = cuda.cuCtxGetCurrent()
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxGetCurrent failed: {error}")
        metadata = tuple((tuple(t.shape), tuple(t.stride()), t.dtype, math.gcd(t.data_ptr(), 16))
                         for t in tensors)
        program = _compiled_post_scale((n, d, p), float(eps), bias is not None, warps, metadata,
                                      (x.device.index, int(context), torch.cuda.get_device_capability(x.device)))
        stream = torch.cuda.current_stream(x.device)
        for tensor in tensors:
            tensor.record_stream(stream)
        program(*tensors)
    return a, r


def _ceildiv(a, b):
    return (a + b - 1) // b


@cute.jit
def _warp_sum(value: Float32):
    for bit in cutlass.range_constexpr(5):
        value = value + cute.arch.shuffle_sync_bfly(value, 1 << bit)
    return value


@cute.jit
def _delta_value(bias: cute.Tensor, u: cute.Tensor, r: cute.Tensor, da: cute.Tensor,
                 row: Int64, col: Int64, has_bias: cutlass.Constexpr):
    """delta[row, col] = BF16(da * silu'(z)) with z = BF16(r*u + b)."""
    value = r[row] * u[row, col]
    if cutlass.const_expr(has_bias):
        value = value + Float32(bias[col])
    z = Float32(BFloat16(value))
    sigmoid = _sigmoid(z)
    derivative = sigmoid * (Float32(1.0) + z * (Float32(1.0) - sigmoid))
    return BFloat16(Float32(da[row, col]) * derivative)


class AddressNetworkBackward:
    """The kernel of one shape: ``prepare`` writes delta, s and BF16(r*delta) with one warp per
    row."""

    def __init__(self, n, d, p, has_bias, warps=4):
        if type(has_bias) is not bool:
            raise TypeError("has_bias must be a compile-time bool")
        self.n, self.d, self.p = n, d, p
        self.has_bias = has_bias
        self.warps, self.threads = warps, 32 * warps

    @cute.kernel
    def prepare_kernel(self, bias: cute.Tensor, u: cute.Tensor, r: cute.Tensor,
                       da: cute.Tensor, delta: cute.Tensor, s: cute.Tensor,
                       scaled_delta: cute.Tensor):
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        row = Int64(cute.arch.block_idx()[0]) * self.warps + tid // 32
        if row < self.n:
            total = Float32(0.0)
            for col in cutlass.range(lane, self.p, 32):
                grad = _delta_value(bias, u, r, da, row, Int64(col), self.has_bias)
                delta[row, col] = grad
                # The dW GEMM takes BF16(r * delta), rounded here.
                scaled_delta[row, col] = BFloat16(r[row] * Float32(grad))
                total = total + Float32(grad) * u[row, col]
            total = _warp_sum(total)
            if lane == 0:
                s[row] = total

    @cute.jit
    def prepare(self, bias: cute.Tensor, u: cute.Tensor, r: cute.Tensor,
                da: cute.Tensor, delta: cute.Tensor, s: cute.Tensor,
                scaled_delta: cute.Tensor, stream):
        for tensor, shape, dtype in (
            (u, (self.n, self.p), Float32), (r, (self.n,), Float32),
            (da, (self.n, self.p), BFloat16), (delta, (self.n, self.p), BFloat16),
            (s, (self.n,), Float32), (scaled_delta, (self.n, self.p), BFloat16),
        ):
            assert tuple(tensor.shape) == shape and tensor.element_type == dtype
        if cutlass.const_expr(self.has_bias):
            assert tuple(bias.shape) == (self.p,) and bias.element_type == BFloat16
        self.prepare_kernel(bias, u, r, da, delta, s, scaled_delta).launch(
            grid=(_ceildiv(self.n, self.warps), 1, 1), block=(self.threads, 1, 1), stream=stream)


def _validate_backward(da, x, w, u, r, bias):
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("x and w must have shape [N,D] and [P,D]")
    n, d = x.shape
    p = w.shape[0]
    if min(n, d, p) <= 0 or w.shape != (p, d):
        raise ValueError("inconsistent or empty N/D/P sizes")
    if da.shape != (n, p) or u.shape != (n, p) or r.shape != (n,):
        raise ValueError("da/u must have shape [N,P] and r must have shape [N]")
    if bias is not None and bias.shape != (p,):
        raise ValueError("bias must have shape [P]")
    bf = (x, w, da, *((bias,) if bias is not None else ()))
    if any(t.dtype != torch.bfloat16 for t in bf):
        raise TypeError("x/w/da/bias must be BF16; cast master parameters outside")
    if u.dtype != torch.float32 or r.dtype != torch.float32:
        raise TypeError("saved u/r must be FP32")
    if any(t.device != x.device for t in (*bf, u, r)):
        raise ValueError("all tensors must share a device")
    return n, d, p


@lru_cache(maxsize=128)
def _compiled_prepare(sizes, has_bias, warps, metadata, capability, device, context):
    # The device and context only key the cache: a compiled module belongs to one context.
    del device, context
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    types = {torch.bfloat16: BFloat16, torch.float32: Float32}
    args = [make_fake_tensor(types[dtype], shape, stride, assumed_align=alignment)
            for dtype, shape, stride, alignment in metadata]
    operation = AddressNetworkBackward(*sizes, has_bias, warps)
    arch = {(9, 0): "sm_90a", (10, 0): "sm_100a"}.get(
        capability, f"sm_{capability[0]}{capability[1]}")
    return cute.compile(operation.prepare, *args, make_fake_stream(use_tvm_ffi_env_stream=True),
                        options=f"--gpu-arch={arch} --enable-tvm-ffi")


def backward(da, x, w, u, r, bias=None, *, input_grad, config=None, warps=4):
    """(dx, dw, db) of the address network, outside autograd; db is None without a bias.

    The prepare kernel writes delta, s and BF16(r * delta). dW is one cuBLAS GEMM on BF16
    operands with an FP32 output, rounded once to BF16; a BF16-output GEMM would depend on the
    global reduced-precision flags. dB is the same kind of GEMM of delta against a column of
    ones, much faster than a column reduction. ``input_grad(x, w, delta, r, s, out=dx,
    config=config)`` writes dx. The scratch is two BF16 [N, P] tensors, an FP32 [N] vector and
    the FP32 [P, D] GEMM output.
    """
    sizes = _validate_backward(da, x, w, u, r, bias)
    if not callable(input_grad):
        raise TypeError("input_grad must be callable")
    n, _, p = sizes

    with torch.no_grad(), torch.cuda.device(x.device), torch.autocast("cuda", enabled=False):
        delta = torch.empty((n, p), dtype=torch.bfloat16, device=x.device)
        scaled_delta = torch.empty_like(delta)
        s = torch.empty((n,), dtype=torch.float32, device=x.device)
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        tensors = (da if bias is None else bias, u, r, da, delta, s, scaled_delta)
        error, context = cuda.cuCtxGetCurrent()
        if error != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxGetCurrent failed: {error}")
        metadata = tuple((t.dtype, tuple(t.shape), tuple(t.stride()), math.gcd(t.data_ptr(), 16))
                         for t in tensors)
        program = _compiled_prepare(sizes, bias is not None, warps, metadata,
                                    torch.cuda.get_device_capability(x.device),
                                    x.device.index, int(context))
        stream = torch.cuda.current_stream(x.device)
        for tensor in (*tensors, x, w, dx):
            tensor.record_stream(stream)
        program(*tensors)
        del tensors
        dw = torch.mm(scaled_delta.T, x, out_dtype=torch.float32).bfloat16()
        del scaled_delta
        db = None
        if bias is not None:
            ones = torch.ones((n, 1), dtype=torch.bfloat16, device=x.device)
            db = torch.mm(delta.T, ones, out_dtype=torch.float32).squeeze(1).bfloat16()
        result = input_grad(x, w, delta, r, s, out=dx, config=config)
        if result is not dx:
            raise ValueError("input_grad must write and return the provided BF16 out tensor")
    return dx, dw, db
