"""The address networks' first layer of one edge in prefill, fused with the stream's residual add.

One kernel per edge adds the pending branch to the stream and rounds it to BF16 (``source + alpha *
delta``, the stream the backbone continues with). It takes the row RMS of that BF16 stream, runs the
first layer of both address networks (``first_layer``, context gains folded in) with FP32 accumulation
and writes the BF16 hidden rows ``SiLU(BF16(u * rms + bias))`` ``[2, N, H]``. The head and gate
projections of both sites then come from one batched BF16 GEMM with FP32 output (`head_gate`). Against
cuBLAS, only the FP32 summation order of the first layer differs.

Each architecture has its own kernel: `projection_sm90` (WGMMA) and `projection_sm100` (tcgen05 with the
accumulator in tensor memory).
"""

import math
from functools import lru_cache

import torch
from cutlass import Float32, cute

from ..runtime import check_operands, compile_options, device_key, fake_stream, fake_tensor, on_device, spec


def _kernel(model_dim, hidden, context_eps, has_delta, capability):
    if capability == (9, 0):
        from .projection_sm90 import FirstProjection
    elif capability == (10, 0):
        from .projection_sm100 import FirstProjection
    else:
        raise ValueError(f"fast-inference runs on SM90 or SM100 GPUs, not compute capability {capability}")
    return FirstProjection(model_dim, hidden, context_eps, has_delta)


@lru_cache(maxsize=32)
def _program(model_dim, hidden, context_eps, has_delta, capability, device_index):
    del device_index
    kernel = _kernel(model_dim, hidden, context_eps, has_delta, capability)
    rows, bf16 = (None, model_dim), torch.bfloat16
    specs = (spec(bf16, rows), spec(bf16, rows), spec(bf16, (model_dim // 128, 2 * hidden, 128)),
             spec(bf16, (2 * hidden,)), spec(bf16, (2, None, hidden)))
    return cute.compile(kernel, *(fake_tensor(item) for item in specs), Float32(1.0), fake_stream(),
                        options=compile_options(capability))


def hidden_rows(source, delta, alpha, edge, *, context_eps):
    """The BF16 hidden rows ``[2, N, H]`` of both sites. With `delta`, the updated stream
    ``source + alpha * delta`` is written over `source`."""
    if source.ndim != 2 or not source.is_cuda or source.dtype != torch.bfloat16 or source.shape[0] <= 0:
        raise ValueError("nonempty CUDA BF16 rows required")
    n, d = source.shape
    h = edge.hidden
    if delta is not None and (delta.shape != source.shape or delta.dtype != source.dtype
                              or delta.device != source.device):
        raise ValueError("delta must match the source shape, dtype and device")
    if edge.first_layer.shape[0] * 128 != d or edge.first_layer.device != source.device:
        raise ValueError("the edge's first layer must match the source width and device")
    if not math.isfinite(context_eps) or context_eps <= 0 or not math.isfinite(alpha):
        raise ValueError("positive finite RMS epsilon and finite residual scale required")
    check_operands(source, *(() if delta is None else (delta,)))
    hidden = torch.empty((2, n, h), device=source.device, dtype=torch.bfloat16)
    with on_device(source.device):
        program = _program(d, h, float(context_eps), delta is not None, *device_key(source.device))
        program(source, source if delta is None else delta, edge.first_layer, edge.first_bias, hidden, float(alpha))
    return hidden


def head_gate(hidden, edge):
    """The FP32 head and gate projections ``[2, N, G (K + 1)]`` of both sites, per group the K head
    columns then the gate column, from one batched GEMM on ``second_layer`` without its row padding."""
    _, groups, heads, _ = edge.second_layer.shape
    weight = edge.second_layer[..., :edge.hidden].reshape(2, groups * heads, edge.hidden).transpose(1, 2)
    if not torch.is_autocast_enabled("cuda"):
        return torch.bmm(hidden, weight, out_dtype=torch.float32)
    with torch.autocast(device_type="cuda", enabled=False):
        return torch.bmm(hidden, weight, out_dtype=torch.float32)
