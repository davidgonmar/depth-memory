"""Copies the kernels share: global to shared memory in bulk, and vector loads and stores of BF16 and
FP32 elements through registers."""

import math

import cutlass
from cutlass import BFloat16, Float32, Int32, cute


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@cute.jit
def bulk_copy(dst: cute.Tensor, dst_offset, src: cute.Tensor, src_offset, nbytes: Int32, mbar: cute.Pointer):
    cute.arch.inline_ptx(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}], {$r2}, [{$r3}];",
        read_only_args=[Int32((dst.iterator + dst_offset).toint()),
                        (src.iterator + src_offset).llvm_ptr, nbytes, Int32(mbar.toint())])


@cute.jit
def fetch_bf16(tensor: cute.Tensor, offset, width: cutlass.Constexpr):
    """`width` BF16 elements (a multiple of 8) from `offset` into BF16 registers, in 128-bit accesses."""
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
    value = cute.make_rmem_tensor(width, BFloat16)
    for part in cutlass.range_constexpr(width // 8):
        source = cute.make_tensor(tensor.iterator + cute.assume(offset + part * 8, divby=8), cute.make_layout(8))
        piece = cute.make_rmem_tensor(8, BFloat16)
        cute.copy(atom, source, piece)
        for v in cutlass.range_constexpr(8):
            value[part * 8 + v] = piece[v]
    return value


@cute.jit
def load_bf16(tensor: cute.Tensor, offset, width: cutlass.Constexpr):
    chunk: cutlass.Constexpr = math.gcd(width, 8)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=16 * chunk)
    value = cute.make_rmem_tensor(width, Float32)
    for part in cutlass.range_constexpr(width // chunk):
        source = cute.make_tensor(tensor.iterator + cute.assume(offset + part * chunk, divby=chunk),
                                  cute.make_layout(chunk))
        raw = cute.make_rmem_tensor(chunk, BFloat16)
        cute.copy(atom, source, raw)
        piece = cute.make_rmem_tensor(chunk, Float32)
        piece.store(raw.load().to(Float32))
        for v in cutlass.range_constexpr(chunk):
            value[part * chunk + v] = piece[v]
    return value


@cute.jit
def load_fp32(tensor: cute.Tensor, offset, count: cutlass.Constexpr):
    out = cute.make_rmem_tensor(count, Float32)
    chunk: cutlass.Constexpr = math.gcd(count, 4)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=32 * chunk)
    for part in cutlass.range_constexpr(count // chunk):
        source = cute.make_tensor(tensor.iterator + cute.assume(offset + part * chunk, divby=chunk),
                                  cute.make_layout(chunk))
        piece = cute.make_rmem_tensor(chunk, Float32)
        cute.copy(atom, source, piece)
        for v in cutlass.range_constexpr(chunk):
            out[part * chunk + v] = piece[v]
    return out


@cute.jit
def store(tensor: cute.Tensor, offset, value: cute.Tensor, width: cutlass.Constexpr):
    destination = cute.make_tensor(tensor.iterator + cute.assume(offset, divby=width), cute.make_layout(width))
    raw = cute.make_rmem_tensor(width, tensor.element_type)
    raw.store(value.load().to(tensor.element_type))
    bits = min(width * tensor.element_type.width, tensor.iterator.max_alignment * 8, 128)
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), tensor.element_type, num_bits_per_copy=bits),
              raw, destination)
