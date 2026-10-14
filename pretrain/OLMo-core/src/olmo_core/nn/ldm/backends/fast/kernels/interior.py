"""The interior of the window backward: one launch per edge of a window other than its last.

Every warp stages its groups through two cp.async stages of its own. A BF16 panel holds the rows
[M (K rows) | vi as three BF16 parts | the pool: residuals r_0..r_edge, this edge's read gradient
y_edge, DY rows y_edge+1..y_E-1] at a padded stride; an FP32 block holds the future value
gradient, the covariance gradient C, the keys, gains and queries of the window, the
future gain gradient, the forgetting factors, the denominator and the write scale. One mma.sync
panel A . B^T over D gives every contraction of the M rows and the pool rows. The covariance VJP
and the outputs after it (the query, key, forgetting and scale gradients and the new C) run for
32 // K groups at once from per-warp finish slots.

vi, the value gradient, is summed in FP32 per lane; dv and the raw value gradient are rounded from
it. The contractions take vi as the exact sum of its three BF16 parts and accumulate in FP32 on
the tensor cores.

P from before this edge's update comes from the saved covariance tape. ``current_dy``, this edge's
read gradient, is staged into the pool and written once into the edge's slot of the DY history.
"""

import math

import cutlass
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, cute
from cutlass.utils import SmemAllocator

from .window_end import (
    _load_bf16,
    _load_dense,
    _load_fp32_vector,
    _store_bf16_vector,
    _store_fp32_row,
    covariance_vjp,
    default_lane_width,
    lane_sum,
)


@cute.jit
def _cp_async_16(smem_addr: Int32, gmem_addr: Int64):
    """One 16-byte cp.async.cg from raw addresses (both 16-byte aligned)."""
    cute.arch.inline_ptx("cp.async.cg.shared.global [{$r0}], [{$r1}], 16;",
                         read_only_args=[smem_addr, gmem_addr])


@cute.jit
def _cp_async_4(smem_addr: Int32, gmem_addr: Int64):
    """One 4-byte cp.async.ca from raw addresses (4-byte aligned)."""
    cute.arch.inline_ptx("cp.async.ca.shared.global [{$r0}], [{$r1}], 4;",
                         read_only_args=[smem_addr, gmem_addr])


@cute.jit
def _cp_async_8(smem_addr: Int32, gmem_addr: Int64):
    """One 8-byte cp.async.ca from raw addresses (8-byte aligned)."""
    cute.arch.inline_ptx("cp.async.ca.shared.global [{$r0}], [{$r1}], 8;",
                         read_only_args=[smem_addr, gmem_addr])


@cute.jit
def _store_bf16_raw(dst: cute.Tensor, offset, raw: cute.Tensor, width: cutlass.Constexpr):
    """Store an already-rounded BF16 register vector (no further rounding)."""
    target = cute.make_tensor(dst.iterator + cute.assume(offset, divby=width),
                              cute.make_layout(width))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                               num_bits_per_copy=16 * width)
    cute.copy(atom, raw, target)


class Interior:
    """The interior of one window shape and schedule (see the module docstring): a persistent
    grid in which every warp stages and computes its own groups. Only this edge's gradient slices
    and C are written; an edge outside the window writes nothing."""

    def __init__(self, E, N, G, K, D, *, sms, warps, blocks_per_sm, lane_width=None):
        self.E, self.N, self.G, self.K, self.D = E, N, G, K, D
        self.warps = warps
        self.blocks = min((N * G + warps - 1) // warps, blocks_per_sm * sms)
        self.static_edge = None
        if D % 8 or K % 4:
            raise ValueError("the interior kernel requires D % 8 == 0 and K % 4 == 0")
        # The D loop's lane width only maps columns to lanes: every vi element keeps its own
        # FP32 expression and the contractions run on the MMA panel, so any width is exact.
        self.lane_width = default_lane_width(D) if lane_width is None else lane_width
        self.width = math.gcd(D, self.lane_width)
        # Two shared stages per warp: one group computed while the next is staged.
        self.stages = 2
        # BF16 panel rows [M (K rows) | vi in three BF16 parts | the pool (E + 1 rows): r_0..r_edge,
        # y_edge..y_E-1], padded to row_stride elements so that ldmatrix reads are free of bank
        # conflicts.
        self.d_pad = (D + 15) // 16 * 16
        self.row_stride = self.d_pad + 8
        self.k_steps = self.d_pad // 16
        self.m_row = 0
        self.split_row = K
        self.pool_row = K + 3
        self.pool_rows = E + 1
        self.a_rows = self.pool_row + self.pool_rows
        self.m_tiles = (self.a_rows + 15) // 16
        self.b_row = K
        self.b_rows = 3 + self.pool_rows
        self.n_tiles = (self.b_rows + 7) // 8
        self.dense_rows = max(self.m_tiles * 16, self.b_row + self.n_tiles * 8, self.a_rows)
        self.packed_words = K * (K + 1) // 2
        self.p_off = self.dense_rows * self.row_stride
        self.dense_elems = self.p_off + (self.packed_words + 7) // 8 * 8
        # FP32 words per stage; every 16-byte copy lands on a 16-byte boundary.
        self.future_value_grad_off = 0
        self.c_off = (D + 3) // 4 * 4
        self.c_words = self.packed_words
        self.keys_off = self.c_off + (self.c_words + 3) // 4 * 4
        self.gains_off = self.keys_off + E * K
        self.u_off = self.gains_off + E * K
        self.future_gain_grad_off = self.u_off + E * K
        self.lam_off = self.future_gain_grad_off + K
        self.den_off = (self.lam_off + E + 3) // 4 * 4
        self.scale_off = self.den_off + 1
        self.compact_words = (self.scale_off + 1 + 3) // 4 * 4
        self.stage_bytes = 2 * self.dense_elems + 4 * self.compact_words
        # Copy classes (16-byte chunks, 32 per instruction); see prepare_issue.
        per_row = max(D // 8, 1)
        self.m_parts = (K * per_row + 31) // 32
        self.pool_parts = (self.pool_rows * per_row + 31) // 32
        self.v_parts = (max(D // 4, 1) + 31) // 32
        self.c_copy_bytes = math.gcd(16, 4 * self.c_words)
        self.c_chunks = 4 * self.c_words // self.c_copy_bytes
        self.c_parts = (self.c_chunks + 31) // 32
        self.p_copy_bytes = math.gcd(16, 2 * self.packed_words)
        self.p_chunks = 2 * self.packed_words // self.p_copy_bytes
        self.p_parts = (self.p_chunks + 31) // 32
        self.k_vector_rows = 1 + 3 * E
        self.k_parts = (self.k_vector_rows * (K // 4) + 31) // 32
        self.s_parts = (E + 2 + 31) // 32
        self.copy_classes = (self.m_parts + self.pool_parts + self.v_parts + self.c_parts
                             + self.p_parts + self.k_parts + self.s_parts)
        # The covariance VJP and the outputs after it keep K lanes busy per group, so they run
        # for 32 // K groups at once. Each group waits for its pack in a per-warp finish slot:
        # FP32 words [C | key | gain | forgetting, denominator, scale, r_i.vi] and BF16 P.
        self.pack = 32 // K
        self.slot_c = 0
        self.slot_key = (self.packed_words + 3) // 4 * 4
        self.slot_gain = self.slot_key + K
        self.slot_scalars = self.slot_gain + K
        self.slot_words = self.slot_scalars + 4
        self.slot_p = (self.packed_words + 7) // 8 * 8
        self.slot_bytes = self.pack * (4 * self.slot_words + 2 * self.slot_p)
        self.shared_bytes = self.warps * (self.stages * self.stage_bytes + self.slot_bytes) + 128

    @cute.jit
    def pool_address(self, r, edge, group, residuals, dy, current_dy):
        """Pool row r: residual r up to this edge, this edge's read gradient, then DY row r - 1."""
        ng = self.N * self.G
        address = residuals.iterator.toint() + (Int64(r) * ng + group) * (2 * self.D)
        if r > edge:
            address = dy.iterator.toint() + (Int64(r - 1) * ng + group) * (2 * self.D)
        if r == edge + 1:
            address = current_dy.iterator.toint() + Int64(group) * (2 * self.D)
        return address, Int64(2 * self.D)

    @cute.jit
    def prepare_issue(self, first, stride, edge: Int32, lane, gaddr: cute.Tensor, gdelta: cute.Tensor,
                      soff: cute.Tensor, m: cute.Tensor, residuals: cute.Tensor, dy: cute.Tensor,
                      keys: cute.Tensor, gains: cute.Tensor, queries: cute.Tensor,
                      future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor, covariance_grad: cute.Tensor,
                      forgetting_tape: cute.Tensor, denominators: cute.Tensor, write_scales: cute.Tensor,
                      saved_p, current_dy):
        """This lane's staging state for its warp's groups first, first + stride, ...: for each
        chunk it copies, the global address for the first group, the address step from one group
        to the next and the chunk's byte offset in a stage. The chunk classes, 32 chunks per copy
        instruction: the M rows (K*D/8 16-byte chunks onto the padded rows), the pool rows
        ((E+1)*D/8, see pool_address), the future value gradient (D/4), the packed covariance
        gradient C and the packed covariance P, the K-vectors ((1+3E)*K/4: key rows, the future
        gain gradient, gain rows and query rows) and the scalars (E+2 4-byte chunks: the
        forgetting rows, the denominator and the write scale)."""
        K: cutlass.Constexpr = self.K
        D: cutlass.Constexpr = self.D
        E: cutlass.Constexpr = self.E
        S: cutlass.Constexpr = self.row_stride
        ng = self.N * self.G
        g0 = Int64(first)
        step = Int64(stride)
        current = Int64(edge) * ng + g0
        per_row: cutlass.Constexpr = D // 8
        c = 0
        for part in cutlass.range_constexpr(self.m_parts):
            chunk = part * 32 + lane
            gaddr[c] = m.iterator.toint() + g0 * (2 * K * D) + chunk * 16
            gdelta[c] = step * (2 * K * D)
            soff[c] = (chunk // per_row) * (2 * S) + (chunk % per_row) * 16
            c += 1
        for part in cutlass.range_constexpr(self.pool_parts):
            chunk = part * 32 + lane
            r = chunk // per_row
            col = chunk % per_row
            address, delta = self.pool_address(r, edge, g0, residuals, dy, current_dy)
            address += col * 16
            gaddr[c] = address
            gdelta[c] = step * delta
            soff[c] = (self.pool_row + r) * (2 * S) + col * 16
            c += 1
        for part in cutlass.range_constexpr(self.v_parts):
            chunk = part * 32 + lane
            gaddr[c] = future_value_grad.iterator.toint() + current * (4 * D) + chunk * 16
            gdelta[c] = step * (4 * D)
            soff[c] = 4 * self.future_value_grad_off + chunk * 16
            c += 1
        for part in cutlass.range_constexpr(self.c_parts):
            chunk = part * 32 + lane
            gaddr[c] = covariance_grad.iterator.toint() + g0 * (4 * self.c_words) + chunk * self.c_copy_bytes
            gdelta[c] = step * (4 * self.c_words)
            soff[c] = 4 * self.c_off + chunk * self.c_copy_bytes
            c += 1
        for part in cutlass.range_constexpr(self.p_parts):
            chunk = part * 32 + lane
            gaddr[c] = saved_p.iterator.toint() + current * (2 * self.packed_words) + chunk * self.p_copy_bytes
            gdelta[c] = step * (2 * self.packed_words)
            soff[c] = 2 * self.p_off + chunk * self.p_copy_bytes
            c += 1
        kchunks: cutlass.Constexpr = K // 4
        for part in cutlass.range_constexpr(self.k_parts):
            chunk = part * 32 + lane
            r = chunk // kchunks
            col = chunk % kchunks
            address = keys.iterator.toint() + (Int64(r) * ng + g0) * (4 * K) + col * 16
            offset = Int32(4 * self.keys_off) + r * (4 * K) + col * 16
            if r == E:
                address = future_gain_grad.iterator.toint() + current * (4 * K) + col * 16
                offset = Int32(4 * self.future_gain_grad_off) + col * 16
            if (r > E) & (r <= 2 * E):
                j = r - (E + 1)
                address = gains.iterator.toint() + (Int64(j) * ng + g0) * (4 * K) + col * 16
                offset = Int32(4 * self.gains_off) + j * (4 * K) + col * 16
            if r > 2 * E:
                j = r - (2 * E + 1)
                address = queries.iterator.toint() + (Int64(j) * ng + g0) * (4 * K) + col * 16
                offset = Int32(4 * self.u_off) + j * (4 * K) + col * 16
            gaddr[c] = address
            gdelta[c] = step * (4 * K)
            soff[c] = offset
            c += 1
        for part in cutlass.range_constexpr(self.s_parts):
            index = part * 32 + lane
            address = forgetting_tape.iterator.toint() + (Int64(index) * ng + g0) * 4
            offset = Int32(4 * self.lam_off) + index * 4
            if index == E:
                address = denominators.iterator.toint() + current * 4
                offset = Int32(4 * self.den_off)
            if index == E + 1:
                address = write_scales.iterator.toint() + g0 * 4
                offset = Int32(4 * self.scale_off)
            gaddr[c] = address
            gdelta[c] = step * 4
            soff[c] = offset
            c += 1

    @cute.jit
    def issue(self, edge: Int32, dense: cute.Tensor, compact: cute.Tensor, doff, coff, lane,
              gaddr: cute.Tensor, gdelta: cute.Tensor, soff: cute.Tensor):
        """Stage one group; completion is the caller's cp.async commit group.

        One cp.async per lane per chunk class (16, 8 or 4 bytes) from the precomputed state."""
        K: cutlass.Constexpr = self.K
        D: cutlass.Constexpr = self.D
        E: cutlass.Constexpr = self.E
        S: cutlass.Constexpr = self.row_stride
        ng = self.N * self.G
        dense_addr = Int32((dense.iterator + doff).toint())
        compact_addr = Int32((compact.iterator + coff).toint())
        per_row: cutlass.Constexpr = D // 8
        c = 0
        for part in cutlass.range_constexpr(self.m_parts):
            if cutlass.const_expr((part + 1) * 32 <= K * per_row):
                _cp_async_16(dense_addr + soff[c], gaddr[c])
            else:
                if part * 32 + lane < K * per_row:
                    _cp_async_16(dense_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        for part in cutlass.range_constexpr(self.pool_parts):
            if cutlass.const_expr((part + 1) * 32 <= self.pool_rows * per_row):
                _cp_async_16(dense_addr + soff[c], gaddr[c])
            else:
                if part * 32 + lane < self.pool_rows * per_row:
                    _cp_async_16(dense_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        for part in cutlass.range_constexpr(self.v_parts):
            if cutlass.const_expr((part + 1) * 32 <= D // 4):
                _cp_async_16(compact_addr + soff[c], gaddr[c])
            else:
                if part * 32 + lane < D // 4:
                    _cp_async_16(compact_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        for part in cutlass.range_constexpr(self.c_parts):
            if part * 32 + lane < self.c_chunks:
                if cutlass.const_expr(self.c_copy_bytes == 16):
                    _cp_async_16(compact_addr + soff[c], gaddr[c])
                else:
                    _cp_async_8(compact_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        for part in cutlass.range_constexpr(self.p_parts):
            if part * 32 + lane < self.p_chunks:
                if cutlass.const_expr(self.p_copy_bytes == 16):
                    _cp_async_16(dense_addr + soff[c], gaddr[c])
                elif cutlass.const_expr(self.p_copy_bytes == 8):
                    _cp_async_8(dense_addr + soff[c], gaddr[c])
                else:
                    _cp_async_4(dense_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        kchunks: cutlass.Constexpr = K // 4
        for part in cutlass.range_constexpr(self.k_parts):
            chunk = part * 32 + lane
            r = chunk // kchunks
            # Key rows j >= edge, the future gain gradient, every gain row and query rows j >= edge.
            wanted = ((r < E) & (r >= edge)) | ((r >= E) & (r <= 2 * E)) | ((r > 2 * E) & (r - (2 * E + 1) >= edge))
            if (chunk < self.k_vector_rows * kchunks) & wanted:
                _cp_async_16(compact_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1
        for part in cutlass.range_constexpr(self.s_parts):
            if part * 32 + lane < E + 2:
                _cp_async_4(compact_addr + soff[c], gaddr[c])
            gaddr[c] = gaddr[c] + gdelta[c]
            c += 1

    @cute.jit
    def __call__(self, m: cute.Tensor, residuals: cute.Tensor, dy: cute.Tensor,
                 keys: cute.Tensor, gains: cute.Tensor, queries: cute.Tensor,
                 future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor, dv: cute.Tensor,
                 dq: cute.Tensor, covariance_grad: cute.Tensor,
                 forgetting_tape: cute.Tensor, denominators: cute.Tensor,
                 grad_forgetting: cute.Tensor, write_scales: cute.Tensor,
                 grad_rawvalue: cute.Tensor, grad_rawkey: cute.Tensor,
                 grad_write_scale: cute.Tensor, saved_p, current_dy: cute.Tensor, edge: Int32, stream):
        """The interior's arguments, with the saved covariance and this edge's read gradient."""
        words: cutlass.Constexpr = self.packed_words
        for tensor, shape, dtype in (
            (m, (self.N, self.G, self.K, self.D), BFloat16),
            (residuals, (self.E, self.N, self.G, self.D), BFloat16),
            (dy, (self.E, self.N, self.G, self.D), BFloat16),
            (keys, (self.E, self.N, self.G, self.K), Float32),
            (gains, (self.E, self.N, self.G, self.K), Float32),
            (queries, (self.E, self.N, self.G, self.K), Float32),
            (future_value_grad, (self.E, self.N, self.G, self.D), Float32),
            (future_gain_grad, (self.E, self.N, self.G, self.K), Float32),
            (dv, (self.E, self.N, self.G, self.D), BFloat16),
            (forgetting_tape, (self.E, self.N, self.G), Float32),
            (denominators, (self.E, self.N, self.G), Float32),
            (saved_p, (self.E, self.N, self.G, words), BFloat16),
            (covariance_grad, (self.N, self.G, words), Float32),
        ):
            assert tensor.element_type == dtype
            assert tuple(tensor.shape) == shape
            stride = 1
            for axis in cutlass.range_constexpr(len(shape) - 1, -1, -1):
                assert tensor.stride[axis] == stride
                stride *= shape[axis]
        # The query and forgetting gradients may be single-edge tensors behind a zero-stride edge
        # axis: each interior writes one edge.
        assert dq.element_type == Float32
        assert tuple(dq.shape) == (self.E, self.N, self.G, self.K)
        assert tuple(dq.stride[1:]) == (self.G * self.K, self.K, 1)
        assert dq.stride[0] in (0, self.N * self.G * self.K)
        assert grad_forgetting.element_type == Float32
        assert tuple(grad_forgetting.shape) == (self.E, self.N, self.G)
        assert tuple(grad_forgetting.stride[1:]) == (self.G, 1)
        assert grad_forgetting.stride[0] in (0, self.N * self.G)
        for tensor, shape, dtype in (
            (current_dy, (self.N, self.G, self.D), BFloat16),
            (write_scales, (self.N, self.G), Float32),
            (grad_rawvalue, (self.N, self.G, self.D), BFloat16),
            (grad_rawkey, (self.N, self.G, self.K), Float32),
            (grad_write_scale, (self.N, self.G), Float32),
        ):
            assert tensor.element_type == dtype
            assert tuple(tensor.shape) == shape
            stride = 1
            for axis in cutlass.range_constexpr(len(shape) - 1, -1, -1):
                assert tensor.stride[axis] == stride
                stride *= shape[axis]
        self.kernel(m, residuals, dy, keys, gains, queries,
                    future_value_grad, future_gain_grad, dv, dq, covariance_grad, forgetting_tape,
                    denominators, grad_forgetting, edge, write_scales,
                    grad_rawvalue, grad_rawkey, grad_write_scale, saved_p, current_dy).launch(
                        grid=(self.blocks, 1, 1), block=(32 * self.warps, 1, 1), stream=stream)

    def _fragment(self, acc, row, col):
        """(register value, source lane) of panel element C[row, col], both static.

        mma.m16n8k16 C fragment: lane holds rows lane//4 and lane//4+8, columns
        2*(lane%4) and 2*(lane%4)+1 of each (m-tile, n-tile) accumulator, in
        registers (row//8)*2 + col%2.
        """
        mt, r = divmod(row, 16)
        nt, c = divmod(col, 8)
        return acc[2 * (r // 8) + (c % 2), mt, nt], 4 * (r % 8) + c // 2

    @cute.jit
    def _fragment_col(self, acc: cute.Tensor, row: cutlass.Constexpr, nt_dyn, parity):
        """Register value of C[row, col] for a static row and a warp-uniform
        runtime column (n-tile nt_dyn, parity col % 2)."""
        mt, r = divmod(row, 16)
        value = acc[2 * (r // 8), mt, 0]
        for nt in cutlass.range_constexpr(self.n_tiles):
            for p in cutlass.range_constexpr(2):
                if cutlass.const_expr(nt > 0 or p > 0):  # noqa: SIM102
                    if (nt_dyn == nt) & (parity == p):
                        value = acc[2 * (r // 8) + p, mt, nt]
        return value

    @cute.jit
    def _fragment_row(self, acc: cute.Tensor, row, col: cutlass.Constexpr):
        """Register value of C[row, col] for a warp-uniform runtime row."""
        nt, c = divmod(col, 8)
        value = acc[c % 2, 0, nt]
        for mt in cutlass.range_constexpr(self.m_tiles):
            for h in cutlass.range_constexpr(2):
                if cutlass.const_expr(mt > 0 or h > 0):  # noqa: SIM102
                    if (row // 16 == mt) & (row % 16 // 8 == h):
                        value = acc[2 * h + c % 2, mt, nt]
        return value

    @cute.jit
    def compute(self, group, edge: Int32, dense: cute.Tensor, compact: cute.Tensor, doff, coff,
                lane, width: cutlass.Constexpr, dv: cute.Tensor, grad_rawvalue: cute.Tensor,
                dy: cute.Tensor, slots: cute.Tensor, slot_p: cute.Tensor, sw, sp, slot,
                carry: cute.Tensor):
        ng = self.N * self.G
        current = Int64(edge) * ng + group
        S: cutlass.Constexpr = self.row_stride
        K: cutlass.Constexpr = self.K
        E: cutlass.Constexpr = self.E
        uo = coff + self.u_off
        # One vector load: every scalar shared load costs a whole wavefront.
        scalars = _load_fp32_vector(compact, coff + self.den_off, 4)  # denominator, scale, padding
        write_scale = scalars[1]
        gain_i = _load_fp32_vector(compact, coff + self.gains_off + edge * K, K)
        # alpha_j = gain_i . U_ij, summed over k in ascending order; one lane per j, in banks of 32
        # lanes for windows longer than a warp.
        alphas = cute.make_rmem_tensor(E, Float32)
        for bank in cutlass.range_constexpr(cute.ceil_div(E, 32)):
            coefficient = 32 * bank + lane
            alpha_lane = Float32(0.0)
            # Each active lane transforms one staged query row into U_ij in place.
            if (coefficient < E) & (coefficient >= edge):
                u_row = _load_fp32_vector(compact, uo + coefficient * K, K)
                for t in cutlass.range_constexpr(E - 1, -1, -1):
                    if (t > edge) & (t <= coefficient):
                        gain_t = _load_fp32_vector(compact, coff + self.gains_off + t * K, K)
                        key_t = _load_fp32_vector(compact, coff + self.keys_off + t * K, K)
                        pairing = Float32(0.0)
                        for k in cutlass.range_constexpr(K):
                            pairing += gain_t[k] * u_row[k]
                        for k in cutlass.range_constexpr(K):
                            u_row[k] -= key_t[k] * pairing
                for k in cutlass.range_constexpr(K):
                    alpha_lane += gain_i[k] * u_row[k]
                _store_fp32_row(compact, Int64(uo + coefficient * K), u_row, K, math.gcd(K, 4))
            for j in cutlass.range_constexpr(min(32, E - 32 * bank)):
                alphas[32 * bank + j] = cute.arch.shuffle_sync(alpha_lane, j)
        # vi for this lane's columns, the dv and raw value gradient outputs, and vi's exact split
        # into three BF16 rows of the panel.
        full: cutlass.Constexpr = self.D % (32 * width) == 0
        for tile in cutlass.range_constexpr(cute.ceil_div(self.D, 32 * width)):
            d = tile * (32 * width) + lane * width
            inside = (d < self.D) | Boolean(full)
            if inside:
                offset = current * self.D + Int64(d)
                # y_i, staged from current_dy, into its DY history slot.
                y_raw = _load_bf16(dense, doff + (self.pool_row + edge + 1) * S + d, width)
                _store_bf16_raw(dy, offset, y_raw, width)
                # One vector load of the future value gradient keeps the live ranges short.
                vi = _load_fp32_vector(compact, coff + self.future_value_grad_off + d, width)
                for j in cutlass.range_constexpr(E):
                    if j >= edge:
                        yj = _load_dense(dense, doff + (self.pool_row + j + 1) * S + d, width)
                        for v in cutlass.range_constexpr(width):
                            vi[v] += alphas[j] * yj[v]
                hi = cute.make_rmem_tensor(width, BFloat16)
                hi.store(vi.load().to(BFloat16))
                rem = cute.make_rmem_tensor(width, Float32)
                rem.store(vi.load() - hi.load().to(Float32))
                mid = cute.make_rmem_tensor(width, BFloat16)
                mid.store(rem.load().to(BFloat16))
                low = cute.make_rmem_tensor(width, BFloat16)
                low.fill(0.0)
                rem2 = cute.make_rmem_tensor(width, Float32)
                rem2.store(rem.load() - mid.load().to(Float32))
                low.store(rem2.load().to(BFloat16))
                _store_bf16_raw(dense, doff + self.split_row * S + d, hi, width)
                _store_bf16_raw(dense, doff + (self.split_row + 1) * S + d, mid, width)
                _store_bf16_raw(dense, doff + (self.split_row + 2) * S + d, low, width)
                _store_bf16_vector(dv, current * self.D + Int64(d), vi, width)
                raw = cute.make_rmem_tensor(width, Float32)
                raw.store(vi.load() * write_scale)
                _store_bf16_vector(grad_rawvalue, Int64(group) * self.D + Int64(d), raw, width)
        cute.arch.sync_warp()
        acc = cute.make_rmem_tensor((4, self.m_tiles, self.n_tiles), Float32)
        acc.fill(0.0)
        self.contract(acc, dense, doff, lane)
        # Panel coordinates: M[k] is row k; pool entry r is row pool_row + r of A and column 3 + r
        # of B; r_j is pool entry j (j <= edge) and y_j pool entry j + 1 (j >= edge), so y_i is
        # column 4 + edge and r_i row pool_row + edge.
        yi_col = Int32(4) + edge
        yi_nt = yi_col // 8
        yi_parity = yi_col % 2
        yi_lane = (yi_col % 8) // 2
        ri_row = Int32(self.pool_row) + edge
        # M[k].y_i and M[k].vi for k = lane % K, the row this lane holds in the covariance VJP.
        m_row = lane % K
        halves: cutlass.Constexpr = (K + 7) // 8
        m_y = Float32(0.0)
        m_v = Float32(0.0)
        for h in cutlass.range_constexpr(halves):
            send_y = self._fragment_col(acc, 8 * h, yi_nt, yi_parity)
            send_hm0, _ = self._fragment(acc, 8 * h, 0)
            send_hm1, _ = self._fragment(acc, 8 * h, 1)
            got_y = cute.arch.shuffle_sync(send_y, 4 * (m_row % 8) + yi_lane)
            got_hm = cute.arch.shuffle_sync(send_hm0 + send_hm1, 4 * (m_row % 8))
            send_lo, _ = self._fragment(acc, 8 * h, 2)
            got_lo = cute.arch.shuffle_sync(send_lo, 4 * (m_row % 8) + 1)
            if cutlass.const_expr(h == 0):
                m_y = got_y
                m_v = got_hm + got_lo
            else:
                if m_row >= 8 * h:
                    m_y = got_y
                    m_v = got_hm + got_lo
        # The same in every lane: r_j.y_i and r_j.vi (j <= edge), r_i.y_j (j >= edge).
        ry = cute.make_rmem_tensor(E, Float32)
        rv = cute.make_rmem_tensor(E, Float32)
        ryf = cute.make_rmem_tensor(E, Float32)
        residual_pairing = Float32(0.0)
        for j in cutlass.range_constexpr(E):
            row: cutlass.Constexpr = self.pool_row + j
            value = self._fragment_col(acc, row, yi_nt, yi_parity)
            ry[j] = cute.arch.shuffle_sync(value, 4 * (row % 8) + yi_lane)
            hm0, source_hm = self._fragment(acc, row, 0)
            hm1, _ = self._fragment(acc, row, 1)
            rv[j] = cute.arch.shuffle_sync(hm0 + hm1, source_hm)
            low_value, source_lo = self._fragment(acc, row, 2)
            rv[j] += cute.arch.shuffle_sync(low_value, source_lo)
            value_f = self._fragment_row(acc, ri_row, 4 + j)
            ryf[j] = cute.arch.shuffle_sync(value_f, 4 * (ri_row % 8) + ((4 + j) % 8) // 2)
            if j == edge:
                residual_pairing = rv[j]
        self.stash(edge, compact, coff, dense, doff, lane, m_y, m_v, ry, rv, ryf,
                   residual_pairing, scalars, slots, slot_p, sw, sp, slot, carry)

    @cute.jit
    def contract(self, acc: cute.Tensor, dense: cute.Tensor, doff, lane):
        """One mma.sync panel per group: acc[(v, m-tile, n-tile)] = A . B^T over d_pad.

        A [m_tiles*16 x d_pad] rows = M rows, split rows (unused), pool rows;
        B [n_tiles*8 x d_pad] rows = vi_hi, vi_mid, vi_lo, pool rows.
        """
        S: cutlass.Constexpr = self.row_stride
        mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)))
        thr = mma.get_slice(lane)
        ac = thr.partition_A(cute.make_identity_tensor((16, 16)))
        bc = thr.partition_B(cute.make_identity_tensor((8, 16)))
        ld_a = cute.make_tiled_copy_A(cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), BFloat16), mma).get_slice(lane)
        ld_b = cute.make_tiled_copy_B(cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), BFloat16), mma).get_slice(lane)
        sa = cute.make_tensor(dense.iterator + cute.assume(doff, divby=8),
                              cute.make_layout((self.m_tiles * 16, self.d_pad), stride=(S, 1)))
        sb = cute.make_tensor(dense.iterator + cute.assume(doff + self.b_row * S, divby=8),
                              cute.make_layout((self.n_tiles * 8, self.d_pad), stride=(S, 1)))
        # Even and odd k-steps accumulate separately, which halves the chain of dependent MMAs;
        # the two FP32 panels are added once at the end.
        acc_odd = cute.make_rmem_tensor((4, self.m_tiles, self.n_tiles), Float32)
        acc_odd.fill(0.0)
        for step in cutlass.range_constexpr(self.k_steps):
            a_frags = []
            for mt in cutlass.range_constexpr(self.m_tiles):
                a = cute.make_rmem_tensor(ac.shape, BFloat16)
                cute.copy(ld_a, ld_a.partition_S(cute.local_tile(sa, (16, 16), (mt, step))),
                          ld_a.retile(a))
                a_frags.append(a)
            b_frags = []
            for nt in cutlass.range_constexpr(self.n_tiles):
                b = cute.make_rmem_tensor(bc.shape, BFloat16)
                cute.copy(ld_b, ld_b.partition_S(cute.local_tile(sb, (8, 16), (nt, step))),
                          ld_b.retile(b))
                b_frags.append(b)
            target = acc if step % 2 == 0 else acc_odd
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.m_tiles):
                    c = cute.make_tensor(target.iterator + 4 * (mt + self.m_tiles * nt),
                                         cute.make_layout((4, 1, 1)))
                    cute.gemm(mma, c, a_frags[mt], b_frags[nt], c)
        if cutlass.const_expr(self.k_steps > 1):
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.m_tiles):
                    for v in cutlass.range_constexpr(4):
                        acc[v, mt, nt] = acc[v, mt, nt] + acc_odd[v, mt, nt]

    @cute.jit
    def stash(self, edge: Int32, compact: cute.Tensor, coff, dense: cute.Tensor, doff, lane,
              m_y, m_v, ry: cute.Tensor, rv: cute.Tensor, ryf: cute.Tensor, residual_pairing,
              scalars: cute.Tensor, slots: cute.Tensor, slot_p: cute.Tensor, sw, sp, slot,
              carry: cute.Tensor):
        """The panel sums the covariance VJP needs, and the group's copy into its finish slot.

        The lanes of run ``slot`` (lanes slot*K .. slot*K + K - 1) keep, for row k = lane % K:
          carry[0] = M[k].y_i + sum_{j<=edge} g_j[k] (r_j.y_i)
          carry[1] = -(M[k].vi) - sum_{j<edge} g_j[k] (r_j.vi)
          carry[2] = sum_{j>=edge} U_ij[k] (r_i.y_j) + the future gain gradient
        The slot receives the covariance gradient C, P, the edge's key and gain rows, the
        forgetting factor, the denominator, the write scale and r_i.vi.
        """
        K: cutlass.Constexpr = self.K
        E: cutlass.Constexpr = self.E
        go = coff + self.gains_off
        uo = coff + self.u_off
        row = lane % K
        mine = lane // K == slot
        on_y = m_y
        on_v = Float32(0.0) - m_v
        on_future = Float32(0.0)
        for j in cutlass.range_constexpr(E):
            if j >= edge:
                if mine:
                    on_future += compact[uo + j * K + row] * ryf[j]
            if j <= edge:  # noqa: SIM102
                if mine:
                    gj = compact[go + j * K + row]
                    on_y += gj * ry[j]
                    if j < edge:
                        on_v -= gj * rv[j]
        if mine:
            carry[0] = on_y
            carry[1] = on_v
            carry[2] = on_future + compact[coff + self.future_gain_grad_off + row]
        words: cutlass.Constexpr = self.packed_words
        for part in cutlass.range_constexpr(cute.ceil_div(words, 32)):
            index = part * 32 + lane
            if index < words:
                slots[sw + self.slot_c + index] = compact[coff + self.c_off + index]
                slot_p[sp + index] = dense[doff + self.p_off + index]
        if lane < K:
            slots[sw + self.slot_key + lane] = compact[coff + self.keys_off + edge * K + lane]
            slots[sw + self.slot_gain + lane] = compact[go + edge * K + lane]
        if lane == 0:
            slots[sw + self.slot_scalars] = compact[coff + self.lam_off + edge]
            slots[sw + self.slot_scalars + 1] = scalars[0]
            slots[sw + self.slot_scalars + 2] = scalars[1]
            slots[sw + self.slot_scalars + 3] = residual_pairing

    @cute.jit
    def finish(self, first, stride, t0, count, edge: Int32, slots: cute.Tensor,
               slot_p: cute.Tensor, slot_base, slot_p_base, lane, carry: cute.Tensor,
               dq: cute.Tensor, covariance_grad: cute.Tensor, grad_forgetting: cute.Tensor,
               grad_rawkey: cute.Tensor, grad_write_scale: cute.Tensor):
        """The covariance VJP and the key, query, forgetting, scale and C outputs of the ``count``
        groups in the finish slots, one run of K lanes per group. Their panel sums are in carry."""
        K: cutlass.Constexpr = self.K
        words: cutlass.Constexpr = self.packed_words
        cute.arch.sync_warp()
        row = lane % K
        run = lane // K
        active = run < count
        sw = slot_base + run * self.slot_words
        sp = slot_p_base + run * self.slot_p
        group = Int64(first + (t0 + run) * stride)
        n, g = group // self.G, group % self.G
        p_row = cute.make_rmem_tensor(K, Float32)
        h_row = cute.make_rmem_tensor(K, Float32)
        p_row.fill(0.0)
        h_row.fill(0.0)
        key_lane, gain_lane = Float32(0.0), Float32(0.0)
        if active:
            key_lane = slots[sw + self.slot_key + row]
            gain_lane = slots[sw + self.slot_gain + row]
            for column in cutlass.range_constexpr(K):
                first_index = cutlass.min(row, Int32(column))
                last_index = cutlass.max(row, Int32(column))
                packed = first_index * K - first_index * (first_index - 1) // 2 + last_index - first_index
                p_row[column] = Float32(slot_p[sp + packed])
                h_row[column] = slots[sw + self.slot_c + packed]
        key_row = _load_fp32_vector(slots, sw + self.slot_key, K)
        gain_row = _load_fp32_vector(slots, sw + self.slot_gain, K)
        scalars = _load_fp32_vector(slots, sw + self.slot_scalars, 4)
        dk_cov, dforget, previous = covariance_vjp(
            p_row, h_row, key_row, gain_row, key_lane, gain_lane, carry[2], scalars[1], scalars[0],
            row, active, run * K, K)
        pairing = Float32(0.0)
        if active:
            key_gradient = carry[1] + dk_cov
            dq[edge, n, g, row] = carry[0]
            grad_rawkey[n, g, row] = scalars[2] * key_gradient
            pairing = dk_cov * key_lane
            for column in cutlass.range_constexpr(K):
                if column >= row:
                    packed = row * K - row * (row - 1) // 2 + column - row
                    slots[sw + self.slot_c + packed] = previous[column]
        # Every lane's rows of the new C are in the slots; store them as contiguous vectors.
        cute.arch.sync_warp()
        store_width: cutlass.Constexpr = math.gcd(words, 4)
        chunks: cutlass.Constexpr = words // store_width
        for part in cutlass.range_constexpr(cute.ceil_div(self.pack * chunks, 32)):
            chunk = part * 32 + lane
            slot = chunk // chunks
            if slot < count:
                piece = (chunk % chunks) * store_width
                values = _load_fp32_vector(slots, slot_base + slot * self.slot_words + self.slot_c + piece,
                                           store_width)
                _store_fp32_row(covariance_grad, Int64(first + (t0 + slot) * stride) * words + piece,
                                values, store_width, store_width)
        pairing = lane_sum(pairing, K)
        if active & (row == 0):
            grad_forgetting[edge, n, g] = dforget
            scale_gradient = Float32(0.0)
            if scalars[2] != Float32(0.0):
                scale_gradient = (scalars[3] + pairing) / scalars[2]
            grad_write_scale[n, g] = scale_gradient

    @cute.kernel
    def kernel(self, m: cute.Tensor, residuals: cute.Tensor, dy: cute.Tensor,
               keys: cute.Tensor, gains: cute.Tensor, queries: cute.Tensor,
               future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor, dv: cute.Tensor,
               dq: cute.Tensor, covariance_grad: cute.Tensor,
               forgetting_tape: cute.Tensor, denominators: cute.Tensor,
               grad_forgetting: cute.Tensor, edge: Int32, write_scales: cute.Tensor,
               grad_rawvalue: cute.Tensor, grad_rawkey: cute.Tensor, grad_write_scale: cute.Tensor,
               saved_p, current_dy):
        if cutlass.const_expr(self.static_edge is not None):
            edge = Int32(self.static_edge)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        K, D = self.K, self.D
        output_alignment = min(grad_rawvalue.iterator.max_alignment, dv.iterator.max_alignment)
        # Staging is 16-byte cp.async: every staged tensor 16-byte aligned, D % 8 == 0, K % 4 == 0.
        for tensor in (m, residuals, dy, future_value_grad, covariance_grad, keys, gains,
                       queries, future_gain_grad, current_dy):
            assert tensor.iterator.max_alignment >= 16
        width = math.gcd(D, self.lane_width, output_alignment // 2)
        assert D % 8 == 0 and K % 4 == 0
        smem = SmemAllocator()
        dense = smem.allocate_tensor(BFloat16, cute.make_layout(
            self.warps * self.stages * self.dense_elems), byte_alignment=128)
        compact = smem.allocate_tensor(Float32, cute.make_layout(
            self.warps * self.stages * self.compact_words), byte_alignment=16)
        slots = smem.allocate_tensor(Float32, cute.make_layout(
            self.warps * self.pack * self.slot_words), byte_alignment=16)
        slot_p = smem.allocate_tensor(BFloat16, cute.make_layout(
            self.warps * self.pack * self.slot_p), byte_alignment=16)
        S: cutlass.Constexpr = self.stages
        dense_base = warp * (S * self.dense_elems)
        compact_base = warp * (S * self.compact_words)
        slot_base = warp * (self.pack * self.slot_words)
        slot_p_base = warp * (self.pack * self.slot_p)
        # A short last pack leaves slots unfilled; its idle lanes then read zeros.
        for part in cutlass.range_constexpr(cute.ceil_div(self.pack * self.slot_words, 32)):
            index = part * 32 + lane
            if index < self.pack * self.slot_words:
                slots[slot_base + index] = Float32(0.0)
        for part in cutlass.range_constexpr(cute.ceil_div(self.pack * self.slot_p, 32)):
            index = part * 32 + lane
            if index < self.pack * self.slot_p:
                slot_p[slot_p_base + index] = BFloat16(0.0)
        # The padding columns D..d_pad of every panel row are zero and no copy or store writes
        # them.
        if cutlass.const_expr(self.d_pad > D):
            tail: cutlass.Constexpr = self.d_pad - D
            total: cutlass.Constexpr = S * self.dense_rows * tail
            for part in cutlass.range_constexpr((total + 31) // 32):
                index = part * 32 + lane
                if index < total:
                    row = index // tail
                    offset = (row // self.dense_rows) * self.dense_elems + (row % self.dense_rows) * self.row_stride
                    dense[dense_base + offset + D + index % tail] = BFloat16(0.0)
        cute.arch.sync_warp()
        stride = Int32(self.blocks) * self.warps
        first = Int32(bid) * self.warps + warp
        mine = Int32(0)
        if first < self.N * self.G:
            mine = (Int32(self.N * self.G) - first + stride - 1) // stride
        classes: cutlass.Constexpr = self.copy_classes
        gaddr = cute.make_rmem_tensor(classes, Int64)
        gdelta = cute.make_rmem_tensor(classes, Int64)
        soff = cute.make_rmem_tensor(classes, Int32)
        if (edge >= 0) & (edge < self.E):
            self.prepare_issue(first, stride, edge, lane, gaddr, gdelta, soff, m, residuals, dy, keys, gains,
                               queries, future_value_grad, future_gain_grad, covariance_grad,
                               forgetting_tape, denominators, write_scales, saved_p, current_dy)
            for s in cutlass.range_constexpr(S - 1):
                if s < mine:
                    self.issue(edge, dense, compact, dense_base + s * self.dense_elems,
                               compact_base + s * self.compact_words, lane, gaddr, gdelta, soff)
                cute.arch.cp_async_commit_group()
            carry = cute.make_rmem_tensor(3, Float32)
            carry.fill(0.0)
            for t in cutlass.range(0, mine, 1):
                stage = t % S
                ahead = t + (S - 1)
                if ahead < mine:
                    other = ahead % S
                    self.issue(edge, dense, compact, dense_base + other * self.dense_elems,
                               compact_base + other * self.compact_words, lane, gaddr, gdelta, soff)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(S - 1)
                cute.arch.sync_warp()
                slot = t % self.pack
                self.compute(Int64(first + t * stride), edge, dense, compact,
                             dense_base + stage * self.dense_elems,
                             compact_base + stage * self.compact_words, lane, width, dv,
                             grad_rawvalue, dy, slots, slot_p, slot_base + slot * self.slot_words,
                             slot_p_base + slot * self.slot_p, slot, carry)
                if (slot == self.pack - 1) | (t == mine - 1):
                    self.finish(first, stride, t - slot, slot + 1, edge, slots, slot_p, slot_base,
                                slot_p_base, lane, carry, dq, covariance_grad, grad_forgetting,
                                grad_rawkey, grad_write_scale)
                cute.arch.sync_warp()
            cute.arch.cp_async_wait_group(0)


def make_interior(E: int, N: int, G: int, K: int, D: int, *, sms: int, warps: int = 2,
                  blocks_per_sm: int = 6, lane_width: int | None = None):
    """The tensor-core interior for one window shape with its launch schedule: warps per CTA,
    resident CTAs per SM and the D-loop lane width (default: default_lane_width(D))."""
    if any(type(size) is not int or size <= 0 for size in (E, N, G, K, D)):
        raise ValueError("E, N, G, K, D must be positive Python integers")
    if K not in (4, 8, 16):
        raise ValueError("K must be 4, 8, or 16")
    if any(type(v) is not int or v <= 0 for v in (sms, warps, blocks_per_sm)) or warps > 32:
        raise ValueError("sms and blocks_per_sm must be positive integers and warps 1..32")
    if lane_width is not None and lane_width not in (1, 2, 4, 8):
        raise ValueError("lane_width must be 1, 2, 4 or 8")
    return Interior(E, N, G, K, D, sms=sms, warps=warps, blocks_per_sm=blocks_per_sm,
                    lane_width=lane_width)
