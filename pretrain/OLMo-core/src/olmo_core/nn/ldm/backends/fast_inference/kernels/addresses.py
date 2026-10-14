"""Per (token, group) routines of an edge: both addresses, the gain, the coefficient row and the
covariance update.

A K-lane subgroup handles one (token, group) row. `produce` finishes both addresses from the head and gate
projections, forms the gain against the packed covariance and writes the FP32 coefficient row the dense
step reads, ``[key[K], query[K], gain[K], scale, gamma, alpha[E-1], beta[E-1]]`` padded to four words,
where only a commit stores the gain. `update_covariance` then writes ``(P - gain Pk^T) / forgetting``.
Both edge kernels (`edge_state`, `edge_decode`) call them on shared-memory copies of the covariance and
the gain history.
"""

import operator
from typing import NamedTuple

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute

from .elementwise import sigmoid

KEY_WIDTHS = (4, 8)     # key widths of the released models


class LaneParameters(NamedTuple):
    """The FP32 parameters one lane's addresses and covariance update read."""

    bias_key: Float32             # head bias of the write site at the lane's key component
    bias_query: Float32           # head bias of the read site
    static_key: Float32           # static address of the write site
    static_query: Float32         # static address of the read site
    gate_bias_key: Float32        # the group's write-gate bias
    gate_bias_query: Float32      # the group's read-gate bias
    gain: Float32                 # the group's read gain, BF16-rounded
    forgetting: Float32           # the group's forgetting factor


class Projections(NamedTuple):
    """One lane's FP32 head projections of both sites, and its group's gate projections (the same on
    every lane of the subgroup)."""

    head_key: Float32
    head_query: Float32
    gate_key: Float32
    gate_query: Float32


def coefficient_width(E, K, *, commit):
    """FP32 words of a coefficient row, padded to 16 bytes. Only a commit needs the gain for the M update."""
    return ((2 + int(commit)) * K + 2 + 2 * (E - 1) + 3) // 4 * 4


def packed_width(K):
    """BF16 words of one packed upper-triangular K x K covariance, padded to 16 bytes."""
    return (K * (K + 1) // 2 + 7) // 8 * 8


class Addresses:
    """The routines of one edge's window position."""

    def __init__(self, E, K, *, commit, terminal, eps):
        if not 1 <= E <= 8:
            raise ValueError(f"span {E}: 1 through 8 is required")
        if K not in KEY_WIDTHS:
            raise ValueError(f"key width {K}: one of {KEY_WIDTHS} is required")
        if terminal and commit:
            raise ValueError("a terminal read must not commit dead state")
        self.E, self.K = E, K
        self.commit, self.terminal = bool(commit), bool(terminal)
        self.eps = float(eps)
        self.past = E - 1
        self.scalar_offset = (2 + int(self.commit)) * K
        self.coeff_width = coefficient_width(E, K, commit=self.commit)
        self.triangle = K * (K + 1) // 2
        self.update_parts = (self.triangle + K - 1) // K
        # For each part of the update, the covariance entry (i, j) each lane writes, 4 bits per index in one
        # word per axis.
        indices = [(i, j) for i in range(K) for j in range(i, K)]
        indices += [(0, 0)] * (self.update_parts * K - self.triangle)
        tables = [[], []]
        for part in range(self.update_parts):
            for axis in range(2):
                word = sum(indices[part * K + lane][axis] << (4 * lane) for lane in range(K))
                tables[axis].append(word - (1 << 32) if word >= 1 << 31 else word)
        self.update_rows, self.update_columns = tuple(tables[0]), tuple(tables[1])

    @cute.jit
    def lane_parameters(self, g, kidx, active, head_bias: cute.Tensor, gate_bias: cute.Tensor,
                        static_address: cute.Tensor, read_gain: cute.Tensor, forgetting: cute.Tensor):
        """The `LaneParameters` of group `g` at key component `kidx` (inactive lanes: zeros, forgetting 1)."""
        bias_key, bias_query, static_key, static_query = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        gate_bias_key, gate_bias_query, gain, factor = Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0)
        if active:
            bias_key, bias_query = Float32(head_bias[0, g, kidx]), Float32(head_bias[1, g, kidx])
            static_key, static_query = Float32(static_address[0, g, kidx]), Float32(static_address[1, g, kidx])
            gate_bias_key, gate_bias_query = Float32(gate_bias[0, g]), Float32(gate_bias[1, g])
            gain, factor = read_gain[g], forgetting[g]
        return LaneParameters(bias_key, bias_query, static_key, static_query, gate_bias_key, gate_bias_query, gain,
                              factor)

    @cute.jit
    def lane_addresses(self, active, projections: Projections, parameters: LaneParameters):
        """This lane's weighted key and query components and the write scale. The two sites' computations
        are independent, so the compiler interleaves them."""
        K = self.K
        raw_key, raw_query = Float32(0.0), Float32(0.0)
        if active:
            raw_key = Float32(BFloat16(projections.head_key + parameters.bias_key)) + parameters.static_key
            raw_query = Float32(BFloat16(projections.head_query + parameters.bias_query)) + parameters.static_query
        norm_key = cute.arch.warp_reduction(raw_key * raw_key, operator.add, threads_in_group=K)
        norm_query = cute.arch.warp_reduction(raw_query * raw_query, operator.add, threads_in_group=K)
        raw_key = raw_key * cute.rsqrt(norm_key + Float32(self.eps))
        raw_query = raw_query * cute.rsqrt(norm_query + Float32(self.eps))
        address_key, address_query = Float32(BFloat16(raw_key)), Float32(BFloat16(raw_query))
        write_weight = Float32(BFloat16(Float32(2.0) * sigmoid(
            Float32(BFloat16(projections.gate_key + parameters.gate_bias_key)))))
        write_scale = cute.math.sqrt(cutlass.max(write_weight, Float32(self.eps)))
        read_weight = Float32(BFloat16(Float32(2.0) * sigmoid(
            Float32(BFloat16(projections.gate_query + parameters.gate_bias_query)))))
        key_lane, query_lane = Float32(0.0), Float32(0.0)
        if active:
            key_lane = address_key * write_scale
            # The read gain is BF16-rounded already.
            query_lane = Float32(BFloat16(Float32(BFloat16(address_query * read_weight)) * parameters.gain))
        return key_lane, query_lane, write_scale

    @cute.jit
    def produce(self, active, edge: Int32, projections: Projections, parameters: LaneParameters,
                packed: cute.Tensor, history: cute.Tensor, history_offset: Int64, history_stride: Int64,
                gain_history: cute.Tensor, gain_offset: Int64, coeff: cute.Tensor, coeff_offset: Int64):
        """Write one subgroup's coefficient row at `coeff_offset` words of `coeff` and return this lane's
        gain and ``P k`` component for `update_covariance`.

        Every lane of a participating warp calls it, with `active` and the offsets uniform within each
        aligned K-lane subgroup, and inactive subgroups only take part in the shuffles. `packed` holds the
        group's packed covariance and `history` the window's earlier gains, at ``history_offset + j *
        history_stride`` words. This edge's gain goes to `gain_offset` of `gain_history`.
        """
        K = self.K
        lane = cute.arch.thread_idx()[0] % 32
        sub, kidx = lane // K, lane % K
        key_lane, query_lane, scale = self.lane_addresses(active, projections, parameters)
        kk = cute.make_rmem_tensor(K, Float32)
        for k in cutlass.range_constexpr(K):
            kk[k] = cute.arch.shuffle_sync(key_lane, sub * K + k)

        row = cute.make_rmem_tensor(K, Float32)
        row.fill(0.0)
        pk_lane = Float32(0.0)
        if active:
            row_bias = kidx * K - ((kidx * (kidx + 1)) >> 1)
            for k in cutlass.range_constexpr(K):
                offset = row_bias + k
                if kidx > k:
                    offset = (k * K - k * (k + 1) // 2) + kidx
                row[k] = Float32(packed[offset])
            for k in cutlass.range_constexpr(K):
                pk_lane += row[k] * kk[k]
        denominator = parameters.forgetting + cute.arch.warp_reduction(
            key_lane * pk_lane, operator.add, threads_in_group=K)
        gain_lane = pk_lane / denominator
        if cutlass.const_expr(not self.commit and not self.terminal):
            if active:
                gain_row = cute.make_tensor(gain_history.iterator + gain_offset, cute.make_layout(K))
                gain_row[kidx] = gain_lane
        gamma = cute.arch.warp_reduction(gain_lane * query_lane, operator.add, threads_in_group=K)
        coeff_row = cute.make_tensor(coeff.iterator + coeff_offset, cute.make_layout(self.coeff_width))
        if active:
            coeff_row[kidx] = key_lane
            coeff_row[K + kidx] = query_lane
            if cutlass.const_expr(self.commit):
                coeff_row[2 * K + kidx] = gain_lane
            if kidx == 0:
                coeff_row[self.scalar_offset] = scale
                coeff_row[self.scalar_offset + 1] = gamma

        qq = cute.make_rmem_tensor(K, Float32)
        for k in cutlass.range_constexpr(K):
            qq[k] = cute.arch.shuffle_sync(query_lane, sub * K + k)

        # Each lane takes one earlier edge of the window per pass and sums its dot products over K in order.
        for jt in cutlass.range_constexpr(cute.ceil_div(self.past, K)):
            history_edge = jt * K + kidx
            a, b = Float32(0.0), Float32(0.0)
            if active & (history_edge < edge) & (history_edge < self.past):
                gj = cute.make_rmem_tensor(K, Float32)
                history_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32,
                                                   num_bits_per_copy=128)
                for history_part in cutlass.range_constexpr(K // 4):
                    offset = history_offset + Int64(history_edge) * history_stride + 4 * history_part
                    history_source = cute.make_tensor(
                        history.iterator + cute.assume(offset, divby=4), cute.make_layout(4))
                    history_values = cute.make_rmem_tensor(4, Float32)
                    cute.copy(history_copy, history_source, history_values)
                    for component in cutlass.range_constexpr(4):
                        gj[4 * history_part + component] = history_values[component]
                for k in cutlass.range_constexpr(K):
                    a += gj[k] * kk[k]
                    b += gj[k] * qq[k]
            if active & (history_edge < self.past):
                coeff_row[self.scalar_offset + 2 + history_edge] = a
                coeff_row[self.scalar_offset + 2 + self.past + history_edge] = b
        return gain_lane, pk_lane

    @cute.jit
    def update_covariance(self, active, gain_lane: Float32, pk_lane: Float32, forgetting: Float32,
                          packed: cute.Tensor, packed_out: cute.Tensor):
        """The covariance update ``(P - gain pk^T) / forgetting`` of the subgroup's group, from the gain and
        ``P k`` of `produce`. The lanes that called `produce` call it afterwards. `packed` and `packed_out`
        may be the same memory."""
        K = self.K
        lane = cute.arch.thread_idx()[0] % 32
        sub, kidx = lane // K, lane % K
        # Every lane must have read its row of the old covariance before any lane writes the update.
        cute.arch.sync_warp()
        # One register buffer holds the old covariance entries and then the numerators of the update.
        updates = cute.make_rmem_tensor(self.update_parts, Float32)
        updates.fill(0.0)
        for part in cutlass.range_constexpr(self.update_parts):
            p = part * K + kidx
            if active & (p < self.triangle):
                updates[part] = Float32(packed[p])
        for part in cutlass.range_constexpr(self.update_parts):
            row_word = Int32(self.update_rows[part])
            column_word = Int32(self.update_columns[part])
            i = Int32((row_word >> (4 * kidx)) & 15)
            j = Int32((column_word >> (4 * kidx)) & 15)
            # Lanes past the triangle still take part in both shuffles. The parts write disjoint entries.
            gain_i = cute.arch.shuffle_sync(gain_lane, sub * K + i)
            pk_j = cute.arch.shuffle_sync(pk_lane, sub * K + j)
            p = part * K + kidx
            if active & (p < self.triangle):
                # One fused multiply-add rounds the numerator once, as the fast backend does.
                updates[part] = cute.arch.inline_ptx(
                    "fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                    write_only_types=[Float32],
                    read_only_args=[-gain_i, pk_j, updates[part]],
                )
        for part in cutlass.range_constexpr(self.update_parts):
            p = part * K + kidx
            if active & (p < self.triangle):
                packed_out[p] = BFloat16(updates[part] / forgetting)
