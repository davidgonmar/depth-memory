"""The per-row stages both edge kernels share, on CTA clusters.

A cluster of C CTAs takes T tokens. CTA c handles groups ``[c G/C, (c + 1) G/C)`` of them, their
``W = G/C * D`` stream columns and their (token, group) rows, token-major. Bulk copies bring the rows'
dense state into shared memory: the packed covariance, M, the value, the stream slice and the window
history of residuals and gains. One K-lane subgroup per row then forms both addresses, the gain and the
coefficient row (`addresses.Addresses`). `_dense_step` forms the residual and the read per (row, 8
columns) and writes the residual history. `_branch_norm` sends each token's sum of squares of ``stream +
read`` to every CTA of the cluster through distributed shared memory, updates the covariance and, at a
prefill commit past the first window, writes the new M (`_commit`) while the sums travel, and writes the
CTA's slice of the branch RMSNorm.

`edge_state.EdgeState` (prefill) reads the head and gate projections and the updated stream from global
memory. `edge_decode.EdgeDecode` computes them itself.
"""

import math
from typing import NamedTuple

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute

from .addresses import Addresses, LaneParameters, packed_width
from .copies import bulk_copy, fetch_bf16, load_bf16, round_up, store

THREADS = 256
WARPS = THREADS // 32
VECTOR = 8                    # BF16 elements per 128-bit access
MAX_CLUSTER = 16


@cute.jit
def _push(dst: cute.Tensor, dst_offset, value: Float32, bar: cute.Pointer, peer: Int32):
    """Store `value` at `dst_offset` of `dst` in the shared memory of cluster CTA `peer` (``st.async``). The
    four bytes complete on that CTA's copy of `bar`."""
    remote = cute.arch.inline_ptx("mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};", write_only_types=[Int32],
                                  read_only_args=[Int32((dst.iterator + dst_offset).toint()), peer])
    remote_bar = cute.arch.inline_ptx("mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};", write_only_types=[Int32],
                                      read_only_args=[Int32(bar.toint()), peer])
    cute.arch.inline_ptx("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [{$r0}], {$r1}, [{$r2}];",
                         read_only_args=[remote, value, remote_bar])


class Place(NamedTuple):
    """Where a thread works: its lane and warp, the cluster's tokens and the CTA's groups."""

    tid: Int32
    lane: Int32
    warp: Int32
    rank: Int32                   # the CTA's index in its cluster
    token0: Int64                 # the cluster's first token
    live: Int32                   # the cluster's tokens that exist (the last cluster may be partial)
    first_group: Int32            # the CTA's first group
    flat0: Int64                  # (token, group) index of the CTA's first row
    groups: Int64                 # (token, group) rows of the whole forward


class Rows(NamedTuple):
    """The shared memory of the CTA's rows."""

    norm_gain: cute.Tensor        # [W] BF16 branch RMSNorm weight of the CTA's slice (none with `commit_pass`)
    packed: cute.Tensor           # [P, packed width] BF16 covariance of every row
    dense: cute.Tensor            # BF16 M rows, value, stream slice and residual history
    coefficients: cute.Tensor     # FP32 coefficient rows, then the gain history
    norm_parts: cute.Tensor       # [T, W / 8] FP32 sums of squares of the branch input per 8 columns
    norms: cute.Tensor            # [C, T] FP32 per-CTA sums of squares, from every CTA of the cluster
    address_bar: cute.Pointer     # what the addresses read (in decode, the state barrier)
    state_bar: cute.Pointer       # the dense state
    norm_bar: cute.Pointer        # sums of squares from every CTA of the cluster


class EdgeRows:
    """Sizes, shared memory and the per-row stages of both edge kernels. A subclass sets `prefill`. The
    prefill kernel copies the stream slice from global memory, loads the norm weight with the dense state
    and forms the addresses while the dense state arrives. The decode kernel writes the stream slice itself
    and loads the norm weight with its parameters, before the grid-dependency wait."""

    prefill: bool

    def __init__(self, E, G, K, D, *, tokens, cluster, key_lanes, first_window, commit, terminal, norm, norm_eps,
                 eps, history=None):
        T, C = tokens, cluster
        if not 1 <= C <= MAX_CLUSTER or G % C:
            raise ValueError(f"a cluster of {C} CTAs must divide the {G} groups and hold at most {MAX_CLUSTER}")
        if D % 64:
            raise ValueError(f"value width {D}: a multiple of 64 is required")
        if key_lanes < 1 or K % key_lanes or key_lanes & (key_lanes - 1):
            raise ValueError(f"{key_lanes} key lanes: a power of two dividing K={K} is required")
        if T < 1 or T * (G // C) * K > THREADS:
            raise ValueError(f"{T} tokens on {G // C} groups per CTA: every (token, group) row of a CTA needs its "
                             f"own K-lane subgroup of the {THREADS} threads")
        if norm and (not math.isfinite(norm_eps) or norm_eps <= 0):
            raise ValueError("norm_eps must be positive and finite")
        self.addresses = Addresses(E, K, commit=commit, terminal=terminal, eps=eps)
        self.E, self.G, self.K, self.D = E, G, K, D
        self.tokens, self.cluster, self.local_groups, self.key_lanes = T, C, G // C, key_lanes
        self.width, self.model_dim, self.pairs = (G // C) * D, G * D, T * (G // C)
        self.first_window, self.commit, self.terminal = bool(first_window), bool(commit), bool(terminal)
        self.norm, self.norm_eps = bool(norm), float(norm_eps)
        # Past the first window, the prefill kernel writes the new M in a pass of its own (`_commit`) and reads
        # the norm weight from global memory after it. Everywhere else each item writes its new M right after
        # its residual, and the norm weight comes into shared memory with the state or the parameters.
        self.commit_pass = self.commit and self.prefill and not self.first_window
        self.past = E - 1
        # Shared memory holds `history` rows of the window history, and an edge reads the first `edge` of
        # them. The prefill kernel knows the edge's place in the window and holds just those rows. The decode
        # kernel has room for the whole window.
        self.history = self.past if history is None else history
        if not 0 <= self.history <= self.past:
            raise ValueError(f"{history} history rows for a window of {E}")
        self.pieces = self.width // VECTOR                 # 8-column pieces of the CTA's slice
        self.covariance_width = packed_width(K)
        # The dense region (BF16) holds M [P, K, D] (none in the first window), the value and the stream
        # [P, D] and the window history of residuals [history, P, D]. The coefficient region (FP32) holds the
        # coefficient rows, then the window history of gains [history, P, K]. P is the CTA's rows, token-major.
        P = self.pairs
        self.value_offset = 0 if self.first_window else P * K * D
        self.stream_offset = self.value_offset + P * D
        self.residual_offset = self.stream_offset + P * D
        self.dense_elements = self.residual_offset + self.history * P * D
        self.gain_offset = P * self.addresses.coeff_width
        self.coefficient_words = self.gain_offset + max(1, self.history) * P * K

    def _row_regions(self):
        """(bytes, alignment) of the row allocations, in `_allocate_rows`'s order."""
        norm_gain = () if self.commit_pass else ((2 * self.width, 16),)
        return norm_gain + ((2 * self.pairs * self.covariance_width, 16), (2 * self.dense_elements, 128),
                            (4 * self.coefficient_words, 16), (4 * self.tokens * self.pieces, 16),
                            (4 * self.cluster * self.tokens, 16), (24, 8))

    @staticmethod
    def _total(regions):
        total = 0
        for size, alignment in regions:
            total = round_up(total, alignment) + size
        return total + 128                   # the dynamic base is only guaranteed 16-byte aligned

    @cute.jit
    def _allocate_rows(self, smem):
        norm_gain = None
        if cutlass.const_expr(not self.commit_pass):
            norm_gain = smem.allocate_tensor(BFloat16, cute.make_layout(self.width), byte_alignment=16)
        packed = smem.allocate_tensor(BFloat16, cute.make_layout(self.pairs * self.covariance_width),
                                      byte_alignment=16)
        dense = smem.allocate_tensor(BFloat16, cute.make_layout(self.dense_elements), byte_alignment=128)
        coefficients = smem.allocate_tensor(Float32, cute.make_layout(self.coefficient_words), byte_alignment=16)
        norm_parts = smem.allocate_tensor(Float32, cute.make_layout(self.tokens * self.pieces), byte_alignment=16)
        norms = smem.allocate_tensor(Float32, cute.make_layout(self.cluster * self.tokens), byte_alignment=16)
        bars = smem.allocate_array(Int64, 3, byte_alignment=8)
        address_bar = bars
        if cutlass.const_expr(self.prefill):
            address_bar = bars + 2
        return Rows(norm_gain, packed, dense, coefficients, norm_parts, norms, address_bar, bars, bars + 1)

    @cute.jit
    def _place(self, n):
        tid = cute.arch.thread_idx()[0]
        # The grid of clusters is one-dimensional because its y extent is limited to 65535 blocks.
        block = Int64(cute.arch.block_idx()[0])
        rank = Int32(block % self.cluster)
        token0 = block // self.cluster * self.tokens
        live = Int32(cutlass.min(Int64(self.tokens), Int64(n) - token0))
        first_group = rank * self.local_groups
        return Place(tid, tid % 32, cute.arch.make_warp_uniform(tid // 32), rank, token0, live, first_group,
                     token0 * self.G + first_group, Int64(n) * self.G)

    @cute.jit
    def _init_row_barriers(self, place: Place, rows: Rows):
        if place.tid == 0:
            cute.arch.mbarrier_init(rows.state_bar, 1)
            cute.arch.mbarrier_init(rows.norm_bar, 1)
            if cutlass.const_expr(self.prefill):
                cute.arch.mbarrier_init(rows.address_bar, 1)

    @cute.jit
    def _request_state(self, place: Place, rows: Rows, stream: cute.Tensor, value: cute.Tensor,
                       covariance: cute.Tensor, memory: cute.Tensor, residual_history: cute.Tensor,
                       gain_history: cute.Tensor, norm_weight: cute.Tensor, edge: Int32):
        """Warp 1 requests the rows' state: first what the addresses read (the packed covariance and the gain
        history, on `address_bar`), then the dense state (on `state_bar`, in prefill with the norm weight of
        the CTA's slice). Prefill starts the addresses while the dense state arrives. Decode forms its hidden
        units meanwhile, and both come on its state barrier."""
        K = self.K
        G, D = self.G, self.D
        GC, PW, W, P = self.local_groups, self.covariance_width, self.width, self.pairs
        if place.warp == 1:
            if place.lane == 0:
                address_bytes = place.live * (Int32(2 * GC * PW) + edge * Int32(4 * GC * K))
                per_token = edge * Int32(2 * W)
                if cutlass.const_expr(not self.first_window):
                    per_token += Int32(2 * GC * K * D)
                per_token += Int32(2 * W)
                if cutlass.const_expr(self.prefill):
                    per_token += Int32(2 * W)
                total = place.live * per_token
                if cutlass.const_expr(self.norm and self.prefill and not self.commit_pass):
                    total += Int32(2 * W)
                if cutlass.const_expr(self.prefill):
                    cute.arch.mbarrier_arrive_and_expect_tx(rows.address_bar, address_bytes)
                    cute.arch.mbarrier_arrive_and_expect_tx(rows.state_bar, total)
                else:
                    cute.arch.mbarrier_arrive_and_expect_tx(rows.state_bar, address_bytes + total)
                for t in cutlass.range_constexpr(self.tokens):
                    if t < place.live:
                        flat = place.flat0 + t * G
                        row = t * GC
                        bulk_copy(rows.packed, row * PW, covariance, flat * PW, Int32(2 * GC * PW), rows.address_bar)
                        for j in cutlass.range_constexpr(self.history):
                            if j < edge:
                                bulk_copy(rows.coefficients, self.gain_offset + j * P * K + row * K, gain_history,
                                          (Int64(j) * place.groups + flat) * K, Int32(4 * GC * K), rows.address_bar)
                if cutlass.const_expr(self.norm and self.prefill and not self.commit_pass):
                    bulk_copy(rows.norm_gain, 0, norm_weight, Int64(place.rank) * W, Int32(2 * W), rows.state_bar)
                for t in cutlass.range_constexpr(self.tokens):
                    if t < place.live:
                        flat = place.flat0 + t * G
                        row = t * GC
                        column = (place.token0 + t) * self.model_dim + Int64(place.rank) * W
                        if cutlass.const_expr(not self.first_window):
                            bulk_copy(rows.dense, row * K * D, memory, flat * K * D, Int32(2 * GC * K * D),
                                      rows.state_bar)
                        if cutlass.const_expr(self.prefill):
                            bulk_copy(rows.dense, self.stream_offset + row * D, stream, column, Int32(2 * W),
                                      rows.state_bar)
                        bulk_copy(rows.dense, self.value_offset + row * D, value, flat * D, Int32(2 * W),
                                  rows.state_bar)
                        for j in cutlass.range_constexpr(self.history):
                            if j < edge:
                                bulk_copy(rows.dense, self.residual_offset + j * P * D + row * D, residual_history,
                                          (Int64(j) * place.groups + flat) * D, Int32(2 * W), rows.state_bar)

    @cute.jit
    def _addresses(self, place: Place, rows: Rows, parameters: LaneParameters, projections, pair, pair_active,
                   gain_history: cute.Tensor, edge: Int32):
        """Addresses and coefficient rows, one K-lane subgroup per row. Returns the lane's gain and ``P k``
        component for the covariance update."""
        K = self.K
        gain_lane, pk_lane = Float32(0.0), Float32(0.0)
        if place.warp * 32 < self.pairs * K:
            pair_flat = place.flat0 + (pair // self.local_groups) * self.G + pair % self.local_groups
            packed = cute.make_tensor(rows.packed.iterator + pair * self.covariance_width,
                                      cute.make_layout(self.covariance_width))
            gain_lane, pk_lane = self.addresses.produce(
                pair_active, edge, projections, parameters, packed, rows.coefficients,
                Int64(self.gain_offset + pair * K), Int64(self.pairs * K), gain_history,
                (Int64(edge) * place.groups + pair_flat) * K, rows.coefficients,
                Int64(pair * self.addresses.coeff_width))
        return gain_lane, pk_lane

    @cute.jit
    def _dense_step(self, place: Place, rows: Rows, residual_history: cute.Tensor, memory: cute.Tensor,
                    output: cute.Tensor, edge: Int32):
        """The rows' dense step over (row, 8 columns) items of `key_lanes` lanes. Each lane contracts K /
        key_lanes keys of M in turn, and a butterfly over the item's lanes completes the projection ``M k``
        and the read ``M q``. Then come the residual ``scale * v - M k - sum_j alpha_j r_j`` rounded to BF16,
        the read ``M q + sum_j beta_j r_j + gamma * residual``, the residual history and the branch input
        ``stream + BF16(read)`` and, at a commit without `commit_pass`, the new M. Returns the lane's residual
        of every item it took, for `_commit`."""
        K = self.K
        D, G, P = self.D, self.G, self.pairs
        GC, KL = self.local_groups, self.key_lanes
        KPL: cutlass.Constexpr = K // self.key_lanes
        DP: cutlass.Constexpr = D // VECTOR
        CW, SO = self.addresses.coeff_width, self.addresses.scalar_offset
        V_OFF, X_OFF, R_OFF = self.value_offset, self.stream_offset, self.residual_offset
        residuals = []
        for turn in cutlass.range_constexpr(cute.ceil_div(P * DP * KL, THREADS)):
            slot = turn * THREADS + place.tid
            item, key_lane = slot // KL, slot % KL
            row = cutlass.min(item // DP, P - 1)
            d = (item % DP) * VECTOR
            row_live = (item < P * DP) & (row // GC < place.live)
            c_at = row * CW
            flat = place.flat0 + (row // GC) * G + row % GC
            projection, read = cute.make_rmem_tensor(VECTOR, Float32), cute.make_rmem_tensor(VECTOR, Float32)
            projection.fill(0.0)
            read.fill(0.0)
            residual = cute.make_rmem_tensor(VECTOR, Float32)
            residuals.append(residual)
            if cutlass.const_expr(not self.first_window):
                if row_live:
                    for key_step in cutlass.range_constexpr(KPL):
                        k = key_lane * KPL + key_step
                        key, query = rows.coefficients[c_at + k], rows.coefficients[c_at + K + k]
                        m_row = load_bf16(rows.dense, (row * K + k) * D + d, VECTOR)
                        for v in cutlass.range_constexpr(VECTOR):
                            projection[v] = m_row[v] * key + projection[v]
                            read[v] = m_row[v] * query + read[v]
                for bit in cutlass.range_constexpr(int(math.log2(KL))):
                    for v in cutlass.range_constexpr(VECTOR):
                        projection[v] = projection[v] + cute.arch.shuffle_sync_bfly(projection[v], 1 << bit)
                        read[v] = read[v] + cute.arch.shuffle_sync_bfly(read[v], 1 << bit)
            if row_live:
                scale, gamma = rows.coefficients[c_at + SO], rows.coefficients[c_at + SO + 1]
                value = load_bf16(rows.dense, V_OFF + row * D + d, VECTOR)
                for v in cutlass.range_constexpr(VECTOR):
                    residual[v] = value[v] * scale - projection[v]
                for j in cutlass.range_constexpr(self.history):
                    if j < edge:
                        alpha_j = rows.coefficients[c_at + SO + 2 + j]
                        beta_j = rows.coefficients[c_at + SO + 2 + self.past + j]
                        history = load_bf16(rows.dense, R_OFF + j * P * D + row * D + d, VECTOR)
                        for v in cutlass.range_constexpr(VECTOR):
                            residual[v] = history[v] * -alpha_j + residual[v]
                            read[v] = history[v] * beta_j + read[v]
                rounded = cute.make_rmem_tensor(VECTOR, BFloat16)
                rounded.store(residual.load().to(BFloat16))
                residual.store(rounded.load().to(Float32))
                for v in cutlass.range_constexpr(VECTOR):
                    read[v] = gamma * residual[v] + read[v]
                if key_lane == 0:
                    if cutlass.const_expr(not self.commit and not self.terminal):
                        store(residual_history, (Int64(edge) * place.groups + flat) * D + d, residual, VECTOR)
                    stream = load_bf16(rows.dense, X_OFF + row * D + d, VECTOR)
                    rounded.store(read.load().to(BFloat16))
                    read.store(rounded.load().to(Float32))
                    for v in cutlass.range_constexpr(VECTOR):
                        read[v] = stream[v] + read[v]
                    if cutlass.const_expr(self.norm):
                        store(rows.dense, X_OFF + row * D + d, read, VECTOR)
                        # This piece's sum of squares of the BF16 branch input, for the RMSNorm.
                        rounded.store(read.load().to(BFloat16))
                        squares = Float32(0.0)
                        for v in cutlass.range_constexpr(VECTOR):
                            squares += Float32(rounded[v]) * Float32(rounded[v])
                        rows.norm_parts[(row // GC) * self.pieces + (row % GC) * DP + item % DP] = squares
                    else:
                        store(output, flat * D + d, read, VECTOR)
                if cutlass.const_expr(self.commit and not self.commit_pass):
                    self._commit_item(rows, memory, row, d, flat, key_lane, residual, edge)
        return residuals

    @cute.jit
    def _commit(self, place: Place, rows: Rows, memory: cute.Tensor, residuals, edge: Int32):
        """The new M of the dense step's items in a pass of its own after a CTA barrier, so M and the history
        rows come again from shared memory instead of being held in registers through the dense step. The
        first window has no M to read again, and the decode kernel runs too few CTAs to be limited by its
        registers, so both write each item's new M in the dense step."""
        D, P = self.D, self.pairs
        GC, KL = self.local_groups, self.key_lanes
        DP: cutlass.Constexpr = D // VECTOR
        for turn in cutlass.range_constexpr(cute.ceil_div(P * DP * KL, THREADS)):
            slot = turn * THREADS + place.tid
            item, key_lane = slot // KL, slot % KL
            row = cutlass.min(item // DP, P - 1)
            flat = place.flat0 + (row // GC) * self.G + row % GC
            if (item < P * DP) & (row // GC < place.live):
                self._commit_item(rows, memory, row, (item % DP) * VECTOR, flat, key_lane, residuals[turn], edge)

    @cute.jit
    def _commit_item(self, rows: Rows, memory: cute.Tensor, row, d, flat, key_lane, residual: cute.Tensor,
                     edge: Int32):
        """The new M ``M + sum_j gain_j r_j + gain * residual`` of one (row, 8 columns) item at the lane's
        keys."""
        K = self.K
        D, P = self.D, self.pairs
        KPL: cutlass.Constexpr = K // self.key_lanes
        R_OFF, G_OFF = self.residual_offset, self.gain_offset
        for key_step in cutlass.range_constexpr(KPL):
            k = key_lane * KPL + key_step
            column = cute.make_rmem_tensor(VECTOR, Float32)
            column.fill(0.0)
            if cutlass.const_expr(not self.first_window):
                m_row = load_bf16(rows.dense, (row * K + k) * D + d, VECTOR)
                for v in cutlass.range_constexpr(VECTOR):
                    column[v] = m_row[v]
            for j in cutlass.range_constexpr(self.history):
                if j < edge:
                    history_gain = rows.coefficients[G_OFF + j * P * K + row * K + k]
                    history = load_bf16(rows.dense, R_OFF + j * P * D + row * D + d, VECTOR)
                    for v in cutlass.range_constexpr(VECTOR):
                        column[v] = history_gain * history[v] + column[v]
            commit_gain = rows.coefficients[row * self.addresses.coeff_width + 2 * K + k]
            for v in cutlass.range_constexpr(VECTOR):
                column[v] = commit_gain * residual[v] + column[v]
            store(memory, (flat * K + k) * D + d, column, VECTOR)

    @cute.jit
    def _norm_weight(self, place: Place, norm_weight: cute.Tensor):
        """The norm weight pieces of the CTA's slice, one per thread and output step, in BF16 registers, read
        from global memory (every CTA finds the weight in L2). A commit pass requests them once its registers
        are free, and without the weight's shared memory the commit of a 40-group model fits three CTAs per
        SM instead of two."""
        gains = []
        for step in cutlass.range_constexpr(cute.ceil_div(self.tokens * self.pieces, THREADS)):
            piece = (step * THREADS + place.tid) % self.pieces
            gains.append(fetch_bf16(norm_weight, Int64(place.rank) * self.width + piece * VECTOR, VECTOR))
        return gains

    @cute.jit
    def _branch_norm(self, place: Place, rows: Rows, parameters: LaneParameters, pair, pair_active,
                     gain_lane: Float32, pk_lane: Float32, covariance: cute.Tensor, output: cute.Tensor,
                     memory: cute.Tensor, norm_weight: cute.Tensor, residuals, edge: Int32):
        """The branch RMSNorm over the cluster's slices of each token, with the covariance update and, with
        `commit_pass`, the new M running while the sums of squares travel."""
        K = self.K
        T, C, PIECES = self.tokens, self.cluster, self.pieces
        if cutlass.const_expr(self.norm):
            cute.arch.sync_threads()
            if place.tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(rows.norm_bar, Int32(4 * C * T))
            for step in cutlass.range_constexpr(cute.ceil_div(T * C, THREADS)):
                index = step * THREADS + place.tid
                if index < T * C:
                    token = index // C
                    total = Float32(0.0)
                    for part in cutlass.range(0, PIECES, 1, unroll_full=True):
                        total += rows.norm_parts[token * PIECES + part]
                    _push(rows.norms, place.rank * T + token, total, rows.norm_bar, index % C)
        if cutlass.const_expr(not self.terminal):
            if place.warp * 32 < self.pairs * K:
                self.addresses.update_covariance(
                    pair_active, gain_lane, pk_lane, parameters.forgetting,
                    cute.make_tensor(rows.packed.iterator + pair * self.covariance_width,
                                     cute.make_layout(self.covariance_width)),
                    cute.make_tensor(covariance.iterator + (place.flat0 + (pair // self.local_groups) * self.G
                                                            + pair % self.local_groups) * self.covariance_width,
                                     cute.make_layout(self.covariance_width)))
        gains = []
        if cutlass.const_expr(self.commit_pass):
            self._commit(place, rows, memory, residuals, edge)
            gains = self._norm_weight(place, norm_weight)
        if cutlass.const_expr(self.norm):
            cute.arch.mbarrier_wait(rows.norm_bar, 0)
            for step in cutlass.range_constexpr(cute.ceil_div(T * PIECES, THREADS)):
                index = step * THREADS + place.tid
                token, piece = index // PIECES, index % PIECES
                if (index < T * PIECES) & (token < place.live):
                    total = Float32(0.0)
                    for c in cutlass.range_constexpr(C):
                        total += rows.norms[c * T + token]
                    scale = cute.math.rsqrt(total / Float32(self.model_dim) + Float32(self.norm_eps))
                    at = (token * self.local_groups + (piece * VECTOR) // self.D) * self.D + (piece * VECTOR) % self.D
                    joined = load_bf16(rows.dense, self.stream_offset + at, VECTOR)
                    gain = cute.make_rmem_tensor(VECTOR, Float32)
                    if cutlass.const_expr(self.commit_pass):
                        gain.store(gains[step].load().to(Float32))
                    else:
                        gain.store(load_bf16(rows.norm_gain, piece * VECTOR, VECTOR).load())
                    result = cute.make_rmem_tensor(VECTOR, Float32)
                    for i in cutlass.range_constexpr(VECTOR):
                        result[i] = (joined[i] * scale) * gain[i]
                    store(output, (place.token0 + token) * self.model_dim + Int64(place.rank) * self.width
                           + piece * VECTOR, result, VECTOR)
