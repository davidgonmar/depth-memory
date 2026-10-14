"""The forward of the windowed state: one launch per edge.

A producer warp copies batches of consecutive groups into two shared stages with bulk copies
(full and empty mbarriers). Consumer warps update the packed covariance, form the residual
against the window's start state and the window's earlier residuals, read, and at the window's
last edge commit the window's residuals into M. ``saved`` receives the packed BF16 covariance of
every edge from before its update (rows of K(K+1)/2 entries).

The kernel takes the raw per-edge inputs: each consumer warp forms its groups' keys as
raw_key * scale (one FP32 product) in the staged copy and writes the key, query, scale and
forgetting tapes. A forgetting operand with a zero token stride (one factor per group) is read
from global memory instead of being staged.

The arithmetic is FP32 with a fixed summation order over the K lanes and a fixed column update
order. Residuals are rounded to BF16 before they are reused; the reads and the committed M
are stored in BF16. The running covariance (16-byte padded rows) is stored in BF16 or, with
``running_covariance_dtype="float32"``, in FP32; the saved covariance is BF16 either way.
"""

import math
import operator

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.utils import SmemAllocator

from .window_end import (
    _bulk_g2s,
    _load_dense,
    _load_fp32_vector,
    _round_up,
    _stage_span,
    _store_vector,
    _symmetric_offset,
)


# The dtypes the running covariance may be stored in.
COVARIANCE_TYPES = {"bfloat16": BFloat16, "float32": Float32}


class Forward:
    def __init__(self, E, N, G, K, D, *, sms, smem_budget, consumers=8, batch=16, blocks_per_sm=1,
                 lane_width=8, first_window=False, running_covariance_dtype="bfloat16"):
        self.E, self.N, self.G, self.K, self.D = E, N, G, K, D
        self.static_edge = None
        self.lane_width = lane_width
        # In the first window the start state is zero: M is not staged or read, only committed.
        self.first_window = first_window
        self.covariance_type = COVARIANCE_TYPES[running_covariance_dtype]
        self.covariance_bytes = self.covariance_type.width // 8
        # The stages hold the rows of at most E - 1 earlier edges of the window.
        self.past = E - 1
        self.gain_tape_shape = (E, N, G, K)
        self.residual_tape_shape = (E, N, G, D)
        self.denominator_tape_shape = (E, N, G)
        groups = N * G
        # Two shared stages: the producer fills one while the consumers compute the other.
        self.stages = 2
        # Fit the schedule into the shared-memory budget: the largest batch (then consumer
        # count) whose two stages fit.
        while True:
            self.consumers, self.batch = consumers, batch
            self.per_warp = batch // consumers
            self._layout()
            if self.shared_bytes <= smem_budget:
                break
            if batch > consumers:
                batch //= 2
            elif consumers > 1:
                consumers //= 2
                batch //= 2
            else:
                raise ValueError("one group of this shape does not fit two shared stages")
        if self.batch % 4 != 0:
            raise ValueError("the forward batch must be a multiple of 4 groups")
        self.warps = self.consumers + 1
        self.batches = (groups + self.batch - 1) // self.batch
        self.blocks = min(self.batches, blocks_per_sm * sms)
        # Groups per pass of the K-lane phase of a consumer warp.
        self.pack = min(self.per_warp, 32 // K)
        self.compact_passes = (self.per_warp + self.pack - 1) // self.pack

    def _layout(self):
        K, D, batch = self.K, self.D, self.batch
        # BF16 part of a stage: M blocks, value rows and the earlier residual rows of the window;
        # every region starts 16-byte aligned so bulk copies can target it.
        self.m_off = 0
        self.v_off = 0 if self.first_window else _round_up(batch * K * D, 8)
        self.r_stride = _round_up(batch * D, 8)
        self.r_off = self.v_off + self.r_stride
        self.dense_elems = self.r_off + self.past * self.r_stride
        # FP32 part of a stage: keys, queries, the earlier gains of the window, forgetting factors
        # and write scales.
        self.key_off = 0
        self.g_stride = _round_up(batch * K, 4)
        self.q_off = self.key_off + self.g_stride
        self.g_off = self.q_off + self.g_stride
        self.lam_off = self.g_off + self.past * self.g_stride
        self.scale_off = self.lam_off + _round_up(batch, 4)
        self.compact_words = self.scale_off + _round_up(batch, 4)
        # Packed P: the upper triangle (K(K+1)/2 entries per group, upper-row-major) padded to 16
        # bytes; the lower triangle is read from the upper one. Its arithmetic is FP32. A BF16 P is
        # staged after the BF16 rows and rounds once per edge; an FP32 P is staged after the FP32
        # words.
        self.pw16 = _round_up(K * (K + 1) // 2, 16 // self.covariance_bytes)
        if self.covariance_type is BFloat16:
            self.p_off = self.dense_elems
            self.dense_elems += batch * self.pw16
        else:
            self.p_off = self.compact_words
            self.compact_words += batch * self.pw16
        self.stage_bytes = 2 * self.dense_elems + 4 * self.compact_words
        self.shared_bytes = self.stages * self.stage_bytes + 16 * self.stages + 256

    @cute.jit
    def __call__(
        self, m: cute.Tensor, covariance: cute.Tensor,
        rawkey: cute.Tensor, query: cute.Tensor, rawvalue: cute.Tensor,
        write_scale: cute.Tensor, forgetting: cute.Tensor,
        gain_tape: cute.Tensor, residual_tape: cute.Tensor,
        denominator_tape: cute.Tensor, read_out: cute.Tensor, saved,
        key_tape: cute.Tensor, query_tape: cute.Tensor, scale_tape: cute.Tensor,
        forgetting_tape: cute.Tensor, edge: Int32, stream: cuda.CUstream,
    ):
        for tensor, shape, dtype in (
            (m, (self.N, self.G, self.K, self.D), BFloat16),
            (covariance, (self.N, self.G, self.pw16), self.covariance_type),
            (rawkey, (self.N, self.G, self.K), Float32),
            (query, (self.N, self.G, self.K), Float32),
            (rawvalue, (self.N, self.G, self.D), BFloat16),
            (write_scale, (self.N, self.G), Float32),
            (gain_tape, self.gain_tape_shape, Float32),
            (residual_tape, self.residual_tape_shape, BFloat16),
            (denominator_tape, self.denominator_tape_shape, Float32),
            (read_out, (self.N, self.G, self.D), BFloat16),
            (saved, (self.E, self.N, self.G, packed_size(self.K)), BFloat16),
            (key_tape, (self.N, self.G, self.K), Float32),
            (query_tape, (self.N, self.G, self.K), Float32),
            (scale_tape, (self.N, self.G), Float32),
            (forgetting_tape, (self.N, self.G), Float32),
        ):
            assert tensor.element_type == dtype
            assert tuple(tensor.shape) == shape
            stride = 1
            for axis in cutlass.range_constexpr(len(shape) - 1, -1, -1):
                assert tensor.stride[axis] == stride
                stride *= shape[axis]
        # Forgetting is staged per group, or broadcast over tokens.
        assert forgetting.element_type == Float32
        assert tuple(forgetting.shape) == (self.N, self.G)
        assert forgetting.stride[1] == 1
        assert forgetting.stride[0] in (self.G, 0)
        self.kernel(
            m, covariance, rawkey, query, rawvalue, write_scale, forgetting,
            gain_tape, residual_tape, denominator_tape, read_out, saved,
            key_tape, query_tape, scale_tape, forgetting_tape, edge,
        ).launch(grid=(self.blocks, 1, 1), block=(32 * self.warps, 1, 1),
                 smem=self.shared_bytes, stream=stream)

    @cute.jit
    def issue(self, batch_index, dense: cute.Tensor, compact: cute.Tensor, doff, coff,
              bar: cute.Pointer, lane, edge: Int32,
              m: cute.Tensor, covariance: cute.Tensor, rawkey: cute.Tensor,
              query: cute.Tensor, rawvalue: cute.Tensor, write_scale: cute.Tensor,
              forgetting: cute.Tensor, gain_tape: cute.Tensor, residual_tape: cute.Tensor):
        """Producer warp: stage one batch at (doff, coff) and complete it on ``bar``.

        Lane 0 issues one bulk copy per block. A short last batch copies its per-group scalars
        per lane instead, and the warp waits for those copies before it arrives, so the
        consumers' one barrier wait covers the whole stage.
        """
        K, D = self.K, self.D
        groups = self.N * self.G
        g0 = Int64(batch_index) * self.batch
        count = cutlass.min(Int32(self.batch), Int32(groups - g0))
        full = count == self.batch
        tail = count < self.batch
        staged_forgetting: cutlass.Constexpr = forgetting.stride[0] != 0
        p_stage, p_at = self.covariance_stage(dense, compact, doff, coff)
        if tail:
            if cutlass.const_expr(staged_forgetting):
                _stage_span(forgetting, g0, count, compact, coff + self.lam_off, self.batch, 1, lane)
            _stage_span(write_scale, g0, count, compact, coff + self.scale_off, self.batch, 1, lane)
        if lane == 0:
            tx = Int32(0)
            if cutlass.const_expr(not self.first_window):
                tx += count * (2 * K * D)
            tx += count * (2 * D)
            if cutlass.const_expr(self.past > 0):
                tx += edge * (count * (2 * D))
            tx += count * (self.covariance_bytes * self.pw16)
            tx += 2 * count * (4 * K)
            if cutlass.const_expr(self.past > 0):
                tx += edge * (count * (4 * K))
            if full:
                tx += (2 if staged_forgetting else 1) * count * 4
            cute.arch.mbarrier_expect_tx(bar, tx)
            if cutlass.const_expr(not self.first_window):
                _bulk_g2s(dense, doff + self.m_off, m, g0 * (K * D), count * (2 * K * D), bar)
            _bulk_g2s(dense, doff + self.v_off, rawvalue, g0 * D, count * (2 * D), bar)
            _bulk_g2s(p_stage, p_at, covariance, g0 * self.pw16,
                      count * (self.covariance_bytes * self.pw16), bar)
            _bulk_g2s(compact, coff + self.key_off, rawkey, g0 * K, count * (4 * K), bar)
            _bulk_g2s(compact, coff + self.q_off, query, g0 * K, count * (4 * K), bar)
            for j in cutlass.range_constexpr(self.past):
                if j < edge:
                    _bulk_g2s(dense, doff + self.r_off + j * self.r_stride, residual_tape,
                              (Int64(j) * groups + g0) * D, count * (2 * D), bar)
                    _bulk_g2s(compact, coff + self.g_off + j * self.g_stride, gain_tape,
                              (Int64(j) * groups + g0) * K, count * (4 * K), bar)
            if full:
                if cutlass.const_expr(staged_forgetting):
                    _bulk_g2s(compact, coff + self.lam_off, forgetting, g0, count * 4, bar)
                _bulk_g2s(compact, coff + self.scale_off, write_scale, g0, count * 4, bar)
        if tail:
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        # Every lane arrives, which orders its own per-lane copies before the consumers' reads;
        # the bulk copies complete through the expected byte count.
        cute.arch.mbarrier_arrive(bar)

    def covariance_stage(self, dense, compact, doff, coff):
        """The shared tensor that stages P and the offset of the stage's first P row in it."""
        if self.covariance_type is BFloat16:
            return dense, doff + self.p_off
        return compact, coff + self.p_off

    @cute.jit
    def store_packed(self, p_stage: cute.Tensor, p_at, base, cp, count, lane,
                     g0, destination: cute.Tensor, group_offset,
                     stride: cutlass.Constexpr):
        """Store the staged packed P rows of this pass's groups to ``destination`` in its dtype,
        consecutive lanes on consecutive entries."""
        triangle: cutlass.Constexpr = self.K * (self.K + 1) // 2
        for part in cutlass.range_constexpr(cute.ceil_div(self.pack * triangle, 32)):
            flat = part * 32 + lane
            gi = base + cp * self.pack + flat // triangle
            pi = flat % triangle
            if (flat < self.pack * triangle) & (gi < count):
                value = p_stage[p_at + gi * self.pw16 + pi]
                target = cute.make_tensor(
                    destination.iterator + (Int64(group_offset) + g0 + gi) * stride + pi,
                    cute.make_layout(1))
                target[0] = destination.element_type(value)

    @cute.jit
    def write_tapes(self, g0, count, base: Int32, compact: cute.Tensor, coff, lane,
                    forgetting: cute.Tensor, key_tape: cute.Tensor, query_tape: cute.Tensor,
                    scale_tape: cute.Tensor, forgetting_tape: cute.Tensor):
        """The input tapes of this warp's groups [base, base + per_warp) of the stage: key =
        raw_key * scale in the staged copy, then the key, query, scale and forgetting tapes.
        Broadcast forgetting fills its stage slots from global memory. The caller synchronizes
        the warp before the compute."""
        K: cutlass.Constexpr = self.K
        P: cutlass.Constexpr = self.per_warp
        total: cutlass.Constexpr = P * K
        for part in cutlass.range_constexpr(cute.ceil_div(total, 32)):
            flat = part * 32 + lane
            gi = base + flat // K
            k = flat % K
            if (flat < total) & (gi < count):
                offset = (g0 + gi) * K + k
                scaled = compact[coff + self.key_off + gi * K + k] * compact[coff + self.scale_off + gi]
                compact[coff + self.key_off + gi * K + k] = scaled
                key_target = cute.make_tensor(key_tape.iterator + offset, cute.make_layout(1))
                key_target[0] = scaled
                query_target = cute.make_tensor(query_tape.iterator + offset, cute.make_layout(1))
                query_target[0] = compact[coff + self.q_off + gi * K + k]
        for part in cutlass.range_constexpr(cute.ceil_div(P, 32)):
            li = part * 32 + lane
            gi = base + li
            if (li < P) & (gi < count):
                group = g0 + gi
                if cutlass.const_expr(forgetting.stride[0] == 0):
                    compact[coff + self.lam_off + gi] = forgetting[group // self.G, group % self.G]
                forgetting_target = cute.make_tensor(forgetting_tape.iterator + group, cute.make_layout(1))
                forgetting_target[0] = compact[coff + self.lam_off + gi]
                scale_target = cute.make_tensor(scale_tape.iterator + group, cute.make_layout(1))
                scale_target[0] = compact[coff + self.scale_off + gi]

    @cute.jit
    def compute_warp(self, g0, count, base: Int32, dense: cute.Tensor, compact: cute.Tensor,
                     doff, coff, lane, width: cutlass.Constexpr, edge: Int32,
                     commit: cutlass.Constexpr, m: cute.Tensor, covariance: cute.Tensor,
                     gain_tape: cute.Tensor, residual_tape: cute.Tensor,
                     denominator_tape: cute.Tensor, read_out: cute.Tensor,
                     saved: cute.Tensor):
        """This warp's groups [base, base + per_warp) of the stage. The K-lane phase computes
        ``pack`` groups per pass with lane = (group, k); the D phase gives every lane ``width``
        columns and covers ``dense_groups`` groups per pass."""
        K, D = self.K, self.D
        groups = self.N * self.G
        P: cutlass.Constexpr = self.per_warp
        pack: cutlass.Constexpr = self.pack
        passes: cutlass.Constexpr = self.compact_passes
        sub = lane // K
        kidx = lane % K
        sub_active = sub < pack
        p_stage, p_at = self.covariance_stage(dense, compact, doff, coff)
        gain_keep = cute.make_rmem_tensor(passes, Float32)
        gamma_keep = cute.make_rmem_tensor(passes, Float32)
        alpha_keep = cute.make_rmem_tensor((passes, max(self.past, 1)), Float32)
        beta_keep = cute.make_rmem_tensor((passes, max(self.past, 1)), Float32)
        gain_keep.fill(0.0)
        gamma_keep.fill(0.0)
        alpha_keep.fill(0.0)
        beta_keep.fill(0.0)
        for cp in cutlass.range_constexpr(passes):
            li = cp * pack + cutlass.min(sub, Int32(pack - 1))     # in-warp group (clamped)
            gi = base + li                                          # in-stage group
            active = sub_active & (gi < count)
            key_at = coff + self.key_off + gi * K
            q_at = coff + self.q_off + gi * K
            kk = _load_fp32_vector(compact, key_at, K)
            qq = _load_fp32_vector(compact, q_at, K)
            key_lane, query_lane = Float32(0.0), Float32(0.0)
            row = cute.make_rmem_tensor(K, Float32)
            row.fill(0.0)
            pk_lane = Float32(0.0)
            if active:
                # This lane's entries come from shared memory: indexing kk and qq by the lane would
                # put them in local memory.
                key_lane = compact[key_at + kidx]
                query_lane = compact[q_at + kidx]
                for k in cutlass.range_constexpr(K):
                    row[k] = Float32(p_stage[p_at + gi * self.pw16 + _symmetric_offset(kidx, k, K)])
                for k in cutlass.range_constexpr(K):
                    pk_lane += row[k] * kk[k]
            forgetting_value = compact[coff + self.lam_off + gi]
            denominator = forgetting_value + cute.arch.warp_reduction(
                key_lane * pk_lane, operator.add, threads_in_group=K,
            )
            gain_lane = pk_lane / denominator
            pk = cute.make_rmem_tensor(K, Float32)
            for k in cutlass.range_constexpr(K):
                pk[k] = cute.arch.shuffle_sync(pk_lane, sub * K + k)
            group = g0 + gi
            n, g = group // self.G, group % self.G
            # The saved tape is BF16: an FP32 P is rounded only for this copy.
            self.store_packed(p_stage, p_at, base, cp, count, lane, g0,
                              saved, Int64(edge) * groups, packed_size(K))
            # Every lane has read P and stored its saved copy before any lane overwrites it.
            cute.arch.sync_warp()
            if active:
                quotient = cute.make_rmem_tensor(K, Float32)
                for k in cutlass.range_constexpr(K):
                    quotient[k] = (row[k] - gain_lane * pk[k]) / forgetting_value
                for j in cutlass.range_constexpr(K):
                    if j >= kidx:
                        p_stage[p_at + gi * self.pw16
                                + _symmetric_offset(kidx, j, K)] = self.covariance_type(quotient[j])
            cute.arch.sync_warp()
            self.store_packed(p_stage, p_at, base, cp, count, lane, g0,
                              covariance, Int64(0), self.pw16)
            if active:
                gain_tape[edge, n, g, kidx] = gain_lane
                if kidx == 0:
                    denominator_tape[edge, n, g] = denominator
            gamma = cute.arch.warp_reduction(gain_lane * query_lane, operator.add,
                                             threads_in_group=K)
            gain_keep[cp] = gain_lane
            gamma_keep[cp] = gamma
            # The lanes of a group take different earlier edges; each dot keeps its sequential K
            # order.
            for jt in cutlass.range_constexpr(cute.ceil_div(self.past, K)):
                if jt * K < edge:
                    jlane = jt * K + kidx
                    a, b = Float32(0.0), Float32(0.0)
                    if active & (jlane < edge) & (jlane < self.past):
                        gj = _load_fp32_vector(
                            compact, coff + self.g_off + jlane * self.g_stride + gi * K, K)
                        for k in cutlass.range_constexpr(K):
                            a += gj[k] * kk[k]
                            b += gj[k] * qq[k]
                    for j in cutlass.range_constexpr(jt * K, min((jt + 1) * K, self.past)):
                        alpha_keep[cp, j] = a
                        beta_keep[cp, j] = b

        # The D phase.
        dense_groups: cutlass.Constexpr = self.dense_groups(width)
        row_lanes: cutlass.Constexpr = 32 // dense_groups
        tiles: cutlass.Constexpr = cute.ceil_div(D, row_lanes * width)
        dq = lane // row_lanes
        dl = lane % row_lanes
        dense_passes: cutlass.Constexpr = (P + dense_groups - 1) // dense_groups
        for dp in cutlass.range_constexpr(dense_passes):
            li = dp * dense_groups + dq                     # in-warp group of this lane
            li_clamped = cutlass.min(li, Int32(P - 1))
            gi = base + li_clamped
            group = g0 + gi
            cp: cutlass.Constexpr = (dp * dense_groups) // pack   # dense_groups divides pack
            src = ((dp * dense_groups + dq) % pack) * K         # the group's first K-phase lane
            gain_src = gain_keep[cp]
            gamma = cute.arch.shuffle_sync(gamma_keep[cp], src)
            alpha = cute.make_rmem_tensor(max(self.past, 1), Float32)
            beta = cute.make_rmem_tensor(max(self.past, 1), Float32)
            alpha.fill(0.0)
            beta.fill(0.0)
            for j in cutlass.range_constexpr(self.past):
                if j < edge:
                    alpha[j] = cute.arch.shuffle_sync(alpha_keep[cp, j], src + j % K)
                    beta[j] = cute.arch.shuffle_sync(beta_keep[cp, j], src + j % K)
            kk = _load_fp32_vector(compact, coff + self.key_off + gi * K, K)
            qq = _load_fp32_vector(compact, coff + self.q_off + gi * K, K)
            scale = compact[coff + self.scale_off + gi]
            m_at = doff + self.m_off + gi * (K * D)
            v_at = doff + self.v_off + gi * D
            for tile in cutlass.range_constexpr(tiles):
                d = Int32(tile) * (row_lanes * width) + dl * width
                inside = (d < D) & (li < P) & (gi < count)
                residual = cute.make_rmem_tensor(width, Float32)
                read = cute.make_rmem_tensor(width, Float32)
                residual.fill(0.0)
                read.fill(0.0)
                if cutlass.const_expr(commit):
                    base_columns = cute.make_rmem_tensor((K, width), Float32)
                    base_columns.fill(0.0)
                if inside:
                    value = _load_dense(dense, v_at + d, width)
                    projection = cute.make_rmem_tensor(width, Float32)
                    projection.fill(0.0)
                    for k in cutlass.range_constexpr(K):
                        if cutlass.const_expr(self.first_window):
                            base_row = cute.make_rmem_tensor(width, Float32)
                            base_row.fill(0.0)
                        else:
                            base_row = _load_dense(dense, m_at + k * D + d, width)
                        for v in cutlass.range_constexpr(width):
                            projection[v] += base_row[v] * kk[k]
                            read[v] += base_row[v] * qq[k]
                            if cutlass.const_expr(commit):
                                base_columns[k, v] = base_row[v]
                    for v in cutlass.range_constexpr(width):
                        residual[v] = value[v] * scale - projection[v]
                for j in cutlass.range_constexpr(self.past):
                    if j < edge:
                        rj = cute.make_rmem_tensor(width, Float32)
                        rj.fill(0.0)
                        if inside:
                            rj = _load_dense(dense, doff + self.r_off + j * self.r_stride
                                             + gi * D + d, width)
                        for v in cutlass.range_constexpr(width):
                            residual[v] -= alpha[j] * rj[v]
                            read[v] += beta[j] * rj[v]
                        if cutlass.const_expr(commit):
                            gj = _load_fp32_vector(compact, coff + self.g_off + j * self.g_stride
                                                   + gi * K, K)
                            for k in cutlass.range_constexpr(K):
                                for v in cutlass.range_constexpr(width):
                                    base_columns[k, v] += gj[k] * rj[v]
                # The residual is rounded to BF16 before it is used, exactly as it is stored.
                rounded = cute.make_rmem_tensor(width, BFloat16)
                rounded.store(residual.load().to(BFloat16))
                residual.store(rounded.load().to(Float32))
                for v in cutlass.range_constexpr(width):
                    read[v] += gamma * residual[v]
                if inside:
                    _store_vector(residual_tape, (Int64(edge) * groups + group) * D + Int64(d),
                                  residual, width)
                    _store_vector(read_out, Int64(group) * D + Int64(d), read, width)
                if cutlass.const_expr(commit):
                    for k in cutlass.range_constexpr(K):
                        gik = cute.arch.shuffle_sync(gain_src, src + k)
                        column = cute.make_rmem_tensor(width, Float32)
                        for v in cutlass.range_constexpr(width):
                            column[v] = base_columns[k, v] + gik * residual[v]
                        if inside:
                            _store_vector(m, (Int64(group) * K + k) * D + Int64(d), column, width)

    def dense_groups(self, width):
        """Groups per dense warp instruction: largest power of two <= min(pack, 32 / lanes-per-row)."""
        lanes = (self.D + width - 1) // width
        q = 1
        while q * 2 <= min(self.pack, max(1, 32 // lanes)):
            q *= 2
        return q

    @cute.jit
    def produce(self, first, stride, mine, dense: cute.Tensor, compact: cute.Tensor,
                full_bars: cute.Pointer, empty_bars: cute.Pointer, lane, edge: Int32,
                m: cute.Tensor, covariance: cute.Tensor, rawkey: cute.Tensor,
                query: cute.Tensor, rawvalue: cute.Tensor, write_scale: cute.Tensor,
                forgetting: cute.Tensor, gain_tape: cute.Tensor, residual_tape: cute.Tensor):
        S: cutlass.Constexpr = self.stages
        for t in cutlass.range(0, mine, 1):
            s = t % S
            if t >= S:
                # Stage s was consumed for batch t - S (round t // S - 1).
                cute.arch.mbarrier_wait(empty_bars + s, ((t // S) - 1) & 1)
            self.issue(first + t * stride, dense, compact, s * self.dense_elems,
                       s * self.compact_words, full_bars + s, lane, edge,
                       m, covariance, rawkey, query, rawvalue,
                       write_scale, forgetting, gain_tape, residual_tape)

    @cute.jit
    def consume(self, first, stride, mine, dense: cute.Tensor, compact: cute.Tensor,
                full_bars: cute.Pointer, empty_bars: cute.Pointer, lane, warp,
                width: cutlass.Constexpr, edge: Int32, commit: cutlass.Constexpr,
                m: cute.Tensor, covariance: cute.Tensor, gain_tape: cute.Tensor,
                residual_tape: cute.Tensor, denominator_tape: cute.Tensor,
                read_out: cute.Tensor, saved: cute.Tensor, forgetting: cute.Tensor,
                key_tape, query_tape, scale_tape, forgetting_tape):
        S: cutlass.Constexpr = self.stages
        base = warp * self.per_warp
        for t in cutlass.range(0, mine, 1):
            s = t % S
            cute.arch.mbarrier_wait(full_bars + s, (t // S) & 1)
            batch_index = first + t * stride
            g0 = Int64(batch_index) * self.batch
            count = cutlass.min(Int32(self.batch), Int32(self.N * self.G - g0))
            self.write_tapes(g0, count, base, compact, s * self.compact_words, lane,
                             forgetting, key_tape, query_tape, scale_tape, forgetting_tape)
            cute.arch.sync_warp()
            self.compute_warp(g0, count, base, dense, compact, s * self.dense_elems,
                              s * self.compact_words, lane, width, edge, commit, m,
                              covariance, gain_tape, residual_tape, denominator_tape, read_out,
                              saved)
            # Every lane finished reading this stage before it is released to
            # the async proxy for refilling.
            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.shared", space="cta")
            if lane == 0:
                cute.arch.mbarrier_arrive(empty_bars + s)

    @cute.kernel
    def kernel(
        self, m: cute.Tensor, covariance: cute.Tensor,
        rawkey: cute.Tensor, query: cute.Tensor, rawvalue: cute.Tensor,
        write_scale: cute.Tensor, forgetting: cute.Tensor,
        gain_tape: cute.Tensor, residual_tape: cute.Tensor,
        denominator_tape: cute.Tensor, read_out: cute.Tensor, saved: cute.Tensor,
        key_tape: cute.Tensor, query_tape: cute.Tensor, scale_tape: cute.Tensor,
        forgetting_tape: cute.Tensor, edge: Int32,
    ):
        if cutlass.const_expr(self.static_edge is not None):
            edge = Int32(self.static_edge)
        K, D = self.K, self.D
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        width = math.gcd(D, self.lane_width, max(1, min(
            m.iterator.max_alignment, rawvalue.iterator.max_alignment,
            residual_tape.iterator.max_alignment, read_out.iterator.max_alignment) // 2))
        # Every staged block is a bulk copy: 16-byte-aligned bases, 16-byte multiples.
        for tensor in (m, rawvalue, residual_tape, covariance, rawkey, query, gain_tape,
                       write_scale):
            assert tensor.iterator.max_alignment >= 16
        if cutlass.const_expr(forgetting.stride[0] != 0):
            assert forgetting.iterator.max_alignment >= 16
        assert D % 8 == 0 and K % 4 == 0
        assert (self.covariance_bytes * self.pw16) % 16 == 0
        smem = SmemAllocator()
        dense = smem.allocate_tensor(BFloat16, cute.make_layout(
            self.stages * self.dense_elems), byte_alignment=128)
        compact = smem.allocate_tensor(Float32, cute.make_layout(
            self.stages * self.compact_words), byte_alignment=16)
        full_bars = smem.allocate_array(Int64, self.stages, byte_alignment=8)
        empty_bars = smem.allocate_array(Int64, self.stages, byte_alignment=8)
        if tid == 0:
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(full_bars + s, 32)
                cute.arch.mbarrier_init(empty_bars + s, self.consumers)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        stride = Int32(self.blocks)
        first = Int32(bid)
        mine = Int32(0)
        if first < self.batches:
            mine = (Int32(self.batches) - first + stride - 1) // stride
        if (edge >= 0) & (edge < self.E):
            if warp == self.consumers:
                self.produce(first, stride, mine, dense, compact, full_bars, empty_bars, lane, edge,
                             m, covariance, rawkey, query, rawvalue,
                             write_scale, forgetting, gain_tape, residual_tape)
            else:
                if edge == self.E - 1:
                    self.consume(first, stride, mine, dense, compact, full_bars, empty_bars, lane,
                                 warp, width, edge, True, m, covariance, gain_tape,
                                 residual_tape, denominator_tape, read_out, saved, forgetting,
                                 key_tape, query_tape, scale_tape, forgetting_tape)
                else:
                    self.consume(first, stride, mine, dense, compact, full_bars, empty_bars, lane,
                                 warp, width, edge, False, m, covariance, gain_tape,
                                 residual_tape, denominator_tape, read_out, saved, forgetting,
                                 key_tape, query_tape, scale_tape, forgetting_tape)


def make_forward(E: int, N: int, G: int, K: int, D: int, *, sms: int, smem_budget: int,
                 consumers: int = 8, batch: int = 16, blocks_per_sm: int = 1,
                 lane_width: int = 8, first_window: bool = False,
                 running_covariance_dtype: str = "bfloat16"):
    """The forward for one window shape: a persistent grid of at most blocks_per_sm * sms CTAs
    of ``consumers`` + 1 warps. The producer warp streams ``batch`` consecutive groups per stage
    through two shared stages and every consumer warp computes batch // consumers groups of each
    stage. The batch is fitted to the shared-memory budget (a smaller batch, then fewer
    consumers). M is committed at the window's last edge. With ``first_window`` the start state
    is zero: M is neither staged nor read, only committed. ``running_covariance_dtype`` ("bfloat16"
    or "float32") is the dtype of the covariance operand the forward reads and updates.
    """
    if type(first_window) is not bool:
        raise ValueError("first_window must be bool")
    if running_covariance_dtype not in COVARIANCE_TYPES:
        raise ValueError(f"running_covariance_dtype must be one of {sorted(COVARIANCE_TYPES)}")
    if K not in (4, 8, 16):
        raise ValueError("K must be 4, 8 or 16")
    if type(consumers) is not int or not 1 <= consumers <= 31:
        raise ValueError("consumers must be an integer in 1..31 (plus one producer warp)")
    if type(batch) is not int or batch <= 0 or batch % consumers:
        raise ValueError("batch must be a positive multiple of consumers")
    per_warp = batch // consumers
    if per_warp & (per_warp - 1):
        raise ValueError("groups per consumer warp must be a power of two")
    if any(type(v) is not int or v <= 0 for v in (blocks_per_sm, sms, smem_budget)):
        raise ValueError("blocks_per_sm, sms and smem_budget must be positive Python integers")
    if lane_width not in (1, 2, 4, 8):
        raise ValueError("lane_width must be 1, 2, 4 or 8")
    return Forward(E, N, G, K, D, sms=sms, smem_budget=smem_budget, consumers=consumers, batch=batch,
                   blocks_per_sm=blocks_per_sm, lane_width=lane_width,
                   first_window=first_window, running_covariance_dtype=running_covariance_dtype)


def packed_size(k: int) -> int:
    """Entries of the packed upper triangle of a K x K symmetric matrix."""
    return k * (k + 1) // 2
