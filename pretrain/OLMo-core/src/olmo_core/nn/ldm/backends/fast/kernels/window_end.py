"""The window-end kernel of the backward, and what the state kernels share: the staging and
store helpers (BF16 and FP32 row loads and stores, the bulk global-to-shared copy, the packed
symmetric offset) and the covariance VJP of one edge."""

import math
import operator

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, cute
from cutlass.utils import SmemAllocator


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


@cute.jit
def _load_fp32_vector(tensor: cute.Tensor, offset, count: cutlass.Constexpr):
    """Load `count` consecutive FP32 words (16-byte pieces when aligned) from shared."""
    out = cute.make_rmem_tensor(count, Float32)
    chunk: cutlass.Constexpr = math.gcd(count, 4)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32,
                               num_bits_per_copy=32 * chunk)
    for part in cutlass.range_constexpr(count // chunk):
        source = cute.make_tensor(
            tensor.iterator + cute.assume(offset + part * chunk, divby=chunk),
            cute.make_layout(chunk))
        piece = cute.make_rmem_tensor(chunk, Float32)
        cute.copy(atom, source, piece)
        for v in cutlass.range_constexpr(chunk):
            out[part * chunk + v] = piece[v]
    return out


@cute.jit
def _bulk_g2s(dst: cute.Tensor, dst_offset, src: cute.Tensor, src_offset,
              nbytes: Int32, mbar: cute.Pointer):
    """One cp.async.bulk global->shared copy of `nbytes` (runtime, multiple of 16,
    both addresses 16-byte aligned) completing on the mbarrier `mbar`."""
    smem_addr = Int32((dst.iterator + dst_offset).toint())
    gmem_ptr = (src.iterator + src_offset).llvm_ptr
    mbar_addr = Int32(mbar.toint())
    cute.arch.inline_ptx(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [{$r0}], [{$r1}], {$r2}, [{$r3}];",
        read_only_args=[smem_addr, gmem_ptr, nbytes, mbar_addr],
    )


@cute.jit
def _store_fp32_row(tensor: cute.Tensor, offset: Int64, row: cute.Tensor,
                    words: cutlass.Constexpr, chunk: cutlass.Constexpr):
    """Store `words` FP32 registers contiguously with `chunk`-word copies."""
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32,
                               num_bits_per_copy=32 * chunk)
    for part in cutlass.range_constexpr(words // chunk):
        destination = cute.make_tensor(
            tensor.iterator + cute.assume(offset + part * chunk, divby=chunk),
            cute.make_layout(chunk))
        piece = cute.make_rmem_tensor(chunk, Float32)
        for v in cutlass.range_constexpr(chunk):
            piece[v] = row[part * chunk + v]
        cute.copy(atom, piece, destination)


@cute.jit
def _wide(raw: cute.Tensor, width: cutlass.Constexpr):
    value = cute.make_rmem_tensor(width, Float32)
    value.store(raw.load().to(Float32))
    return value


@cute.jit
def _load_bf16(tensor: cute.Tensor, offset: Int64, width: cutlass.Constexpr):
    source = cute.make_tensor(tensor.iterator + cute.assume(offset, divby=width),
                              cute.make_layout(width))
    raw = cute.make_rmem_tensor(width, BFloat16)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                               num_bits_per_copy=16 * width)
    cute.copy(atom, source, raw)
    return raw


@cute.jit
def _store_vector(tensor: cute.Tensor, offset: Int64, value: cute.Tensor,
                  width: cutlass.Constexpr):
    destination = cute.make_tensor(
        tensor.iterator + cute.assume(offset, divby=width), cute.make_layout(width))
    raw = cute.make_rmem_tensor(width, tensor.element_type)
    raw.store(value.load().to(tensor.element_type))
    bits = min(width * tensor.element_type.width, tensor.iterator.max_alignment * 8, 128)
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), tensor.element_type,
                               num_bits_per_copy=bits)
    cute.copy(atom, raw, destination)


@cute.jit
def _symmetric_offset(row, column, k: cutlass.Constexpr):
    """Upper-row-major packed offset of the symmetric K x K entry (row, column)."""
    first = cutlass.min(row, column)
    last = cutlass.max(row, column)
    return first * k - first * (first - 1) // 2 + (last - first)


@cute.jit
def _sigmoid(logit: Float32):
    """Strict FP32 sigmoid: 1 / (1 + exp(-x)) with the exact exponential."""
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-logit))


@cute.jit
def lane_sum(value: Float32, lanes: cutlass.Constexpr):
    """Sum of ``value`` over each aligned run of ``lanes`` lanes; every lane gets its run's sum.
    A K-lane dot whose other lanes hold zeros gets the same sum as the 32-lane butterfly, which
    adds those zeros first."""
    return cute.arch.warp_reduction(value, operator.add, threads_in_group=lanes)


@cute.jit
def covariance_vjp(p_row: cute.Tensor, h_row: cute.Tensor, key_row: cute.Tensor,
                   gain_row: cute.Tensor, key_lane: Float32, gain_lane: Float32, dg_lane: Float32,
                   denominator: Float32, forgetting: Float32, row, active, first_lane,
                   K: cutlass.Constexpr):
    """The covariance VJP of one edge of one group, computed by the K lanes first_lane ..
    first_lane + K - 1: lane first_lane + row holds row ``row`` of P and of the covariance
    gradient C in p_row and h_row, and ``active`` is true in those lanes (the other lanes of the
    run contribute zeros to every sum). key_row and gain_row are the edge's key and gain, key_lane
    and gain_lane this lane's entries. Returns (the key gradient through P, the forgetting
    gradient, this lane's row of the previous edge's C)."""
    inv_forgetting = Float32(1.) / forgetting
    inv_den = Float32(1.) / denominator
    p_lane = Float32(0.)
    for column in cutlass.range_constexpr(K):
        if active:
            p_lane += p_row[column] * key_row[column]
    # Shuffle every column first: interleaved with the row-dependent updates below, each shuffle
    # would wait for the previous branch to reconverge.
    pk = cute.make_rmem_tensor(K, Float32)
    for column in cutlass.range_constexpr(K):
        pk[column] = cute.arch.shuffle_sync(p_lane, first_lane + column)
    up, utg, upper_pair = Float32(0.), Float32(0.), Float32(0.)
    if active:
        for column in cutlass.range_constexpr(K):
            if column >= row:
                up += h_row[column] * pk[column]
                upper_pair += h_row[column] * ((p_row[column] - gain_lane * pk[column]) * inv_forgetting)
            if column <= row:
                utg += h_row[column] * gain_row[column]
    d_gain = dg_lane - up * inv_forgetting
    reduce_in = Float32(0.)
    if active:
        reduce_in = d_gain * p_lane
    dden = -lane_sum(reduce_in, K) * (inv_den * inv_den)
    d_p = -utg * inv_forgetting + d_gain * inv_den + dden * key_lane
    dk_cov = dden * p_lane
    dp = cute.make_rmem_tensor(K, Float32)
    for column in cutlass.range_constexpr(K):
        dp[column] = cute.arch.shuffle_sync(d_p, first_lane + column)
    for column in cutlass.range_constexpr(K):
        if active:
            dk_cov += p_row[column] * dp[column]
    pair_in = Float32(0.)
    if active:
        pair_in = upper_pair
    d_forgetting = -lane_sum(pair_in, K) * inv_forgetting + dden
    previous = cute.make_rmem_tensor(K, Float32)
    previous.fill(0.)
    for column in cutlass.range_constexpr(K):
        if active:
            if column >= row:
                value = h_row[column] * inv_forgetting + d_p * key_row[column]
                if column != row:
                    value += dp[column] * key_lane
                previous[column] = value
    return dk_cov, d_forgetting, previous


@cute.jit
def _dot(left: cute.Tensor, right: cute.Tensor, width: cutlass.Constexpr):
    value = Float32(0.0)
    for v in cutlass.range_constexpr(width):
        value += left[v] * right[v]
    return value


@cute.jit
def _reduce_scatter_columns(values: cute.Tensor, width: cutlass.Constexpr,
                            bit: cutlass.Constexpr, lane: Int32):
    # Each butterfly step keeps half of this lane's rows and receives the same half from its
    # partner, which keeps the other half. After log2(K) steps lane % K holds one row's partial
    # sum, and the remaining steps add it up over the lanes of the D columns.
    if cutlass.const_expr(width > 1):
        partial = cute.make_rmem_tensor((width // 2, values.shape[1]), Float32)
        high = (lane & bit) != 0
        for k in cutlass.range_constexpr(width // 2):
            for component in cutlass.range_constexpr(values.shape[1]):
                lo, hi = values[2 * k, component], values[2 * k + 1, component]
                own = hi if high else lo
                send = lo if high else hi
                partial[k, component] = own + cute.arch.shuffle_sync_bfly(send, bit)
        result = _reduce_scatter_columns(partial, width // 2, bit * 2, lane)
    else:
        result = cute.make_rmem_tensor(values.shape[1], Float32)
        for component in cutlass.range_constexpr(values.shape[1]):
            value = values[0, component]
            for stage in cutlass.range_constexpr(int(math.log2(32 // bit))):
                value += cute.arch.shuffle_sync_bfly(value, bit << stage)
            result[component] = value
    return result


@cute.jit
def _store_bf16_vector(tensor: cute.Tensor, offset: Int64, values: cute.Tensor,
                       width: cutlass.Constexpr):
    rounded = cute.make_rmem_tensor(width, BFloat16)
    rounded.store(values.load().to(BFloat16))
    destination = cute.make_tensor(tensor.iterator + cute.assume(offset, divby=width),
                                    cute.make_layout(width))
    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                               num_bits_per_copy=16 * width)
    cute.copy(atom, rounded, destination)


@cute.jit
def _load_dense(dense: cute.Tensor, offset, width: cutlass.Constexpr):
    return _wide(_load_bf16(dense, offset, width), width)


@cute.jit
def _stage_span(src: cute.Tensor, src_offset, elements, dst: cute.Tensor, dst_offset,
                capacity: cutlass.Constexpr, chunk: cutlass.Constexpr, lane):
    """cp.async ``elements`` (runtime, at most ``capacity``, a multiple of ``chunk``) contiguous
    elements, ``chunk`` per lane copy. Completion is the caller's cp.async commit group."""
    atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), src.element_type,
                               num_bits_per_copy=src.element_type.width * chunk)
    for part in cutlass.range_constexpr(cute.ceil_div(capacity, 32 * chunk)):
        index = (part * 32 + lane) * chunk
        if index < elements:
            source = cute.make_tensor(src.iterator + cute.assume(src_offset + index, divby=chunk),
                                      cute.make_layout(chunk))
            target = cute.make_tensor(dst.iterator + cute.assume(dst_offset + index, divby=chunk),
                                      cute.make_layout(chunk))
            cute.copy(atom, source, target)


class WindowEnd:
    """The window end, which also handles the window's last edge. Tensor order: m_end, dm_end,
    keys, queries, gains, residuals, forgetting_tape, denominators, future_queries, future_keys,
    future_dy, future_dv, m_start, dm_start, future_value_grad,
    future_gain_grad, current_dy, write_scales, dy, dv, dq, covariance_grad, grad_forgetting,
    grad_rawvalue, grad_rawkey, grad_write_scale, the saved covariance, then the stream.

    E is the window length and F the length of the following window, already reversed (with
    F = 0 the future tapes are one-edge placeholders). Of the per-edge gradients only the last
    edge's are written, and the future value and gain gradients of edges 0..E-2 for the interior
    launches that follow. m_end and
    m_start may be the same tensor, and so may dm_end and dm_start; the current and future DY
    tapes (and DV tapes) may be slices of one history buffer; current_dy must not alias a
    written buffer. P comes from the saved covariance tape, and covariance_grad is the FP32
    gradient of the packed upper triangle of P.

    The last edge's dq and -M dv use the committed BF16 m_end. In the first window the start state
    is zero: the end state is rebuilt as sum(g r^T) in FP32 and m_start is written as exact zeros.

    Persistent CTAs: a producer warp stages batches of ``batch`` consecutive groups into two
    shared stages with bulk copies (the batch's rows of every input are contiguous), and each of
    the ``consumers`` warps computes groups of the staged batch, one at a time. A group's inputs
    are staged before any of its outputs is written, which makes the aliasing above safe.
    """

    def __init__(self, E, F, N, G, K, D, *, sms, batch, consumers, blocks_per_sm, first_window=False):
        self.E, self.F, self.N, self.G, self.K, self.D = E, F, N, G, K, D
        self.first_window = first_window
        self.batch, self.consumers = batch, consumers
        self.per_warp = batch // consumers
        self.warps = consumers + 1
        self.stages = 2
        self.batches = (N * G + batch - 1) // batch
        self.blocks = min(self.batches, blocks_per_sm * sms)
        self.lane_width = default_lane_width(D)
        self.words = K * (K + 1) // 2
        # One block per staged input, [rows][batch][row elements]; every block starts 16-byte
        # aligned. BF16 elements:
        self.g_off = 0                                    # dM at the window's end, [batch][K][D]
        self.m_off = batch * K * D                        # M at the window's end, [batch][K][D]
        self.rr_off = self.m_off + (0 if first_window else batch * K * D)
        self.fdy_off = self.rr_off + E * batch * D
        self.fdv_off = self.fdy_off + F * batch * D
        self.cdy_off = self.fdv_off + F * batch * D
        self.p_off = self.cdy_off + batch * D             # the saved P of the last edge
        self.dense_elems = _round_up(self.p_off + batch * self.words, 8)
        # FP32 words:
        self.c_off = 0
        self.keys_off = _round_up(batch * self.words, 4)
        self.queries_off = self.keys_off + E * batch * K
        self.gains_off = self.queries_off + E * batch * K
        self.fq_off = self.gains_off + E * batch * K
        self.fk_off = self.fq_off + F * batch * K
        self.lam_off = self.fk_off + F * batch * K
        self.den_off = self.lam_off + E * batch
        self.scale_off = self.den_off + E * batch
        self.compact_words = _round_up(self.scale_off + batch, 4)
        self.stage_bytes = 2 * self.dense_elems + 4 * self.compact_words
        # The last edge's covariance VJP and the outputs after it keep K lanes busy per group, so
        # they run for 32 // K groups at once. Each group waits for its pack in a per-warp finish
        # slot: FP32 words [C | key | gain | forgetting, denominator, scale, r.vi], BF16 P and the
        # group index.
        self.pack = 32 // K
        self.slot_c = 0
        self.slot_key = _round_up(self.words, 4)
        self.slot_gain = self.slot_key + K
        self.slot_scalars = self.slot_gain + K
        self.slot_words = self.slot_scalars + 4
        self.slot_p = _round_up(self.words, 8)
        self.slot_bytes = self.pack * (4 * self.slot_words + 2 * self.slot_p + 4)
        self.shared_bytes = self.stages * (self.stage_bytes + 16) + consumers * self.slot_bytes + 128

    def _state(self, offset, b, k):
        """Shared offset of row k of group b in a [batch][K][D] block."""
        return offset + (b * self.K + k) * self.D

    def _row(self, offset, row, b, width):
        """Shared offset of group b's piece of row ``row`` in a [rows][batch][width] block."""
        return offset + (row * self.batch + b) * width

    def _blocks(self, m_end, dm_end, keys, queries, gains, residuals, forgetting_tape,
                denominators, future_queries, future_keys, future_dy, future_dv, current_dy,
                write_scales, covariance_grad, saved_p):
        """(source, first element, rows, row stride, group width, dense?, shared offset) of every
        staged block; group g of row r is at first + r * row stride + g * group width."""
        ng = self.N * self.G
        K, D, E, F = self.K, self.D, self.E, self.F
        blocks = [(dm_end, 0, 1, 0, K * D, True, self.g_off)]
        if not self.first_window:
            blocks.append((m_end, 0, 1, 0, K * D, True, self.m_off))
        blocks += [(residuals, 0, E, ng * D, D, True, self.rr_off)]
        if F > 0:
            blocks += [(future_dy, 0, F, future_dy.stride[0], D, True, self.fdy_off),
                       (future_dv, 0, F, future_dv.stride[0], D, True, self.fdv_off)]
        blocks += [(current_dy, 0, 1, 0, D, True, self.cdy_off),
                   (saved_p, (E - 1) * ng * self.words, 1, 0, self.words, True, self.p_off),
                   (covariance_grad, 0, 1, 0, self.words, False, self.c_off),
                   (keys, 0, E, ng * K, K, False, self.keys_off),
                   (queries, 0, E, ng * K, K, False, self.queries_off),
                   (gains, 0, E, ng * K, K, False, self.gains_off)]
        if F > 0:
            blocks += [(future_queries, 0, F, ng * K, K, False, self.fq_off),
                       (future_keys, 0, F, ng * K, K, False, self.fk_off)]
        blocks += [(forgetting_tape, 0, E, ng, 1, False, self.lam_off),
                   (denominators, 0, E, ng, 1, False, self.den_off),
                   (write_scales, 0, 1, 0, 1, False, self.scale_off)]
        return blocks

    def _plan(self, source, first, rows, row_stride, width):
        """(bulk for a full batch, bulk for a short one, per-lane chunk) of one block: a bulk
        copy needs 16-byte-aligned rows of whole 16-byte multiples."""
        size = source.element_type.width // 8
        alignment = source.iterator.max_alignment
        starts = math.gcd(alignment, first * size, row_stride * size if rows > 1 else 0,
                          self.batch * width * size)
        full = starts % 16 == 0
        short = full and (width * size) % 16 == 0
        chunk = math.gcd(16 // size, width, starts // size if starts else 16 // size)
        assert chunk * size >= 4, "cp.async copies whole 4-byte words"
        return full, short, chunk

    @cute.jit
    def issue(self, batch_index, dense: cute.Tensor, compact: cute.Tensor, doff, coff,
              bar: cute.Pointer, lane, m_end: cute.Tensor, dm_end: cute.Tensor,
              keys: cute.Tensor, queries: cute.Tensor, gains: cute.Tensor, residuals: cute.Tensor,
              forgetting_tape: cute.Tensor, denominators: cute.Tensor, future_queries: cute.Tensor,
              future_keys: cute.Tensor, future_dy: cute.Tensor, future_dv: cute.Tensor,
              current_dy: cute.Tensor, write_scales: cute.Tensor, covariance_grad: cute.Tensor,
              saved_p: cute.Tensor):
        """Producer warp: stage one batch into (doff, coff) and complete it on ``bar``. Lane 0
        bulk-copies every block it can; the producer lanes copy the rest (unaligned shapes and a
        short last batch) and wait for them before arriving."""
        B = self.batch
        g0 = Int64(batch_index) * B
        count = cutlass.min(Int32(B), Int32(self.N * self.G - g0))
        full = count == B
        blocks = self._blocks(m_end, dm_end, keys, queries, gains, residuals, forgetting_tape,
                              denominators, future_queries, future_keys, future_dy, future_dv,
                              current_dy, write_scales, covariance_grad, saved_p)
        short = count < B
        per_lane = False
        for source, first, rows, row_stride, width, is_dense, offset in blocks:
            bulk_full, bulk_short, chunk = self._plan(source, first, rows, row_stride, width)
            target, base = (dense, doff) if is_dense else (compact, coff)
            if cutlass.const_expr(not bulk_full):
                per_lane = True
                for row in cutlass.range_constexpr(rows):
                    _stage_span(source, first + row * row_stride + g0 * width, count * width,
                                target, base + offset + row * B * width, B * width, chunk, lane)
            elif cutlass.const_expr(not bulk_short):
                per_lane = True
                if short:
                    for row in cutlass.range_constexpr(rows):
                        _stage_span(source, first + row * row_stride + g0 * width, count * width,
                                    target, base + offset + row * B * width, B * width, chunk, lane)
        if lane == 0:
            size = Int32(0)
            for source, first, rows, row_stride, width, is_dense, offset in blocks:
                bulk_full, bulk_short, chunk = self._plan(source, first, rows, row_stride, width)
                element: cutlass.Constexpr = source.element_type.width // 8
                if cutlass.const_expr(bulk_short):
                    size += rows * count * (width * element)
                elif cutlass.const_expr(bulk_full):
                    if full:
                        size += rows * count * (width * element)
            cute.arch.mbarrier_expect_tx(bar, size)
            for source, first, rows, row_stride, width, is_dense, offset in blocks:
                bulk_full, bulk_short, chunk = self._plan(source, first, rows, row_stride, width)
                element: cutlass.Constexpr = source.element_type.width // 8
                target, base = (dense, doff) if is_dense else (compact, coff)
                if cutlass.const_expr(bulk_short):
                    for row in cutlass.range_constexpr(rows):
                        _bulk_g2s(target, base + offset + row * B * width, source,
                                  first + row * row_stride + g0 * width, count * (width * element), bar)
                elif cutlass.const_expr(bulk_full):
                    if full:
                        for row in cutlass.range_constexpr(rows):
                            _bulk_g2s(target, base + offset + row * B * width, source,
                                      first + row * row_stride + g0 * width, count * (width * element), bar)
        if cutlass.const_expr(per_lane):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_warp()
        cute.arch.mbarrier_arrive(bar)

    @cute.jit
    def produce(self, first, stride, mine, dense: cute.Tensor, compact: cute.Tensor,
                full_bars: cute.Pointer, empty_bars: cute.Pointer, lane, m_end: cute.Tensor,
                dm_end: cute.Tensor, keys: cute.Tensor, queries: cute.Tensor, gains: cute.Tensor,
                residuals: cute.Tensor, forgetting_tape: cute.Tensor, denominators: cute.Tensor,
                future_queries: cute.Tensor, future_keys: cute.Tensor, future_dy: cute.Tensor,
                future_dv: cute.Tensor, current_dy: cute.Tensor, write_scales: cute.Tensor,
                covariance_grad: cute.Tensor, saved_p: cute.Tensor):
        S: cutlass.Constexpr = self.stages
        for t in cutlass.range(0, mine, 1):
            s = t % S
            if t >= S:
                # Stage s was consumed for batch t - S (round t // S - 1).
                cute.arch.mbarrier_wait(empty_bars + s, ((t // S) - 1) & 1)
            self.issue(first + t * stride, dense, compact, s * self.dense_elems,
                       s * self.compact_words, full_bars + s, lane, m_end, dm_end, keys,
                       queries, gains, residuals, forgetting_tape, denominators, future_queries,
                       future_keys, future_dy, future_dv, current_dy, write_scales, covariance_grad,
                       saved_p)

    @cute.jit
    def consume(self, first, stride, mine, dense: cute.Tensor, compact: cute.Tensor,
                full_bars: cute.Pointer, empty_bars: cute.Pointer, lane, warp,
                width: cutlass.Constexpr, m_start: cute.Tensor, dm_start: cute.Tensor,
                future_value_grad: cute.Tensor,
                future_gain_grad: cute.Tensor, dy: cute.Tensor, dv: cute.Tensor, dq: cute.Tensor,
                covariance_grad: cute.Tensor, grad_forgetting: cute.Tensor, grad_rawvalue: cute.Tensor,
                grad_rawkey: cute.Tensor, grad_write_scale: cute.Tensor, slots: cute.Tensor,
                slot_p: cute.Tensor, slot_groups: cute.Tensor):
        S: cutlass.Constexpr = self.stages
        pack: cutlass.Constexpr = self.pack
        slot_base = warp * (pack * self.slot_words)
        slot_p_base = warp * (pack * self.slot_p)
        group_base = warp * pack
        # A short last pack leaves slots unfilled; its idle lanes then read zeros.
        for part in cutlass.range_constexpr(cute.ceil_div(pack * self.slot_words, 32)):
            index = part * 32 + lane
            if index < pack * self.slot_words:
                slots[slot_base + index] = Float32(0.0)
        for part in cutlass.range_constexpr(cute.ceil_div(pack * self.slot_p, 32)):
            index = part * 32 + lane
            if index < pack * self.slot_p:
                slot_p[slot_p_base + index] = BFloat16(0.0)
        if lane < pack:
            slot_groups[group_base + lane] = Int32(0)
        # carry holds, in the lanes of each group's run, the two terms of the last edge's gain
        # gradient (zi and the future term), dq and the key gradient without its covariance term;
        # done counts this warp's groups.
        carry = cute.make_rmem_tensor(4, Float32)
        carry.fill(0.0)
        done = cute.make_rmem_tensor(1, Int32)
        done[0] = Int32(0)
        for t in cutlass.range(0, mine, 1):
            s = t % S
            cute.arch.mbarrier_wait(full_bars + s, (t // S) & 1)
            g0 = Int64(first + t * stride) * self.batch
            count = cutlass.min(Int32(self.batch), Int32(self.N * self.G - g0))
            for i in cutlass.range(0, self.per_warp, 1):
                b = warp * self.per_warp + i
                if b < count:
                    slot = done[0] % pack
                    self.compute(g0 + b, b, dense, compact, s * self.dense_elems,
                                 s * self.compact_words, lane, width, m_start, dm_start,
                                 future_value_grad, future_gain_grad, dy, dv,
                                 grad_rawvalue, slots, slot_p, slot_groups,
                                 slot_base + slot * self.slot_words,
                                 slot_p_base + slot * self.slot_p, group_base + slot, slot, carry)
                    done[0] = done[0] + 1
                    if slot == pack - 1:
                        self.finish(pack, slots, slot_p, slot_groups, slot_base, slot_p_base,
                                    group_base, lane, carry, dq, covariance_grad, grad_forgetting,
                                    grad_rawkey, grad_write_scale)
            # Every lane finished with this stage (it also wrote to it) before it is released
            # to the async proxy for refilling.
            cute.arch.sync_warp()
            cute.arch.fence_proxy("async.shared", space="cta")
            if lane == 0:
                cute.arch.mbarrier_arrive(empty_bars + s)
        if done[0] % pack != 0:
            self.finish(done[0] % pack, slots, slot_p, slot_groups, slot_base, slot_p_base,
                        group_base, lane, carry, dq, covariance_grad, grad_forgetting, grad_rawkey,
                        grad_write_scale)

    @cute.jit
    def compute(self, group, b, dense: cute.Tensor, compact: cute.Tensor, doff, coff, lane,
                width: cutlass.Constexpr, m_start: cute.Tensor, dm_start: cute.Tensor,
                future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor,
                dy: cute.Tensor, dv: cute.Tensor, grad_rawvalue: cute.Tensor, slots: cute.Tensor,
                slot_p: cute.Tensor, slot_groups: cute.Tensor, sw, sp, sg, slot,
                carry: cute.Tensor):
        """Group ``group``, staged as row ``b`` of every block of the batch stage at (doff, coff),
        up to its finish slot ``slot``."""
        n, g = group // self.G, group % self.G
        ng = self.N * self.G
        last = self.E - 1
        words = self.words
        # The K-vector chains run one edge per K-lane segment, 32 // K edges at a time; each
        # chain keeps its own order and K-lane sums.
        per_bank: cutlass.Constexpr = 32 // self.K
        key_index = lane % self.K
        mine = lane // self.K == slot
        u_last = Float32(0.0)
        if mine:
            u_last = compact[coff + self._row(self.queries_off, last, b, self.K) + key_index]
        alpha_part = Float32(0.0)
        for bank in cutlass.range_constexpr(cute.ceil_div(self.E, per_bank)):
            edge = bank * per_bank + lane // self.K
            active = edge < self.E
            h, query = Float32(0.), Float32(0.)
            if active:
                h = compact[coff + self._row(self.gains_off, edge, b, self.K) + key_index]
                query = compact[coff + self._row(self.queries_off, edge, b, self.K) + key_index]
            # The factors of the later edges, applied to this edge's gain in ascending edge order.
            for future in cutlass.range_constexpr(bank * per_bank + 1, self.E):
                applies = active & (future > edge)
                dot = Float32(0.)
                if applies:
                    dot = compact[coff + self._row(self.keys_off, future, b, self.K) + key_index] * h
                dot = lane_sum(dot, self.K)
                if applies:
                    h -= compact[coff + self._row(self.gains_off, future, b, self.K) + key_index] * dot
            if active:
                if edge == last:
                    alpha_part = h * query
                # The query row has been read; it now holds this edge's gain carried to the
                # window's end, which every lane reads back as a vector in the D loop.
                compact[coff + self._row(self.queries_off, edge, b, self.K) + key_index] = h
        alpha = cute.arch.warp_reduction(alpha_part, operator.add)
        cute.arch.sync_warp()
        future_grad_accum = cute.make_rmem_tensor(self.E, Float32)
        future_grad_accum.fill(0.0)
        q_accum, k_accum = Float32(0.0), Float32(0.0)
        residual_dy, residual_dv = Float32(0.0), Float32(0.0)
        write_scale = compact[coff + self.scale_off + b]
        full: cutlass.Constexpr = self.D % (32 * width) == 0
        for tile in cutlass.range_constexpr(cute.ceil_div(self.D, 32 * width)):
            # Shared addressing stays 32-bit; only global offsets widen.
            d = Int32(tile) * (32 * width) + lane * width
            inside = (d < self.D) | Boolean(full)
            gradient = cute.make_rmem_tensor((self.K, width), Float32)
            gradient.fill(0.0)
            if inside:
                for k in cutlass.range_constexpr(self.K):
                    loaded = _load_dense(dense, doff + self._state(self.g_off, b, k) + d, width)
                    for v in cutlass.range_constexpr(width):
                        gradient[k, v] = loaded[v]
                for edge in cutlass.range_constexpr(self.F):
                    yy = _load_dense(dense, doff + self._row(self.fdy_off, edge, b, self.D) + d, width)
                    query_row = _load_fp32_vector(compact, coff + self._row(self.fq_off, edge, b, self.K), self.K)
                    vv = _load_dense(dense, doff + self._row(self.fdv_off, edge, b, self.D) + d, width)
                    key_row = _load_fp32_vector(compact, coff + self._row(self.fk_off, edge, b, self.K), self.K)
                    for k in cutlass.range_constexpr(self.K):
                        query = query_row[k]
                        key = key_row[k]
                        for v in cutlass.range_constexpr(width):
                            gradient[k, v] += query * yy[v]
                            gradient[k, v] -= key * vv[v]
                for k in cutlass.range_constexpr(self.K):
                    rounded = cute.make_rmem_tensor(width, BFloat16)
                    rounded.store(gradient[k, None].load().to(BFloat16))
                    _store_vector(dm_start, (Int64(group) * self.K + k) * self.D + Int64(d), rounded, width)
                    for v in cutlass.range_constexpr(width):
                        gradient[k, v] = Float32(rounded[v])
            # Column e of the contraction is gradient . r_e; all E columns are reduced together.
            contraction = cute.make_rmem_tensor((self.K, self.E), Float32)
            contraction.fill(0.0)
            for edge in cutlass.range_constexpr(self.E - 1):
                future_grad = cute.make_rmem_tensor(width, Float32)
                future_grad.fill(0.0)
                pk = _load_fp32_vector(compact, coff + self._row(self.queries_off, edge, b, self.K), self.K)
                for k in cutlass.range_constexpr(self.K):
                    for v in cutlass.range_constexpr(width):
                        future_grad[v] += gradient[k, v] * pk[k]
                if inside:
                    rr = _load_dense(dense, doff + self._row(self.rr_off, edge, b, self.D) + d, width)
                    _store_vector(future_value_grad, (Int64(edge) * ng + group) * self.D + Int64(d), future_grad, width)
                    for k in cutlass.range_constexpr(self.K):
                        value = Float32(0.0)
                        for v in cutlass.range_constexpr(width):
                            value += gradient[k, v] * rr[v]
                        contraction[k, edge] = value
            yi = cute.make_rmem_tensor(width, Float32)
            ri = cute.make_rmem_tensor(width, Float32)
            vi = cute.make_rmem_tensor(width, Float32)
            yi.fill(0.0)
            ri.fill(0.0)
            vi.fill(0.0)
            pk = _load_fp32_vector(compact, coff + self._row(self.queries_off, last, b, self.K), self.K)
            for k in cutlass.range_constexpr(self.K):
                g_row = gradient[k, None]
                for v in cutlass.range_constexpr(width):
                    vi[v] += g_row[v] * pk[k]
            if inside:
                yi = _load_dense(dense, doff + self._row(self.cdy_off, 0, b, self.D) + d, width)
                ri = _load_dense(dense, doff + self._row(self.rr_off, last, b, self.D) + d, width)
                for k in cutlass.range_constexpr(self.K):
                    g_row = gradient[k, None]
                    value = Float32(0.0)
                    for v in cutlass.range_constexpr(width):
                        value += g_row[v] * ri[v]
                    contraction[k, last] = value
                for v in cutlass.range_constexpr(width):
                    vi[v] += alpha * yi[v]
                residual_dy += _dot(ri, yi, width)
                residual_dv += _dot(ri, vi, width)
            reduced_future_grad = _reduce_scatter_columns(contraction, self.K, 1, lane)
            for edge in cutlass.range_constexpr(self.E):
                future_grad_accum[edge] += reduced_future_grad[edge]
            # This group's future DY and DV rows were staged before, so they can be overwritten.
            if inside:
                dy_current = Int64(last) * dy.stride[0] + Int64(group) * dy.stride[2] + Int64(d)
                _store_bf16_vector(dy, dy_current, yi, width)
                dv_current = Int64(last) * dv.stride[0] + Int64(group) * dv.stride[2] + Int64(d)
                _store_bf16_vector(dv, dv_current, vi, width)
                raw = cute.make_rmem_tensor(width, Float32)
                raw.store(vi.load() * write_scale)
                _store_bf16_vector(grad_rawvalue, Int64(group) * self.D + Int64(d), raw, width)
            tile_accum = cute.make_rmem_tensor((self.K, 2), Float32)
            tile_accum.fill(0.0)
            memory = cute.make_rmem_tensor((self.K, width), Float32)
            memory.fill(0.0)
            if inside:
                if cutlass.const_expr(not self.first_window):
                    for k in cutlass.range_constexpr(self.K):
                        loaded = _load_dense(dense, doff + self._state(self.m_off, b, k) + d, width)
                        for v in cutlass.range_constexpr(width):
                            memory[k, v] = loaded[v]
                            tile_accum[k, 0] += loaded[v] * yi[v]
                            tile_accum[k, 1] -= loaded[v] * vi[v]
                for edge in cutlass.range_constexpr(self.E):
                    rr = _load_dense(dense, doff + self._row(self.rr_off, edge, b, self.D) + d, width)
                    gain_row = _load_fp32_vector(compact, coff + self._row(self.gains_off, edge, b, self.K), self.K)
                    for k in cutlass.range_constexpr(self.K):
                        gain = gain_row[k]
                        for v in cutlass.range_constexpr(width):
                            if cutlass.const_expr(self.first_window):
                                memory[k, v] += gain * rr[v]
                            else:
                                memory[k, v] -= gain * rr[v]
                for k in cutlass.range_constexpr(self.K):
                    if cutlass.const_expr(self.first_window):
                        for v in cutlass.range_constexpr(width):
                            tile_accum[k, 0] += memory[k, v] * yi[v]
                            tile_accum[k, 1] -= memory[k, v] * vi[v]
                        zero = cute.make_rmem_tensor(width, BFloat16)
                        zero.fill(0.0)
                        _store_vector(m_start, (Int64(group) * self.K + k) * self.D + Int64(d), zero, width)
                    else:
                        _store_bf16_vector(m_start, (Int64(group) * self.K + k) * self.D + Int64(d),
                                           memory[k, None], width)
            reduced_memory = _reduce_scatter_columns(tile_accum, self.K, 1, lane)
            q_accum += reduced_memory[0]
            k_accum += reduced_memory[1]
        # Every lane holds the future gain gradient sums of row lane % K.
        for bank in cutlass.range_constexpr(cute.ceil_div(self.E - 1, per_bank)):
            edge = bank * per_bank + lane // self.K
            active = edge < self.E - 1
            value = future_grad_accum[bank * per_bank]
            for other in cutlass.range_constexpr(bank * per_bank + 1, min((bank + 1) * per_bank, self.E - 1)):
                if edge == other:
                    value = future_grad_accum[other]
            # The transposed factors of the later edges apply in descending edge order.
            for future in cutlass.range_constexpr(self.E - 1, bank * per_bank, -1):
                applies = active & (future > edge)
                dot = Float32(0.)
                if applies:
                    dot = compact[coff + self._row(self.gains_off, future, b, self.K) + key_index] * value
                dot = lane_sum(dot, self.K)
                if applies:
                    value -= compact[coff + self._row(self.keys_off, future, b, self.K) + key_index] * dot
            if active:
                future_gain_grad[edge, n, g, key_index] = value
        ry = cute.arch.warp_reduction(residual_dy, operator.add)
        rv = cute.arch.warp_reduction(residual_dv, operator.add)
        zi = u_last * ry
        if mine:
            # zi is added up in finish: here the compiler would fuse the product and the sum into
            # one FMA, which rounds once instead of twice.
            carry[0] = zi
            carry[1] = future_grad_accum[last]
            carry[2] = q_accum
            carry[3] = k_accum
        for part in cutlass.range_constexpr(cute.ceil_div(words, 32)):
            index = part * 32 + lane
            if index < words:
                slots[sw + self.slot_c + index] = compact[coff + self._row(self.c_off, 0, b, words) + index]
                slot_p[sp + index] = dense[doff + self._row(self.p_off, 0, b, words) + index]
        if lane < self.K:
            slots[sw + self.slot_key + lane] = compact[coff + self._row(self.keys_off, last, b, self.K) + lane]
            slots[sw + self.slot_gain + lane] = compact[coff + self._row(self.gains_off, last, b, self.K) + lane]
        if lane == 0:
            slots[sw + self.slot_scalars] = compact[coff + self._row(self.lam_off, last, b, 1)]
            slots[sw + self.slot_scalars + 1] = compact[coff + self._row(self.den_off, last, b, 1)]
            slots[sw + self.slot_scalars + 2] = write_scale
            slots[sw + self.slot_scalars + 3] = rv
            slot_groups[sg] = Int32(group)

    @cute.jit
    def finish(self, count, slots: cute.Tensor, slot_p: cute.Tensor, slot_groups: cute.Tensor,
               slot_base, slot_p_base, group_base, lane, carry: cute.Tensor, dq: cute.Tensor,
               covariance_grad: cute.Tensor, grad_forgetting: cute.Tensor, grad_rawkey: cute.Tensor,
               grad_write_scale: cute.Tensor):
        """The last edge's covariance VJP and the key, query, forgetting, scale and C outputs of
        the ``count`` groups in the finish slots, one run of K lanes per group."""
        K: cutlass.Constexpr = self.K
        words: cutlass.Constexpr = self.words
        last: cutlass.Constexpr = self.E - 1
        cute.arch.sync_warp()
        row = lane % K
        run = lane // K
        active = run < count
        sw = slot_base + run * self.slot_words
        sp = slot_p_base + run * self.slot_p
        group = Int64(slot_groups[group_base + run])
        n, g = group // self.G, group % self.G
        end_state_row = cute.make_rmem_tensor(K, Float32)
        crow = cute.make_rmem_tensor(K, Float32)
        end_state_row.fill(0.)
        crow.fill(0.)
        key_lane, gain_lane = Float32(0.), Float32(0.)
        if active:
            key_lane = slots[sw + self.slot_key + row]
            gain_lane = slots[sw + self.slot_gain + row]
            for k in cutlass.range_constexpr(K):
                packed = _symmetric_offset(row, k, K)
                end_state_row[k] = Float32(slot_p[sp + packed])
                crow[k] = slots[sw + self.slot_c + packed]
        key_row = _load_fp32_vector(slots, sw + self.slot_key, K)
        gain_row = _load_fp32_vector(slots, sw + self.slot_gain, K)
        scalars = _load_fp32_vector(slots, sw + self.slot_scalars, 4)
        dk_cov, d_forgetting, previous = covariance_vjp(
            end_state_row, crow, key_row, gain_row, key_lane, gain_lane, carry[0] + carry[1], scalars[1],
            scalars[0], row, active, run * K, K)
        local_pairing = Float32(0.)
        if active:
            key_gradient = carry[3] + gain_lane * scalars[3] + dk_cov
            dq[last, n, g, row] = carry[2]
            grad_rawkey[n, g, row] = scalars[2] * key_gradient
            local_pairing = dk_cov * key_lane
            for k in cutlass.range_constexpr(K):
                if k >= row:
                    slots[sw + self.slot_c + _symmetric_offset(row, k, K)] = previous[k]
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
                _store_fp32_row(covariance_grad, Int64(slot_groups[group_base + slot]) * words + piece,
                                values, store_width, store_width)
        covariance_pairing = lane_sum(local_pairing, K)
        if active & (row == 0):
            grad_forgetting[last, n, g] = d_forgetting
            scale_gradient = Float32(0.)
            if scalars[2] != Float32(0.):
                scale_gradient = (scalars[3] + covariance_pairing) / scalars[2]
            grad_write_scale[n, g] = scale_gradient
        # Every lane is done with the slots before the next group's copy overwrites them.
        cute.arch.sync_warp()

    @cute.jit
    def __call__(self, m_end: cute.Tensor, dm_end: cute.Tensor,
                 keys: cute.Tensor, queries: cute.Tensor,
                 gains: cute.Tensor, residuals: cute.Tensor, forgetting_tape: cute.Tensor,
                 denominators: cute.Tensor, future_queries: cute.Tensor,
                 future_keys: cute.Tensor, future_dy: cute.Tensor, future_dv: cute.Tensor,
                 m_start: cute.Tensor, dm_start: cute.Tensor,
                 future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor,
                 current_dy: cute.Tensor, write_scales: cute.Tensor, dy: cute.Tensor,
                 dv: cute.Tensor, dq: cute.Tensor, covariance_grad: cute.Tensor,
                 grad_forgetting: cute.Tensor, grad_rawvalue: cute.Tensor,
                 grad_rawkey: cute.Tensor, grad_write_scale: cute.Tensor, saved_p, stream):
        for tensor in (future_dy, future_dv, dy, dv):
            assert tensor.stride[3] == 1
            assert tensor.stride[1] == self.G * tensor.stride[2]
        # A batch's staged rows are contiguous: consecutive groups follow each other.
        for tensor in (future_dy, future_dv):
            assert tensor.stride[2] == self.D
        assert tuple(current_dy.stride) == (self.G * self.D, self.D, 1)
        for tensor, shape, dtype in (
            (saved_p, (self.E, self.N, self.G, self.words), BFloat16),
            (covariance_grad, (self.N, self.G, self.words), Float32),
        ):
            assert tensor.element_type == dtype
            assert tuple(tensor.shape) == shape
            stride = 1
            for axis in cutlass.range_constexpr(len(shape) - 1, -1, -1):
                assert tensor.stride[axis] == stride
                stride *= shape[axis]
        self.kernel(m_end, dm_end, keys, queries, gains, residuals,
                    forgetting_tape, denominators, future_queries, future_keys, future_dy,
                    future_dv, m_start, dm_start, future_value_grad,
                    future_gain_grad, current_dy, write_scales, dy, dv, dq, covariance_grad,
                    grad_forgetting, grad_rawvalue, grad_rawkey, grad_write_scale, saved_p).launch(
                        grid=(self.blocks, 1, 1), block=(self.warps * 32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, m_end: cute.Tensor, dm_end: cute.Tensor,
               keys: cute.Tensor, queries: cute.Tensor,
               gains: cute.Tensor, residuals: cute.Tensor, forgetting_tape: cute.Tensor,
               denominators: cute.Tensor, future_queries: cute.Tensor,
               future_keys: cute.Tensor, future_dy: cute.Tensor, future_dv: cute.Tensor,
               m_start: cute.Tensor, dm_start: cute.Tensor,
               future_value_grad: cute.Tensor, future_gain_grad: cute.Tensor,
               current_dy: cute.Tensor, write_scales: cute.Tensor, dy: cute.Tensor,
               dv: cute.Tensor, dq: cute.Tensor, covariance_grad: cute.Tensor,
               grad_forgetting: cute.Tensor, grad_rawvalue: cute.Tensor,
               grad_rawkey: cute.Tensor, grad_write_scale: cute.Tensor, saved_p: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        # The lane width is limited by the alignment of the BF16 tapes it writes.
        width = math.gcd(self.D, self.lane_width, min(
            dm_start.iterator.max_alignment, m_start.iterator.max_alignment,
            dy.iterator.max_alignment,
            grad_rawvalue.iterator.max_alignment) // 2)
        width = math.gcd(width, dy.stride[0], dy.stride[2])
        width = math.gcd(width, dv.iterator.max_alignment // 2, dv.stride[0], dv.stride[2])
        smem = SmemAllocator()
        dense = smem.allocate_tensor(BFloat16, cute.make_layout(
            self.stages * self.dense_elems), byte_alignment=128)
        compact = smem.allocate_tensor(Float32, cute.make_layout(
            self.stages * self.compact_words), byte_alignment=16)
        slots = smem.allocate_tensor(Float32, cute.make_layout(
            self.consumers * self.pack * self.slot_words), byte_alignment=16)
        slot_p = smem.allocate_tensor(BFloat16, cute.make_layout(
            self.consumers * self.pack * self.slot_p), byte_alignment=16)
        slot_groups = smem.allocate_tensor(Int32, cute.make_layout(
            self.consumers * self.pack), byte_alignment=16)
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
        if warp == self.consumers:
            self.produce(first, stride, mine, dense, compact, full_bars, empty_bars, lane,
                         m_end, dm_end, keys, queries, gains, residuals, forgetting_tape,
                         denominators, future_queries, future_keys, future_dy, future_dv,
                         current_dy, write_scales, covariance_grad, saved_p)
        else:
            self.consume(first, stride, mine, dense, compact, full_bars, empty_bars, lane, warp,
                         width, m_start, dm_start, future_value_grad,
                         future_gain_grad, dy, dv, dq, covariance_grad, grad_forgetting, grad_rawvalue,
                         grad_rawkey, grad_write_scale, slots, slot_p, slot_groups)


def default_lane_width(D: int) -> int:
    """Lane width of the D loops (32 lanes x width columns per tile): 4 fills a warp at D >= 128;
    at D <= 64 it would leave lanes 16-31 idle in every tile, so D <= 64 uses 2."""
    return 2 if D <= 64 else 4


def make_window_end(E: int, F: int, N: int, G: int, K: int, D: int, *, sms: int, batch: int = 4,
                    consumers: int = 4, blocks_per_sm: int = 2, first_window: bool = False):
    """The window-end kernel for one window shape with its launch schedule: groups per shared
    stage, consumer warps and CTAs per SM. The D-loop lane width is fixed by default_lane_width
    because it decides which columns each lane adds up before the butterfly, which is part of the
    summation order."""
    if any(type(size) is not int or size <= 0 for size in (E, N, G, K, D)):
        raise ValueError("E, N, G, K, D must be positive Python integers")
    if type(F) is not int or F < 0:
        raise ValueError("F must be a nonnegative integer")
    if K not in (4, 8, 16) or D % 8:
        raise ValueError("the window end requires K in {4, 8, 16} and D % 8 == 0")
    if type(first_window) is not bool:
        raise ValueError("first_window must be a Python bool")
    if any(type(v) is not int or v <= 0 for v in (sms, batch, consumers, blocks_per_sm)):
        raise ValueError("sms, batch, consumers and blocks_per_sm must be positive integers")
    if consumers > 31 or batch % consumers:
        raise ValueError("consumers must divide batch and be at most 31")
    return WindowEnd(E, F, N, G, K, D, sms=sms, batch=batch, consumers=consumers,
                     blocks_per_sm=blocks_per_sm, first_window=first_window)
