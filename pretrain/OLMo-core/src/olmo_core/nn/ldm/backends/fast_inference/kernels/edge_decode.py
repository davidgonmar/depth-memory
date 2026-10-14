"""The decode edge kernel: a whole edge for few tokens in one launch, the address networks included.

The grid holds ``ceil(N / tokens)`` clusters of `cluster` CTAs (`edge.EdgeRows`), and CTA c also computes
``R = ceil(2H / C)`` rows of the first layer. Before the grid-dependency wait it requests the head and gate
rows of its groups (``second_layer``), the first-layer bias and the first stages of its first-layer rows.

After the wait, `_stream_rows` adds the branch to the stream rows of the cluster's tokens, giving the BF16
stream the backbone continues with. `_hidden_units` multiplies each warp's first-layer rows with the stream
rows (lanes over 8 columns, then a butterfly) and sums the rows' squares, which gives the controller RMS
and the BF16 hidden units ``SiLU(BF16(u * rms + bias))``. The first layer streams through `weight_stages`
shared-memory stages of `weight_chunk` columns, and a single stage of d_model columns keeps it resident.
Every CTA stores its hidden units into every CTA of the cluster through distributed shared memory. Once
every CTA of the cluster has read its stream rows, `_write_stream` writes the CTA's slice of the updated
stream over the source. `_head_gate_columns` forms the head and gate projections of the CTA's rows, each
summed over `slices` lanes, and the row stages of `edge.EdgeRows` finish the edge on the dense state,
which was requested right after the wait.

The kernel is a programmatic dependent launch, so its parameter loads overlap the previous kernel, and the
next kernel may launch once every CTA has passed the grid-dependency wait.
"""

import operator
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.utils import SmemAllocator

from .addresses import Projections
from .copies import bulk_copy, load_bf16, load_fp32, store
from .edge import THREADS, VECTOR, WARPS, EdgeRows, _push
from .elementwise import sigmoid

TILE = 128                    # first-layer columns per global tile
ROW_PAD = 8                   # BF16 padding of a second-layer row


class Networks(NamedTuple):
    """The shared memory of the address networks, which only the decode kernel computes."""

    weights: cute.Tensor          # [weight_stages, weight_chunk / 128, R, 128] BF16 first-layer rows
    second: cute.Tensor           # [2, G/C, K + 1, H + 8] BF16 head and gate rows of the CTA's groups
    bias: cute.Tensor             # [2H] BF16 first-layer bias
    stream: cute.Tensor           # [T, d_model] BF16 stream rows
    hidden: cute.Tensor           # [2, T, H + 4] FP32 hidden units of both sites
    columns: cute.Tensor          # FP32 head and gate projections of every row and site
    parameters_bar: cute.Pointer  # the parameters requested before the grid-dependency wait
    hidden_bar: cute.Pointer      # hidden units from every CTA of the cluster
    weights_full: cute.Pointer    # [weight_stages] a first-layer stage has landed
    weights_empty: cute.Pointer   # [weight_stages] every warp is done with a first-layer stage


class EdgeDecode(EdgeRows):
    prefill = False

    def __init__(self, E, G, K, D, hidden, tiling, *, has_delta, context_eps, **options):
        super().__init__(E, G, K, D, tokens=tiling.tokens, cluster=tiling.cluster, key_lanes=tiling.key_lanes,
                         **options)
        C = tiling.cluster
        if hidden not in (32, 64) or hidden % K:
            raise ValueError(f"hidden width {hidden}: 32 or 64, a multiple of K={K}, is required")
        if not 1 <= tiling.slices <= 32 or hidden % tiling.slices or tiling.slices & (tiling.slices - 1):
            raise ValueError(f"{tiling.slices} head/gate slices: a power of two dividing {hidden} is required")
        if tiling.weight_chunk % TILE or self.model_dim % tiling.weight_chunk:
            raise ValueError(f"first-layer chunk of {tiling.weight_chunk} columns: a multiple of {TILE} dividing "
                             f"d_model={self.model_dim} is required")
        if not 1 <= tiling.weight_stages <= self.model_dim // tiling.weight_chunk:
            raise ValueError(f"{tiling.weight_stages} first-layer stages for "
                             f"{self.model_dim // tiling.weight_chunk} chunks")
        self.tiling, self.hidden = tiling, hidden
        self.has_delta, self.context_eps = bool(has_delta), float(context_eps)
        self.slices = tiling.slices
        self.weight_chunk, self.weight_stages = tiling.weight_chunk, tiling.weight_stages
        self.weight_chunks = self.model_dim // tiling.weight_chunk
        self.chunk_tiles = tiling.weight_chunk // TILE
        self.lane_pieces = -(-tiling.weight_chunk // VECTOR // 32)
        self.row_pieces = self.model_dim // VECTOR
        self.rows = -(-2 * hidden // C)
        self.warp_rows = -(-self.rows // WARPS)
        self.hidden_row = hidden + 4
        self.second_row = hidden + ROW_PAD
        self.second_block = self.local_groups * (K + 1) * self.second_row     # one site's rows of the CTA
        self.columns = self.pairs * 2 * (K + 1)
        self.shared_bytes = self._total(self._network_regions() + self._row_regions())

    def _network_regions(self):
        T = self.tokens
        return ((2 * self.weight_stages * self.rows * self.weight_chunk, 128), (2 * 2 * self.second_block, 16),
                (4 * self.hidden, 16), (2 * T * self.model_dim, 16), (4 * 2 * T * self.hidden_row, 16),
                (4 * self.columns, 16), (8 * (2 + 2 * self.weight_stages), 8))

    @cute.jit
    def __call__(self, source: cute.Tensor, delta: cute.Tensor, value: cute.Tensor,
                 first_layer: cute.Tensor, first_bias: cute.Tensor, second_layer: cute.Tensor,
                 head_bias: cute.Tensor, gate_bias: cute.Tensor, static_address: cute.Tensor,
                 read_gain: cute.Tensor, forgetting: cute.Tensor, covariance: cute.Tensor, memory: cute.Tensor,
                 residual_history: cute.Tensor, gain_history: cute.Tensor, output: cute.Tensor,
                 norm_weight: cute.Tensor, residual_scale: Float32, edge: Int32, stream: cuda.CUstream):
        clusters = cute.ceil_div(source.shape[0], self.tokens)
        self.kernel(source, delta, value, first_layer, first_bias, second_layer, head_bias, gate_bias,
                    static_address, read_gain, forgetting, covariance, memory, residual_history, gain_history,
                    output, norm_weight, residual_scale, edge).launch(
            grid=(self.cluster * clusters, 1, 1), block=(THREADS, 1, 1), cluster=(self.cluster, 1, 1),
            smem=self.shared_bytes, stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, source: cute.Tensor, delta: cute.Tensor, value: cute.Tensor,
               first_layer: cute.Tensor, first_bias: cute.Tensor, second_layer: cute.Tensor,
               head_bias: cute.Tensor, gate_bias: cute.Tensor, static_address: cute.Tensor,
               read_gain: cute.Tensor, forgetting: cute.Tensor, covariance: cute.Tensor, memory: cute.Tensor,
               residual_history: cute.Tensor, gain_history: cute.Tensor, output: cute.Tensor,
               norm_weight: cute.Tensor, residual_scale: Float32, edge: Int32):
        K = self.K
        place = self._place(source.shape[0])
        smem = SmemAllocator()
        networks = self._allocate_networks(smem)
        rows = self._allocate_rows(smem)
        if place.tid == 0:
            cute.arch.mbarrier_init(networks.parameters_bar, 1)
            cute.arch.mbarrier_init(networks.hidden_bar, 1)
            for s in cutlass.range_constexpr(self.weight_stages):
                cute.arch.mbarrier_init(networks.weights_full + s, 1)
                cute.arch.mbarrier_init(networks.weights_empty + s, WARPS)
        self._init_row_barriers(place, rows)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Nothing earlier in the stream writes the parameters, so they load before the grid-dependency wait.
        self._request_parameters(place, networks, rows, first_layer, second_layer, first_bias, norm_weight)
        pair, kidx = place.tid // K, place.tid % K
        pair_active = (pair < self.pairs) & (pair // self.local_groups < place.live)
        parameters = self.addresses.lane_parameters(place.first_group + pair % self.local_groups, kidx,
                                                    pair_active, head_bias, gate_bias, static_address, read_gain,
                                                    forgetting)
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        self._request_state(place, rows, source, value, covariance, memory, residual_history, gain_history,
                            norm_weight, edge)
        self._stream_rows(place, networks, rows, source, delta, residual_scale)
        cute.arch.sync_threads()
        # This CTA has read its stream rows and initialized its barriers. Peers may now push into them, and
        # once every CTA has arrived, the updated stream may overwrite the source.
        cute.arch.cluster_arrive_relaxed()
        self._hidden_units(place, networks, first_layer)
        if cutlass.const_expr(self.has_delta):
            self._write_stream(place, rows, source)
        self._head_gate_columns(place, networks)
        cute.arch.mbarrier_wait(rows.state_bar, 0)
        cute.arch.sync_threads()
        head_key, head_query, gate_key, gate_query = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        if pair_active:
            head_key = networks.columns[(pair * 2) * (K + 1) + kidx]
            head_query = networks.columns[(pair * 2 + 1) * (K + 1) + kidx]
            gate_key = networks.columns[(pair * 2) * (K + 1) + K]
            gate_query = networks.columns[(pair * 2 + 1) * (K + 1) + K]
        gain_lane, pk_lane = self._addresses(place, rows, parameters, Projections(head_key, head_query, gate_key,
                                                                                  gate_query),
                                             pair, pair_active, gain_history, edge)
        cute.arch.sync_threads()
        residuals = self._dense_step(place, rows, residual_history, memory, output, edge)
        self._branch_norm(place, rows, parameters, pair, pair_active, gain_lane, pk_lane, covariance, output,
                          memory, norm_weight, residuals, edge)

    @cute.jit
    def _allocate_networks(self, smem):
        weights = smem.allocate_tensor(BFloat16, cute.make_layout(self.weight_stages * self.rows * self.weight_chunk),
                                       byte_alignment=128)
        second = smem.allocate_tensor(BFloat16, cute.make_layout(2 * self.second_block), byte_alignment=16)
        bias = smem.allocate_tensor(BFloat16, cute.make_layout(2 * self.hidden), byte_alignment=16)
        stream = smem.allocate_tensor(BFloat16, cute.make_layout(self.tokens * self.model_dim), byte_alignment=16)
        hidden = smem.allocate_tensor(Float32, cute.make_layout(2 * self.tokens * self.hidden_row), byte_alignment=16)
        columns = smem.allocate_tensor(Float32, cute.make_layout(self.columns), byte_alignment=16)
        bars = smem.allocate_array(Int64, 2 + 2 * self.weight_stages, byte_alignment=8)
        return Networks(weights, second, bias, stream, hidden, columns, bars, bars + 1, bars + 2,
                     bars + 2 + self.weight_stages)

    @cute.jit
    def _request_weights(self, place, networks: Networks, first_layer: cute.Tensor, stage, chunk):
        """Warp 0 requests the CTA's first-layer rows of chunk `chunk` into `stage`: one bulk copy per
        128-column tile (a tile's rows are contiguous in ``first_layer``)."""
        first_row = place.rank * self.rows
        own_rows = cutlass.max(cutlass.min(Int32(self.rows), Int32(2 * self.hidden) - first_row), Int32(0))
        full = networks.weights_full + stage
        if place.lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(full, own_rows * Int32(2 * TILE * self.chunk_tiles))
        cute.arch.sync_warp()
        for part in cutlass.range_constexpr(cute.ceil_div(self.chunk_tiles, 32)):
            tile = place.lane + 32 * part
            if (tile < self.chunk_tiles) & (own_rows > 0):
                bulk_copy(networks.weights, (stage * self.chunk_tiles + tile) * self.rows * TILE, first_layer,
                          (Int64(chunk * self.chunk_tiles + tile) * (2 * self.hidden) + first_row) * TILE,
                          own_rows * Int32(2 * TILE), full)

    @cute.jit
    def _request_parameters(self, place, networks: Networks, rows, first_layer: cute.Tensor,
                            second_layer: cute.Tensor, first_bias: cute.Tensor, norm_weight: cute.Tensor):
        """Warp 0 requests the head and gate rows of the CTA's groups (one block per site), the first-layer
        bias, the norm weight of the CTA's slice and the first first-layer stages."""
        K = self.K
        if place.warp == 0:
            if place.lane == 0:
                nbytes = 2 * 2 * self.second_block + 4 * self.hidden + (2 * self.width if self.norm else 0)
                cute.arch.mbarrier_arrive_and_expect_tx(networks.parameters_bar, Int32(nbytes))
                if cutlass.const_expr(self.norm):
                    bulk_copy(rows.norm_gain, 0, norm_weight, Int64(place.rank) * self.width, Int32(2 * self.width),
                              networks.parameters_bar)
                for site in cutlass.range_constexpr(2):
                    bulk_copy(networks.second, site * self.second_block, second_layer,
                              (Int64(site) * self.G + place.first_group) * (K + 1) * self.second_row,
                              Int32(2 * self.second_block), networks.parameters_bar)
                bulk_copy(networks.bias, 0, first_bias, 0, Int32(4 * self.hidden), networks.parameters_bar)
            for stage in cutlass.range_constexpr(self.weight_stages):
                self._request_weights(place, networks, first_layer, stage, stage)

    @cute.jit
    def _stream_rows(self, place, networks: Networks, rows, source: cute.Tensor, delta: cute.Tensor,
                     residual_scale: Float32):
        """The stream rows of the CTA's tokens with the branch added, for the first layer. The CTA's slice
        also goes to its dense rows."""
        D, MD, RP, PIECES = self.D, self.model_dim, self.row_pieces, self.pieces
        vector = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=16 * VECTOR)
        for step in cutlass.range_constexpr(cute.ceil_div(self.tokens * RP, THREADS)):
            index = step * THREADS + place.tid
            if index < self.tokens * RP:
                row_token, row_piece = index // RP, index % RP
                slice_piece = row_piece - place.rank * PIECES
                in_slice = (slice_piece >= 0) & (slice_piece < PIECES)
                piece = cute.make_rmem_tensor(VECTOR, BFloat16)
                piece.fill(0.0)
                if row_token < place.live:
                    address = (place.token0 + row_token) * MD + row_piece * VECTOR
                    cute.copy(vector, cute.make_tensor(source.iterator + cute.assume(address, divby=VECTOR),
                                                       cute.make_layout(VECTOR)), piece)
                    if cutlass.const_expr(self.has_delta):
                        branch = cute.make_rmem_tensor(VECTOR, BFloat16)
                        cute.copy(vector, cute.make_tensor(delta.iterator + cute.assume(address, divby=VECTOR),
                                                           cute.make_layout(VECTOR)), branch)
                        for i in cutlass.range_constexpr(VECTOR):
                            piece[i] = BFloat16(Float32(piece[i]) + residual_scale * Float32(branch[i]))
                cute.copy(vector, piece, cute.make_tensor(
                    networks.stream.iterator + cute.assume(row_token * MD + row_piece * VECTOR, divby=VECTOR),
                    cute.make_layout(VECTOR)))
                if in_slice:
                    slice_row = (row_token * self.local_groups + (slice_piece * VECTOR) // D) * D \
                        + (slice_piece * VECTOR) % D
                    cute.copy(vector, piece, cute.make_tensor(
                        rows.dense.iterator + cute.assume(self.stream_offset + slice_row, divby=VECTOR),
                        cute.make_layout(VECTOR)))

    @cute.jit
    def _write_stream(self, place, rows, source: cute.Tensor):
        """The CTA's slice of the updated stream, from its dense rows, over the source."""
        D, PIECES = self.D, self.pieces
        for step in cutlass.range_constexpr(cute.ceil_div(self.tokens * PIECES, THREADS)):
            index = step * THREADS + place.tid
            token, piece = index // PIECES, index % PIECES
            if (index < self.tokens * PIECES) & (token < place.live):
                at = (token * self.local_groups + (piece * VECTOR) // D) * D + (piece * VECTOR) % D
                store(source, (place.token0 + token) * self.model_dim + Int64(place.rank) * self.width + piece * VECTOR,
                      load_bf16(rows.dense, self.stream_offset + at, VECTOR), VECTOR)

    @cute.jit
    def _hidden_units(self, place, networks: Networks, first_layer: cute.Tensor):
        """Each warp's first-layer rows against the stream rows, chunk by chunk as the stages land, lanes over
        8 columns and then a butterfly. Every warp also forms the rows' sums of squares. Warp 0 refills a
        stage once every warp is done with it. The hidden units go to every CTA of the cluster."""
        T, R, RW, H, MD = self.tokens, self.rows, self.warp_rows, self.hidden, self.model_dim
        S, CHUNK = self.weight_stages, self.weight_chunk
        acc = cute.make_rmem_tensor(RW * T, Float32)
        squares = cute.make_rmem_tensor(T, Float32)
        acc.fill(0.0)
        squares.fill(0.0)
        for chunk in cutlass.range(0, self.weight_chunks, 1):
            stage = chunk % S
            phase = (chunk // S) % 2
            cute.arch.mbarrier_wait(networks.weights_full + stage, phase)
            for lane_piece in cutlass.range_constexpr(self.lane_pieces):
                piece = place.lane + 32 * lane_piece
                if piece < CHUNK // VECTOR:
                    column = chunk * CHUNK + piece * VECTOR
                    tile, within = (piece * VECTOR) // TILE, (piece * VECTOR) % TILE
                    elements = cute.make_rmem_tensor(T * VECTOR, Float32)
                    for t in cutlass.range_constexpr(T):
                        row_values = load_bf16(networks.stream, t * MD + column, VECTOR)
                        for i in cutlass.range_constexpr(VECTOR):
                            elements[t * VECTOR + i] = row_values[i]
                            squares[t] = row_values[i] * row_values[i] + squares[t]
                    for rw in cutlass.range_constexpr(RW):
                        weight_row = cutlass.min(rw * WARPS + place.warp, R - 1)
                        weights = load_bf16(networks.weights,
                                            ((stage * self.chunk_tiles + tile) * R + weight_row) * TILE + within,
                                            VECTOR)
                        for t in cutlass.range_constexpr(T):
                            for i in cutlass.range_constexpr(VECTOR):
                                acc[rw * T + t] = weights[i] * elements[t * VECTOR + i] + acc[rw * T + t]
            cute.arch.sync_warp()
            if place.lane == 0:
                cute.arch.mbarrier_arrive(networks.weights_empty + stage)
            if cutlass.const_expr(self.weight_chunks > S):
                if place.warp == 0:
                    if chunk + S < self.weight_chunks:
                        cute.arch.mbarrier_wait(networks.weights_empty + stage, phase)
                        self._request_weights(place, networks, first_layer, stage, chunk + S)
        for bit in cutlass.range_constexpr(5):
            for v in cutlass.range_constexpr(RW * T):
                acc[v] = acc[v] + cute.arch.shuffle_sync_bfly(acc[v], 1 << bit)
            for t in cutlass.range_constexpr(T):
                squares[t] = squares[t] + cute.arch.shuffle_sync_bfly(squares[t], 1 << bit)
        cute.arch.mbarrier_wait(networks.parameters_bar, 0)
        # Every CTA of the cluster has initialized its barriers and read its stream rows once this returns.
        cute.arch.cluster_wait()
        if place.tid == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(networks.hidden_bar, Int32(4 * 2 * H * T))
        first_row = place.rank * R
        for rw in cutlass.range_constexpr(RW):
            local_row = rw * WARPS + place.warp
            unit = first_row + local_row
            if (local_row < R) & (unit < 2 * H):
                for t in cutlass.range_constexpr(T):
                    z = acc[rw * T + t] * cute.math.rsqrt(squares[t] / Float32(MD) + Float32(self.context_eps))
                    z = Float32(BFloat16(z + Float32(networks.bias[unit])))
                    hidden_unit = Float32(BFloat16(z * sigmoid(z)))
                    if place.lane < self.cluster:
                        _push(networks.hidden, ((unit // H) * T + t) * self.hidden_row + unit % H, hidden_unit,
                              networks.hidden_bar, place.lane)
        cute.arch.mbarrier_wait(networks.hidden_bar, 0)

    @cute.jit
    def _head_gate_columns(self, place, networks: Networks):
        """The head and gate projections of the CTA's rows over the whole CTA: (row, site, column) items, each
        a sum of `slices` slices of the hidden width (four accumulators per slice, then a tree over slices)."""
        K = self.K
        H, T, SL = self.hidden, self.tokens, self.slices
        GC, SR, HR = self.local_groups, self.second_row, self.hidden_row
        TERMS: cutlass.Constexpr = H // SL
        for turn in cutlass.range_constexpr(cute.ceil_div(self.columns * SL, THREADS)):
            item, column_slice = (turn * THREADS + place.tid) // SL, place.tid % SL
            item_pair, item_site, item_column = item // (2 * (K + 1)), (item // (K + 1)) % 2, item % (K + 1)
            sums = cute.make_rmem_tensor(4, Float32)
            sums.fill(0.0)
            if (item < self.columns) & (item_pair // GC < place.live):
                weight_at = ((item_site * GC + item_pair % GC) * (K + 1) + item_column) * SR + column_slice * TERMS
                hidden_at = (item_site * T + item_pair // GC) * HR + column_slice * TERMS
                for part in cutlass.range_constexpr(cute.ceil_div(TERMS, VECTOR)):
                    width: cutlass.Constexpr = min(VECTOR, TERMS)
                    column_weight = load_bf16(networks.second, weight_at + part * width, width)
                    column_hidden = load_fp32(networks.hidden, hidden_at + part * width, width)
                    for i in cutlass.range_constexpr(width):
                        sums[i % 4] = column_hidden[i] * column_weight[i] + sums[i % 4]
            column_sum = (sums[0] + sums[1]) + (sums[2] + sums[3])
            if cutlass.const_expr(SL > 1):
                column_sum = cute.arch.warp_reduction(column_sum, operator.add, threads_in_group=SL)
            if (item < self.columns) & (column_slice == 0):
                networks.columns[item] = column_sum
