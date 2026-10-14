"""The SM90 projection kernel: residual add, row RMS, the first layer on WGMMA tensor cores and the
activation. `projection` describes what it computes. The weights are the edge's ``first_layer``, BF16 in
128-column tiles ``[D/128, 2H, 128]`` (`edges.InferenceEdge`).
"""

import math

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Float32, Int64, cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.utils import SmemAllocator

from .elementwise import sigmoid


class FirstProjection:
    def __init__(self, model_dim, hidden, context_eps, has_delta):
        if model_dim <= 0 or model_dim % 8 or not 0 < hidden <= 64:
            raise ValueError("positive width divisible by eight and hidden width at most 64 required")
        if not math.isfinite(context_eps) or context_eps <= 0:
            raise ValueError("context_eps must be positive and finite")
        self.model_dim, self.hidden = model_dim, hidden
        self.context_eps, self.has_delta = float(context_eps), bool(has_delta)
        self.tokens, self.threads, self.reduction, self.stages = 64, 256, 128, 2
        self.warps = self.threads // 32
        self.columns = (2 * hidden + 15) // 16 * 16
        self.shared_bytes = (2 * (2 * self.stages * self.tokens + self.columns) * self.reduction
                             + 4 * self.tokens + 8 * (self.stages + 1))

    @cute.jit
    def __call__(self, source: cute.Tensor, delta: cute.Tensor, weight: cute.Tensor,
                 bias: cute.Tensor, hidden: cute.Tensor, alpha: Float32, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(warpgroup.MmaF16BF16Op(
            BFloat16, Float32, (self.tokens, self.columns, 16),
            warpgroup.OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K))
        layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, BFloat16)
        a_layout = cute.tile_to_shape(
            layout_atom, (self.tokens, self.reduction, self.stages), order=(0, 1, 2))
        b_layout = cute.tile_to_shape(
            layout_atom, (self.columns, self.reduction, 1), order=(0, 1, 2))
        a_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), source, a_layout,
            (self.tokens, self.reduction))
        delta_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), delta, a_layout,
            (self.tokens, self.reduction))
        packed = cute.make_tensor(weight.iterator, cute.make_layout(
            (self.columns, self.reduction, weight.shape[0]),
            stride=(self.reduction, 1, self.columns * self.reduction)))
        b_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), packed, b_layout,
            (self.columns, self.reduction))
        self.kernel(source, bias, hidden, alpha,
                    mma, a_layout, b_layout, a_tma, delta_tma, b_tma).launch(
            grid=(cute.ceil_div(source.shape[0], self.tokens), 1, 1),
            block=(self.threads, 1, 1), smem=self.shared_bytes, stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor,
               bias: cute.Tensor, hidden: cute.Tensor,
               alpha: Float32, mma: cute.TiledMma,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               a_tma: cpasync.TmaInfo, delta_tma: cpasync.TmaInfo,
               b_tma: cpasync.TmaInfo):
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        warp = cute.arch.make_warp_uniform(tid // 32)
        token0 = Int64(cute.arch.block_idx()[0]) * self.tokens
        smem = SmemAllocator()
        a_s = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=128,
                                   swizzle=a_layout.inner)
        b_s = smem.allocate_tensor(BFloat16, b_layout.outer, byte_alignment=128,
                                   swizzle=b_layout.inner)
        delta_s = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=128,
                                       swizzle=a_layout.inner)
        inverse_s = smem.allocate_tensor(Float32, cute.make_layout(self.tokens), byte_alignment=16)
        b_ready = smem.allocate_array(Int64, 1)
        input_ready = smem.allocate_array(Int64, self.stages)
        if tid == 0:
            cute.arch.mbarrier_init(b_ready, 1)
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(input_ready + stage, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tma_sB, tma_gB = cpasync.tma_partition(
            b_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(b_s, 0, 2),
            cute.group_modes(b_tma.tma_tensor, 0, 2))
        gA = cute.local_tile(a_tma.tma_tensor, (self.tokens, self.reduction), (None, None))
        gDelta = cute.local_tile(delta_tma.tma_tensor,
                                 (self.tokens, self.reduction), (None, None))
        tma_sA, tma_gA = cpasync.tma_partition(
            a_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(a_s, 0, 2), cute.group_modes(gA, 0, 2))
        tma_sDelta, tma_gDelta = cpasync.tma_partition(
            delta_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(delta_s, 0, 2), cute.group_modes(gDelta, 0, 2))
        if warp == 0:
            cpasync.prefetch_descriptor(a_tma.atom)
            if cutlass.const_expr(self.has_delta):
                cpasync.prefetch_descriptor(delta_tma.atom)
            cpasync.prefetch_descriptor(b_tma.atom)
        vector_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                                          num_bits_per_copy=128)
        thread = mma.get_slice(tid % 128)
        a_frag = mma.make_fragment_A(thread.partition_A(a_s))
        b_frag = mma.make_fragment_B(thread.partition_B(b_s))
        cc = thread.partition_C(cute.make_identity_tensor((self.tokens, self.columns)))
        acc = cute.make_rmem_tensor(cc.shape, Float32)
        acc.fill(0.0)
        sums = cute.make_rmem_tensor(self.tokens // self.warps, Float32)
        sums.fill(0.0)
        mma.set(warpgroup.Field.ACCUMULATE, True)
        if warp < 4:
            warpgroup.fence()
        if warp == 0:
            self.load_input(a_tma.atom, delta_tma.atom, tma_gA, tma_sA,
                            tma_gDelta, tma_sDelta, input_ready, 0)

        for block_k in cutlass.range(cute.ceil_div(self.model_dim, self.reduction)):
            stage = block_k % self.stages
            # The previous MMA group still reads the weight tile and the stage the next load
            # overwrites, so it has to finish before either copy is issued.
            if block_k > 0:
                if warp < 4:
                    warpgroup.wait_group(0)
                cute.arch.sync_threads()
            if warp == 0:
                if block_k + 1 < cute.ceil_div(self.model_dim, self.reduction):
                    self.load_input(a_tma.atom, delta_tma.atom, tma_gA, tma_sA,
                                    tma_gDelta, tma_sDelta, input_ready, block_k + 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        b_ready, 2 * self.columns * self.reduction)
                # The copy elects its own issuing lane, so the whole warp calls it.
                cute.copy(b_tma.atom, tma_gB[None, block_k], tma_sB[None, 0],
                          tma_bar_ptr=b_ready)
            cute.arch.mbarrier_wait(input_ready + stage, (block_k // self.stages) & 1)

            # Lane l sums the squares of columns 8l to 8l + 7 of every 256, as the decode kernel does, so
            # the two half-warps trade rows on odd tiles.
            owner = (lane // 16) ^ (block_k % 2)
            column = block_k * self.reduction + (lane % 16) * 8
            for pair in cutlass.range_constexpr(self.tokens // (2 * self.warps)):
                row = warp + (2 * pair + owner) * self.warps
                token = token0 + row
                offset_s = a_s.layout((row, (lane % 16) * 8, stage))
                tile_values = cute.make_tensor(a_s.iterator + cute.assume(offset_s, divby=8),
                                               cute.make_layout(8))
                values = cute.make_rmem_tensor(8, BFloat16)
                values.fill(0.0)
                if (token < source.shape[0]) & (column < self.model_dim):
                    offset = token * self.model_dim + column
                    cute.copy(vector_copy, tile_values, values)
                    if cutlass.const_expr(self.has_delta):
                        branch = cute.make_rmem_tensor(8, BFloat16)
                        src_delta = cute.make_tensor(delta_s.iterator + cute.assume(offset_s, divby=8),
                                                     cute.make_layout(8))
                        cute.copy(vector_copy, src_delta, branch)
                        for i in cutlass.range_constexpr(8):
                            values[i] = BFloat16(Float32(values[i]) + alpha * Float32(branch[i]))
                        dst = cute.make_tensor(source.iterator + cute.assume(offset, divby=8),
                                               cute.make_layout(8))
                        cute.copy(vector_copy, values, dst)
                    if owner == 0:
                        for i in cutlass.range_constexpr(8):
                            value = Float32(values[i])
                            sums[2 * pair] += value * value
                    else:
                        for i in cutlass.range_constexpr(8):
                            value = Float32(values[i])
                            sums[2 * pair + 1] += value * value
                cute.copy(vector_copy, values, tile_values)

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp < 4:
                cute.arch.mbarrier_wait(b_ready, block_k & 1)
                for k in cutlass.range_constexpr(self.reduction // 16):
                    cute.gemm(mma, acc, a_frag[None, None, k, stage],
                              b_frag[None, None, k, 0], acc)
                warpgroup.commit_group()

        for row_part in cutlass.range_constexpr(self.tokens // self.warps):
            total = sums[row_part]
            for bit in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total, 1 << bit)
            if lane == 0:
                inverse_s[warp + row_part * self.warps] = cute.math.rsqrt(
                    total / Float32(self.model_dim) + Float32(self.context_eps))
        if warp < 4:
            warpgroup.wait_group(0)
        cute.arch.sync_threads()
        if warp < 4:
            for i in cutlass.range_constexpr(cute.size(cc)):
                row, column = cc[i]
                token = token0 + row
                if (token < source.shape[0]) & (column < 2 * self.hidden):
                    z = acc[i] * inverse_s[row]
                    z = z + Float32(bias[column])
                    z = Float32(BFloat16(z))
                    hidden[column // self.hidden, token, column % self.hidden] = BFloat16(z * sigmoid(z))

    @cute.jit
    def load_input(self, a_atom: cute.CopyAtom, delta_atom: cute.CopyAtom,
                   gA: cute.Tensor, sA: cute.Tensor, gDelta: cute.Tensor,
                   sDelta: cute.Tensor, ready: cute.Pointer, block_k):
        stage = block_k % self.stages
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                ready + stage, 2 * self.tokens * self.reduction * (1 + int(self.has_delta)))
        token_tile = cute.arch.block_idx()[0]
        cute.copy(a_atom, gA[None, token_tile, block_k], sA[None, stage],
                  tma_bar_ptr=ready + stage)
        if cutlass.const_expr(self.has_delta):
            cute.copy(delta_atom, gDelta[None, token_tile, block_k], sDelta[None, stage],
                      tma_bar_ptr=ready + stage)
