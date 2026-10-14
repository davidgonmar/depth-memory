"""The SM100 projection kernel: residual add, row RMS, the first layer on tcgen05 tensor cores with FP32
accumulation in tensor memory, and the activation. `projection` describes what it computes."""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.utils import LayoutEnum, SmemAllocator, TmemAllocator, get_num_tmem_alloc_cols
import cutlass.utils.blackwell_helpers as sm100

from .elementwise import sigmoid


class FirstProjection:
    def __init__(self, model_dim, hidden, context_eps, has_delta):
        if model_dim <= 0 or model_dim % 64:
            raise ValueError("model_dim must be positive and divisible by 64")
        if not 1 <= hidden <= 64:
            raise ValueError("SM100 first projection supports hidden widths 1 through 64")
        if not math.isfinite(context_eps) or context_eps <= 0:
            raise ValueError("context_eps must be positive and finite")
        self.model_dim = model_dim
        self.hidden = hidden
        self.context_eps = float(context_eps)
        self.has_delta = bool(has_delta)
        self.columns = (2 * hidden + 15) // 16 * 16
        self.rows = 64
        self.k_tile = 128
        self.stages = 2
        self.threads = 256
        self.input_warps = self.threads // 32
        self.shared_bytes = (2 * self.stages * self.rows * self.k_tile * (1 + self.has_delta)
                             + 2 * self.columns * self.k_tile
                             + 4 * self.rows + 256)

    @cute.jit
    def __call__(self, source: cute.Tensor, delta: cute.Tensor, weight: cute.Tensor,
                 bias: cute.Tensor, hidden: cute.Tensor, alpha: Float32, stream: cuda.CUstream):
        for tensor in (source, delta, weight, bias, hidden):
            assert tensor.element_type == BFloat16
        assert source.shape[1] == self.model_dim
        assert weight.shape[0] == cute.ceil_div(self.model_dim, self.k_tile)
        assert weight.shape[1] == self.columns
        assert weight.shape[2] == self.k_tile
        assert bias.shape[0] == 2 * self.hidden
        assert hidden.shape[0] == 2
        assert hidden.shape[2] == self.hidden
        mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            BFloat16, Float32, (self.rows, self.columns, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            OperandMajorMode.K, OperandMajorMode.K))
        a_layout = sm100.make_smem_layout(
            OperandMajorMode.K, (self.rows, self.k_tile), BFloat16, self.stages)
        b_layout = sm100.make_smem_layout(
            OperandMajorMode.K, (self.columns, self.k_tile), BFloat16, 1)
        a_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), source, a_layout,
            (self.rows, self.k_tile))
        delta_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), delta, a_layout,
            (self.rows, self.k_tile))
        packed = cute.make_tensor(weight.iterator, cute.make_layout(
            (self.columns, self.k_tile, weight.shape[0]),
            stride=(self.k_tile, 1, self.columns * self.k_tile)))
        b_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), packed, b_layout,
            (self.columns, self.k_tile))
        self.kernel(source, bias, hidden, alpha,
                    mma, a_layout, b_layout, a_tma, delta_tma, b_tma).launch(
            grid=(cute.ceil_div(source.shape[0], self.rows), 1, 1),
            block=(self.threads, 1, 1), smem=self.shared_bytes,
            stream=stream)

    @cute.kernel
    def kernel(self, source: cute.Tensor, bias: cute.Tensor,
               hidden: cute.Tensor,
               alpha: Float32, mma: cute.TiledMma,
               a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
               a_tma: cpasync.TmaInfo, delta_tma: cpasync.TmaInfo,
               b_tma: cpasync.TmaInfo):
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        warp = cute.arch.make_warp_uniform(tid // 32)
        first = Int64(cute.arch.block_idx()[0]) * self.rows
        D: cutlass.Constexpr = self.model_dim
        H: cutlass.Constexpr = self.hidden
        KT: cutlass.Constexpr = cute.ceil_div(D, self.k_tile)
        smem = SmemAllocator()
        sA = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=128,
                                  swizzle=a_layout.inner)
        sDelta = sA
        if cutlass.const_expr(self.has_delta):
            sDelta = smem.allocate_tensor(BFloat16, a_layout.outer, byte_alignment=128,
                                         swizzle=a_layout.inner)
        sB = smem.allocate_tensor(BFloat16, b_layout.outer, byte_alignment=128,
                                  swizzle=b_layout.inner)
        inverse = smem.allocate_tensor(Float32, cute.make_layout(self.rows))
        done = smem.allocate_array(Int64, 1)
        b_ready = smem.allocate_array(Int64, 1)
        input_ready = smem.allocate_array(Int64, self.stages)
        tmem_address = smem.allocate_array(Int32, 1)
        if tid == 0:
            cute.arch.mbarrier_init(done, 1)
            cute.arch.mbarrier_init(b_ready, 1)
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(input_ready + stage, 1)
        cute.arch.mbarrier_init_fence()
        tmem = TmemAllocator(tmem_address, barrier_for_retrieve=pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads))
        acc_shape = mma.partition_shape_C((self.rows, self.columns))
        acc_layout = mma.make_fragment_C(acc_shape).layout
        acc_fake = mma.make_fragment_C(acc_shape)
        tmem_columns: cutlass.Constexpr = get_num_tmem_alloc_cols(acc_fake)
        tmem.allocate(tmem_columns)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        acc = cute.make_tensor(tmem_ptr, acc_layout)
        tmem.relinquish_alloc_permit()
        thr_mma = mma.get_slice(0)
        a_frag = mma.make_fragment_A(thr_mma.partition_A(sA))
        b_frag = mma.make_fragment_B(thr_mma.partition_B(sB))
        gA = cute.local_tile(a_tma.tma_tensor, (self.rows, self.k_tile),
                            (cute.arch.block_idx()[0], None))
        gDelta = cute.local_tile(delta_tma.tma_tensor, (self.rows, self.k_tile),
                                (cute.arch.block_idx()[0], None))
        tma_sA, tma_gA = cpasync.tma_partition(
            a_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tma_sDelta, tma_gDelta = cpasync.tma_partition(
            delta_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(sDelta, 0, 2), cute.group_modes(gDelta, 0, 2))
        tma_sB, tma_gB = cpasync.tma_partition(
            b_tma.atom, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(b_tma.tma_tensor, 0, 2))
        if warp == 0:
            cpasync.prefetch_descriptor(a_tma.atom)
            if cutlass.const_expr(self.has_delta):
                cpasync.prefetch_descriptor(delta_tma.atom)
            cpasync.prefetch_descriptor(b_tma.atom)
            for kt in cutlass.range_constexpr(min(self.stages, KT)):
                self.prefetch_input(kt, kt, input_ready, a_tma.atom, delta_tma.atom,
                                    tma_gA, tma_sA, tma_gDelta, tma_sDelta)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                                   num_bits_per_copy=128)
        sums = cute.make_rmem_tensor(self.rows // self.input_warps, Float32)
        sums.fill(0.0)

        for kt in cutlass.range(KT):
            stage = kt % self.stages
            if warp == 0:
                if kt > 0:
                    cute.arch.mbarrier_wait(done, (kt - 1) & 1)
                    # The previous MMA is done with its stage of the stream and with the weight tile.
                    if kt + 1 < KT:
                        self.prefetch_input(kt + 1, (kt + 1) % self.stages,
                                            input_ready, a_tma.atom, delta_tma.atom,
                                            tma_gA, tma_sA, tma_gDelta, tma_sDelta)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        b_ready, 2 * self.columns * self.k_tile)
                cute.copy(b_tma.atom, tma_gB[None, kt], tma_sB[None, 0],
                          tma_bar_ptr=b_ready)
            cute.arch.mbarrier_wait(input_ready + stage, (kt // self.stages) & 1)

            # Lane l sums the squares of columns 8l to 8l + 7 of every 256, as the decode kernel does, so
            # the two half-warps trade rows on odd tiles.
            owner = (lane // 16) ^ (kt % 2)
            column = kt * self.k_tile + (lane % 16) * 8
            for pair in cutlass.range_constexpr(self.rows // (2 * self.input_warps)):
                row = (2 * pair + owner) * self.input_warps + warp
                token = first + row
                piece = cute.make_rmem_tensor(8, BFloat16)
                shared_offset = sA.layout((row, (lane % 16) * 8, stage))
                shared_piece = cute.make_tensor(
                    sA.iterator + cute.assume(shared_offset, divby=8), cute.make_layout(8))
                cute.copy(atom, shared_piece, piece)
                if (token < source.shape[0]) & (column < D):
                    offset = token * D + column
                    if cutlass.const_expr(self.has_delta):
                        branch = cute.make_rmem_tensor(8, BFloat16)
                        cute.copy(atom, cute.make_tensor(
                            sDelta.iterator + cute.assume(shared_offset, divby=8),
                            cute.make_layout(8)), branch)
                        for i in cutlass.range_constexpr(8):
                            piece[i] = BFloat16(Float32(piece[i]) + alpha * Float32(branch[i]))
                        cute.copy(atom, piece, cute.make_tensor(
                            source.iterator + cute.assume(offset, divby=8),
                            cute.make_layout(8)))
                    if owner == 0:
                        for i in cutlass.range_constexpr(8):
                            value = Float32(piece[i])
                            sums[2 * pair] = sums[2 * pair] + value * value
                    else:
                        for i in cutlass.range_constexpr(8):
                            value = Float32(piece[i])
                            sums[2 * pair + 1] = sums[2 * pair + 1] + value * value
                if cutlass.const_expr(self.has_delta):
                    cute.copy(atom, piece, shared_piece)

            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if warp == 0:
                cute.arch.mbarrier_wait(b_ready, kt & 1)
                for k in cutlass.range_constexpr(self.k_tile // 16):
                    mma.set(tcgen05.Field.ACCUMULATE, (kt != 0) | (k != 0))
                    cute.gemm(mma, acc, a_frag[None, None, k, stage],
                              b_frag[None, None, k, 0], acc)
                with cute.arch.elect_one():
                    tcgen05.commit(done)

        for r in cutlass.range_constexpr(self.rows // self.input_warps):
            total = sums[r]
            for bit in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total, 1 << bit)
            if lane == 0:
                inverse[r * self.input_warps + warp] = cute.math.rsqrt(
                    total / Float32(D) + Float32(self.context_eps))
        cute.arch.mbarrier_wait(done, (KT - 1) & 1)
        cute.arch.barrier()

        if warp < 4:
            acc_matrix = acc[(None, None), 0, 0]
            load_atom = sm100.get_tmem_load_op(
                (self.rows, self.columns, self.k_tile), LayoutEnum.ROW_MAJOR,
                Float32, Float32, (self.rows, self.columns), False)
            t2r = tcgen05.make_tmem_copy(load_atom, acc_matrix)
            thread_copy = t2r.get_slice(tid)
            t_acc = thread_copy.partition_S(acc_matrix)
            coords = thread_copy.partition_D(cute.make_identity_tensor((self.rows, self.columns)))
            registers = cute.make_rmem_tensor(coords.shape, Float32)
            cute.copy(t2r, t_acc, registers)
            for i in cutlass.range_constexpr(cute.size(registers)):
                row, column = coords[i]
                token = first + row
                if (token < source.shape[0]) & (column < 2 * H):
                    z = registers[i] * inverse[row]
                    z = z + Float32(bias[column])
                    z = Float32(BFloat16(z))
                    hidden[column // H, token, column % H] = BFloat16(z * sigmoid(z))
            cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier()
        tmem.free(tmem_ptr)

    @cute.jit
    def prefetch_input(self, kt, stage, ready: cute.Pointer,
                       a_atom: cute.CopyAtom, delta_atom: cute.CopyAtom,
                       gA: cute.Tensor, sA: cute.Tensor,
                       gDelta: cute.Tensor, sDelta: cute.Tensor):
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                ready + stage, 2 * self.rows * self.k_tile * (1 + self.has_delta))
        cute.copy(a_atom, gA[None, kt], sA[None, stage], tma_bar_ptr=ready + stage)
        if cutlass.const_expr(self.has_delta):
            cute.copy(delta_atom, gDelta[None, kt], sDelta[None, stage],
                      tma_bar_ptr=ready + stage)
