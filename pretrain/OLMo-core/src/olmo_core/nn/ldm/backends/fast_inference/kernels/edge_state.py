"""The prefill edge kernel: addresses, covariance update, dense step and branch RMSNorm of an edge for many
tokens, after the projection kernel (updated stream, hidden units) and the head/gate GEMM.

The grid holds ``ceil(N / tokens)`` clusters of `cluster` CTAs (`edge.EdgeRows`). Each subgroup reads its
row's head and gate projections from the GEMM's output ``[2, N, G (K + 1)]`` (per group, the K head columns
then the gate column) while bulk copies bring the rows' dense state and the updated stream slice.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, cute
from cutlass.utils import SmemAllocator

from .addresses import Projections
from .edge import THREADS, EdgeRows


class EdgeState(EdgeRows):
    prefill = True

    def __init__(self, E, G, K, D, tiling, *, history, **options):
        super().__init__(E, G, K, D, tokens=tiling.tokens, cluster=tiling.cluster, key_lanes=tiling.key_lanes,
                         history=history, **options)
        self.tiling = tiling
        self.shared_bytes = self._total(self._row_regions())      # the caller checks it against the device

    @cute.jit
    def __call__(self, stream: cute.Tensor, value: cute.Tensor, projected: cute.Tensor, head_bias: cute.Tensor,
                 gate_bias: cute.Tensor, static_address: cute.Tensor, read_gain: cute.Tensor,
                 forgetting: cute.Tensor, covariance: cute.Tensor, memory: cute.Tensor,
                 residual_history: cute.Tensor, gain_history: cute.Tensor, output: cute.Tensor,
                 norm_weight: cute.Tensor, edge: Int32, cuda_stream: cuda.CUstream):
        clusters = cute.ceil_div(stream.shape[0], self.tokens)
        self.kernel(stream, value, projected, head_bias, gate_bias, static_address, read_gain, forgetting,
                    covariance, memory, residual_history, gain_history, output, norm_weight, edge).launch(
            grid=(self.cluster * clusters, 1, 1), block=(THREADS, 1, 1), cluster=(self.cluster, 1, 1),
            smem=self.shared_bytes, stream=cuda_stream)

    @cute.kernel
    def kernel(self, stream: cute.Tensor, value: cute.Tensor, projected: cute.Tensor, head_bias: cute.Tensor,
               gate_bias: cute.Tensor, static_address: cute.Tensor, read_gain: cute.Tensor,
               forgetting: cute.Tensor, covariance: cute.Tensor, memory: cute.Tensor,
               residual_history: cute.Tensor, gain_history: cute.Tensor, output: cute.Tensor,
               norm_weight: cute.Tensor, edge: Int32):
        K = self.K
        place = self._place(stream.shape[0])
        rows = self._allocate_rows(SmemAllocator())
        self._init_row_barriers(place, rows)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        # Peers push into these barriers only after every CTA of the cluster has initialized its own.
        cute.arch.cluster_arrive_relaxed()
        self._request_state(place, rows, stream, value, covariance, memory, residual_history, gain_history,
                            norm_weight, edge)
        pair, kidx = place.tid // K, place.tid % K
        pair_active = (pair < self.pairs) & (pair // self.local_groups < place.live)
        g = place.first_group + pair % self.local_groups
        parameters = self.addresses.lane_parameters(g, kidx, pair_active, head_bias, gate_bias, static_address,
                                                    read_gain, forgetting)
        head_key, head_query, gate_key, gate_query = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        if pair_active:
            token = place.token0 + pair // self.local_groups
            head_key, head_query = projected[0, token, g * (K + 1) + kidx], projected[1, token, g * (K + 1) + kidx]
            gate_key, gate_query = projected[0, token, g * (K + 1) + K], projected[1, token, g * (K + 1) + K]
        cute.arch.mbarrier_wait(rows.address_bar, 0)
        gain_lane, pk_lane = self._addresses(place, rows, parameters, Projections(head_key, head_query, gate_key,
                                                                                  gate_query),
                                             pair, pair_active, gain_history, edge)
        cute.arch.mbarrier_wait(rows.state_bar, 0)
        cute.arch.sync_threads()
        residuals = self._dense_step(place, rows, residual_history, memory, output, edge)
        # Once this returns, every CTA of the cluster has initialized its barriers.
        cute.arch.cluster_wait()
        self._branch_norm(place, rows, parameters, pair, pair_active, gain_lane, pk_lane, covariance, output,
                          memory, norm_weight, residuals, edge)
