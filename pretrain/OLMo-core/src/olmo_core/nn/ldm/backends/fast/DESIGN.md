# Fast backend

The LDM training backend of the paper's main runs: the windowed state with a taped backward.

## Layout
- `backend.py`: the settings (`span`, `read_checkpoint`, `running_covariance_dtype`), the
  transformer block and the segment runner.
- `sites.py`, `address_network.py`: the address sites and the `ldm::address_network` custom op.
- `state.py`: `SegmentState` (tapes, the reverse order of the backward, buffer release) and the
  kernel launcher.
- `schedules.py`, `tune.py`: the launch schedules, their per-GPU tables and the offline tuner.
- `kernels/forward.py`: the forward, one launch per edge; it commits M at each window's last edge.
- `kernels/window_end.py`, `kernels/interior.py`: the backward, one window-end launch per window
  and one interior launch per other edge of the window.
- `kernels/address_network.py`: the address-network epilogues and the input-gradient GEMM.
- `tuned/`: the per-GPU schedule tables, read at launch.

## Kernels
The forward and the window end are persistent producer/consumer kernels. A producer warp
bulk-copies a batch of consecutive groups (every input's rows of a batch are contiguous) into one
of two shared stages, and the consumer warps compute the batch's groups. Per-lane cp.async covers
only the blocks a bulk copy cannot move (unaligned shapes, a short last batch). The interior gives each warp its own two cp.async stages and contracts one group at a
time on a tensor-core panel.

The covariance VJP of an edge and the outputs that follow it (the query, key, forgetting and scale
gradients and the covariance gradient C) need only K lanes per group. Both backward kernels
therefore run that part for 32 // K groups at once: while a warp works through its groups, each
group's partial sums stay in its own run of K lanes and its rows of P and C, its key and gain and
its scalars wait in a per-warp finish slot; when the slots are full, one pass finishes them all.
Each group's arithmetic and summation order are the same as when it is finished alone.

## Group width
The kernels move BF16 rows in 16-byte vectors, so they need a group width D that is a multiple of
8. `SegmentState` pads a narrower width at its boundary, the way FlashAttention pads head
dimensions: values and read gradients get zero columns up to the next multiple of 8, M, the
residual tape and the reverse buffers are allocated at that width, and the reads and value
gradients are sliced back to D. Zero columns stay zero through the recurrence (each column of M
is updated from its own value column) and add exact zeros to the backward's sums over D. Nothing
outside the state sees the padding, and a width that is already a multiple of 8 is not copied.
Launch shapes, and so the tuned tables, name the padded width: a group width of 94 runs as D96.

## Precision
M, values, residuals and reads are stored in BF16. Live P is the packed upper triangle, padded to
16 bytes per group, in the dtype of the `running_covariance_dtype` setting: BF16 by default, or
FP32. Every edge saves its P from before its update in BF16. P updates, keys, queries, gains,
denominators and scales use FP32 arithmetic. A BF16 live P is rounded once per edge, residuals are
rounded before they are reused and M is rounded when a window commits. The order of the column
updates and of the sums over the K lanes is fixed. The backward reads the saved P instead of
inverting the next one. The forgetting and write-scale gradients and the covariance gradients stay
FP32.

In BF16 the rounding errors of the live P accumulate from edge to edge. In deep stacks the stored
P can become indefinite; covariance-form RLS then amplifies the negative
direction, and the denominator lambda + k'Pk falls below lambda, even to zero. With
`running_covariance_dtype` set to `"float32"` the forward keeps P in FP32, staged with the FP32
words of each stage, and rounds it to BF16 only for the saved copy. The backward is the same for
both settings: it uses each saved copy once, at its own edge, with the FP32 key, gain and
denominator tapes.

## Launch schedules
Every kernel launch runs the schedule that the table for its GPU (`tuned/<arch>.json`: `sm100`,
`sm90`) names for the launch shape, for example
`window_end E4 F4 N65536 G16 K8 D128` or `forward E4 N32768 G16 K8 D128 first_window edge2`. The
axes (`schedules.AXES`) only move groups between warps, CTAs and lanes, so every schedule gives
bitwise the same result; the tuner skips a candidate the kernel would not run as named and stops
with an error if a candidate's outputs differ. A training launch whose shape is not in the table raises and names the
tuner command:

    python -m olmo_core.nn.ldm.backends.fast.tune --tokens N --groups G --key-dim K \
        --value-dim D --edges L [--kernels interior window_end] [--running-covariance-dtype float32]

A forward with an FP32 running covariance is looked up under keys that carry the word
`float32_covariance` (for example `forward E4 N32768 G16 K8 D96 first_window float32_covariance
edge2`); its stages hold more shared memory per group, so its best schedules can differ. The
backward entries serve both settings.

A launch without gradients (evaluation at any batch size) whose shape is not in the table runs the
default schedule. The tables cover every trained shape on B200 and H100.

The forward forms key = raw_key * scale itself and writes the key, query, scale and forgetting
tapes. Both backward kernels stage their edge's read gradient directly and write it into the DY
history; the query and forgetting gradients land in each edge's own outputs.

## Storage
With L edges, span E and state shape N x G x K x D (D padded to a multiple of 8):
- M: N*G*K*D BF16, no per-edge snapshots.
- Live P: N*G rows of K(K+1)/2 entries padded to 16 bytes, BF16 by default (40 entries, 80 bytes
  for K = 8) or FP32 (36 entries, 144 bytes for K = 8).
- Residual tape: L*N*G*D BF16.
- Keys, queries and gains: FP32 [L, N, G, K]; denominators, forgetting and scales: FP32 [L, N, G].
- Saved covariance: L*N*G*K(K+1)/2 BF16, the P of every edge from before its update.
- Reverse buffers, allocated at the first backward: BF16 dM and E-row DY and DV histories, FP32
  future value and gain gradients and the covariance gradient.

With the read checkpoint (`read_checkpoint`, on by default) the backward adds each read to the
stream again and reruns the branch RMSNorm instead of keeping their outputs. It keeps only the
norm's FP32 row scale (rsqrt, 4 bytes per token and read), so the backward does not recompute the
row's mean square.

Buffers are released when a backward without `retain_graph=True` completes. A backward with
`retain_graph=True` keeps them, plus a copy of the end state, so that it can run again.

## Window start state
Each window-end launch writes the window's start M, computed from the committed end M and the
window's residual and gain tapes as M_start = BF16(M_end - sum_j g_j r_j^T); in the first window
the start state is zero. The residuals and the P of every edge from before its update are read from
the tapes.

## Benchmarking
The paper's training-step, prefill and decode measurements are in `paper-experiments/throughput/`.
