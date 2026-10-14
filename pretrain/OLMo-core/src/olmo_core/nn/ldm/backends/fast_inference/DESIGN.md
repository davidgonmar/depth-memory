# fast-inference

Forward-only execution of the released LDM models (the depth mixer the `fast` backend trains), for
prefill and KV-cached decode. Nothing is kept for a backward, and `run_segment` refuses to run with
gradients enabled.

## The model it runs

`backend.validate_quality` accepts exactly the released depth mixer: RLS with key width 4 or 8, address
hidden width 32 or 64, dynamic normalized keys and queries on the stream, sigmoid write and read gates,
the branch output as the written value, a learned linear read gain per group, a learned forgetting factor
and both residual additions. Anything else raises. The model width must be a multiple of 128 and the
value width a multiple of 64. The edge kernels keep the running covariance in BF16, so a model trained
with the fast backend's `running_covariance_dtype: "float32"` raises too.

## From a checkpoint to the inference form

A model is built with the `fast` backend's parameter modules, so its checkpoints load unchanged. After
loading, `backend.prepare_inference(model)` replaces them once with the tensors the kernels read. There
is one `edges.InferenceEdge` per edge, an edge being the write of one site followed by the read of the
next. It holds:

- `first_layer` `[D/128, 2H, 128]` BF16, the first layer of both address networks (write rows, then read
  rows) in 128-column tiles, with the context gains folded in as `BF16(W * g)`, which is how training
  rounds them;
- `first_bias` `[2H]`;
- `second_layer` `[2, G, K+1, H+8]` BF16, per site and group the K head rows and the gate row, padded so
  that neighbouring rows start in different shared-memory banks;
- `head_bias` `[2, G, K]`, `gate_bias` `[2, G]` and `static_address` `[2, G, K]`;
- `forgetting` `[G]` FP32, `max(sigmoid(logit), eps)`;
- `read_gain` `[G]` FP32 holding the BF16-rounded gain.

A block's `ldm_rule` becomes an `InferenceRule` with the edges that end at its attention and
feed-forward inputs, and the output boundary becomes the final edge. The input boundary and the
forgetting parameters are removed. Loading training parameters into the inference form raises. The eval
harness (`eval/olmo_lm.py`) calls `prepare_inference` after loading a checkpoint.

## One edge

`state.InferenceState.step` runs one edge, and the token count `n` of the forward picks the path.

A prefill (more tokens than the tuned decode classes) runs three kernels. The projection kernel
(`kernels/projection.py`) does the residual add, the row RMS, the first layer on tensor cores (WGMMA on
SM90, tcgen05 on SM100) and the activation. One batched cuBLAS GEMM on `second_layer` without its padding
gives the head and gate projections. The prefill edge kernel (`kernels/edge_state.py`) does the rest.

A decode (few tokens) runs the decode edge kernel alone (`kernels/edge_decode.py`). It also does the
residual add, the first layer and the head and gate projections.

Both paths write the updated stream (`source + alpha * delta`) over `source`, so an edge allocates only
its output and, in prefill, the hidden units and the head and gate projections. `InferenceState.begin`
allocates M, the packed covariance and both window histories as views of one buffer: the allocator gets
the state back as one block after the forward, where four separate allocations would be carved out of
cached blocks of other sizes and keep them split (a large batch after a small one could then find no
block for its logits).

Both edge kernels share the per-row stages of `kernels/edge.py`. A cluster of C CTAs takes T tokens, and
CTA c handles groups `[c G/C, (c+1) G/C)` of them with their (token, group) rows. Bulk copies bring the
rows' dense state into shared memory: the packed covariance, M, the value, the stream slice and the
window history. One K-lane subgroup per row forms both addresses, the gain and the coefficient row
(`kernels/addresses.py`). In prefill it starts as soon as the packed covariance and the gain history have
arrived, while M and the rest are still on their way. The dense step then forms, per (row, 8 columns),
the residual `scale v - M k - sum_j alpha_j r_j` rounded to BF16, the read
`M q + sum_j beta_j r_j + gamma residual` and the window history. The per-token sums of squares of
`stream + read` go to every CTA of the cluster through distributed shared memory while the covariance
update runs, and each CTA writes its slice of the branch RMSNorm. The final read has no norm.

At a commit the new M is `M + sum_j gain_j r_j + gain residual`. In the first window and in the decode
kernel each item writes it right after its residual. Past the first window the prefill kernel writes it
in a pass of its own after the barrier that ends the dense step, while the sums of squares travel, and
reads the norm weight from global memory after that pass.

The prefill kernel is compiled per place in the window: its shared memory holds only the history rows
the edge reads, so the early edges of a window fit more CTAs per SM. The decode kernel holds the whole
window's history.

The decode kernel streams the first layer through `weight_stages` shared-memory stages of `weight_chunk`
columns (a single stage of d_model columns keeps it resident), exchanges the hidden units across the
cluster through distributed shared memory and, as a programmatic dependent launch, loads its parameters
while the previous kernel finishes.

## Tuned launches

`tuning.py` reads one JSON table per GPU (`tuned/sm90.json`, `tuned/sm100.json`), keyed by the model
shape (groups, key, value and hidden width). An entry holds the prefill edge kernel's tiling (tokens per
cluster, cluster size, key lanes) and the decode classes. A decode class is a token-count bound with the
decode kernel's tiling (tokens per cluster, cluster size, key lanes, head and gate slices, first-layer
chunk and stages). `tune.py` writes the tables. It times a segment of random weights by CUDA-graph replay
for every candidate, checks each against a reference tiling and keeps the fastest, and the decode classes
end where the prefill kernels are as fast. A shape without a table, or a tiling that does not fit the
GPU's shared memory, raises.

The cluster size, the key lanes and the head and gate slices set the order of some FP32 sums, so they
change results at the rounding level. The other axes only divide the work and give bitwise the same
outputs.

## Runtime token count

Every kernel is compiled with the token axis as a runtime extent (`runtime.tensor_spec`,
`runtime.fake_tensor`) and the TVM-FFI launcher, which is called with the PyTorch tensors on PyTorch's
current stream. The compile keys are the edge's window position, the norm, the sizes and the tiling,
never the token count, so a new batch size or sequence length reuses the compiled programs. A decode step
can be captured in a CUDA graph.
