# Vendored TileKernels (mHC subset)

Upstream: https://github.com/deepseek-ai/TileKernels, commit 36d9e45d38e204ebb87e6f6e833821eee0482fe5
(2026-04-23), MIT license (see LICENSE). Only the mHC kernels (`mhc/`), the mHC modeling layer
(`modeling/mhc/`) and the PyTorch references (`torch/mhc.py`) are included.

Local changes relative to upstream:
- `modeling/mhc/functional.py::mhc_pre`: merge an optional RMS weight into the projection
  before fused inference, using the same merge operation as training. The upstream no-grad
  dispatcher ignores this weight. The stored projection is never modified or cached.
- `__init__.py`, `modeling/__init__.py`, `torch/__init__.py`: import only the mHC subset (the
  upstream package also imports quantization kernels that require a newer TileLang API).
- `modeling/mhc/functional.py::mhc_head`: the sliced mixes are made contiguous (the padded-fn head
  path violated the kernel's packed-stride ABI) and `fuse_grad_acc` is a parameter (default True,
  upstream behavior; False for a head that is not preceded by `mhc_post`).
- `mhc/norm_fn_kernel.py::_mhc_pre_norm_fn_fwd_mul`: bound output stores by the declared
  projection width rather than the literal 24.
- `modeling/mhc/ops/norm_fn.py`: return one backward entry per forward argument and allocate
  scratch for the one forward split actually used, instead of allocating and slicing 16 splits.
- `modeling/mhc/ops/pre_big_fuse.py` and `mhc/pre_big_fuse_kernel.py`: the API's `n_splits` is
  honoured: the projection splits its n*D reduction that many ways (each part a multiple of the
  kernel's 256-column block, otherwise it raises), and the big-fuse kernel adds the partial sums in
  split order. Upstream allocates 16 splits but computes one and grids the projection over tokens
  only. The default (`None`, also in `functional.mhc_pre`) takes the split count from
  `modeling/mhc/ops/tuned/sm<arch>.json`, tuned per width and token count by
  `modeling/mhc/ops/tune_projection.py`: a few tokens (a decode step) use the most splits the width
  allows, and the largest token counts one split. The tables hold the widths 1024, 2048, 2560, 4096
  and 5120; a width missing from the table raises with the tune command. Only the FP32 summation
  order changes. The kernel reads the partials token-major, the layout `_mhc_pre_norm_fn_fwd_mul`
  writes.
  Scratch is allocated for the splits actually used. `fn_rounded=True` skips the per-call
  `round_to_tf32` for a projection rounded once at load.

Runtime: `tilelang` (pinned in the project lock) and a CUDA toolkit with `nvcc` on PATH;
kernels are JIT-compiled into `TILELANG_CACHE_DIR`.

## Integration in this repository

`olmo_core.nn.hyper_connections.MHC` drives these kernels through `torch.library` custom ops
(`olmo_core/nn/mhc_ops.py`) that reproduce upstream's compositions `mhc_pre` (training branch),
`mhc_post` and `mhc_head` one kernel call at a time, so a block compiles without graph breaks.
Upstream's `autograd.Function` wrappers and their storage side channel (`grad_from_mhc_post`) are
not used on the training path: the stream gradient of `pre` is accumulated in one buffer and
autograd adds `post`'s residual gradient (one bf16 add per connector, which changes the bf16
rounding order of the stream gradient). Without gradients the controller runs upstream's fused
no-grad kernel (`mhc_pre_big_fuse`), see below.

The projection backward matmuls (`d_fn = d_mixes^T x`, `d_x += d_mixes fn`) run as TF32 GEMMs
compiled by Inductor, which autotunes cuBLAS against its Triton GEMM templates per shape, instead of
upstream's `_mhc_pre_norm_fn_bwd_mul`, whose grid is serial over tokens. The operands are the
same TF32 values (upstream's `round_to_tf32` leaves 13 low mantissa bits that the tensor cores
discard; they are cleared explicitly). `HyperConnectionsConfig.upstream_projection_backward`
selects upstream's kernel. Supported geometry is four streams with a per-stream width divisible
by 64. Streams are BF16; controller and learned-head parameters are FP32.

Controller and head parameters are slots of one model-owned FP32 `MHCParameterBank` (one
`decay_<block>` / `no_decay_<block>` pair per block, one FSDP unit, unsharded once per step at
the start of the model forward). Slots are reserved large-first so every kernel-facing tensor
is 16-byte aligned. Checkpoints therefore store `mhc_parameter_bank.*` tensors, not per-controller
parameters.

Between transformer blocks, streams are concatenated on the last axis (`[..., 4 * D]`);
the kernels receive a view with shape `[..., 4, D]`.

Each controller and the learned head own a trainable FP32 `norm_weight` of length `nD`,
initialized to ones, like the default affine weight of the `torch.nn.RMSNorm(nD, eps=1e-20)` in
Algorithm 1 (Appendix A.6, page 17) of the ICML version of mHC. These RMS weights are merged into
the projection for kernel execution and remain separate trainable parameters. The learned final
head follows DeepSeek-V4 and is not specified by Algorithm 1.

Controller `b`, `alpha` and `norm_weight` are excluded from weight decay; `fn` is not. Controller
initialization follows Appendix A.6 (Algorithm 1) of the ICML version of mHC: zero projection,
selected-stream pre bias and diagonal residual bias at +3 (other entries -3), zero post bias, and
the configured gating factors (0.01 in the paper configurations); selected streams cycle by
connector index modulo four. The controller RMS epsilon is `controller_norm_eps`, 1e-6 in the
paper configurations, as in the upstream functional API. The learned head is the only stream
reduction (`learned_final_reduce` must be true); it has independent sigmoid gates and its own
normalization epsilon, and decays `weight` but not `bias`, `scale` or `norm_weight`. Parameter
counts include the `nD` weights per controller and `nD` for the head.

With gradients disabled, the controller runs upstream's no-grad `mhc_pre` path (RMS-weight merge,
then `mhc_pre_big_fuse`) as one traceable op (`olmo_mhc::pre_inference` in `nn/mhc_ops.py`), so a
compiled block has no graph break around the TileLang calls. This applies to both prompt
processing and single-token inference. `HyperConnectionsConfig.upstream_inference_projection`
selects upstream's one-split projection at every token count (the `mhc_upstream` arm of
`paper-experiments/throughput`); by default no-grad forwards use the tuned split count above. After a checkpoint is
loaded, `hyper_connections.prepare_inference` folds each controller's RMS weight into its
projection and rounds it to TF32 once, with upstream's merge kernel and `round_to_tf32`, so no-grad
forwards skip the per-call merge and rounding (`mhc_pre_big_fuse(..., fn_rounded=True)`; the model is
inference only afterwards). Training continues to use
the autograd-capable kernels and merges per call, as upstream does. Both paths use the same
configured Sinkhorn iteration count, epsilons, post multiplier, and controller parameters.

## Paper efficiency coverage

Reference: [mHC, Section 4.3](https://arxiv.org/html/2512.24880v1#S4.SS3).

| Technique | Implementation |
| --- | --- |
| Project before dividing by RMS; share input reads for the projection and norm | `mhc/norm_fn_kernel.py` forward and backward |
| Mixed precision: BF16 streams, TF32 projection operands, FP32 accumulators and coefficients | `mhc/norm_fn_kernel.py` and `nn/mhc_ops.py` |
| Fuse scale, bias and sigmoid coefficient operations | `mhc/pre_split_mixes_kernel.py` |
| Fused Sinkhorn iterations with backward intermediates recomputed on chip | `mhc/sinkhorn_kernel.py` |
| Fused post mapping and residual merge | `mhc/post_kernel.py` |
| Groupwise activation reconstruction without rerunning attention or FFN | `nn/mhc_ops.py` recompute ops (`pre_r`, `post_r`, `group_replay`, `mhc_branch_input`) + `MHCGroup`, on upstream `mhc_multilayer_recompute`: Table 3 exactly (`x_l`, `H^pre x_l`, `RMSNorm(H^pre x_l)` rebuilt in backward, one replay per group), traceable end to end |
| DualPipe communication overlap | Not implemented; the paper's runs use data parallelism without pipeline parallelism |

Recomputation: the paper configurations group six mHC connectors (three transformer
blocks). It saves the entry residual, branch outputs
and mixes, then calls upstream replay with six pre operations and five post operations.
Replay preserves the intervening BF16 residual rounding; branch functions are not rerun.
The local recompute-aware ops, the group object and the branch-input function (which
rebuilds the pre-norm and feeds the attention / FFN input projections) are not upstream code;
the replay kernel they call is. The branch's first projections are given precomputed
activations (`Attention.forward(projections=...)`, `FeedForward.forward(projections=...)`)
so that their input is rebuilt rather than stored.

Location: this package lives at `olmo_core/kernels/tile_kernels`. Its internal imports carry that
package prefix (`from olmo_core.kernels.tile_kernels.mhc.post_kernel import ...` where upstream has
`from tile_kernels.mhc.post_kernel import ...`). `package-data` in `pyproject.toml` names the
package under this path, and `packages.find` picks it up through `olmo_core*`.
