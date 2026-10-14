# Flexible LDM backend

The address networks, normalization, gates and residual joins are ordinary PyTorch and compile with
the transformer block. Only the state write/read step and its reverse are Triton kernels
(`kernels.py`), wrapped as mutating custom ops with a first-order autograd node (`state.py`). This is
the backend of the ablations: every model option of `LDMQualityConfig` is implemented here.

## Configuration

`execution.backend.name: flexible` with the setting

- `covariance_dtype`: `bfloat16` (default) or `float32`, the storage dtype of the RLS covariance P.

`quality.recurrence` selects `rls`, `delta_rule` or `outer_product`. The last two keep no covariance.
All three share the address networks, the read and write gates, the read gain, the boundary write
and read, the address sources, the residual options, static addresses and parameter banks.

## Recurrences

For a normalized key k, a value v and a write weight w, the controller passes `kw = sqrt(w) k` and
`vw = sqrt(w) v`, rounded to the working dtype, to the kernel. The memory has shape
`[tokens, groups, K, D/groups]`.

- RLS: `u = P kw`, `s = lambda + kw.u`, `g = u / s`, `e = vw - M^T kw`, `M' = M + g e^T`,
  `P' = (P - u u^T / s) / lambda`.
- Delta rule: `M' = lambda M + kw (vw - (lambda M)^T kw)^T`.
- Outer product: `M' = lambda M + kw vw^T`.

The read is `M'^T q`. Gates are `2 sigmoid(logit)`, so they start at the identity. M is stored in
BF16 and rounded after each write, before its read.

## Precision and autograd

RMS normalization and the L2 normalization of the addresses run in FP32 with an explicit epsilon.
Projections run in the working dtype. Kernel arithmetic and reductions are FP32; keys, values, reads
and the residual tape use the working dtype; M and its cotangent are BF16; P is stored in
`covariance_dtype` while its cotangent, the gains, denominators and forgetting factors stay FP32.
The covariance each write starts from is kept on a tape in its storage dtype, so the reverse restores
P exactly and differentiates the FP32 update at the stored values; only M is unwound.

The backward unwinds the shared state in place from the tapes (covariance before the write,
residual, gain, denominator, forgetting factor) instead of keeping a dense M per layer. Unwinding a
rounded M is approximate; the first write of a segment is restored from the exact initial M = 0.
Each forward supports one first-order backward: a second backward through an unwound graph raises,
and a backward that skips newer writes fails the order check. Whole-block activation checkpointing is
rejected because recomputing a block would repeat its state writes; checkpointing the address
networks (`execution.common.checkpoint_address_projections`) is supported.
