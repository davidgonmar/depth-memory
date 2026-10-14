# LDM ablations (400M, 15B tokens)

One-change-at-a-time ablations of the 400M LDM model, as reported in the paper's ablation table and
the group-count figure. Everything here derives from `pretrain/configs/400m/ldm.json`:

- `generate_configs.py` defines every variant as the 400M LDM config plus one `quality` change and
  writes `configs/400m/<variant>.json`. Rerun it after changing the base config or the table.
- `launch.sh VARIANT [b200|h100]` trains one variant on the `flexible` backend
  (`pretrain/configs/execution/ldm_flexible.json`) at learning rate 1e-3, seed 42, for 15B tokens:
  14,305 optimizer steps of 512 x 2048 tokens, warmup 1,000 steps and cosine decay to 10% at step
  14,305. `VARIANT=preln` trains the PreLN reference on the same recipe. The global batch is fixed;
  the GPU count and the microbatch only change gradient accumulation.

The held-out loss of each run is logged by the training recipe at the final step.

## Variants

| Paper row | Variant | Change to `quality` |
| --- | --- | --- |
| LDM | `ldm` | none |
| (a) $F(\mathbf r)$ | `replace_read` | `replace_read=true`: each branch reads only the memory read |
| (a) $\mathbf x=\boldsymbol\delta$ | `no_post_function_residual` | `no_post_function_residual=true` |
| (b) Delta | `delta_rule` | `recurrence=delta_rule` (the learned forgetting factor decays M) |
| (b) Outer | `outer_product` | `recurrence=outer_product` (the learned forgetting factor decays M) |
| (c) $\mathbf v=\mathbf x$ | `value_stream` | `value_source=updated_stream` |
| (c) $\mathbf{QK}(\boldsymbol\delta)$ | `query_key_delta` | `addresses.query.source=branch_delta`, `addresses.key.source=branch_delta` |
| (c) Static $\mathbf q$ | `static_query` | `addresses.query.mode=static`, no read gate |
| (c) $G=1$ | `groups_1` | `groups=1` |
| (c) $G=8$ | `groups_8` | `groups=8` |
| (d) $-\gamma$ | `no_read_gain` | `learned_read_gain.enabled=false` |
| (d) $\mathbf g=\mathbf 1$ | `no_read_gate` | `read_gate.enabled=false` |
| (d) $\boldsymbol\omega=\mathbf 1$ | `no_write_gate` | `write_gate.enabled=false` |
| (d) $\lambda=1$ | `no_forgetting` | `rls.lambda_init=1.0`, `rls.learn_lambda=false` |

The definition of each row is in the paper's appendix "Ablation Definitions".
