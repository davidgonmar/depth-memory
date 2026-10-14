# LDM backends

A model selects one backend by name in `execution.backend.name`; `integration._BACKEND_MODULES` maps
each name to its module. All backends share the configuration in `config.py`; `fast` and
`fast-inference` implement a subset of its options.

| name | module | purpose |
|---|---|---|
| `reference` | `backends/reference.py` | Pure PyTorch memory M and covariance P that materializes every intermediate. It is the numerical and gradient reference for the other backends and runs on CPU. |
| `flexible` | `backends/flexible/` | The ablation backend. The address networks and residual joins are PyTorch modules compiled with the block; only the state write/read step and its first-order backward are Triton kernels. It implements every model option of the configuration (see `flexible/README.md`). The paper's 400M ablations ran on it. |
| `fast` | `backends/fast/` | The training backend of the paper's main runs. State unwinding over windows of E edges: one forward launch per edge that commits M at each window's last edge, and a backward of one window-end launch per window and one interior launch per other edge. Address networks run as the custom op `ldm::address_network`. See `fast/DESIGN.md`. |
| `fast-inference` | `backends/fast_inference/` | Forward-only execution of `fast` checkpoints for prefill and KV-cached decode, with launch shapes from per-GPU tuned tables. See `fast_inference/DESIGN.md`. |

`reference` and `flexible` build their parameters from `components.py`; `fast` and `fast-inference`
build theirs from `parameters/`.
