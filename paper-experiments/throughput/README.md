# Throughput and memory measurements

The efficiency results of the paper: training step, prefill and decode of PreLN, LDM, mHC and AttnRes
on Qwen3 shapes. Every measurement runs the code paths of the training recipe
(`paper-experiments/pretrain/train.py`) and of inference (every model prepared once, as a loaded
checkpoint is); none is an isolated kernel benchmark.

## Models

The configs under `configs/<size>/` have the Qwen3 shapes and the Qwen3 vocabulary of 151,936
tokens. The 1.7B shape is the 1.5B pretraining model with this vocabulary.

| size | width | layers | MLP width | heads / KV heads | head width | tied embeddings | LDM groups |
|---|---|---|---|---|---|---|---|
| 1.7B | 2048 | 28 | 6144 | 16 / 8 | 128 | yes | 16 |
| 4B | 2560 | 36 | 9728 | 32 / 8 | 128 | yes | 20 |
| 8B | 4096 | 36 | 12288 | 32 / 8 | 128 | no | 32 |
| 14B | 5120 | 40 | 17408 | 40 / 8 | 128 | no | 40 |

| arm | config | implementation |
|---|---|---|
| PreLN | `preln.json` | the backbone and the baseline of every comparison |
| LDM | `ldm.json` | `fast` backend for training, `fast-inference` backend for prefill and decode, both with tuned launch tables (a shape missing from a table raises) |
| mHC | `mhc.json` | DeepSeek TileKernels with the projection backward as two cuBLAS GEMMs and, without gradients, the projection reduction split as the tuned table gives; streams recomputed per 6 layers. The mHC row of the paper's figures. |
| mHC, released kernels | `mhc_upstream.json` | TileKernels as released: TileLang projection backward and a single split at decode; streams recomputed per 6 layers |
| mHC, no recompute | `mhc_no_recompute.json` | as mHC, keeping every stream (training only) |
| AttnRes (Block) | `attnres_block.json` | blocks of 8 sites, FLA Triton kernels |
| AttnRes (Full) | `attnres_full.json` | block size 1, FLA Triton kernels |

Before inference, mHC folds each controller's RMS weight into its projection once
(`olmo_core.nn.hyper_connections.prepare_inference`) and LDM converts its parameters into the tensors
its kernels read (`olmo_core.nn.ldm.backends.fast_inference.backend.prepare_inference`).

## Measurements

**Training step** (`measure.py --mode train`): one node of 8 GPUs, FSDP with FP32 master parameters
and optimizer state, BF16 compute, FP32 gradient reduction, AdamW with the recipe's weight-decay
groups, gradient clipping at 1.0 and compiled blocks. A step is the recipe's global batch of 1M
tokens. Every arm of a size runs at the same microbatch, the largest power of two at which every arm
fits (`measure.TRAIN_MICROBATCH`; B200: 1.7B 16, 4B 8, 8B 8, 14B 4). At 4096-token sequences the
microbatch and the number of sequences per step are halved, so a step still trains on 1M tokens.
Reported: tokens per second per GPU and peak allocated memory per GPU.

**Prefill** (`measure.py --mode prefill`): one GPU, BF16 weights, compiled blocks, inference mode,
the whole prompt in one forward and the logits of the last position only. Lengths 2048, 4096 and
8192; at each length the batches of `measure.PREFILL_BATCHES` (powers of two up to 32k tokens per
forward, then the largest batch at which every arm fits).

**Decode** (`decode.py`): one GPU, FlashAttention 4 over a preallocated KV cache for every arm,
contexts 2048, 4096 and 8192, batches 1 to 64. The prompt is prefilled in chunks of 512 positions
(time to first token); a whole decode step is captured in a CUDA graph and 128 replays are timed.
A cell that does not fit on one GPU is recorded as such.

Inputs are fixed synthetic tokens drawn with a fixed seed, the same for every arm.

**Interleaving.** One job per mode and size (`rounds.sh`) runs every arm in three rounds, each arm
in a fresh process, reversing the arm order every other round. A round's value is the median of its
measured steps; the reported value is the median over rounds.

**Validity.** A measurement raises instead of writing a result when Dynamo reports a graph break
other than the ones by design (FlashAttention 4 at decode, LDM's fast-inference edge calls), when
anything compiles or recompiles inside the measured steps, when a Triton autotuner tunes inside the
measured steps, or when a training step or prefill forward takes more than 1.5 times the median.

## Commands

```bash
# Training on every GPU of one node; prefill and decode on one GPU.
bash paper-experiments/throughput/rounds.sh train 1p7b b200 3
bash paper-experiments/throughput/rounds.sh train 1p7b b200 3 --length 4096
bash paper-experiments/throughput/rounds.sh prefill 1p7b b200 3
bash paper-experiments/throughput/rounds.sh decode 1p7b b200 3
```

Repeat for the sizes `4b`, `8b` and `14b`. Each measurement appends one JSON line to
`results/<gpu>/<mode>.jsonl`. One cell can also be measured directly, for example
`python paper-experiments/throughput/measure.py --mode prefill --size 1p7b --arm ldm --gpu b200 --batch 16`
or `python paper-experiments/throughput/decode.py --size 1p7b --arm ldm --gpu b200 --batch 1 --context 2048`.
