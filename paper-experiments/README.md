# Paper experiments

Everything needed to reproduce the experiments of the paper: the pretraining comparison at 400M and
1.5B, the 400M ablations, the downstream evaluation, and the throughput and memory measurements.
Methods are named `preln`, `attnres_block`, `attnres_full`, `mhc` and `ldm` in configs and run
names.

## Layout

```text
paper-experiments/
├── pretrain/
│   ├── train.py                the training recipe and entry point
│   ├── launch.sh               one training run on every GPU of a node
│   ├── configs/400m/, 1p5b/    model configs (backbone and method) and train.json (length, parallelism)
│   ├── configs/execution/      LDM backend configs: ldm_fast.json (main runs), ldm_flexible.json (ablations)
│   └── evaluation/             run.sh (lm-eval suite)
├── ablations/                  generate_configs.py, configs/400m/, configs/train.json, launch.sh
└── throughput/                 measure.py, decode.py, rounds.sh, configs/ (README.md)
```

## Models

The backbones are those of `configs/llama_400m_L24.json` and `configs/qwen3_1500m_L28.json` at the
repository root; every method config of a scale repeats them exactly and adds only the method's own
fields.

| | 400M Llama-like | 1.5B Qwen3-like |
|---|---:|---:|
| Layers | 24 | 28 |
| Width | 1024 | 2048 |
| Attention heads / KV heads | 16 / 16 | 16 / 8 |
| Head width | 64 | 128 |
| MLP width (SwiGLU) | 2736 | 6144 |
| QK normalization | no | yes |
| RoPE theta | 10,000 | 1,000,000 |
| Tied embeddings | no | yes |
| Vocabulary | 50,280 | 50,280 |

LDM uses 16 groups, key dimension 8 (4 for `ldm_k4`) and address networks of hidden width 32 at
400M and 64 at 1.5B. AttnRes (Block) uses 8 depth blocks at 400M and 7 at 1.5B; mHC uses four
streams, 20 Sinkhorn iterations, recomputation every 6 layers and the learned final head.

## Training recipe

`pretrain/train.py` fixes everything but the model, the training config (`train.json`: length,
warmup, checkpoint interval and data parallelism) and the learning rate:

- FineWeb-Edu (100BT sample) tokenized with the GPT-NeoX OLMo/Dolma v1.5 tokenizer, packed into
  sequences of 2048 tokens; the shard `013_00008.parquet` is held out for validation.
- Global batch of 512 sequences (1,048,576 tokens), independent of the GPU count and microbatch; the
  data order depends only on the seed (42).
- 400M: 47,684 steps (50.0B tokens), warmup 1,000 steps. 1.5B: 94,972 steps (99.6B tokens),
  warmup 2,000 steps.
- AdamW (betas 0.9 and 0.95, epsilon 1e-8, weight decay 0.1), gradient clipping at 1.0, linear
  warmup from zero and cosine decay to 10% of the peak learning rate. No weight decay on RMSNorm
  weights, the static mHC parameters, or LDM's forgetting logits, read gains and gate biases.
- FP32 parameters and optimizer state, BF16 compute, FP32 gradient reduction (replicated data
  parallelism at 400M, FSDP at 1.5B), compiled model.
- Held-out validation every 2,500 steps and at the end; checkpoints every 2,500 (400M) or 5,000
  (1.5B) steps and at the end.

The peak learning rates selected by the sweeps of the paper, given to `launch.sh`:

| | PreLN | AttnRes (Full) | AttnRes (Block) | mHC | LDM | LDM, D_k = 4 |
|---|---|---|---|---|---|---|
| 400M | 1e-3 | 4e-3 | 2e-3 | 2e-3 | 1e-3 | 1e-3 |
| 1.5B | 1e-3 | – | 1e-3 | 1e-3 | 1e-3 | – |

The ablations train the 400M LDM model with one change each for 15B tokens (14,305 steps) at 1e-3 on
the flexible backend, with the same recipe otherwise (the cosine schedule ends at step 14,305).

## Commands

```bash
export DATA_DIR=/path/to/fineweb-edu/tokenized WANDB_ENTITY=your-entity CHECKPOINT_DIR=/path/to/runs
bash paper-experiments/pretrain/launch.sh 400m ldm 0.001          # one run, every GPU of the node
bash paper-experiments/ablations/launch.sh delta_rule b200         # one ablation
bash paper-experiments/pretrain/evaluation/run.sh "$CHECKPOINT_DIR/<run>/step47684" ldm.json
```

`launch.sh` takes `--gpu h100` to halve the default microbatch, `--microbatch N` to set it and
`--backend flexible` to train LDM on the flexible backend. The evaluation runs lm-eval zero-shot on
ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, OpenBookQA, LAMBADA and WikiText.
