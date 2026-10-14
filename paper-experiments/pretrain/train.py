#!/usr/bin/env python3
"""The pretraining recipe of the paper on OLMo-core: data, schedule, optimizer, precision, evaluation
and checkpointing are fixed here, and the model comes from a JSON config."""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
OLMO_CORE_SRC = REPO_ROOT / "pretrain" / "OLMo-core" / "src"
for path in (REPO_ROOT, OLMO_CORE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rich
import torch.nn as nn
from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.ldm.config import LDMConfig
from olmo_core.nn.layer_norm import RMSNorm
from olmo_core.nn.transformer import (
    HyperConnectionsConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.checkpoint import Checkpointer, CheckpointerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CheckpointRemovalStrategy,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from config_utils.model_config import load_llama_like_kwargs
from config_utils.tokenizer_config import build_tokenizer_config, copy_tokenizer_to_dir

SEED = 42
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_SEQUENCES = 512
GLOBAL_BATCH_TOKENS = GLOBAL_BATCH_SEQUENCES * SEQUENCE_LENGTH
VALIDATION_INTERVAL = 2_500
TOKENIZER = (
    REPO_ROOT / "pretrain/OLMo-core/src/olmo_core/data/tokenizers/"
    "allenai_gpt-neox-olmo-dolma-v1_5.json"
)


@dataclass(frozen=True)
class TrainingConfig:
    """The training config of a run (``configs/<scale>/train.json``): its length in optimizer steps,
    the warmup, the checkpoint interval and the data parallelism, ``ddp`` (replicated) or ``fsdp``
    (sharded)."""

    max_steps: int
    warmup_steps: int
    checkpoint_interval: int
    data_parallel: str


def read_training_config(path: Path) -> TrainingConfig:
    config = TrainingConfig(**read_json(path))
    if not 0 < config.warmup_steps < config.max_steps:
        raise ValueError(f"{path}: the warmup must be positive and shorter than the run")
    if config.data_parallel not in {"ddp", "fsdp"}:
        raise ValueError(f"{path}: data_parallel must be ddp or fsdp")
    return config


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = SEED


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def model_method(raw: dict[str, Any]) -> str:
    """The method of a model config, from its block type."""
    block_name = raw.get("block_name")
    if block_name is None:
        return "preln"
    methods = {
        TransformerBlockType.attnres.value: "attnres",
        TransformerBlockType.mhc.value: "mhc",
        TransformerBlockType.ldm.value: "ldm",
    }
    if block_name not in methods:
        raise ValueError(f"unsupported block_name: {block_name!r}")
    return methods[block_name]


def build_model(
    model_path: Path,
    raw: dict[str, Any],
    *,
    method: str,
    tokenizer_vocab_size: int,
    ldm_execution_path: Path | None,
) -> TransformerConfig:
    kwargs, _ = load_llama_like_kwargs(
        model_path,
        tokenizer_vocab_size=tokenizer_vocab_size,
    )

    # Parameters are stored in FP32; FSDP casts them to BF16 for the forward and backward passes.
    kwargs["dtype"] = DType.float32
    if kwargs.get("feed_forward") is not None:
        kwargs["feed_forward"].dtype = DType.float32
    model = TransformerConfig.llama_like(**kwargs)

    if method == "preln":
        if model.block.name != TransformerBlockType.default:
            raise ValueError("PreLN must use OLMo's standard default block")
    elif method == "attnres":
        if model.block.name != TransformerBlockType.attnres:
            raise ValueError("AttnRes JSON did not build an AttnRes block")
    elif method == "mhc":
        hyper_connections = raw.get("hyper_connections")
        if not isinstance(hyper_connections, dict):
            raise ValueError("mHC JSON requires hyper_connections")
        model.block.hyper_connections = HyperConnectionsConfig.from_dict(
            hyper_connections
        )
    elif method == "ldm":
        if ldm_execution_path is None:
            raise ValueError("LDM requires --ldm-execution-config")
        ldm_json = raw.get("ldm")
        if not isinstance(ldm_json, dict) or not isinstance(ldm_json.get("quality"), dict):
            raise ValueError("LDM JSON requires ldm.quality")
        execution = read_json(ldm_execution_path).get("execution")
        if not isinstance(execution, dict):
            raise ValueError("LDM execution JSON requires execution")
        ldm = LDMConfig.from_dict(
            {"quality": ldm_json["quality"], "execution": execution}
        )
        ldm.validate()
        model.block.ldm = ldm
    else:
        raise AssertionError(method)

    if method != "ldm" and ldm_execution_path is not None:
        raise ValueError("--ldm-execution-config is only valid for LDM")
    return model


def mhc_weight_decay_override(method: str):
    if method != "mhc":
        return None
    # The no-decay tensors of the model's mHC parameter bank hold exactly the static parameters
    # (biases, gating factors, RMS weights, and the head's bias and scale).
    patterns = ["mhc_parameter_bank.no_decay_*"]
    return [OptimGroupOverride(params=patterns, opts={"weight_decay": 0.0})]


def configure_weight_decay(model: nn.Module, optim: AdamWConfig) -> None:
    """Exclude RMSNorm weights, the static mHC parameters and the LDM control parameters from weight
    decay. Embeddings and AttnRes queries are decayed."""
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    no_decay = {
        parameter_names[id(module.weight)]
        for module in model.modules()
        if isinstance(module, (RMSNorm, nn.RMSNorm))
        and module.weight is not None
        and module.weight.requires_grad
    }
    no_decay.update(ldm_control_no_decay(parameter_names.values()))
    for override in optim.group_overrides or []:
        if override.opts != {"weight_decay": 0.0}:
            raise ValueError("Unexpected optimizer override in the paper training recipe")
        no_decay.update(override.params)
    optim.group_overrides = [
        OptimGroupOverride(params=sorted(no_decay), opts={"weight_decay": 0.0})
    ]


# Like the decay and skip parameters of state-space models, the forgetting logits and the read gains
# are not decayed. Decay would pull the gate biases towards a gate of one, so they are excluded too.
LDM_CONTROL_NO_DECAY = (
    r"ldm_retention\.(root_logit|block_logits|ridge_raw)$",
    r"read_gain$",
    r"ldm\w*\..*\.gate\.bias$",
    # With execution.common.parameter_bank, the no-decay bank of a block or boundary holds only
    # biases, context-norm gains and read gains.
    r"ldm\w*\..*parameter_bank\.no_decay(_fp32)?$",
)


def ldm_control_no_decay(parameter_names) -> set[str]:
    """The LDM control parameters excluded from weight decay; empty for the other methods."""
    names = list(parameter_names)
    if not any("ldm_retention" in name for name in names):
        return set()
    selected: set[str] = set()
    for pattern in LDM_CONTROL_NO_DECAY:
        matched = {name for name in names if re.search(pattern, name)}
        if pattern.startswith("ldm_retention") and not matched:
            raise ValueError("LDM model without forgetting logits")
        selected.update(matched)
    return selected


def configure_compiler(model_config: TransformerConfig) -> None:
    """Allow AttnRes's depth-dependent source lists to remain statically compiled."""
    if model_config.block.name != TransformerBlockType.attnres:
        return
    import torch._dynamo.config as dynamo_config

    # One specialization per block for its number of sources, with room for training and
    # evaluation and for full and partial batches.
    variants = 4 * model_config.n_layers
    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, variants)
    dynamo_config.accumulated_recompile_limit = max(
        dynamo_config.accumulated_recompile_limit, variants
    )


def validate_resume(config: ExperimentConfig, run_dir: Path | None = None, *,
                    resume_checkpoint: Path | None = None) -> Path:
    """Resume only a complete checkpoint from the same training recipe."""
    if (run_dir is None) == (resume_checkpoint is None):
        raise ValueError("Specify exactly one of --resume-run-dir and --resume-checkpoint")
    if resume_checkpoint is None:
        checkpoint = Path(Checkpointer.latest_checkpoint(run_dir.resolve()))
    else:
        checkpoint = resume_checkpoint.resolve()
        if (not Checkpointer.dir_is_checkpoint(checkpoint)
                or not (checkpoint / "train/rank0.pt").is_file()
                or not (checkpoint / "model_and_optim/.metadata").is_file()):
            raise ValueError(f"Cannot resume: {checkpoint} is not a complete training checkpoint")
    previous = read_json(checkpoint / "config.json")
    current = config.as_config_dict()
    if resume_checkpoint is not None:
        save_folder = Path(current["trainer"]["save_folder"]).resolve()
        source_run = checkpoint.parent
        if save_folder.is_relative_to(source_run) or source_run.is_relative_to(save_folder):
            raise ValueError("--resume-checkpoint requires a new --save-folder outside the source run")
        if save_folder.exists() and (not save_folder.is_dir() or any(save_folder.iterdir())):
            raise ValueError("--resume-checkpoint requires a new or empty --save-folder")
    checks = [
        ("model", previous["model"], current["model"]),
        ("optimizer", previous["train_module"]["optim"], current["train_module"]["optim"]),
        ("schedule", previous["train_module"]["scheduler"], current["train_module"]["scheduler"]),
        # The micro-batch only splits the global batch (checked below) into accumulation steps.
        ("train module", _without_microbatch(previous["train_module"]), _without_microbatch(current["train_module"])),
        ("duration", previous["trainer"]["max_duration"], current["trainer"]["max_duration"]),
    ]
    for key in ("global_batch_size", "seed"):
        checks.append((key, previous["data_loader"][key], current["data_loader"][key]))
    checks.append(("dataset",
                   {k: v for k, v in previous["dataset"].items() if k != "work_dir"},
                   {k: v for k, v in current["dataset"].items() if k != "work_dir"}))
    for name, saved, requested in checks:
        if saved != requested:
            raise ValueError(f"Cannot resume: {name} differs from {checkpoint}/config.json")
    return checkpoint


def _without_microbatch(train_module: dict) -> dict:
    return {k: v for k, v in train_module.items() if k != "rank_microbatch_size"}


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    model_path = args.model_config.resolve()
    raw = read_json(model_path)
    method = model_method(raw)
    training = read_training_config(args.train_config.resolve())
    replicated = training.data_parallel == "ddp"
    learning_rate = args.learning_rate
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if int(raw.get("max_sequence_length", -1)) != SEQUENCE_LENGTH:
        raise ValueError(f"model JSON must set max_sequence_length={SEQUENCE_LENGTH}")

    tokenizer = build_tokenizer_config(str(TOKENIZER))
    model = build_model(
        model_path,
        raw,
        method=method,
        tokenizer_vocab_size=tokenizer.vocab_size,
        ldm_execution_path=args.ldm_execution_config,
    )
    work_dir = args.work_dir.resolve()
    resume_run_dir = getattr(args, "resume_run_dir", None)

    data_dir = args.data_dir.resolve()
    dataset = NumpyFSLDatasetConfig.glob(
        str(data_dir / "train" / "*.npy"),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        work_dir=work_dir / "dataset",
    )
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_TOKENS,
        seed=SEED,
        num_workers=args.data_loader_num_workers,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=args.device_microbatch_size * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            group_overrides=mhc_weight_decay_override(method),
        ),
        scheduler=CosWithWarmup(
            warmup=training.warmup_steps,
            alpha_f=0.1,
            t_max=training.max_steps,
            warmup_min_lr=0.0,
        ),
        max_grad_norm=1.0,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp if replicated else DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=None if replicated else 1,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            prefetch_factor=0 if replicated else 1,
        ),
        autocast_precision=None,
    )
    trainer = (
        TrainerConfig(
            checkpointer=CheckpointerConfig(
                # Checkpoint shards are written with O_DIRECT, bypassing the page cache of the node.
                local_write_mode="direct",
            ),
            save_folder=str(resume_run_dir.resolve()) if resume_run_dir else args.save_folder,
            load_strategy=LoadStrategy.always if resume_run_dir else LoadStrategy.never,
            load_optim_state=True,
            load_trainer_state=True,
            work_dir=work_dir / "trainer",
            save_overwrite=False,
            max_duration=Duration.steps(training.max_steps),
            metrics_collect_interval=10,
            cancel_check_interval=10,
            bookkeeping_soft_timeout=120,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=training.checkpoint_interval,
                pre_train_checkpoint=False,
                save_async=True,
                remove=CheckpointRemovalStrategy.never,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=args.run_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                cancel_check_interval=10,
                enabled=True,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.glob(
                    str(data_dir / "eval" / "*.npy"),
                    metadata=[{"label": "fineweb-edu-fixed-heldout"}],
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer,
                    work_dir=work_dir / "dataset",
                ),
                eval_interval=VALIDATION_INTERVAL,
                eval_on_startup=False,
                eval_on_finish=True,
                eval_duration=Duration.epochs(1),
            ),
        )
    )
    return ExperimentConfig(
        model=model,
        dataset=dataset,
        data_loader=data_loader,
        train_module=train_module,
        trainer=trainer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--train-config", type=Path, required=True,
        help="The run's length, warmup, checkpoint interval and data parallelism "
        "(pretrain/configs/<scale>/train.json, ablations/configs/train.json).",
    )
    parser.add_argument(
        "--ldm-execution-config", type=Path,
        help="LDM only: the backend and numerics (pretrain/configs/execution/).",
    )
    parser.add_argument(
        "--learning-rate", type=float, required=True, help="Peak learning rate.",
    )
    parser.add_argument("--device-microbatch-size", type=int, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("DATA_DIR"),
        help="Tokenized FineWeb-Edu with train/*.npy and eval/*.npy (default: $DATA_DIR).",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--save-folder", required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-run-dir", type=Path,
        help="Resume the latest complete checkpoint in this run directory and keep saving there.",
    )
    resume.add_argument(
        "--resume-checkpoint", type=Path,
        help="Resume this complete checkpoint and save into a new --save-folder.",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--data-loader-num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("set --data-dir or DATA_DIR")
    return args


def main() -> None:
    args = parse_args()
    config = build_config(args)
    prepare_training_environment()
    try:
        configure_compiler(config.model)
        seed_all(config.init_seed)
        model = config.model.build(init_device="meta")
        configure_weight_decay(model, config.train_module.optim)
        resume_checkpoint = (
            validate_resume(
                config, args.resume_run_dir, resume_checkpoint=args.resume_checkpoint,
            ) if args.resume_run_dir or args.resume_checkpoint else None
        )
        if get_rank() == 0:
            rich.print(config)
        train_module = config.train_module.build(model)
        dataset = config.dataset.build()
        loader = config.data_loader.build(
            dataset,
            dp_process_group=train_module.dp_process_group,
        )
        trainer = config.trainer.build(train_module, loader)
        if resume_checkpoint is not None:
            trainer.load_checkpoint(
                resume_checkpoint, load_optim_state=True, load_trainer_state=True
            )
            if trainer.global_train_tokens_seen != trainer.global_step * GLOBAL_BATCH_TOKENS:
                raise ValueError("Resumed step/token counters do not match the global batch size")
        run_config = config.as_config_dict()
        if resume_checkpoint is not None:
            run_config["resume_source_checkpoint"] = str(resume_checkpoint.resolve())
        cast(
            ConfigSaverCallback, trainer.callbacks["config_saver"]
        ).config = run_config
        if get_rank() == 0:
            copy_tokenizer_to_dir(
                str(config.dataset.tokenizer.identifier),
                Path(trainer.save_folder) / "tokenizer",
            )
        trainer.fit()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
