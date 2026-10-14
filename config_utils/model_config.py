from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from olmo_core.config import DType
from olmo_core.nn.feed_forward import ActivationFunction, FeedForwardConfig, FeedForwardType
from olmo_core.nn.transformer import TransformerBlockType

HF_TO_LLAMA_LIKE_KEY_MAP = {
    "hidden_size": "d_model",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "rms_norm_eps": "layer_norm_eps",
    "torch_dtype": "dtype",
    "initializer_range": "init_std",
}

DIRECT_LLAMA_LIKE_KEYS = {
    "d_model",
    "vocab_size",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "qk_norm",
    "use_head_qk_norm",
    "layer_norm_eps",
    "rope_theta",
    "no_global_rope",
    "hidden_size_multiple_of",
    "hidden_size_multiplier",
    "fused_ops",
    "use_flash",
    "init_std",
    "embedding_init_std",
    "embed_scale",
    "block_name",
    "attnres_block_size",
    "tie_word_embeddings",
}


def _resolve_dtype(value: Any, default: DType = DType.bfloat16) -> DType:
    if value is None:
        return default
    if isinstance(value, DType):
        return value
    if isinstance(value, str):
        normalized = value.removeprefix("torch.")
        try:
            return DType(normalized)
        except ValueError as exc:
            raise ValueError(f"Unsupported dtype in model config: {value}") from exc
    raise TypeError(f"Unsupported dtype value in model config: {value!r}")


def _resolve_activation(value: Any) -> ActivationFunction:
    activation_aliases = {
        "silu": ActivationFunction.silu,
        "swiglu": ActivationFunction.silu,
        "gelu_tanh": ActivationFunction.gelu_tanh,
        "gelu_pytorch_tanh": ActivationFunction.gelu_tanh,
    }

    if value is None:
        return ActivationFunction.silu
    if isinstance(value, ActivationFunction):
        return value
    if isinstance(value, str) and value in activation_aliases:
        return activation_aliases[value]
    raise ValueError(f"Unsupported hidden_act in model config: {value}")


def _build_feed_forward_config(raw_config: dict[str, Any], dtype: DType) -> Optional[FeedForwardConfig]:
    feed_forward = raw_config.get("feed_forward")
    if feed_forward is not None:
        if not isinstance(feed_forward, dict):
            raise TypeError("'feed_forward' in model config must be a JSON object")

        ff_kwargs = dict(feed_forward)
        ff_kwargs["dtype"] = _resolve_dtype(ff_kwargs.get("dtype"), default=dtype)
        ff_kwargs["activation"] = _resolve_activation(
            ff_kwargs.get("activation", raw_config.get("hidden_act"))
        )
        if "name" in ff_kwargs:
            ff_kwargs["name"] = FeedForwardType(ff_kwargs["name"])
        return FeedForwardConfig(**ff_kwargs)

    intermediate_size = raw_config.get("intermediate_size")
    if intermediate_size is None:
        return None

    return FeedForwardConfig(
        hidden_size=intermediate_size,
        bias=raw_config.get("mlp_bias", False),
        dtype=dtype,
        activation=_resolve_activation(raw_config.get("hidden_act")),
    )


def load_llama_like_kwargs(config_path: str | Path, tokenizer_vocab_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    if raw_config.get("attention_bias") not in (None, False):
        raise ValueError(
            "This script only supports attention_bias=false because TransformerConfig.llama_like "
            "builds a bias-free attention module."
        )
    model_kwargs: dict[str, Any] = {}
    for key in DIRECT_LLAMA_LIKE_KEYS:
        value = raw_config.get(key)
        if value is not None:
            model_kwargs[key] = value

    for source_key, target_key in HF_TO_LLAMA_LIKE_KEY_MAP.items():
        if target_key not in model_kwargs and raw_config.get(source_key) is not None:
            model_kwargs[target_key] = raw_config[source_key]

    raw_vocab_size = model_kwargs.get("vocab_size")
    if raw_vocab_size is not None and int(raw_vocab_size) != int(tokenizer_vocab_size):
        raise ValueError(
            f"Model config vocab_size ({raw_vocab_size}) does not match tokenizer vocab_size "
            f"({tokenizer_vocab_size}). Update the JSON or tokenizer so they agree."
        )

    model_kwargs["dtype"] = _resolve_dtype(model_kwargs.get("dtype"), default=DType.bfloat16)
    if "block_name" in model_kwargs:
        model_kwargs["block_name"] = TransformerBlockType(model_kwargs["block_name"])
    elif "attnres_block_size" in model_kwargs:
        model_kwargs["block_name"] = TransformerBlockType.attnres
    model_kwargs["vocab_size"] = int(tokenizer_vocab_size)

    feed_forward = _build_feed_forward_config(raw_config, model_kwargs["dtype"])
    if feed_forward is not None:
        model_kwargs["feed_forward"] = feed_forward

    required_keys = ("d_model", "n_layers", "n_heads", "vocab_size")
    missing_keys = [key for key in required_keys if model_kwargs.get(key) is None]
    if missing_keys:
        raise ValueError(
            f"Missing required model config fields for llama_like(): {', '.join(missing_keys)}"
        )

    return model_kwargs, raw_config
