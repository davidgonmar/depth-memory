from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Import the OLMo-core source tree of this repository even when it is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_OLMO_CORE_SRC = _REPO_ROOT / "pretrain" / "OLMo-core" / "src"
if str(_OLMO_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_OLMO_CORE_SRC))

import torch
import torch.nn.functional as F
from cached_path import cached_path
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.generate import GenerationConfig, TransformerGenerationModule
from olmo_core.io import join_path, normalize_path
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device


def _model_forward(generation_module, input_ids: torch.Tensor):
    """The generation module's forward under ``torch.no_grad``."""
    generation_module._set_model_mode("eval")
    with torch.no_grad():
        return generation_module.model(input_ids.to(generation_module.device))


class OLMoCheckpointConfigError(RuntimeError):
    """Raised when a native OLMo-core checkpoint is missing required metadata."""


def _resolve_checkpoint_dir(checkpoint_dir: str | Path) -> str:
    checkpoint_dir = normalize_path(checkpoint_dir)
    if checkpoint_dir.endswith("/model_and_optim"):
        return str(Path(checkpoint_dir).parent)
    return checkpoint_dir


def _load_checkpoint_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    config_path = join_path(checkpoint_dir, "config.json")
    with cached_path(config_path).open() as f:
        return json.load(f)


def _load_tokenizer_config(checkpoint_dir: str | Path) -> TokenizerConfig:
    config_dict = _load_checkpoint_config(checkpoint_dir)
    try:
        return TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except KeyError as exc:
        raise OLMoCheckpointConfigError(
            f"Missing tokenizer config in checkpoint '{checkpoint_dir}'."
        ) from exc


def _load_hf_tokenizer(
    checkpoint_dir: str | Path,
    tokenizer_name_or_path: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    checkpoint_dir = Path(_resolve_checkpoint_dir(checkpoint_dir))
    tokenizer_cfg = _load_tokenizer_config(checkpoint_dir)

    if tokenizer_name_or_path is None:
        local_tokenizer_dir = checkpoint_dir / "tokenizer"
        if local_tokenizer_dir.exists():
            tokenizer_name_or_path = str(local_tokenizer_dir)
        elif tokenizer_cfg.identifier is not None:
            tokenizer_name_or_path = tokenizer_cfg.identifier
        else:
            raise OLMoCheckpointConfigError(
                "Could not determine tokenizer path. Pass --tokenizer explicitly."
            )

    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    if tokenizer_path.is_file() and tokenizer_path.suffix == ".json":
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    elif tokenizer_path.is_dir() and tokenizer_json_path.is_file():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # Keep tokenizer ids aligned with the checkpoint config even when loading from a generic HF repo.
    if tokenizer.pad_token_id is None and tokenizer_cfg.pad_token_id is not None:
        tokenizer.pad_token_id = tokenizer_cfg.pad_token_id
    if tokenizer.eos_token_id is None and tokenizer_cfg.eos_token_id is not None:
        tokenizer.eos_token_id = tokenizer_cfg.eos_token_id
    if tokenizer.bos_token_id is None and tokenizer_cfg.bos_token_id is not None:
        tokenizer.bos_token_id = tokenizer_cfg.bos_token_id

    return tokenizer


def _infer_model_max_length(
    checkpoint_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    override: Optional[int] = None,
) -> Optional[int]:
    if override is not None:
        return override

    config_dict = _load_checkpoint_config(checkpoint_dir)
    max_sequence_length = config_dict.get("train_module", {}).get("max_sequence_length")
    if isinstance(max_sequence_length, int) and max_sequence_length > 0:
        return max_sequence_length

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None:
        return None

    if isinstance(model_max_length, int) and 0 < model_max_length < 10**6:
        return model_max_length

    return None


def _align_ldm_inference_parameters(model: torch.nn.Module) -> None:
    if not getattr(model, "_ldm_enabled", False):
        return
    if model._ldm_config.execution.backend.name == "fast-inference":
        # The LDM parameter modules become the tensors the inference kernels read.
        from olmo_core.nn.ldm.backends.fast_inference.backend import prepare_inference

        prepare_inference(model)
        return

    from olmo_core.nn.ldm.components import ParameterBank, ReadParameters

    # The read gains are built in FP32 even in a BF16 model, and training casts them through its
    # data-parallel wrapper, so cast them here. Banked gains are views of the bank storage, so the
    # bank is cast as well. The forgetting logits keep their FP32 precision.
    for module in model.modules():
        if isinstance(module, (ReadParameters, ParameterBank)):
            module.to(dtype=model.dtype)


class OLMoNativeLM:
    """
    Thin lm-eval-harness adapter for native OLMo-core checkpoints.

    This class subclasses `TemplateLM` at runtime so that the file can still be imported
    in environments where `lm_eval` is not installed yet.
    """

    @classmethod
    def build(
        cls,
        checkpoint_dir: str | Path,
        *,
        tokenizer_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        from lm_eval.api.model import TemplateLM

        checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
        tokenizer = _load_hf_tokenizer(checkpoint_dir, tokenizer_name_or_path)
        tokenizer_cfg = _load_tokenizer_config(checkpoint_dir)
        resolved_device = torch.device(device) if device is not None else get_default_device()

        generation_kwargs: dict[str, Any] = {}
        transformer_config = None
        if dtype is not None:
            generation_kwargs["dtype"] = DType(dtype)
        resolved_attention_backend = None
        if resolved_device.type != "cuda":
            resolved_attention_backend = AttentionBackendName.torch

        if resolved_attention_backend is not None:
            # The generation module's attention override would also set the field named
            # ``backend`` of LDM execution configs, so the override is applied here to
            # attention configs only.
            resolved_attention_backend.assert_supported()
            config_dict = _load_checkpoint_config(checkpoint_dir)
            transformer_config = TransformerConfig.from_dict(config_dict["model"])
            transformer_config.apply(
                lambda config: setattr(
                    config, "backend", resolved_attention_backend
                )
                if isinstance(config, AttentionConfig)
                else None
            )
        # An LDM model trained with the fast backend is evaluated on the fast-inference kernels.
        from olmo_core.nn.ldm.config import LDMConfig

        model_config = transformer_config or TransformerConfig.from_dict(
            _load_checkpoint_config(checkpoint_dir)["model"]
        )
        trained_with_fast: list[LDMConfig] = []
        model_config.apply(
            lambda config: trained_with_fast.append(config)
            if isinstance(config, LDMConfig) and config.execution.backend.name == "fast"
            else None
        )
        for config in trained_with_fast:
            config.execution.backend.name = "fast-inference"
        if trained_with_fast:
            transformer_config = model_config

        eos_token_id = tokenizer_cfg.eos_token_id or tokenizer.eos_token_id
        if eos_token_id is None:
            raise OLMoCheckpointConfigError("Tokenizer has no EOS token id.")
        pad_token_id = tokenizer_cfg.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generation_config = GenerationConfig(
            do_sample=False,
            use_cache=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        generation_module = TransformerGenerationModule.from_checkpoint(
            checkpoint_dir=checkpoint_dir,
            generation_config=generation_config,
            transformer_config=transformer_config,
            device=resolved_device,
            **generation_kwargs,
        )
        _align_ldm_inference_parameters(generation_module.model)

        model_max_length = _infer_model_max_length(
            checkpoint_dir,
            tokenizer,
            override=max_length,
        )

        class _OLMoNativeTemplateLM(TemplateLM):
            backend = "causal"

            def __init__(self):
                super().__init__()
                self.checkpoint_dir = checkpoint_dir
                self.generation_module = generation_module
                self.tokenizer = tokenizer
                self._batch_size = batch_size
                self._max_length = model_max_length
                self._device = str(self.generation_module.device)

            @property
            def eot_token_id(self) -> int:
                token_id = self.tokenizer.eos_token_id
                if token_id is None:
                    if self.tokenizer.bos_token_id is not None:
                        return int(self.tokenizer.bos_token_id)
                    if self.tokenizer.pad_token_id is not None:
                        return int(self.tokenizer.pad_token_id)
                    raise OLMoCheckpointConfigError("Tokenizer has no EOS/BOS/PAD token id.")
                return int(token_id)

            @property
            def tokenizer_name(self) -> str:
                return str(self.checkpoint_dir)

            @property
            def batch_size(self) -> int:
                return self._batch_size

            @property
            def max_length(self) -> Optional[int]:
                return self._max_length

            @property
            def max_gen_toks(self) -> int:
                return 256

            def tok_encode(
                self,
                string: str,
                add_special_tokens: bool | None = None,
                **kwargs,
            ) -> list[int]:
                if add_special_tokens is None:
                    add_special_tokens = False
                return list(
                    self.tokenizer.encode(
                        string,
                        add_special_tokens=add_special_tokens,
                        **kwargs,
                    )
                )

            def tok_decode(self, tokens: list[int]) -> str:
                return self.tokenizer.decode(tokens, skip_special_tokens=False)

            def _encode_pair(
                self, context: str, continuation: str
            ) -> tuple[list[int], list[int]]:
                context_ids, continuation_ids = super()._encode_pair(context, continuation)
                # The inherited splitter assumes separately encoded context IDs
                # prefix the joint encoding. BPE merges can violate that assumption.
                whole_ids = self.tok_encode(context + continuation)
                if context_ids + continuation_ids != whole_ids:
                    raise ValueError(
                        "The context/continuation boundary changes tokenization. "
                        "Refusing to score a different token sequence or silently "
                        "drop answer tokens; use a token-aligned text boundary. "
                        f"context tail={context[-60:]!r} continuation={continuation!r} "
                        f"context_ids tail={context_ids[-4:]} continuation_ids={continuation_ids} "
                        f"whole tail={whole_ids[len(context_ids) - 4:len(context_ids) + len(continuation_ids) + 2]}"
                    )
                return context_ids, continuation_ids

            def apply_chat_template(
                self,
                chat_history: list[dict[str, str]],
                add_generation_prompt: bool = True,
            ) -> str:
                return self.tokenizer.apply_chat_template(
                    chat_history,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )

            def _score_requests(
                self,
                requests: list[tuple[tuple[str, str], list[int], list[int]]],
                *,
                disable_tqdm: bool = False,
            ) -> list[tuple[float, bool]]:
                del disable_tqdm

                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.eot_token_id

                prepared: list[dict[str, Any]] = []
                for cache_key, context_enc, continuation_enc in requests:
                    if not continuation_enc:
                        prepared.append(
                            {
                                "cache_key": cache_key,
                                "input_ids": [self.prefix_token_id],
                                "labels": [-100],
                                "score_start": 0,
                                "empty_continuation": True
                            }
                        )
                        continue

                    full_ids = list(context_enc) + list(continuation_enc)
                    if self.max_length is not None and len(full_ids) > self.max_length + 1:
                        full_ids = full_ids[-(self.max_length + 1) :]

                    score_start = len(full_ids) - len(continuation_enc) - 1
                    if score_start < 0:
                        raise ValueError(
                            "Continuation is longer than the model context window after truncation."
                        )

                    prepared.append(
                        {
                            "cache_key": cache_key,
                            "input_ids": full_ids[:-1],
                            "labels": full_ids[1:],
                            "score_start": score_start,
                        }
                    )

                results: list[tuple[float, bool]] = []
                for batch_start in range(0, len(prepared), self.batch_size):
                    batch = prepared[batch_start : batch_start + self.batch_size]
                    max_seq_len = max(len(item["input_ids"]) for item in batch)
                    model_device = self.generation_module.device

                    batch_input_ids = torch.full(
                        (len(batch), max_seq_len),
                        fill_value=pad_token_id,
                        dtype=torch.long,
                        device=model_device,
                    )
                    batch_labels = torch.full(
                        (len(batch), max_seq_len),
                        fill_value=-100,
                        dtype=torch.long,
                        device=model_device,
                    )
                    lengths: list[int] = []
                    score_starts: list[int] = []

                    for row, item in enumerate(batch):
                        seq_len = len(item["input_ids"])
                        lengths.append(seq_len)
                        score_starts.append(item["score_start"])
                        if seq_len > 0:
                            batch_input_ids[row, :seq_len] = torch.tensor(
                                item["input_ids"], dtype=torch.long, device=model_device
                            )
                            batch_labels[row, :seq_len] = torch.tensor(
                                item["labels"], dtype=torch.long, device=model_device
                            )

                    logits = _model_forward(self.generation_module, batch_input_ids)
                    if not isinstance(logits, torch.Tensor):
                        logits = logits.logits
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    gathered = torch.gather(
                        log_probs,
                        dim=-1,
                        index=batch_labels.clamp_min(0).unsqueeze(-1),
                    ).squeeze(-1)
                    greedy = logits.argmax(dim=-1)

                    for row, item in enumerate(batch):
                        seq_len = lengths[row]
                        score_start = score_starts[row]
                        if item.get("empty_continuation", False):
                            score = 0.0
                            is_greedy = True
                        else:
                            token_scores = gathered[row, score_start:seq_len]
                            token_labels = batch_labels[row, score_start:seq_len]
                            score = float(token_scores.sum().item())
                            is_greedy = bool(
                                torch.equal(greedy[row, score_start:seq_len], token_labels)
                            )

                        results.append((score, is_greedy))
                        self.cache_hook.add_partial("loglikelihood", item["cache_key"], (score, is_greedy))

                return results

            def _loglikelihood_tokens(
                self,
                requests: list[tuple[tuple[str, str], list[int], list[int]]],
                **kwargs,
            ) -> list[tuple[float, bool]]:
                return self._score_requests(requests, **kwargs)

            def loglikelihood_rolling(
                self,
                requests,
                disable_tqdm: bool = False,
            ) -> list[float]:
                del disable_tqdm

                rolling_scores: list[float] = []
                window = None if self.max_length is None else self.max_length + 1

                for (string,) in [req.args for req in requests]:
                    token_ids = [self.prefix_token_id] + self.tok_encode(
                        string,
                        add_special_tokens=False,
                    )
                    if len(token_ids) <= 1:
                        rolling_scores.append(0.0)
                        continue

                    if window is None or len(token_ids) <= window:
                        pairs = [
                            ((string, string), token_ids[:1], token_ids[1:]),
                        ]
                        rolling_scores.append(self._score_requests(pairs)[0][0])
                        continue

                    total_score = 0.0
                    step = window - 1
                    for continuation_start in range(1, len(token_ids), step):
                        continuation_end = min(len(token_ids), continuation_start + step)
                        start = max(0, continuation_end - window)
                        full_ids = token_ids[start:continuation_end]
                        continuation = token_ids[continuation_start:continuation_end]
                        context = full_ids[: len(full_ids) - len(continuation)]
                        pairs = [((string, string), context, continuation)]
                        total_score += self._score_requests(pairs)[0][0]
                    rolling_scores.append(total_score)

                return rolling_scores

            def generate_until(
                self,
                requests,
                disable_tqdm: bool = False,
            ) -> list[str]:
                del disable_tqdm

                pad_token_id = self.tokenizer.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.eot_token_id

                prepared: list[dict[str, Any]] = []
                for index, (context, gen_kwargs) in enumerate(req.args for req in requests):
                    encoded = self.tok_encode(context, add_special_tokens=False)
                    if not encoded:
                        encoded = [self.prefix_token_id]

                    max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                    if self.max_length is not None:
                        max_context_len = self.max_length - max_gen_toks
                        if max_context_len <= 0:
                            raise ValueError(
                                "max_gen_toks must be smaller than the model context window."
                            )
                        encoded = encoded[-max_context_len:]

                    stop_sequences = gen_kwargs.get("until", []) or []
                    if isinstance(stop_sequences, str):
                        stop_sequences = [stop_sequences]
                    prepared.append(
                        {
                            "index": index,
                            "encoded": encoded,
                            "max_gen_toks": max_gen_toks,
                            "stop_sequences": stop_sequences,
                        }
                    )

                outputs = [""] * len(prepared)
                # TorchAttentionBackend does not support KV caching or padded prompt
                # masks, so only prompts with identical lengths can share a batch.
                group_keys = sorted(
                    {
                        (item["max_gen_toks"], len(item["encoded"]))
                        for item in prepared
                    }
                )
                for max_gen_toks, prompt_len in group_keys:
                    group = [
                        item
                        for item in prepared
                        if item["max_gen_toks"] == max_gen_toks
                        and len(item["encoded"]) == prompt_len
                    ]
                    for batch_start in range(0, len(group), self.batch_size):
                        batch = group[batch_start : batch_start + self.batch_size]
                        model_device = self.generation_module.device
                        input_ids = torch.tensor(
                            [item["encoded"] for item in batch],
                            dtype=torch.long,
                            device=model_device,
                        )

                        generated_ids, _, _ = self.generation_module.generate_batch(
                            input_ids,
                            do_sample=False,
                            max_new_tokens=max_gen_toks,
                            pad_token_id=pad_token_id,
                            eos_token_id=self.eot_token_id,
                            use_cache=False,
                            completions_only=True,
                            log_timing=False,
                        )
                        for row, item in enumerate(batch):
                            text = self.tok_decode(generated_ids[row].tolist())
                            for stop_str in item["stop_sequences"]:
                                if stop_str and stop_str in text:
                                    text = text.split(stop_str)[0]
                                    break
                            outputs[item["index"]] = text

                return outputs

        return _OLMoNativeTemplateLM()
