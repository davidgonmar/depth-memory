from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from olmo_core.data import TokenizerConfig

SPECIAL_TOKEN_CANDIDATES = {
    "eos_token": ("<|endoftext|>", "</s>"),
    "pad_token": ("<|padding|>", "<pad>"),
    "bos_token": ("<s>", "<bos>"),
}


def _token_exists(tokenizer: Any, token: str) -> bool:
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id is not None and token_id != unk_token_id


def _ensure_special_tokens(tokenizer: Any) -> Any:
    for token_attr, candidates in SPECIAL_TOKEN_CANDIDATES.items():
        if getattr(tokenizer, f"{token_attr}_id", None) is not None:
            continue
        for candidate in candidates:
            if _token_exists(tokenizer, candidate):
                setattr(tokenizer, token_attr, candidate)
                break
    return tokenizer


def load_hf_tokenizer(tokenizer_name_or_path: str) -> Any:
    tokenizer_path = Path(tokenizer_name_or_path).expanduser()
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    if tokenizer_path.is_file() and tokenizer_path.suffix == ".json":
        return _ensure_special_tokens(
            PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        )
    if tokenizer_path.is_dir() and tokenizer_json_path.is_file():
        return _ensure_special_tokens(
            PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path))
        )
    return _ensure_special_tokens(
        AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True,
            local_files_only=tokenizer_path.exists(),
        )
    )


def choose_vocab_size(tokenizer: Any) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    return int(len(tokenizer))


def build_tokenizer_config(tokenizer_name_or_path: str) -> TokenizerConfig:
    tokenizer = load_hf_tokenizer(tokenizer_name_or_path)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)

    if eos_token_id is None:
        raise ValueError(
            f"Tokenizer '{tokenizer_name_or_path}' has no eos_token_id. "
            "Use a tokenizer with EOS configured before pretraining."
        )
    if pad_token_id is None:
        raise ValueError(
            f"Tokenizer '{tokenizer_name_or_path}' has no pad_token_id. "
            "Use a tokenizer with PAD configured before pretraining."
        )

    return TokenizerConfig(
        vocab_size=choose_vocab_size(tokenizer),
        eos_token_id=int(eos_token_id),
        pad_token_id=int(pad_token_id),
        bos_token_id=None if bos_token_id is None else int(bos_token_id),
        identifier=tokenizer_name_or_path,
    )


def copy_tokenizer_to_dir(tokenizer_name_or_path: str, destination_dir: str | Path) -> None:
    source_path = Path(tokenizer_name_or_path).expanduser()
    destination_dir = Path(destination_dir)
    if destination_dir.exists():
        return

    if source_path.is_dir():
        tokenizer_json_path = source_path / "tokenizer.json"
        if tokenizer_json_path.is_file() and not (source_path / "tokenizer_config.json").exists():
            load_hf_tokenizer(str(source_path)).save_pretrained(str(destination_dir))
        else:
            shutil.copytree(source_path, destination_dir)
        return

    if source_path.is_file():
        destination_dir.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".json":
            load_hf_tokenizer(tokenizer_name_or_path).save_pretrained(str(destination_dir))
        else:
            shutil.copy2(source_path, destination_dir / source_path.name)
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    load_hf_tokenizer(tokenizer_name_or_path).save_pretrained(str(destination_dir))
