from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from glob import glob
from hashlib import sha1
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from olmo_core.data import TokenizerConfig

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PARQUET_GLOB = "data/fineweb-edu/100BT/*.parquet"
DEFAULT_EVAL_PARQUET_PATH = "data/fineweb-edu/eval/013_00008.parquet"
DEFAULT_OUTPUT_DIR = "data/fineweb-edu/tokenized"
DEFAULT_TOKENIZER_PATH = "pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"


@dataclass
class ShardStats:
    split: str
    source_parquet: str
    output_npy: str
    output_doc_indices: Optional[str]
    num_documents: int
    num_tokens: int
    dtype: str
    eos_token_id: int
    tokenizer_name_or_path: str


def format_worker_label(worker_id: int, num_workers: int) -> str:
    return f"worker {worker_id + 1}/{num_workers}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize parquet pretraining data into OLMo-core native numpy shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-parquet-glob", type=str, default=DEFAULT_TRAIN_PARQUET_GLOB)
    parser.add_argument("--eval-parquet-path", type=str, default=DEFAULT_EVAL_PARQUET_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer-name-or-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--bos-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--pad-token-id", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit-train-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--train-worker-id", type=int, default=0)
    parser.add_argument("--train-num-workers", type=int, default=1)
    parser.add_argument("--progress-log-interval-docs", type=int, default=8192)
    parser.add_argument(
        "--write-doc-indices",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write optional .csv.gz document boundary sidecars next to each shard.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def choose_vocab_size(tokenizer) -> int:
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
        return int(tokenizer.vocab_size)
    if hasattr(tokenizer, "get_vocab_size"):
        return int(tokenizer.get_vocab_size())
    return int(len(tokenizer))


def resolve_tokenizer_field(
    *,
    tokenizer_value: Optional[int],
    override_value: Optional[int],
    field_name: str,
    required: bool,
) -> Optional[int]:
    if tokenizer_value is not None:
        tokenizer_value = int(tokenizer_value)
    if override_value is not None:
        override_value = int(override_value)

    if tokenizer_value is not None and override_value is not None and tokenizer_value != override_value:
        raise ValueError(
            f"Tokenizer {field_name} ({tokenizer_value}) does not match CLI override ({override_value}). "
            f"Update the tokenizer or remove --{field_name.replace('_', '-')}."
        )

    if tokenizer_value is not None:
        return tokenizer_value
    if override_value is not None:
        return override_value
    if required:
        raise ValueError(
            f"Tokenizer is missing {field_name}. Provide --{field_name.replace('_', '-')} explicitly."
        )
    return None


def build_tokenizer_config(args: argparse.Namespace, tokenizer) -> TokenizerConfig:
    vocab_size = choose_vocab_size(tokenizer)
    if args.vocab_size is not None and int(args.vocab_size) != vocab_size:
        raise ValueError(
            f"Tokenizer vocab size ({vocab_size}) does not match CLI override ({args.vocab_size}). "
            "Update the tokenizer or remove --vocab-size."
        )

    bos_token_id = resolve_tokenizer_field(
        tokenizer_value=getattr(tokenizer, "bos_token_id", None),
        override_value=args.bos_token_id,
        field_name="bos_token_id",
        required=False,
    )
    eos_token_id = resolve_tokenizer_field(
        tokenizer_value=getattr(tokenizer, "eos_token_id", None),
        override_value=args.eos_token_id,
        field_name="eos_token_id",
        required=True,
    )
    pad_token_id = resolve_tokenizer_field(
        tokenizer_value=getattr(tokenizer, "pad_token_id", None),
        override_value=args.pad_token_id,
        field_name="pad_token_id",
        required=True,
    )

    return TokenizerConfig(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        identifier=args.tokenizer_name_or_path,
    )


def infer_numpy_dtype(vocab_size: int) -> np.dtype:
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError(f"Unsupported vocab size for numpy unsigned integer dtypes: {vocab_size}")


def load_hf_tokenizer(args: argparse.Namespace):
    tokenizer_path = Path(args.tokenizer_name_or_path)
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    if tokenizer_path.is_file() and tokenizer_path.suffix == ".json":
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    elif tokenizer_path.is_dir() and tokenizer_json_path.is_file():
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            use_fast=True,
            local_files_only=tokenizer_path.exists(),
        )

    return tokenizer


def expand_train_files(train_parquet_glob: str, limit: Optional[int]) -> List[Path]:
    paths = sorted(Path(path) for path in glob(train_parquet_glob))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {train_parquet_glob}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def get_eval_file(eval_parquet_path: str) -> Path:
    path = Path(eval_parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval parquet file does not exist: {eval_parquet_path}")
    return path


def iter_parquet_text_batches(
    parquet_path: Path,
    *,
    text_field: str,
    batch_size: int,
) -> Iterator[List[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to pre-tokenize parquet files") from exc

    parquet_file = pq.ParquetFile(parquet_path)
    if text_field not in parquet_file.schema.names:
        raise KeyError(f"Column '{text_field}' not found in {parquet_path}")

    for record_batch in parquet_file.iter_batches(columns=[text_field], batch_size=batch_size):
        column = record_batch.column(0).to_pylist()
        texts = ["" if value is None else value if isinstance(value, str) else str(value) for value in column]
        if texts:
            yield texts


def normalize_token_ids(token_ids: List[int], eos_token_id: int) -> List[int]:
    if not token_ids:
        return [eos_token_id]
    if token_ids[-1] != eos_token_id:
        return token_ids + [eos_token_id]
    return token_ids


def make_output_name(source_path: Path, seen_names: set[str]) -> str:
    candidate = f"{source_path.stem}.npy"
    if candidate not in seen_names:
        seen_names.add(candidate)
        return candidate

    suffix = sha1(str(source_path).encode()).hexdigest()[:8]
    candidate = f"{source_path.stem}-{suffix}.npy"
    seen_names.add(candidate)
    return candidate


def shard_meta_path(output_npy_path: Path) -> Path:
    return output_npy_path.with_suffix(".meta.json")


def shard_doc_indices_path(output_npy_path: Path) -> Path:
    return output_npy_path.with_suffix(".csv.gz")


def write_json(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def load_shard_stats(meta_path: Path) -> ShardStats:
    return ShardStats(**json.loads(meta_path.read_text()))


def collect_existing_split_stats(split_dir: Path) -> List[ShardStats]:
    stats: List[ShardStats] = []
    if not split_dir.exists():
        return stats
    for meta_path in sorted(split_dir.glob("*.meta.json")):
        stats.append(load_shard_stats(meta_path))
    return stats


def filter_train_files_for_worker(
    train_parquet_paths: List[Path],
    *,
    worker_id: int,
    num_workers: int,
) -> List[Path]:
    if num_workers <= 0:
        raise ValueError("--train-num-workers must be positive")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("--train-worker-id must satisfy 0 <= worker_id < train_num_workers")
    return [path for idx, path in enumerate(train_parquet_paths) if idx % num_workers == worker_id]


def tokenize_parquet_to_memmap(
    parquet_path: Path,
    *,
    split: str,
    output_npy_path: Path,
    tokenizer,
    tokenizer_config: TokenizerConfig,
    np_dtype: np.dtype,
    text_field: str,
    batch_size: int,
    write_doc_indices: bool,
    overwrite: bool,
    skip_existing: bool,
    progress_log_interval_docs: int,
    worker_label: str,
) -> ShardStats:
    meta_path = shard_meta_path(output_npy_path)
    doc_indices_path = shard_doc_indices_path(output_npy_path)

    if output_npy_path.exists() and not overwrite:
        if skip_existing:
            if not meta_path.exists():
                raise FileExistsError(
                    f"Found existing shard without metadata sidecar: {output_npy_path}"
                )
            return ShardStats(**json.loads(meta_path.read_text()))
        raise FileExistsError(
            f"Output shard already exists: {output_npy_path}. Use --overwrite or --skip-existing."
        )

    output_npy_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy_path = output_npy_path.with_suffix(output_npy_path.suffix + ".tmp")
    tmp_doc_indices_path = doc_indices_path.with_suffix(doc_indices_path.suffix + ".tmp")

    for path in (tmp_npy_path, tmp_doc_indices_path):
        if path.exists():
            path.unlink()

    if overwrite:
        for path in (output_npy_path, meta_path):
            if path.exists():
                path.unlink()
        if doc_indices_path.exists():
            doc_indices_path.unlink()

    num_documents = 0
    num_tokens = 0
    doc_start = 0
    docs_since_log = 0
    last_log_time = time.monotonic()
    use_tqdm = sys.stderr.isatty() and worker_label == "worker 1/1"

    progress = tqdm(
        desc=f"{split}:{parquet_path.name}",
        unit="docs",
        dynamic_ncols=True,
        disable=not use_tqdm,
    )

    log.info("%s starting %s shard %s", worker_label, split, parquet_path.name)

    with tmp_npy_path.open("wb") as data_fh:
        doc_index_fh = (
            gzip.open(tmp_doc_indices_path, "wt", compresslevel=6) if write_doc_indices else None
        )
        try:
            for text_batch in iter_parquet_text_batches(
                parquet_path,
                text_field=text_field,
                batch_size=batch_size,
            ):
                encoded = tokenizer(
                    text_batch,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                for token_ids in encoded:
                    normalized = normalize_token_ids(token_ids, tokenizer_config.eos_token_id)
                    np.asarray(normalized, dtype=np_dtype).tofile(data_fh)

                    doc_end = doc_start + len(normalized)
                    if doc_index_fh is not None:
                        doc_index_fh.write(f"{doc_start},{doc_end}\n")

                    doc_start = doc_end
                    num_documents += 1
                    num_tokens += len(normalized)
                    docs_since_log += 1

                progress.update(len(encoded))
                progress.set_postfix(tokens=f"{num_tokens:,}")
                now = time.monotonic()
                if (
                    not use_tqdm
                    and progress_log_interval_docs > 0
                    and docs_since_log >= progress_log_interval_docs
                ) or (not use_tqdm and now - last_log_time >= 60 and docs_since_log > 0):
                    log.info(
                        "%s progress %s: %s docs, %s tokens",
                        worker_label,
                        parquet_path.name,
                        f"{num_documents:,}",
                        f"{num_tokens:,}",
                    )
                    docs_since_log = 0
                    last_log_time = now
        finally:
            if doc_index_fh is not None:
                doc_index_fh.close()
            progress.close()

    tmp_npy_path.replace(output_npy_path)
    if write_doc_indices:
        tmp_doc_indices_path.replace(doc_indices_path)
    elif tmp_doc_indices_path.exists():
        tmp_doc_indices_path.unlink()

    stats = ShardStats(
        split=split,
        source_parquet=str(parquet_path),
        output_npy=str(output_npy_path),
        output_doc_indices=str(doc_indices_path) if write_doc_indices else None,
        num_documents=num_documents,
        num_tokens=num_tokens,
        dtype=np_dtype.name,
        eos_token_id=tokenizer_config.eos_token_id,
        tokenizer_name_or_path=str(tokenizer_config.identifier),
    )
    write_json(meta_path, asdict(stats))
    log.info(
        "%s wrote %s shard %s with %s documents and %s tokens",
        worker_label,
        split,
        output_npy_path,
        f"{num_documents:,}",
        f"{num_tokens:,}",
    )
    return stats


def write_split_manifest(split_dir: Path, stats_list: Iterable[ShardStats]) -> None:
    stats_list = list(stats_list)
    manifest_path = split_dir / "manifest.jsonl"
    lines = "\n".join(json.dumps(asdict(stats), sort_keys=True) for stats in stats_list)
    manifest_path.write_text(lines + ("\n" if lines else ""))

    total_documents = sum(stats.num_documents for stats in stats_list)
    total_tokens = sum(stats.num_tokens for stats in stats_list)
    summary = {
        "split": split_dir.name,
        "num_shards": len(stats_list),
        "num_documents": total_documents,
        "num_tokens": total_tokens,
        "shards": [asdict(stats) for stats in stats_list],
    }
    write_json(split_dir / "summary.json", summary)


def write_top_level_summary(
    output_dir: Path,
    *,
    tokenizer_config: TokenizerConfig,
    train_stats: List[ShardStats],
    eval_stats: List[ShardStats],
) -> None:
    payload = {
        "format": "olmo-core-numpy",
        "train_glob": str(output_dir / "train" / "*.npy"),
        "eval_paths": [stats.output_npy for stats in eval_stats],
        "dtype": infer_numpy_dtype(tokenizer_config.vocab_size).name,
        "tokenizer": asdict(tokenizer_config),
        "train_num_shards": len(train_stats),
        "train_num_documents": sum(stats.num_documents for stats in train_stats),
        "train_num_tokens": sum(stats.num_tokens for stats in train_stats),
        "eval_num_shards": len(eval_stats),
        "eval_num_documents": sum(stats.num_documents for stats in eval_stats),
        "eval_num_tokens": sum(stats.num_tokens for stats in eval_stats),
    }
    write_json(output_dir / "summary.json", payload)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    output_dir = Path(args.output_dir)
    train_output_dir = output_dir / "train"
    eval_output_dir = output_dir / "eval"
    train_output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_hf_tokenizer(args)
    tokenizer_config = build_tokenizer_config(args, tokenizer)
    np_dtype = infer_numpy_dtype(tokenizer_config.vocab_size)

    log.info("Repository root: %s", REPO_ROOT)
    log.info("Tokenizer: %s", args.tokenizer_name_or_path)
    log.info("Resolved tokenizer vocab size: %s", tokenizer_config.vocab_size)
    log.info("Resolved tokenizer eos token id: %s", tokenizer_config.eos_token_id)
    log.info("Resolved tokenizer pad token id: %s", tokenizer_config.pad_token_id)
    log.info("Resolved tokenizer bos token id: %s", tokenizer_config.bos_token_id)
    log.info("Configured dataset dtype: %s", np_dtype.name)
    worker_label = format_worker_label(args.train_worker_id, args.train_num_workers)

    train_parquet_paths: List[Path] = []
    if not args.skip_train:
        train_parquet_paths = expand_train_files(args.train_parquet_glob, args.limit_train_files)
        train_parquet_paths = filter_train_files_for_worker(
            train_parquet_paths,
            worker_id=args.train_worker_id,
            num_workers=args.train_num_workers,
        )
        log.info(
            "%s assigned %s train parquet files",
            worker_label,
            f"{len(train_parquet_paths):,}",
        )

    eval_parquet_path: Optional[Path] = None
    if not args.skip_eval:
        eval_parquet_path = get_eval_file(args.eval_parquet_path)
        log.info("%s will also process eval shard %s", worker_label, eval_parquet_path.name)

    train_stats: List[ShardStats] = []
    eval_stats: List[ShardStats] = []

    seen_train_names: set[str] = set()
    seen_eval_names: set[str] = set()

    for train_parquet_path in train_parquet_paths:
        output_name = make_output_name(train_parquet_path, seen_train_names)
        train_stats.append(
            tokenize_parquet_to_memmap(
                train_parquet_path,
                split="train",
                output_npy_path=train_output_dir / output_name,
                tokenizer=tokenizer,
                tokenizer_config=tokenizer_config,
                np_dtype=np_dtype,
                text_field=args.text_field,
                batch_size=args.batch_size,
                write_doc_indices=args.write_doc_indices,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
                progress_log_interval_docs=args.progress_log_interval_docs,
                worker_label=worker_label,
            )
        )

    if eval_parquet_path is not None:
        eval_output_name = make_output_name(eval_parquet_path, seen_eval_names)
        eval_stats.append(
            tokenize_parquet_to_memmap(
                eval_parquet_path,
                split="eval",
                output_npy_path=eval_output_dir / eval_output_name,
                tokenizer=tokenizer,
                tokenizer_config=tokenizer_config,
                np_dtype=np_dtype,
                text_field=args.text_field,
                batch_size=args.batch_size,
                write_doc_indices=args.write_doc_indices,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
                progress_log_interval_docs=args.progress_log_interval_docs,
                worker_label=worker_label,
            )
        )

    if not args.skip_summary:
        all_train_stats = collect_existing_split_stats(train_output_dir)
        all_eval_stats = collect_existing_split_stats(eval_output_dir)
        write_split_manifest(train_output_dir, all_train_stats)
        write_split_manifest(eval_output_dir, all_eval_stats)
        write_top_level_summary(
            output_dir,
            tokenizer_config=tokenizer_config,
            train_stats=all_train_stats,
            eval_stats=all_eval_stats,
        )

    log.info("Finished pre-tokenization.")
    log.info("Train shards processed in this run: %s", len(train_stats))
    log.info("Eval shards processed in this run: %s", len(eval_stats))
    log.info("Train glob for NumpyFSLDatasetConfig: %s", train_output_dir / "*.npy")
    if eval_stats:
        log.info("Eval path for NumpyPaddedFSLDatasetConfig: %s", eval_stats[0].output_npy)


if __name__ == "__main__":
    main()
