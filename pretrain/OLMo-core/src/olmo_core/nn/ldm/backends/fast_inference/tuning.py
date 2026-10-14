"""Tuned launch settings, one JSON table per GPU under ``tuned/``, keyed by the model's address sizes.

For each model shape (groups, key width, value width, hidden width) a table holds the tiling of the
prefill edge kernel and the decode classes: token-count bounds in increasing order, each with the tiling
of the decode kernel. A forward of `n` tokens runs the decode kernel with the tiling of the first class
whose bound is at least `n`, and the prefill kernels above the last class, where `tune` measured them as
at least as fast. A model shape missing from its GPU's table raises: run `tune` for it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

TUNED = Path(__file__).with_name("tuned")


class PrefillTiling(NamedTuple):
    """How one prefill edge-kernel launch divides its work."""

    tokens: int                   # tokens of one CTA cluster
    cluster: int                  # CTAs of one cluster, a divisor of the group count
    key_lanes: int                # lanes per (row, 8 columns) of the dense step, each over K / key_lanes keys


class DecodeTiling(NamedTuple):
    """How one decode launch divides its work."""

    tokens: int                   # tokens of one CTA cluster
    cluster: int                  # CTAs of one cluster, a divisor of the group count
    key_lanes: int                # lanes per (row, 8 columns) of the dense step, each over K / key_lanes keys
    slices: int                   # lanes per head or gate column, each over hidden / slices terms
    weight_chunk: int             # first-layer columns per shared-memory stage (d_model keeps them resident)
    weight_stages: int            # shared-memory stages the first layer streams through


class DecodeClass(NamedTuple):
    max_tokens: int
    tiling: DecodeTiling


class Launches(NamedTuple):
    """Everything tuned for one model shape on one GPU."""

    prefill: PrefillTiling
    decode: tuple[DecodeClass, ...]

    def decode_tiling(self, tokens: int) -> DecodeTiling | None:
        """The decode tiling for a forward of `tokens` tokens, or None when the prefill kernels run it."""
        return next((entry.tiling for entry in self.decode if tokens <= entry.max_tokens), None)


def shape_key(groups: int, key_dim: int, value_dim: int, hidden: int) -> str:
    return f"groups={groups},key_dim={key_dim},value_dim={value_dim},hidden={hidden}"


def table_path(capability: tuple[int, int]) -> Path:
    major, minor = capability
    return TUNED / f"sm{major}{minor}.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"device": None, "shapes": {}}


@lru_cache(maxsize=None)
def launches(capability: tuple[int, int], groups: int, key_dim: int, value_dim: int, hidden: int) -> Launches:
    """The tuned launches of one model shape on a GPU of `capability`."""
    key = shape_key(groups, key_dim, value_dim, hidden)
    entry = _read(table_path(capability))["shapes"].get(key)
    if entry is None:
        major, minor = capability
        raise ValueError(f"no tuned launches for {key} on sm_{major}{minor} in {table_path(capability)}; "
                         "run python -m olmo_core.nn.ldm.backends.fast_inference.tune")
    return Launches(PrefillTiling(**entry["prefill"]),
                    tuple(DecodeClass(c["max_tokens"], DecodeTiling(**c["tiling"])) for c in entry["decode"]))


def write(path: Path, device: str, key: str, tuned: Launches) -> None:
    """Set one shape's launches in the table at `path`, keeping every other shape (one line per class)."""
    data = _read(path)
    data["device"] = device
    data["shapes"][key] = {"prefill": tuned.prefill._asdict(),
                           "decode": [{"max_tokens": c.max_tokens, "tiling": c.tiling._asdict()} for c in tuned.decode]}
    shapes = []
    for name, entry in sorted(data["shapes"].items()):
        classes = ",\n".join(f"    {json.dumps(c)}" for c in entry["decode"])
        shapes.append(f'  {json.dumps(name)}: {{\n   "prefill": {json.dumps(entry["prefill"])},\n'
                      f'   "decode": [\n{classes}\n   ]}}')
    text = f'{{\n "device": {json.dumps(device)},\n "shapes": {{\n' + ",\n".join(shapes) + "\n }\n}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)
