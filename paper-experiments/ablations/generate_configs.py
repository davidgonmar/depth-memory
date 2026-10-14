#!/usr/bin/env python3
"""Generate the 400M LDM ablation model configs from the canonical LDM config.

Every ablation is the canonical ``pretrain/configs/400m/ldm.json`` with exactly
one change applied to ``ldm.quality``. The table below defines the paper's
ablation rows: the variant name is the config file name, and the group is the
row group of the paper's ablation table.

Run ``python paper-experiments/ablations/generate_configs.py`` after changing
the canonical config or this table.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "paper-experiments/pretrain/configs/400m/ldm.json"
OUTPUT = ROOT / "paper-experiments/ablations/configs/400m"

# Each variant: (row group of the paper table, label, [(path inside quality, value), ...]).
VARIANTS: dict[str, tuple[str, str, list[tuple[tuple[str, ...], object]]]] = {
    "ldm": ("baseline", "LDM", []),
    # (a) Residual stream.
    "replace_read": ("a", "F(r)", [(("replace_read",), True)]),
    "no_post_function_residual": ("a", "x = delta", [(("no_post_function_residual",), True)]),
    # (b) Recurrence. The plain rules keep the learned forgetting factor and apply it to M.
    "delta_rule": ("b", "Delta", [(("recurrence",), "delta_rule")]),
    "outer_product": ("b", "Outer", [(("recurrence",), "outer_product")]),
    # (c) Addressing and stored values.
    "value_stream": ("c", "v = x", [(("value_source",), "updated_stream")]),
    "query_key_delta": ("c", "QK(delta)", [
        (("addresses", "query", "source"), "branch_delta"),
        (("addresses", "key", "source"), "branch_delta"),
    ]),
    # The read gate is a head of the query network, so a static query has no read gate.
    "static_query": ("c", "Static q", [
        (("addresses", "query", "mode"), "static"),
        (("addresses", "query", "source"), None),
        (("read_gate", "enabled"), False),
    ]),
    "groups_1": ("c", "G = 1", [(("groups",), 1)]),
    "groups_8": ("c", "G = 8", [(("groups",), 8)]),
    # (d) Gain, gates and forgetting.
    "no_read_gain": ("d", "-gamma", [(("learned_read_gain", "enabled"), False)]),
    "no_read_gate": ("d", "g = 1", [(("read_gate", "enabled"), False)]),
    "no_write_gate": ("d", "omega = 1", [(("write_gate", "enabled"), False)]),
    "no_forgetting": ("d", "lambda = 1", [
        (("rls", "lambda_init"), 1.0),
        (("rls", "learn_lambda"), False),
    ]),
}


def apply(quality: dict, changes) -> None:
    for path, value in changes:
        node = quality
        for key in path[:-1]:
            node = node[key]
        if path[-1] not in node:
            raise KeyError(f"quality path {list(path)} does not exist in the canonical config")
        node[path[-1]] = value


def variant_config(canonical: dict, name: str) -> dict:
    config = copy.deepcopy(canonical)
    apply(config["ldm"]["quality"], VARIANTS[name][2])
    return config


def main() -> None:
    canonical = json.loads(CANONICAL.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        path = OUTPUT / f"{name}.json"
        path.write_text(json.dumps(variant_config(canonical, name), indent=2) + "\n")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
