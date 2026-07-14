#!/usr/bin/env python3
"""Compare complete experiment TOMLs and report every intentional difference."""

from __future__ import annotations

import argparse
import pprint
import tomllib
from pathlib import Path


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            key_path = f"{prefix}.{key}" if prefix else key
            out.update(flatten(item, key_path))
        return out
    return {prefix: value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    parsed = {path.name: flatten(tomllib.loads(path.read_text())) for path in args.configs}
    keys = sorted(set().union(*(set(values) for values in parsed.values())))
    baseline_name = args.configs[0].name
    baseline = parsed[baseline_name]
    print(f"baseline = {baseline_name}")
    differing = []
    for key in keys:
        values = {name: values.get(key, "<missing>") for name, values in parsed.items()}
        if len(set(map(repr, values.values()))) > 1:
            differing.append((key, values))
            print(f"{key}: {pprint.pformat(values, sort_dicts=True)}")
    print(f"differing_fields = {len(differing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
