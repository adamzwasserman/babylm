"""Aggregate evaluation results across multiple training seeds.

Reads one JSON file per seed, flattens nested dicts to dot-paths, and prints
a markdown (and optionally LaTeX) table with mean +/- std for each metric.

Each seed file should look like:
    {"blimp_average": 0.81, "ewok_average": 0.55, ...}
or nested:
    {"results": {"blimp": {"average": 0.81}, ...}}

Use --key to drill into a sub-tree (dot-path), e.g. --key results.

Usage:
    uv run python scripts/aggregate_seeds.py path/seed1.json path/seed2.json ...
    uv run python scripts/aggregate_seeds.py 'eval_results/seed*.json'
    uv run python scripts/aggregate_seeds.py 'eval_results/seed*.json' --latex
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd


def _flatten(obj: Any, prefix: str = "") -> Iterator[tuple[str, float]]:
    """Yield (dot.path, scalar) pairs for every numeric leaf in obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            yield from _flatten(v, new_prefix)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)


def _drill(obj: Any, dot_path: str) -> Any:
    if not dot_path:
        return obj
    for part in dot_path.split("."):
        obj = obj[part]
    return obj


def load_seed(path: str, dot_key: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if dot_key:
        data = _drill(data, dot_key)
    return dict(_flatten(data))


def expand_paths(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(matched)
        else:
            files.append(pattern)
    return files


def build_summary(rows: list[dict[str, float]], names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, index=names)
    summary = pd.DataFrame(
        {
            "mean": df.mean(),
            "std": df.std(ddof=1),
            "n": df.count().astype(int),
            "min": df.min(),
            "max": df.max(),
        }
    )
    return summary.sort_index()


def print_markdown(summary: pd.DataFrame) -> None:
    print("| Metric | mean +/- std | min | max | n |")
    print("|---|---|---|---|---|")
    for metric, row in summary.iterrows():
        print(
            f"| {metric} | {row['mean']:.4f} +/- {row['std']:.4f} "
            f"| {row['min']:.4f} | {row['max']:.4f} | {int(row['n'])} |"
        )


def print_latex(summary: pd.DataFrame) -> None:
    print()
    print(r"\begin{tabular}{lr}")
    print(r"\toprule")
    print(r"Metric & mean $\pm$ std \\")
    print(r"\midrule")
    for metric, row in summary.iterrows():
        metric_esc = str(metric).replace("_", r"\_")
        print(f"{metric_esc} & ${row['mean']:.4f} \\pm {row['std']:.4f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", help="JSON files or globs (one per seed)")
    p.add_argument("--key", default="", help="Dot-path into JSON (e.g. results.zero_shot)")
    p.add_argument("--latex", action="store_true", help="Also emit a LaTeX tabular block")
    args = p.parse_args()

    files = expand_paths(args.paths)
    if len(files) < 2:
        print(f"Need at least 2 seed files to aggregate, got {len(files)}.", file=sys.stderr)
        sys.exit(1)

    rows = [load_seed(f, args.key) for f in files]
    names = [Path(f).stem for f in files]

    print(f"Loaded {len(files)} seeds: {names}")
    print()

    summary = build_summary(rows, names)
    print_markdown(summary)

    if args.latex:
        print_latex(summary)


if __name__ == "__main__":
    main()
