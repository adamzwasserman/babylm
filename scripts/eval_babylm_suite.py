"""Wrapper around the official BabyLM 2025 evaluation pipeline.

The pipeline lives at https://github.com/babylm/evaluation-pipeline-2025 and
covers BLiMP, BLiMP Supplement, EWoK, and GLUE for the BabyLM weighted
leaderboard score. Two upstream entry points cover everything we need:

    eval_zero_shot.sh <model> causal      # BLiMP, BLiMP-Sup, EWoK, etc.
    eval_finetuning.sh <model> ... <seed> # GLUE tasks

We clone the pipeline on first use, sanity-check that `evaluation_data/`
(downloaded separately from https://osf.io/ryjfm/) is present, shell out to
the two scripts, then harvest every `results.txt` / `best_temperature_report.txt`
the pipeline writes under `results/<model_basename>/`. Each seed's output
directory is renamed to `results/seed{S}/` so concurrent seeds don't collide.

Usage:
    uv run python scripts/eval_babylm_suite.py models/seed42/best --seed 42

Output:
    eval_results/seed{S}_babylm.json with the structure:
    {
      "checkpoint": "...",
      "seed": 42,
      "tasks": {
        "zero_shot/causal/blimp/blimp_filtered": {"accuracy": 0.7628, ...},
        "finetune/boolq": {"accuracy": ..., "f1": ..., "mcc": ...},
        ...
      }
    }

Pre-requisites on the host:
    - The pipeline's requirements installed (it lists requirements.txt at root).
    - evaluation_data/ present at the pipeline root (download from
      https://osf.io/ryjfm/, place it in eval/evaluation-pipeline-2025/).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE_REPO_URL = "https://github.com/babylm/evaluation-pipeline-2025.git"
PIPELINE_DIR_NAME = "evaluation-pipeline-2025"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pipeline_dir() -> Path:
    return _project_root() / "eval" / PIPELINE_DIR_NAME


def ensure_pipeline_cloned() -> Path:
    target = _pipeline_dir()
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {PIPELINE_REPO_URL} into {target}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", PIPELINE_REPO_URL, str(target)],
        check=True,
    )
    return target


def ensure_eval_data(pipeline_dir: Path) -> None:
    data_dir = pipeline_dir / "evaluation_data"
    if data_dir.is_dir() and any(data_dir.iterdir()):
        return
    sys.exit(
        f"Missing {data_dir}. Download evaluation_data/ from "
        "https://osf.io/ryjfm/ and place it under "
        f"{pipeline_dir.relative_to(_project_root())}/."
    )


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ (cd {cwd.relative_to(_project_root())} && {' '.join(cmd)})")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Pipeline command failed: {' '.join(cmd)}")


def _parse_kv_file(path: Path) -> dict:
    out: dict[str, float | str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def harvest_results(pipeline_dir: Path, model_basename: str) -> dict:
    """Glob every results.txt and best_temperature_report.txt the pipeline
    wrote under results/<model_basename>/ and key them by relative path."""
    base = pipeline_dir / "results" / model_basename
    if not base.is_dir():
        raise FileNotFoundError(f"No results under {base}")
    out: dict[str, dict] = {}
    for path in sorted(base.rglob("results.txt")):
        rel = path.parent.relative_to(base).as_posix()
        out[rel] = _parse_kv_file(path)
    for path in sorted(base.rglob("best_temperature_report.txt")):
        rel = path.parent.relative_to(base).as_posix()
        out.setdefault(rel, {})
        out[rel].update(_parse_kv_file(path))
    return out


def evaluate_checkpoint(checkpoint: str, seed: int | None) -> dict:
    pipeline_dir = ensure_pipeline_cloned()
    ensure_eval_data(pipeline_dir)

    model_basename = Path(checkpoint).name
    results_root = pipeline_dir / "results" / model_basename
    if results_root.exists():
        # Clean prior run for this basename so the harvest is unambiguous.
        shutil.rmtree(results_root)

    abs_ckpt = str(Path(checkpoint).resolve())
    seed_arg = str(seed if seed is not None else 42)

    _run(
        ["bash", "eval_zero_shot.sh", abs_ckpt, "causal"],
        cwd=pipeline_dir,
    )
    _run(
        # Args (per upstream eval_finetuning.sh):
        # MODEL_PATH LR BSZ BIG_BSZ MAX_EPOCHS WSC_EPOCHS SEED
        ["bash", "eval_finetuning.sh",
         abs_ckpt, "3e-5", "32", "16", "10", "30", seed_arg],
        cwd=pipeline_dir,
    )

    tasks = harvest_results(pipeline_dir, model_basename)

    # Move results/<basename>/ -> results/seed{S}/ so concurrent seeds don't
    # clobber each other.
    if seed is not None:
        seed_root = pipeline_dir / "results" / f"seed{seed}"
        if seed_root.exists():
            shutil.rmtree(seed_root)
        shutil.move(results_root, seed_root)

    return {
        "checkpoint": checkpoint,
        "seed": seed,
        "tasks": tasks,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a HuggingFace checkpoint dir")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label used for the output filename")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.output_dir) if args.output_dir else _project_root() / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_checkpoint(args.checkpoint, args.seed)

    seed_tag = f"seed{args.seed}" if args.seed is not None else Path(args.checkpoint).name
    out_path = out_dir / f"{seed_tag}_babylm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(f"Tasks captured: {len(summary['tasks'])}")
    for k in sorted(summary["tasks"]):
        print(f"  {k}")


if __name__ == "__main__":
    main()
