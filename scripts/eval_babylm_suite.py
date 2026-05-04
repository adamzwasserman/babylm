"""Wrapper around the official BabyLM 2025 evaluation pipeline.

The pipeline lives at https://github.com/babylm/evaluation-pipeline-2025 and
covers BLiMP, BLiMP Supplement, EWoK, and GLUE for the BabyLM weighted
leaderboard score. We clone it on first use and shell out to its provided
entry points, then collect the per-task scores into a single JSON under
eval_results/seed{S}_babylm.json.

The 2025 pipeline is the one referenced by the paper (Charpentier and Samuel,
2024); the 2026 pipeline (released April 2026) is API-compatible and can be
substituted by changing PIPELINE_REPO_URL.

Usage:
    uv run python scripts/eval_babylm_suite.py models/seed42/chck_92M
    uv run python scripts/eval_babylm_suite.py models/seed42/chck_92M --seed 42

Output:
    eval_results/seed{S}_babylm.json with the structure:
    {
        "checkpoint": "...",
        "seed": 42,
        "blimp": {"average": 0.7628, "per_task": {...}},
        "blimp_supplement": {"average": 0.34, "per_task": {...}},
        "ewok": {"average": 0.50, "per_task": {...}},
        "glue": {"average": 0.6565, "per_task": {...}}
    }
"""

from __future__ import annotations

import argparse
import json
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
    """Clone the BabyLM 2025 eval pipeline on first use; return the path."""
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


def _run_pipeline_task(pipeline_dir: Path, script_name: str, checkpoint: str) -> dict:
    """Invoke a single pipeline shell entry point and parse its JSON output.

    The 2025 pipeline ships per-task scripts (eval_blimp.sh, eval_glue.sh, ...)
    that take a checkpoint path and write JSON to results/<task>.json. We
    shell out, then read the result.
    """
    cmd = ["bash", str(pipeline_dir / script_name), checkpoint]
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=pipeline_dir, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f"{script_name} failed for {checkpoint}")
    # Standard pipeline convention: results land under results/<basename>.json.
    task = script_name.replace("eval_", "").replace(".sh", "")
    result_path = pipeline_dir / "results" / f"{task}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected {result_path} after running {script_name}")
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_checkpoint(checkpoint: str, seed: int | None) -> dict:
    pipeline_dir = ensure_pipeline_cloned()

    summary: dict = {"checkpoint": checkpoint, "seed": seed}
    for task_script, key in [
        ("eval_blimp.sh", "blimp"),
        ("eval_blimp_supplement.sh", "blimp_supplement"),
        ("eval_ewok.sh", "ewok"),
        ("eval_glue.sh", "glue"),
    ]:
        try:
            summary[key] = _run_pipeline_task(pipeline_dir, task_script, checkpoint)
        except FileNotFoundError as e:
            # Some pipeline releases ship a single eval.sh; fall back to skipping
            # the missing per-task entry point and let the user wire it manually.
            print(f"  skip {key}: {e}", file=sys.stderr)
            summary[key] = {"error": str(e)}
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a HuggingFace checkpoint dir")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label used for the output filename (eval_results/seed{S}_babylm.json)")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the JSON (default: <project>/eval_results/)")
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


if __name__ == "__main__":
    main()
