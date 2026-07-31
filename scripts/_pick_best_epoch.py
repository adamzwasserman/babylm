"""Pick the best-performing epoch checkpoint per seed and stage it for
downstream evals.

After running QFrBLiMP on every per-epoch checkpoint
(eval_results/seed{S}_qfrblimp_epoch{E}.json), this helper:

    1. Picks the epoch with the highest QFrBLiMP "overall" accuracy.
    2. Creates models/seed{S}/best -> chck_{N}M_epoch{E*} (relative symlink).
    3. Copies the peak per-epoch JSON to eval_results/seed{S}_qfrblimp.json
       (the canonical filename consumed by aggregate_paper_tables.py).
    4. Writes a short eval_results/seed{S}_best_epoch.json summary recording
       the choice and the per-epoch trajectory for reproducibility.

Usage:
    uv run python scripts/_pick_best_epoch.py --seed 42
    uv run python scripts/_pick_best_epoch.py --seed 42 --metric overall

Importable from the orchestrator. Stays prefixed with an underscore because
it is glue, not a user-facing entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

EPOCH_RE = re.compile(r"_epoch(\d+)\.json$")

# Minimum accuracy gap between the best and the runner-up epoch for the pick to
# count as a peak rather than a plateau. 0.005 (0.5pp) is a heuristic floor: the
# binomial standard error on QFrBLiMP's 1761 items is already ~0.008 at p=0.86.
DEFAULT_MIN_MARGIN = 0.005


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def per_epoch_files(seed: int, eval_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for f in sorted(eval_dir.glob(f"seed{seed}_qfrblimp_epoch*.json")):
        m = EPOCH_RE.search(f.name)
        if m:
            out.append((int(m.group(1)), f))
    return out


def epoch_ckpt_dir(seed: int, epoch: int, models_dir: Path) -> Path:
    # sorted(): Path.glob yields in filesystem order, so an unsorted list would
    # make the "pick the first candidate" branch below resolve differently on
    # two machines holding the same checkpoints.
    candidates = sorted(models_dir.glob(f"seed{seed}/chck_*M_epoch{epoch}"))
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directory matching seed{seed}/chck_*M_epoch{epoch}"
        )
    if len(candidates) > 1:
        # Multiple matches would mean an inconsistent on-disk state; pick the
        # lexicographically first and warn rather than silently dropping.
        print(f"WARN: multiple candidates for seed{seed} epoch{epoch}: {candidates}",
              file=sys.stderr)
    return candidates[0]


def pick(seed: int, metric: str, eval_dir: Path, models_dir: Path,
         min_margin: float = DEFAULT_MIN_MARGIN) -> dict:
    files = per_epoch_files(seed, eval_dir)
    if not files:
        raise SystemExit(
            f"No per-epoch QFrBLiMP results for seed {seed} under {eval_dir}/. "
            "Run phase 2 of run_paper_part1.sh first."
        )
    trajectory = []
    for epoch, path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        score = data.get(metric)
        if score is None:
            raise SystemExit(f"{path} has no '{metric}' field")
        trajectory.append({"epoch": epoch, metric: score, "file": str(path)})

    # Ties resolve to the earliest epoch: sort by score desc, epoch asc.
    ranked = sorted(trajectory, key=lambda t: (-t[metric], t["epoch"]))
    best = ranked[0]
    best_epoch = best["epoch"]
    best_file = Path(best["file"])
    best_ckpt = epoch_ckpt_dir(seed, best_epoch, models_dir)

    # An argmax over epochs is only meaningful if the epochs are actually
    # separated. On QFrBLiMP (n=1761) the binomial standard error alone is
    # ~0.8pp, so a "peak" that beats the runner-up by less than min_margin is
    # a coin flip dressed up as a selection. Report it rather than let the
    # downstream table present it as an empirically located peak.
    margin = (best[metric] - ranked[1][metric]) if len(ranked) > 1 else None
    spread = max(t[metric] for t in trajectory) - min(t[metric] for t in trajectory)
    peak_is_separated = margin is None or margin >= min_margin
    if not peak_is_separated:
        print(f"WARN: seed{seed} epoch {best_epoch} beats epoch {ranked[1]['epoch']} by only"
              f" {margin:.4f} ({metric}), under the {min_margin:.4f} margin;"
              " treat this as a plateau, not a peak.", file=sys.stderr)

    # Symlink models/seed{S}/best -> chck_NM_epoch{E*}.
    seed_dir = models_dir / f"seed{seed}"
    link = seed_dir / "best"
    target_rel = best_ckpt.name
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(target_rel, link)

    # Copy peak JSON to the canonical filename consumed by the aggregator.
    canonical = eval_dir / f"seed{seed}_qfrblimp.json"
    shutil.copy(best_file, canonical)

    summary = {
        "seed": seed,
        "metric": metric,
        "best_epoch": best_epoch,
        "best_score": best[metric],
        "runner_up_margin": margin,
        "epoch_spread": spread,
        "min_margin": min_margin,
        "peak_is_separated": peak_is_separated,
        "best_checkpoint": str(best_ckpt),
        "best_link": str(link),
        "canonical_eval_json": str(canonical),
        "trajectory": [{"epoch": t["epoch"], metric: t[metric]} for t in trajectory],
    }
    summary_path = eval_dir / f"seed{seed}_best_epoch.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--metric", default="overall",
                   help="QFrBLiMP field to argmax over (default: overall)")
    p.add_argument("--min_margin", type=float, default=DEFAULT_MIN_MARGIN,
                   help="Gap to the runner-up epoch below which the pick is reported as a "
                        f"plateau rather than a peak (default: {DEFAULT_MIN_MARGIN})")
    p.add_argument("--eval_dir", default=None)
    p.add_argument("--models_dir", default=None)
    args = p.parse_args()

    root = _project_root()
    eval_dir = Path(args.eval_dir) if args.eval_dir else root / "eval_results"
    models_dir = Path(args.models_dir) if args.models_dir else root / "models"

    summary = pick(args.seed, args.metric, eval_dir, models_dir, args.min_margin)
    print(f"seed={args.seed} best epoch={summary['best_epoch']} "
          f"({args.metric}={summary['best_score']:.4f})")
    if not summary["peak_is_separated"]:
        print(f"  plateau: only {summary['runner_up_margin']:.4f} over the runner-up"
              f" (spread across epochs: {summary['epoch_spread']:.4f})")
    print(f"  symlink : {summary['best_link']} -> {Path(summary['best_checkpoint']).name}")
    print(f"  canonical eval: {summary['canonical_eval_json']}")
    print("  trajectory:")
    for t in summary["trajectory"]:
        marker = "*" if t["epoch"] == summary["best_epoch"] else " "
        print(f"    {marker} epoch {t['epoch']}: {t[args.metric]:.4f}")


if __name__ == "__main__":
    main()
