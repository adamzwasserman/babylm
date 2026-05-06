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
import subprocess
import sys
from pathlib import Path

PIPELINE_REPO_URL = "https://github.com/babylm/evaluation-pipeline-2025.git"
PIPELINE_DIR_NAME = "evaluation-pipeline-2025"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pipeline_dir() -> Path:
    return _project_root() / "eval" / PIPELINE_DIR_NAME


PATCH_MARKER = "# >>> babylm patched: AutoProcessor -> AutoTokenizer fallback"
PAD_TOKEN_MARKER = "# >>> babylm patched: pad_token fallback"


def _patch_file(target: Path, marker: str, needle: str, replacement: str,
                pipeline_dir: Path) -> None:
    src = target.read_text(encoding="utf-8")
    if marker in src:
        return
    if needle not in src:
        print(f"WARN: could not patch {target}; needle not found",
              file=sys.stderr)
        return
    target.write_text(src.replace(needle, replacement), encoding="utf-8")
    print(f"Patched {target.relative_to(pipeline_dir)}.")


def _patch_dataset_autoprocessor(pipeline_dir: Path) -> None:
    """Make the cloned pipeline work for text-only causal LMs.

    Upstream calls AutoProcessor.from_pretrained in two places (sentence
    zero-shot and finetuning trainer). Both fail on tokenizer-only
    checkpoints with 'Unrecognized processing class'. We wrap each call in a
    try/except that falls back to AutoTokenizer, and ensure the tokenizer
    has a pad_token (set to eos_token, fallback unk_token) for fine-tuning.
    Patches are marked with comments so they are idempotent across reruns.
    """
    # 1. sentence_zero_shot/dataset.py
    _patch_file(
        target=pipeline_dir / "evaluation_pipeline" / "sentence_zero_shot" / "dataset.py",
        marker=PATCH_MARKER,
        needle=(
            '        self.processor: ProcessorMixin = AutoProcessor.from_pretrained('
            'args.model_path_or_name, padding_side="right", '
            'revision=args.revision_name, trust_remote_code=True)'
        ),
        replacement=(
            f"        {PATCH_MARKER}\n"
            "        try:\n"
            "            self.processor: ProcessorMixin = AutoProcessor.from_pretrained("
            "args.model_path_or_name, padding_side=\"right\", "
            "revision=args.revision_name, trust_remote_code=True)\n"
            "        except (ValueError, OSError):\n"
            "            from transformers import AutoTokenizer\n"
            "            self.processor = AutoTokenizer.from_pretrained("
            "args.model_path_or_name, padding_side=\"right\", "
            "revision=args.revision_name, trust_remote_code=True)"
        ),
        pipeline_dir=pipeline_dir,
    )

    # 2. finetune/trainer.py: same AutoProcessor issue + ensure pad_token.
    _patch_file(
        target=pipeline_dir / "evaluation_pipeline" / "finetune" / "trainer.py",
        marker=PATCH_MARKER,
        needle=(
            "        self.tokenizer: PreTrainedTokenizerBase = "
            "AutoProcessor.from_pretrained(self.args.model_name_or_path, "
            "trust_remote_code=True, padding_side=self.args.padding_side)"
        ),
        replacement=(
            f"        {PATCH_MARKER}\n"
            "        try:\n"
            "            self.tokenizer: PreTrainedTokenizerBase = "
            "AutoProcessor.from_pretrained(self.args.model_name_or_path, "
            "trust_remote_code=True, padding_side=self.args.padding_side)\n"
            "        except (ValueError, OSError):\n"
            "            from transformers import AutoTokenizer\n"
            "            self.tokenizer = AutoTokenizer.from_pretrained("
            "self.args.model_name_or_path, trust_remote_code=True, "
            "padding_side=self.args.padding_side)\n"
            f"        {PAD_TOKEN_MARKER}\n"
            "        if getattr(self.tokenizer, 'pad_token', None) is None:\n"
            "            self.tokenizer.pad_token = (self.tokenizer.eos_token "
            "or self.tokenizer.unk_token or self.tokenizer.bos_token)\n"
            "            if self.tokenizer.pad_token is None:\n"
            "                self.tokenizer.add_special_tokens({'pad_token': "
            "'[PAD]'})"
        ),
        pipeline_dir=pipeline_dir,
    )


def ensure_pipeline_cloned() -> Path:
    target = _pipeline_dir()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {PIPELINE_REPO_URL} into {target}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", PIPELINE_REPO_URL, str(target)],
            check=True,
        )
    _patch_dataset_autoprocessor(target)
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


def harvest_results(pipeline_dir: Path,
                     pre_snapshot: dict[Path, float]) -> dict:
    """Glob every results.txt and best_temperature_report.txt under
    results/ whose mtime is newer than the snapshot (or that didn't exist
    in the snapshot at all). The pipeline writes paths that depend on the
    model's full name and re-runs overwrite the same files, so existence
    alone isn't enough — we also require mtime > pre-run."""
    results_root = pipeline_dir / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(f"No results dir at {results_root}")
    out: dict[str, dict] = {}
    for pattern in ("results.txt", "best_temperature_report.txt"):
        for path in sorted(results_root.rglob(pattern)):
            mtime = path.stat().st_mtime
            prior = pre_snapshot.get(path)
            if prior is not None and mtime <= prior:
                continue
            rel = path.parent.relative_to(results_root).as_posix()
            out.setdefault(rel, {})
            out[rel].update(_parse_kv_file(path))
    if not out:
        raise FileNotFoundError(
            f"No new or refreshed results.txt / best_temperature_report.txt "
            f"under {results_root} after running the pipeline."
        )
    return out


def _snapshot_results(pipeline_dir: Path) -> dict[Path, float]:
    results_root = pipeline_dir / "results"
    if not results_root.is_dir():
        return {}
    found: dict[Path, float] = {}
    for pattern in ("results.txt", "best_temperature_report.txt"):
        for path in results_root.rglob(pattern):
            found[path] = path.stat().st_mtime
    return found


# (task, data_path_relative_to_full_eval) for the upstream zero-shot suite.
# We invoke each task separately so that a missing data directory (e.g. EWoK,
# which the OSF dump omits and the pipeline ships a separate downloader for)
# results in a skip rather than the whole suite aborting.
ZERO_SHOT_TASKS: list[tuple[str, str]] = [
    ("blimp", "blimp_filtered"),
    ("blimp", "supplement_filtered"),
    ("ewok", "ewok_filtered"),
    ("entity_tracking", "entity_tracking"),
    ("wug_adj", "wug_adj_nominalization"),
    ("wug_past", "wug_past_tense"),
    ("comps", "comps"),
]


def _run_zero_shot(pipeline_dir: Path, abs_ckpt: str, eval_dir: Path) -> None:
    for task, subdir in ZERO_SHOT_TASKS:
        data_path = eval_dir / subdir
        if not data_path.is_dir():
            print(f"  skip zero-shot {task} ({subdir} missing)")
            continue
        _run(
            ["python", "-m", "evaluation_pipeline.sentence_zero_shot.run",
             "--model_path_or_name", abs_ckpt, "--backend", "causal",
             "--task", task, "--data_path", str(data_path),
             "--save_predictions"],
            cwd=pipeline_dir,
        )
    reading_csv = eval_dir / "reading" / "reading_data.csv"
    if reading_csv.is_file():
        _run(
            ["python", "-m", "evaluation_pipeline.reading.run",
             "--model_path_or_name", abs_ckpt, "--backend", "causal",
             "--data_path", str(reading_csv)],
            cwd=pipeline_dir,
        )
    else:
        print("  skip reading (reading_data.csv missing)")


def evaluate_checkpoint(checkpoint: str, seed: int | None) -> dict:
    pipeline_dir = ensure_pipeline_cloned()
    ensure_eval_data(pipeline_dir)

    eval_dir = pipeline_dir / "evaluation_data" / "full_eval"
    if not eval_dir.is_dir():
        sys.exit(f"Missing {eval_dir}; the pipeline shell scripts assume "
                 "evaluation_data/full_eval/ exists.")

    abs_ckpt = str(Path(checkpoint).resolve())
    seed_arg = str(seed if seed is not None else 42)

    # Snapshot existing results so we know which output files this run
    # produces (the pipeline keys outputs by the model name, which can be
    # a basename or full path depending on the entry point).
    pre_snapshot = _snapshot_results(pipeline_dir)

    _run_zero_shot(pipeline_dir, abs_ckpt, eval_dir)
    _run(
        # Args (per upstream eval_finetuning.sh):
        # MODEL_PATH LR BSZ BIG_BSZ MAX_EPOCHS WSC_EPOCHS SEED
        ["bash", "eval_finetuning.sh",
         abs_ckpt, "3e-5", "32", "16", "10", "30", seed_arg],
        cwd=pipeline_dir,
    )

    tasks = harvest_results(pipeline_dir, pre_snapshot)

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
