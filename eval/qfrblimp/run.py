"""QFrBLiMP zero-shot evaluation harness.

QFrBLiMP (Beauchemin et al., 2025c) is a Quebec-French minimal-pair
benchmark of 1,761 pairs across 20 phenomena, hand-crafted from the OQLF
normative grammar resource. For each (good, bad) pair, we compute the
sum-of-token log-likelihoods under the model and predict "good" iff
score(good) > score(bad). Item-level accuracy is aggregated by paradigm
and into the four high-level buckets reported in Table 1 of the paper:

    - Syntactic
    - Semantic
    - Morphological
    - Anglicism-related

Usage:
    uv run python eval/qfrblimp/run.py models/seed42/chck_92M --seed 42

Output:
    eval_results/seed{S}_qfrblimp.json with the structure:
    {
        "checkpoint": "...",
        "seed": 42,
        "overall": 0.8597,
        "buckets": {
            "syntactic": 0.866,
            "semantic": 0.847,
            "morphological": 0.828,
            "anglicism_related": 0.764
        },
        "per_paradigm": { "<paradigm>": {"n": ..., "accuracy": ...}, ... }
    }

Paradigm-to-bucket mapping is read from PARADIGM_BUCKETS at module top so
it can be edited without touching the scoring code; the default mapping
matches the bucketing used in the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The graalul/qfrblimp HF dataset card declares qfrblimp.jsonl but does not
# publish it; the actual release lives in the upstream GitHub repo. We default
# to the raw GitHub URL (load_dataset can fetch json over http) and let users
# override with --dataset for a cached local copy.
DEFAULT_DATASET = (
    "https://raw.githubusercontent.com/davebulaval/QFrBLiMP/main/"
    "datastore/QFrBLiMP/release/qfrblimp.jsonl"
)

# QFrBLiMP's `category` field is itself the paper's high-level bucket
# (syntax/semantic/morphology/anglicism; counts: 380/398/716/267, total 1761).
# We expose this map for normalisation only; new categories will fall through
# to the raw value.
PARADIGM_BUCKETS: dict[str, str] = {
    "syntax": "syntactic",
    "semantic": "semantic",
    "morphology": "morphological",
    "anglicism": "anglicism_related",
}

BUCKETS = ("syntactic", "semantic", "morphological", "anglicism_related")


@torch.no_grad()
def sentence_loglikelihood(model, tokenizer, sentence: str, device) -> float:
    """Sum of log p(t_i | t_<i) over the tokens of `sentence` under `model`."""
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    if input_ids.size(1) < 2:
        return 0.0
    out = model(input_ids=input_ids, labels=input_ids)
    # CrossEntropyLoss reports mean loss over (n_tokens - 1); recover the sum.
    n_targets = input_ids.size(1) - 1
    return float(-out.loss.item() * n_targets)


def evaluate(checkpoint: str, dataset_name: str, seed: int | None,
             paradigm_field: str, good_field: str, bad_field: str,
             split: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()

    print(f"Loading {dataset_name} (split={split})...")
    if dataset_name.startswith(("http://", "https://")) or Path(dataset_name).exists():
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)
    n = len(ds)
    print(f"  {n} pairs")

    correct_per_paradigm: dict[str, list[bool]] = defaultdict(list)
    for i, row in enumerate(ds):
        good = row[good_field]
        bad = row[bad_field]
        paradigm = row[paradigm_field]
        s_good = sentence_loglikelihood(model, tokenizer, good, device)
        s_bad = sentence_loglikelihood(model, tokenizer, bad, device)
        correct_per_paradigm[paradigm].append(s_good > s_bad)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")

    per_paradigm: dict[str, dict] = {}
    bucket_correct: dict[str, list[bool]] = defaultdict(list)
    unknown_paradigms: list[str] = []
    all_correct: list[bool] = []
    for paradigm, hits in correct_per_paradigm.items():
        per_paradigm[paradigm] = {
            "n": len(hits),
            "accuracy": sum(hits) / len(hits),
        }
        bucket = PARADIGM_BUCKETS.get(paradigm)
        if bucket is None:
            unknown_paradigms.append(paradigm)
        else:
            bucket_correct[bucket].extend(hits)
        all_correct.extend(hits)

    buckets = {b: (sum(bc) / len(bc) if bc else None) for b, bc in bucket_correct.items()}
    for b in BUCKETS:
        buckets.setdefault(b, None)

    return {
        "checkpoint": checkpoint,
        "seed": seed,
        "dataset": dataset_name,
        "split": split,
        "overall": sum(all_correct) / len(all_correct) if all_correct else 0.0,
        "n": len(all_correct),
        "buckets": buckets,
        "per_paradigm": per_paradigm,
        "unknown_paradigms": unknown_paradigms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a HuggingFace checkpoint dir")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the output filename")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"HF dataset name (default: {DEFAULT_DATASET})")
    p.add_argument("--split", default="test")
    p.add_argument("--paradigm_field", default="category")
    p.add_argument("--good_field", default="sentence_a")
    p.add_argument("--bad_field", default="sentence_b")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(args.output_dir) if args.output_dir else project_root / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate(
        args.checkpoint, args.dataset, args.seed,
        args.paradigm_field, args.good_field, args.bad_field, args.split,
    )

    seed_tag = f"seed{args.seed}" if args.seed is not None else Path(args.checkpoint).name
    out_path = out_dir / f"{seed_tag}_qfrblimp.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nOverall: {result['overall']:.4f} (n={result['n']})")
    for b in BUCKETS:
        v = result["buckets"].get(b)
        if v is not None:
            print(f"  {b:>22}: {v:.4f}")
    if result["unknown_paradigms"]:
        print(f"\nWARNING: {len(result['unknown_paradigms'])} paradigms not in"
              " PARADIGM_BUCKETS, contributed to overall but no bucket:"
              f" {result['unknown_paradigms']}", file=sys.stderr)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
