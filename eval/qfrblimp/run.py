"""QFrBLiMP zero-shot evaluation harness.

QFrBLiMP (Beauchemin et al., 2025c) is a Quebec-French minimal-pair
benchmark of 1,761 pairs, hand-crafted from the OQLF normative grammar
resource. For each (good, bad) pair, we compute the sum-of-token
log-likelihoods under the model and predict "good" iff
score(good) > score(bad).

The released dataset carries two levels of granularity:

    - `category` (4 values: syntax/semantic/morphology/anglicism,
      counts 380/398/716/267) maps to the four high-level buckets of
      Table 1 and is what `--paradigm_field` reads by default;
    - `type` (20 values) is the fine-grained phenomenon. It is reported
      separately under "per_phenomenon" when the column is present.

Orientation: the dataset randomises which side is grammatical and records
it in `label` (0 -> sentence_a, 1 -> sentence_b). See `orient_pair`.

Usage:
    uv run python eval/qfrblimp/run.py models/seed42/chck_92M --seed 42

Output:
    eval_results/seed{S}_qfrblimp.json. That name is fixed per seed, so
    scoring several epochs of one seed back-to-back overwrites it; the
    orchestrator (scripts/run_paper_part1.sh) renames the file between
    epochs, and ad-hoc runs should pass --tag to get a distinct name.
    Overwriting an existing file is reported on stderr, never silent.
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
        "per_paradigm": { "<category>": {"n": ..., "accuracy": ...}, ... },
        "per_phenomenon": { "<type>": {"n": ..., "accuracy": ...}, ... }
    }

`overall` counts every scoreable item, including the 18 released rows whose
two sentences are byte-identical (which no model can get right, capping
accuracy at 1743/1761 = 98.98%). `overall_excluding_degenerate` reports the
same run with those rows removed; both are emitted so neither convention is
hidden. Items the tokenizer cannot score at all (under two tokens) are
excluded from both and counted in `n_unscoreable`.

Paradigm-to-bucket mapping is read from PARADIGM_BUCKETS at module top so
it can be edited without touching the scoring code; the default mapping
matches the bucketing used in the paper.
"""

from __future__ import annotations

import argparse
import json
import re
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
def sentence_loglikelihood(model, tokenizer, sentence: str, device) -> float | None:
    """Sum of log p(t_i | t_<i) over the tokens of `sentence` under `model`.

    Returns None when the sentence tokenizes to fewer than two tokens: an
    autoregressive model scores no target at all in that case. Returning a
    0.0 sentinel instead would silently beat every real (negative) score and
    hand the pair to whichever side happened to be too short.
    """
    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    if input_ids.size(1) < 2:
        return None
    out = model(input_ids=input_ids, labels=input_ids)
    # CrossEntropyLoss reports mean loss over (n_tokens - 1); recover the sum.
    n_targets = input_ids.size(1) - 1
    return float(-out.loss.item() * n_targets)


def orient_pair(row, sentence_a_field: str, sentence_b_field: str,
                label_field: str, row_index: int | None = None) -> tuple[str, str]:
    """Return (grammatical, ungrammatical) for a QFrBLiMP row.

    QFrBLiMP randomises which side is grammatical and records it in `label`
    (0 -> sentence_a is grammatical, 1 -> sentence_b is grammatical). Ignoring
    this field and always treating sentence_a as "good" scores correctly on
    label==0 items and *inverted* on label==1 items, collapsing any true
    competence to chance (~48.5% at the released 845/916 label split).

    The released labels are floats (0.0/1.0), so the value is checked for
    integrality rather than truncated: a non-integral label would otherwise
    round down to 0 and silently mis-orient the pair.
    """
    a = row[sentence_a_field]
    b = row[sentence_b_field]
    raw = row[label_field]
    where = "" if row_index is None else f" at row {row_index}"
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{label_field}{where} must be 0 or 1, got {raw!r}")
    try:
        label = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_field}{where} must be 0 or 1, got {raw!r}") from exc
    if label not in (0.0, 1.0):
        raise ValueError(f"{label_field}{where} must be 0 or 1, got {raw!r}")
    return (a, b) if label == 0.0 else (b, a)


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-item scoring records into the reported metrics.

    Each record is {"paradigm", "phenomenon", "correct", "degenerate"},
    where `correct` is None for an item the tokenizer could not score.
    Kept pure (no model, no dataset) so the reporting logic is testable on
    its own; `evaluate` supplies the records.
    """
    per_paradigm: dict[str, dict] = {}
    per_phenomenon: dict[str, dict] = {}
    bucket_correct: dict[str, list[bool]] = defaultdict(list)
    paradigm_correct: dict[str, list[bool]] = defaultdict(list)
    phenomenon_correct: dict[str, list[bool]] = defaultdict(list)
    unknown_paradigms: list[str] = []
    all_correct: list[bool] = []
    non_degenerate: list[bool] = []
    n_unscoreable = 0
    n_degenerate = 0

    for rec in records:
        if rec["degenerate"]:
            n_degenerate += 1
        if rec["correct"] is None:
            n_unscoreable += 1
            continue
        hit = bool(rec["correct"])
        paradigm_correct[rec["paradigm"]].append(hit)
        if rec.get("phenomenon") is not None:
            phenomenon_correct[str(rec["phenomenon"])].append(hit)
        all_correct.append(hit)
        if not rec["degenerate"]:
            non_degenerate.append(hit)

    for paradigm, hits in paradigm_correct.items():
        per_paradigm[paradigm] = {"n": len(hits), "accuracy": sum(hits) / len(hits)}
        bucket = PARADIGM_BUCKETS.get(paradigm)
        if bucket is None:
            unknown_paradigms.append(paradigm)
        else:
            bucket_correct[bucket].extend(hits)

    for phenomenon, hits in phenomenon_correct.items():
        per_phenomenon[phenomenon] = {"n": len(hits), "accuracy": sum(hits) / len(hits)}

    buckets = {b: (sum(bc) / len(bc) if bc else None) for b, bc in bucket_correct.items()}
    for b in BUCKETS:
        buckets.setdefault(b, None)

    return {
        "overall": sum(all_correct) / len(all_correct) if all_correct else 0.0,
        "overall_excluding_degenerate": (
            sum(non_degenerate) / len(non_degenerate) if non_degenerate else 0.0
        ),
        "n": len(all_correct),
        "n_degenerate": n_degenerate,
        "n_unscoreable": n_unscoreable,
        "buckets": buckets,
        "per_paradigm": per_paradigm,
        "per_phenomenon": per_phenomenon,
        "unknown_paradigms": unknown_paradigms,
    }


def output_filename(checkpoint: str, seed: int | None, tag: str | None = None) -> str:
    """Stem of the result file for one run (no extension).

    Without --tag this is the legacy `seed{S}_qfrblimp` that
    scripts/run_paper_part1.sh and scripts/_pick_best_epoch.py expect, so the
    orchestrator keeps working; --tag inserts a run label for ad-hoc per-epoch
    runs, which would otherwise all land on the same path. The tag reaches a
    filesystem path, so it is reduced to [A-Za-z0-9_-] rather than trusted.
    """
    parts = []
    if seed is not None:
        parts.append(f"seed{seed}")
    if tag:
        parts.append(re.sub(r"[^A-Za-z0-9_-]+", "_",tag).strip("_"))
    elif seed is None:
        parts.append(re.sub(r"[^A-Za-z0-9_-]+", "_",Path(checkpoint).name).strip("_"))
    parts.append("qfrblimp")
    return "_".join(p for p in parts if p)


def evaluate(checkpoint: str, dataset_name: str, seed: int | None,
             paradigm_field: str, good_field: str, bad_field: str,
             split: str, label_field: str = "label",
             phenomenon_field: str = "type") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()

    # load_dataset("json", ...) exposes a single "train" split whatever the
    # caller asked for; record what was actually read, not what was requested.
    is_file_like = dataset_name.startswith(("http://", "https://")) or Path(dataset_name).exists()
    effective_split = "train" if is_file_like else split
    print(f"Loading {dataset_name} (split={effective_split})...")
    if is_file_like:
        if split != "train":
            print(f"  NOTE: --split {split!r} does not apply to a file/URL dataset;"
                  " reading the single 'train' split.", file=sys.stderr)
        ds = load_dataset("json", data_files=dataset_name, split="train")
    else:
        ds = load_dataset(dataset_name, split=split)
    n = len(ds)
    print(f"  {n} pairs")

    has_label = label_field in ds.column_names
    if not has_label:
        print(f"  WARNING: no '{label_field}' column; assuming {good_field} is"
              " always grammatical (valid only for pre-oriented data).",
              file=sys.stderr)
    has_phenomenon = phenomenon_field in ds.column_names

    records: list[dict] = []
    for i, row in enumerate(ds):
        if has_label:
            good, bad = orient_pair(row, good_field, bad_field, label_field, row_index=i)
        else:
            good, bad = row[good_field], row[bad_field]
        s_good = sentence_loglikelihood(model, tokenizer, good, device)
        s_bad = sentence_loglikelihood(model, tokenizer, bad, device)
        correct = None if s_good is None or s_bad is None else s_good > s_bad
        records.append({
            "paradigm": row[paradigm_field],
            "phenomenon": row[phenomenon_field] if has_phenomenon else None,
            "correct": correct,
            "degenerate": good == bad,
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")

    result = aggregate(records)
    result.update({
        "checkpoint": checkpoint,
        "seed": seed,
        "dataset": dataset_name,
        "split": effective_split,
    })
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Path to a HuggingFace checkpoint dir")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the output filename")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"HF dataset name (default: {DEFAULT_DATASET})")
    p.add_argument("--split", default="test")
    p.add_argument("--paradigm_field", default="category")
    p.add_argument("--good_field", default="sentence_a",
                   help="sentence_a field; oriented via --label_field")
    p.add_argument("--bad_field", default="sentence_b",
                   help="sentence_b field; oriented via --label_field")
    p.add_argument("--label_field", default="label",
                   help="0 -> good_field grammatical, 1 -> bad_field grammatical")
    p.add_argument("--phenomenon_field", default="type",
                   help="Fine-grained phenomenon column (20 values); reported if present")
    p.add_argument("--tag", default=None,
                   help="Run tag in the output filename (default: the checkpoint dir name). "
                        "Keeps per-epoch runs of one seed from overwriting each other.")
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
        args.label_field, args.phenomenon_field,
    )

    out_path = out_dir / (output_filename(args.checkpoint, args.seed, args.tag) + ".json")
    if out_path.exists():
        print(f"  NOTE: overwriting {out_path}. Scoring several epochs of one seed?"
              " pass --tag to keep them apart.", file=sys.stderr)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nOverall: {result['overall']:.4f} (n={result['n']})")
    for b in BUCKETS:
        v = result["buckets"].get(b)
        if v is not None:
            print(f"  {b:>22}: {v:.4f}")
    if result["n_degenerate"]:
        print(f"  {result['n_degenerate']} byte-identical pair(s) counted as incorrect;"
              f" excluding them: {result['overall_excluding_degenerate']:.4f}")
    if result["n_unscoreable"]:
        print(f"  WARNING: {result['n_unscoreable']} item(s) too short to score, excluded.",
              file=sys.stderr)
    if result["unknown_paradigms"]:
        print(f"\nWARNING: {len(result['unknown_paradigms'])} paradigms not in"
              " PARADIGM_BUCKETS, contributed to overall but no bucket:"
              f" {result['unknown_paradigms']}", file=sys.stderr)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
