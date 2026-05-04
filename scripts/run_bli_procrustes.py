"""Bilingual Lexicon Induction via closed-form orthogonal Procrustes.

This is the §4.3 / Table 2 evaluation: take the input-token embeddings of a
French model and learn a 768x768 orthogonal map W minimising
||E_FR @ W - E_EN||_F over a French/English seed dictionary, then evaluate
word-translation precision@k on a held-out split.

Targets reported in the paper (Table 2):
    1. GPT-2 (competent EN)            -> openai-community/gpt2 (HF Hub)
    2. Matched-arch failed-EN model    -> caller supplies via --target_en
    3. Random orthogonal               -> baseline, sampled per seed
    4. Chance (1 / n_test)             -> analytic baseline

Each cell of the table is a number (Fit, p@1, p@5, p@10) we want averaged
across the 5 ablation seeds, so this script writes one JSON per (model, seed,
target). The aggregator stacks them later.

Procrustes: given (X, Y) with X = E_FR[seed_pairs], Y = E_EN[seed_pairs],
the closed-form solution is W = U V^T where U S V^T = SVD(X^T Y). This is
linear, deterministic given the data, fast (seconds), and avoids the
gradient-descent drift of training-based BLI.

Usage:
    uv run python scripts/run_bli_procrustes.py models/seed42/chck_92M --seed 42
    uv run python scripts/run_bli_procrustes.py models/seed42/chck_92M --seed 42 \\
        --target_en path/to/matched_failed_en --target_label matched_failed_en

Output:
    eval_results/seed{S}_bli_<target_label>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

DEFAULT_SEED_DICT = "BaselineQuebec/fr_en_seed_dict"
DEFAULT_HELD_OUT = 48  # paper: 194 train / 48 test on the parsed 242-pair list
GPT2_EN = "openai-community/gpt2"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_input_embeddings(model_path: str) -> tuple[np.ndarray, dict[str, int]]:
    """Return (E, vocab) where E[i] = input embedding of token id i and
    vocab maps surface tokens to ids."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    emb = model.get_input_embeddings().weight.detach().cpu().numpy()
    vocab = tok.get_vocab()
    return emb, vocab


def lookup(word: str, tokenizer_vocab: dict[str, int], embeddings: np.ndarray) -> np.ndarray | None:
    """Return the embedding of `word` if it is a single-token entry of the
    vocab, else None. Both raw and BPE-prefixed (G + word for ByteLevel
    tokenizers) variants are tried."""
    candidates = [word, "Ġ" + word]
    for c in candidates:
        if c in tokenizer_vocab:
            return embeddings[tokenizer_vocab[c]]
    return None


def build_pair_matrices(seed_pairs: list[tuple[str, str]],
                        emb_fr: np.ndarray, vocab_fr: dict[str, int],
                        emb_en: np.ndarray, vocab_en: dict[str, int]
                        ) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """For each (fr, en) pair where both words are single-token in their
    respective vocabs, stack their embeddings. Returns (X, Y, kept_pairs)."""
    X_rows, Y_rows, kept = [], [], []
    for fr, en in seed_pairs:
        x = lookup(fr, vocab_fr, emb_fr)
        y = lookup(en, vocab_en, emb_en)
        if x is None or y is None:
            continue
        X_rows.append(x)
        Y_rows.append(y)
        kept.append((fr, en))
    if not X_rows:
        raise RuntimeError("No seed pair could be mapped to single-token "
                           "embeddings on both sides.")
    return np.stack(X_rows), np.stack(Y_rows), kept


def fit_procrustes(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form orthogonal Procrustes. Returns (W, fit) where
    fit = 1 - ||XW - Y||_F^2 / ||Y||_F^2 (1.0 perfect, 0.0 no alignment)."""
    U, _S, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    W = U @ Vt
    err = np.linalg.norm(X @ W - Y) ** 2
    denom = np.linalg.norm(Y) ** 2
    fit = float(1.0 - err / denom) if denom > 0 else 0.0
    return W, fit


def precision_at_k(X_test: np.ndarray, W: np.ndarray, Y_all: np.ndarray,
                    test_indices: list[int], ks: tuple[int, ...] = (1, 5, 10)
                    ) -> dict[int, float]:
    """For each held-out source row, project via W and look up the k nearest
    neighbours in Y_all (cosine similarity); count a hit if the gold target
    index is in the top-k. Returns {k: precision}.
    """
    proj = X_test @ W  # (n_test, d)
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
    Yn = Y_all / (np.linalg.norm(Y_all, axis=1, keepdims=True) + 1e-12)
    sim = proj @ Yn.T  # (n_test, n_all)
    out: dict[int, float] = {}
    for k in ks:
        topk = np.argsort(-sim, axis=1)[:, :k]
        hits = sum(int(gold_idx in topk[i]) for i, gold_idx in enumerate(test_indices))
        out[k] = hits / len(test_indices) if test_indices else 0.0
    return out


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, d))
    Q, _ = np.linalg.qr(A)
    return Q


def split_seed_dict(pairs: list[tuple[str, str]], n_test: int, seed: int
                    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in test_idx]


def evaluate_target(emb_fr: np.ndarray, vocab_fr: dict[str, int],
                    emb_en: np.ndarray, vocab_en: dict[str, int],
                    seed_pairs: list[tuple[str, str]],
                    n_test: int, seed: int) -> dict:
    train_pairs, test_pairs = split_seed_dict(seed_pairs, n_test, seed)

    X_train, Y_train, _ = build_pair_matrices(train_pairs, emb_fr, vocab_fr, emb_en, vocab_en)
    W, fit = fit_procrustes(X_train, Y_train)

    X_test, Y_test, kept_test = build_pair_matrices(test_pairs, emb_fr, vocab_fr, emb_en, vocab_en)
    test_indices = list(range(len(kept_test)))
    pak = precision_at_k(X_test, W, Y_test, test_indices)

    rng = np.random.default_rng(seed + 1)
    W_rand = random_orthogonal(W.shape[0], rng)
    pak_rand = precision_at_k(X_test, W_rand, Y_test, test_indices)
    err_rand = np.linalg.norm(X_test @ W_rand - Y_test) ** 2
    denom_rand = np.linalg.norm(Y_test) ** 2
    fit_rand = float(1.0 - err_rand / denom_rand) if denom_rand > 0 else 0.0

    n_test_eff = len(kept_test)
    chance = {k: min(k / n_test_eff, 1.0) for k in (1, 5, 10)}

    return {
        "fit": fit,
        "p@1": pak[1],
        "p@5": pak[5],
        "p@10": pak[10],
        "n_train_kept": len(X_train),
        "n_test_kept": n_test_eff,
        "random_orthogonal": {
            "fit": fit_rand,
            "p@1": pak_rand[1],
            "p@5": pak_rand[5],
            "p@10": pak_rand[10],
        },
        "chance": chance,
    }


def load_seed_dict(name_or_path: str, fr_field: str, en_field: str
                    ) -> list[tuple[str, str]]:
    """Load a (fr, en) seed dictionary. Tries an HF dataset name first, then
    a local CSV/TSV path with two columns."""
    p = Path(name_or_path)
    if p.exists():
        delim = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
        out: list[tuple[str, str]] = []
        with open(p, encoding="utf-8") as f:
            header = f.readline().strip().split(delim)
            if fr_field in header and en_field in header:
                fi, ei = header.index(fr_field), header.index(en_field)
            else:
                fi, ei = 0, 1
            for line in f:
                parts = line.strip().split(delim)
                if len(parts) >= max(fi, ei) + 1:
                    out.append((parts[fi].strip(), parts[ei].strip()))
        return out
    ds = load_dataset(name_or_path, split="train")
    return [(row[fr_field], row[en_field]) for row in ds]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="French model checkpoint")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed label for the output filename and split RNG")
    p.add_argument("--target_en", default=GPT2_EN,
                   help="HF id or path of the English target model")
    p.add_argument("--target_label", default="gpt2_en",
                   help="Label used in the output filename")
    p.add_argument("--seed_dict", default=DEFAULT_SEED_DICT,
                   help="HF dataset name or local CSV/TSV with FR/EN columns")
    p.add_argument("--fr_field", default="fr")
    p.add_argument("--en_field", default="en")
    p.add_argument("--n_test", type=int, default=DEFAULT_HELD_OUT)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    out_dir = Path(args.output_dir) if args.output_dir else _project_root() / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FR embeddings from {args.checkpoint}...")
    emb_fr, vocab_fr = get_input_embeddings(args.checkpoint)
    print(f"  vocab={len(vocab_fr)} dim={emb_fr.shape[1]}")

    print(f"Loading EN embeddings from {args.target_en}...")
    emb_en, vocab_en = get_input_embeddings(args.target_en)
    print(f"  vocab={len(vocab_en)} dim={emb_en.shape[1]}")

    if emb_fr.shape[1] != emb_en.shape[1]:
        sys.exit(f"Embedding dim mismatch: FR={emb_fr.shape[1]} EN={emb_en.shape[1]}. "
                 "Procrustes assumes a square map; pick a target with matching d.")

    print(f"Loading seed dictionary from {args.seed_dict}...")
    seed_pairs = load_seed_dict(args.seed_dict, args.fr_field, args.en_field)
    print(f"  {len(seed_pairs)} pairs in source")

    seed_for_split = args.seed if args.seed is not None else 0
    result = evaluate_target(
        emb_fr, vocab_fr, emb_en, vocab_en,
        seed_pairs, args.n_test, seed_for_split,
    )
    result["checkpoint"] = args.checkpoint
    result["seed"] = args.seed
    result["target_en"] = args.target_en
    result["target_label"] = args.target_label

    seed_tag = f"seed{args.seed}" if args.seed is not None else Path(args.checkpoint).name
    out_path = out_dir / f"{seed_tag}_bli_{args.target_label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nBLI Procrustes:")
    print(f"  fit     : {result['fit']:.4f}")
    print(f"  p@1     : {result['p@1']:.4f}")
    print(f"  p@5     : {result['p@5']:.4f}")
    print(f"  p@10    : {result['p@10']:.4f}")
    r = result["random_orthogonal"]
    print(f"  random  : fit={r['fit']:.4f} p@1={r['p@1']:.4f} p@5={r['p@5']:.4f} p@10={r['p@10']:.4f}")
    print(f"  chance  : p@1={result['chance'][1]:.4f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()


# Silence unused-import warning when torch is dragged in only to ensure the
# transformers backend registers properly on some environments.
_ = torch
