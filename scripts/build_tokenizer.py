"""Pre-train the shared BPE tokenizer once.

Run this before launching multiple seeds in parallel to avoid race conditions
on tokenizer.json and redundant tokenizer training.

Usage:
    uv run python scripts/build_tokenizer.py
    uv run python scripts/build_tokenizer.py --vocab_size 50000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import CORPUS_DIR, TOKENIZER_DIR, train_tokenizer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=os.path.join(CORPUS_DIR, "train_french.txt"))
    p.add_argument("--vocab_size", type=int, default=50000)
    p.add_argument("--save_dir", default=TOKENIZER_DIR)
    args = p.parse_args()

    train_tokenizer(args.corpus, args.vocab_size, args.save_dir)


if __name__ == "__main__":
    main()
