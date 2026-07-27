"""Checkpoint schedule and pure helpers, separated from train.py so the
regression test for compute_next_ckpt_idx can run without torch installed.

The schedule itself is a property of the BabyLM evaluation pipeline and is
shared between train.py and any future tooling that wants to reason about
which checkpoint sizes are expected on disk.
"""

from __future__ import annotations

# Per the BabyLM 2026 eval pipeline, save at every million from 1M..10M, every
# ten million up to 100M, then every hundred million up to 1000M. Each entry is
# a target word count (exposure). The 2025 spec stopped at 100M; 2026 extends
# the expected fast-eval curve to 1000M (matching OTHER_FAST_REVISIONS in the
# eval pipeline's collate_preds.py). A run of E epochs over W words saves every
# milestone below E*W and skips the rest, so a 6-epoch run over 92.5M words
# (~555M exposure) fills the curve through chck_500M.
CHECKPOINT_WORDS: list[int] = (
    [i * 1_000_000 for i in range(1, 11)]
    + [i * 10_000_000 for i in range(2, 11)]
    + [i * 100_000_000 for i in range(2, 11)]
)


def compute_next_ckpt_idx(words_processed: int, schedule: list[int]) -> int:
    """Return the index of the first checkpoint threshold strictly greater
    than words_processed. Returns len(schedule) when words_processed already
    meets or exceeds every threshold, so the caller's save loop becomes a
    no-op (otherwise a resumed run past the last threshold would re-save
    every checkpoint in cascade at the first batch).
    """
    for i, w in enumerate(schedule):
        if w > words_processed:
            return i
    return len(schedule)
