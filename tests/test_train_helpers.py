"""Tests for compute_next_ckpt_idx and the BabyLM checkpoint schedule.

This module deliberately has no torch dependency so the regression tests
run in any Python environment, not only on the GPU server. The schedule
helper was extracted from train.py specifically to enable this.

The headline regression: a resume from a checkpoint past the last scheduled
threshold previously left next_ckpt_idx at the 0-default, which caused the
training loop to re-save every checkpoint in cascade at the first batch
after resume.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import _checkpoint_schedule as cs  # noqa: E402


def test_next_ckpt_idx_fresh_run():
    """Fresh run (0 words processed) starts at the first threshold."""
    schedule = [1_000_000, 2_000_000, 5_000_000]
    assert cs.compute_next_ckpt_idx(0, schedule) == 0


def test_next_ckpt_idx_partial_progress():
    """Mid-run we point to the next unmet threshold."""
    schedule = [1_000_000, 2_000_000, 5_000_000, 10_000_000]
    assert cs.compute_next_ckpt_idx(2_500_000, schedule) == 2


def test_next_ckpt_idx_exactly_at_threshold():
    """words_processed == threshold means the threshold has already been
    crossed (the save is triggered by >=), so we point at the next one."""
    schedule = [1_000_000, 2_000_000, 5_000_000]
    assert cs.compute_next_ckpt_idx(2_000_000, schedule) == 2


def test_next_ckpt_idx_past_last_threshold_regression():
    """Resuming from a checkpoint past the last threshold previously left
    next_ckpt_idx = 0 (default), causing the inner save loop to re-save
    every checkpoint in cascade at the first batch. The fix returns
    len(schedule), making the loop a no-op."""
    schedule = [1_000_000, 2_000_000, 5_000_000]
    assert cs.compute_next_ckpt_idx(10_000_000, schedule) == 3


def test_next_ckpt_idx_empty_schedule():
    assert cs.compute_next_ckpt_idx(0, []) == 0
    assert cs.compute_next_ckpt_idx(1_000_000, []) == 0


def test_next_ckpt_idx_real_schedule():
    """Smoke check against the actual training schedule shipped with train.py."""
    real = cs.CHECKPOINT_WORDS
    assert cs.compute_next_ckpt_idx(0, real) == 0
    # 92M (the leaderboard checkpoint size) sits between 90M and 100M
    idx = cs.compute_next_ckpt_idx(92_000_000, real)
    assert real[idx] == 100_000_000
    # Past 100M the schedule now continues on the 2026 hundred-million grid:
    # exposure just over 100M targets 200M next, and just over 200M targets 300M.
    assert real[cs.compute_next_ckpt_idx(100_000_000, real)] == 200_000_000
    assert real[cs.compute_next_ckpt_idx(200_000_000, real)] == 300_000_000
    # 1000M is the last milestone; at or beyond it, no more saves.
    assert cs.compute_next_ckpt_idx(1_000_000_000, real) == len(real)


def test_real_schedule_is_strictly_monotonic():
    """The compute_next_ckpt_idx contract assumes the schedule is sorted
    ascending; a regression that breaks ordering would silently make
    intermediate checkpoints unreachable."""
    real = cs.CHECKPOINT_WORDS
    assert real == sorted(real)
    assert len(real) == len(set(real))
