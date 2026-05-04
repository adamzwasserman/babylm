"""Tests for parse_args and init_wandb in scripts/train.py.

These pull in train.py which imports torch, transformers, and tokenizers,
so the whole module is skipped in environments without the ML stack
installed (e.g. linting boxes). The actual training environment that runs
multi-seed jobs has these installed and will execute the tests.

The properties under test are critical for multi-seed correctness:
- output_dir defaults to a seed-namespaced path so 3 parallel processes
  do not overwrite each other's checkpoints.
- wandb_run_name defaults to seed{S} so wandb runs are distinguishable.
- --wandb_mode disabled short-circuits the wandb import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import train  # noqa: E402


def test_parse_args_default_output_dir_uses_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--seed", "7"])
    args = train.parse_args()
    assert args.seed == 7
    assert args.output_dir.endswith("/seed7") or args.output_dir.endswith("\\seed7")
    assert args.wandb_run_name == "seed7"


def test_parse_args_default_output_dir_distinct_per_seed(monkeypatch):
    """Two seeds must resolve to distinct default output_dirs; collision
    here would silently overwrite checkpoints across the parallel runs."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--seed", "1"])
    a1 = train.parse_args()
    monkeypatch.setattr(sys, "argv", ["train.py", "--seed", "2"])
    a2 = train.parse_args()
    assert a1.output_dir != a2.output_dir
    assert a1.wandb_run_name != a2.wandb_run_name


def test_parse_args_explicit_output_dir_respected(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--seed", "3", "--output_dir", "/tmp/custom-out"]
    )
    args = train.parse_args()
    assert args.output_dir == "/tmp/custom-out"


def test_parse_args_explicit_wandb_run_name_respected(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--seed", "3", "--wandb_run_name", "ablation-A"]
    )
    args = train.parse_args()
    assert args.wandb_run_name == "ablation-A"


def test_parse_args_wandb_mode_choices(monkeypatch):
    """argparse choices guard prevents typos like --wandb_mode=of."""
    monkeypatch.setattr(sys, "argv", ["train.py", "--wandb_mode", "bogus"])
    with pytest.raises(SystemExit):
        train.parse_args()


def test_init_wandb_disabled_returns_none_without_importing_wandb(monkeypatch):
    """--wandb_mode disabled must short-circuit before importing wandb so
    that running without wandb installed is supported."""
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--seed", "1", "--wandb_mode", "disabled"]
    )
    args = train.parse_args()
    # If init_wandb tried to import wandb, this would raise ImportError on
    # a wandb-less box; the function must return None first.
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert train.init_wandb(args, total_steps=10, n_params=1000) is None
