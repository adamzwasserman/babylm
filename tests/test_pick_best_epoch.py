"""Tests for scripts/_pick_best_epoch.py.

This helper decides which checkpoint every downstream evaluation runs on, so a
wrong or unstable pick silently propagates into Table 1 and Figure 1. It had no
test coverage. What matters here: the argmax and its tie-break, the plateau
guard, deterministic checkpoint resolution, and the canonical-copy contract
that aggregate_paper_tables.py depends on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "scripts" / "_pick_best_epoch.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pick_best_epoch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_pick_best_epoch"] = module
    spec.loader.exec_module(module)
    return module


peb = _load()


@pytest.fixture
def workspace(tmp_path):
    """eval_results/ + models/ laid out the way the orchestrator leaves them."""
    eval_dir = tmp_path / "eval_results"
    models_dir = tmp_path / "models"
    eval_dir.mkdir()
    (models_dir / "seed42").mkdir(parents=True)
    return tmp_path, eval_dir, models_dir


def _write_epochs(eval_dir: Path, models_dir: Path, scores: dict[int, float], seed: int = 42):
    for epoch, score in scores.items():
        payload = {"overall": score, "n": 1761, "buckets": {"syntactic": score}}
        (eval_dir / f"seed{seed}_qfrblimp_epoch{epoch}.json").write_text(
            json.dumps(payload), encoding="utf-8")
        (models_dir / f"seed{seed}" / f"chck_92M_epoch{epoch}").mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def test_per_epoch_files_finds_and_orders_epochs(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80, 2: 0.82, 10: 0.83})
    found = peb.per_epoch_files(42, eval_dir)
    assert [e for e, _ in found] == [1, 10, 2]  # glob sort is lexicographic
    assert {e for e, _ in found} == {1, 2, 10}


def test_per_epoch_files_ignores_other_seeds(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80})
    (models_dir / "seed43").mkdir()
    _write_epochs(eval_dir, models_dir, {1: 0.99, 2: 0.99}, seed=43)
    assert [e for e, _ in peb.per_epoch_files(42, eval_dir)] == [1]


def test_missing_results_fail_loudly(workspace):
    _, eval_dir, models_dir = workspace
    with pytest.raises(SystemExit, match="No per-epoch QFrBLiMP results"):
        peb.pick(42, "overall", eval_dir, models_dir)


def test_missing_metric_field_fails_loudly(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80})
    with pytest.raises(SystemExit, match="has no 'nonexistent' field"):
        peb.pick(42, "nonexistent", eval_dir, models_dir)


# --------------------------------------------------------------------------
# the pick itself
# --------------------------------------------------------------------------

def test_picks_the_highest_scoring_epoch(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.8350, 2: 0.8400, 3: 0.8597, 4: 0.8500})
    summary = peb.pick(42, "overall", eval_dir, models_dir)
    assert summary["best_epoch"] == 3
    assert summary["best_score"] == 0.8597


def test_epoch_10_beats_epoch_2_despite_lexicographic_filenames(workspace):
    # seed42_qfrblimp_epoch10.json sorts before ..._epoch2.json; the pick must
    # follow the score, not the filename order.
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {2: 0.80, 10: 0.90})
    assert peb.pick(42, "overall", eval_dir, models_dir)["best_epoch"] == 10


def test_ties_resolve_to_the_earliest_epoch(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.86, 3: 0.86, 5: 0.86})
    summary = peb.pick(42, "overall", eval_dir, models_dir)
    assert summary["best_epoch"] == 1
    assert summary["runner_up_margin"] == 0.0
    assert summary["peak_is_separated"] is False


# --------------------------------------------------------------------------
# plateau guard
# --------------------------------------------------------------------------

def test_a_040pp_band_across_epochs_is_reported_as_a_plateau(workspace, capsys):
    # The paper describes exactly this: accuracy oscillating within ~0.40pp
    # across the five epochs. That is under the binomial noise floor, so the
    # argmax must not be presented as an empirically located peak.
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir,
                  {1: 0.8580, 2: 0.8597, 3: 0.8600, 4: 0.8592, 5: 0.8571})
    summary = peb.pick(42, "overall", eval_dir, models_dir)
    assert summary["best_epoch"] == 3
    assert summary["peak_is_separated"] is False
    assert summary["epoch_spread"] == pytest.approx(0.0029, abs=1e-6)
    assert "plateau, not a peak" in capsys.readouterr().err


def test_a_well_separated_peak_raises_no_warning(workspace, capsys):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.70, 2: 0.75, 3: 0.86})
    summary = peb.pick(42, "overall", eval_dir, models_dir)
    assert summary["peak_is_separated"] is True
    assert summary["runner_up_margin"] == pytest.approx(0.11)
    assert "plateau" not in capsys.readouterr().err


def test_min_margin_is_configurable(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.850, 2: 0.856})
    assert peb.pick(42, "overall", eval_dir, models_dir, min_margin=0.001)[
        "peak_is_separated"] is True
    assert peb.pick(42, "overall", eval_dir, models_dir, min_margin=0.02)[
        "peak_is_separated"] is False


def test_single_epoch_has_no_runner_up_and_is_not_flagged(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.86})
    summary = peb.pick(42, "overall", eval_dir, models_dir)
    assert summary["runner_up_margin"] is None
    assert summary["peak_is_separated"] is True
    assert summary["epoch_spread"] == 0.0


# --------------------------------------------------------------------------
# side effects the orchestrator relies on
# --------------------------------------------------------------------------

def test_canonical_json_is_a_copy_of_the_winning_epoch(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80, 3: 0.86})
    peb.pick(42, "overall", eval_dir, models_dir)
    canonical = json.loads((eval_dir / "seed42_qfrblimp.json").read_text(encoding="utf-8"))
    assert canonical["overall"] == 0.86


def test_best_symlink_points_at_the_winning_checkpoint(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80, 3: 0.86})
    peb.pick(42, "overall", eval_dir, models_dir)
    link = models_dir / "seed42" / "best"
    assert link.is_symlink()
    assert link.readlink().name == "chck_92M_epoch3"
    assert link.resolve().is_dir()


def test_rerunning_replaces_a_stale_symlink(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.90, 3: 0.80})
    peb.pick(42, "overall", eval_dir, models_dir)
    assert (models_dir / "seed42" / "best").readlink().name == "chck_92M_epoch1"

    _write_epochs(eval_dir, models_dir, {1: 0.80, 3: 0.90})
    peb.pick(42, "overall", eval_dir, models_dir)
    assert (models_dir / "seed42" / "best").readlink().name == "chck_92M_epoch3"


def test_missing_checkpoint_dir_is_an_error_not_a_silent_skip(workspace):
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.80, 3: 0.86})
    # The winning epoch's checkpoint disappeared (interrupted sync).
    (models_dir / "seed42" / "chck_92M_epoch3").rmdir()
    with pytest.raises(FileNotFoundError, match="chck_.*M_epoch3"):
        peb.pick(42, "overall", eval_dir, models_dir)


def test_checkpoint_resolution_is_deterministic_across_duplicates(workspace, capsys, monkeypatch):
    # Two checkpoint dirs for one epoch means an inconsistent on-disk state.
    # Whatever we pick, two machines must pick the same one -- but Path.glob
    # yields in filesystem order, which differs between machines. Forcing a
    # reversed order here is what makes this test fail if the sort is dropped;
    # relying on the real directory order would pass either way.
    _, eval_dir, models_dir = workspace
    _write_epochs(eval_dir, models_dir, {1: 0.86})
    for name in ("chck_88M_epoch1", "chck_105M_epoch1"):
        (models_dir / "seed42" / name).mkdir()

    real_glob = Path.glob

    def reversed_glob(self, pattern, *args, **kwargs):
        return iter(sorted(real_glob(self, pattern, *args, **kwargs), reverse=True))

    monkeypatch.setattr(Path, "glob", reversed_glob)
    assert peb.epoch_ckpt_dir(42, 1, models_dir).name == "chck_105M_epoch1"
    assert "multiple candidates" in capsys.readouterr().err
