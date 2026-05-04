"""Tests for scripts/aggregate_seeds.py.

Aggregation across seeds is the bridge between training output and the paper
table. A bug here silently corrupts the headline numbers, so we test:

- _flatten correctly handles flat dicts, nested dicts, and excludes booleans
  (since bool is a subclass of int in Python, naive isinstance(int) checks
  would treat True/False as 0/1).
- _drill correctly walks dot-paths and surfaces a clean error on missing keys.
- expand_paths globs correctly and falls through to the literal string when
  no glob match is found, so the caller gets a clear FileNotFoundError later.
- build_summary computes mean and (sample) std with ddof=1 over present
  values; missing-key seeds do not contribute zeros.
- load_seed reads UTF-8 JSON without depending on the platform default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pd = pytest.importorskip("pandas")  # aggregate_seeds depends on pandas

import aggregate_seeds as ag  # noqa: E402

# ---- _flatten --------------------------------------------------------------

def test_flatten_flat_dict():
    assert dict(ag._flatten({"a": 1, "b": 2.5})) == {"a": 1.0, "b": 2.5}


def test_flatten_nested_dict_uses_dot_paths():
    assert dict(ag._flatten({"a": {"b": 1, "c": {"d": 2}}})) == {"a.b": 1.0, "a.c.d": 2.0}


def test_flatten_excludes_booleans():
    """bool is a subclass of int in Python; without an explicit guard,
    True would be aggregated as 1.0 and skew the mean."""
    assert dict(ag._flatten({"flag": True, "count": 3})) == {"count": 3.0}


def test_flatten_excludes_strings_and_lists():
    assert dict(ag._flatten({"name": "blimp", "values": [1, 2], "score": 0.5})) == {"score": 0.5}


def test_flatten_preserves_int_as_float():
    out = dict(ag._flatten({"steps": 1000}))
    assert out == {"steps": 1000.0}
    assert isinstance(out["steps"], float)


# ---- _drill ----------------------------------------------------------------

def test_drill_empty_path_returns_root():
    assert ag._drill({"a": 1}, "") == {"a": 1}


def test_drill_walks_dot_path():
    assert ag._drill({"a": {"b": {"c": 7}}}, "a.b") == {"c": 7}


def test_drill_missing_key_raises():
    with pytest.raises(KeyError):
        ag._drill({"a": 1}, "missing")


# ---- expand_paths ----------------------------------------------------------

def test_expand_paths_literal_passthrough(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text("{}", encoding="utf-8")
    assert ag.expand_paths([str(p)]) == [str(p)]


def test_expand_paths_glob(tmp_path: Path):
    (tmp_path / "seed1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "seed2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("", encoding="utf-8")
    out = ag.expand_paths([str(tmp_path / "seed*.json")])
    assert sorted(Path(p).name for p in out) == ["seed1.json", "seed2.json"]


def test_expand_paths_unmatched_glob_keeps_literal(tmp_path: Path):
    """When a glob matches nothing, the literal pattern is preserved so
    that load_seed downstream produces a clear FileNotFoundError instead
    of silently dropping the missing seed."""
    pattern = str(tmp_path / "nope*.json")
    assert ag.expand_paths([pattern]) == [pattern]


# ---- load_seed (end to end on disk) ----------------------------------------

def test_load_seed_reads_utf8_with_accents(tmp_path: Path):
    """JSON values with French accents must round-trip; aggregate_seeds
    forces UTF-8 reads so platform default encoding (e.g. cp1252 on
    some Windows hosts) cannot corrupt metric names like 'qfrcola_été'."""
    p = tmp_path / "seed1.json"
    p.write_text(
        json.dumps({"qfrcola_été": 0.81, "blimp_average": 0.77}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = ag.load_seed(str(p), "")
    assert out == {"qfrcola_été": 0.81, "blimp_average": 0.77}


def test_load_seed_drills_nested(tmp_path: Path):
    p = tmp_path / "seed1.json"
    p.write_text(
        json.dumps({"results": {"zero_shot": {"blimp": 0.81, "ewok": 0.55}}}),
        encoding="utf-8",
    )
    out = ag.load_seed(str(p), "results.zero_shot")
    assert out == {"blimp": 0.81, "ewok": 0.55}


# ---- build_summary ---------------------------------------------------------

def test_build_summary_mean_and_sample_std():
    rows = [
        {"blimp": 0.80},
        {"blimp": 0.82},
        {"blimp": 0.84},
    ]
    summary = ag.build_summary(rows, ["seed1", "seed2", "seed3"])
    assert summary.loc["blimp", "mean"] == pytest.approx(0.82)
    # Sample std (ddof=1) of [0.80, 0.82, 0.84] = sqrt(((-0.02)^2+0+0.02^2)/2)
    assert summary.loc["blimp", "std"] == pytest.approx(0.02)
    assert int(summary.loc["blimp", "n"]) == 3
    assert summary.loc["blimp", "min"] == pytest.approx(0.80)
    assert summary.loc["blimp", "max"] == pytest.approx(0.84)


def test_build_summary_missing_key_does_not_zero_fill():
    """A seed missing a metric must reduce n for that metric, not be
    silently treated as zero (which would tank the mean)."""
    rows = [
        {"blimp": 0.80, "ewok": 0.50},
        {"blimp": 0.82},  # ewok missing
        {"blimp": 0.84, "ewok": 0.52},
    ]
    summary = ag.build_summary(rows, ["seed1", "seed2", "seed3"])
    assert int(summary.loc["ewok", "n"]) == 2
    assert summary.loc["ewok", "mean"] == pytest.approx(0.51)
    # Sanity: blimp still has all 3
    assert int(summary.loc["blimp", "n"]) == 3


def test_build_summary_indexes_alphabetically():
    """Sorted index gives stable ordering for paper tables across runs."""
    rows = [{"zeta": 1.0, "alpha": 2.0}, {"zeta": 1.5, "alpha": 2.5}]
    summary = ag.build_summary(rows, ["s1", "s2"])
    assert list(summary.index) == ["alpha", "zeta"]
