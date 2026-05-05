"""Tests for the BabyLM word-budget counter.

The BabyLM Strict track caps total exposure at 100M whitespace-split words.
A miscount in either direction is a real risk: undercounting a corrupted file
could let a submission silently exceed the cap; overcounting from a stray
non-text file would shrink the available budget on a budget-tight submission.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

# Make scripts/ importable as a top-level module path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import count_words  # noqa: E402


@pytest.fixture
def text_file(tmp_path: Path):
    def _make(content: str, name: str = "sample.txt") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


def test_count_simple_ascii(text_file):
    p = text_file("the cat sat on the mat\n")
    assert count_words.count_words_file(str(p)) == 6


def test_count_uses_whitespace_split_not_subwords(text_file):
    # BabyLM rule: whitespace split, not subword/punctuation tokenization.
    # "don't" is one whitespace-token even though a tokenizer would split it.
    p = text_file("don't worry\n")
    assert count_words.count_words_file(str(p)) == 2


def test_count_handles_french_accents(text_file):
    p = text_file("le château était grandement décoré pendant l'été\n")
    # Whitespace split treats l'été as one token: ["le", "château",
    # "était", "grandement", "décoré", "pendant", "l'été"] -> 7 words.
    assert count_words.count_words_file(str(p)) == 7


def test_count_collapses_repeated_whitespace(text_file):
    # str.split() with no separator collapses runs of any whitespace,
    # which is the BabyLM behaviour we want.
    p = text_file("alpha   beta\t\tgamma\n\n\ndelta\n")
    assert count_words.count_words_file(str(p)) == 4


def test_count_empty_file(text_file):
    p = text_file("")
    assert count_words.count_words_file(str(p)) == 0


def test_count_blank_lines_only(text_file):
    p = text_file("\n\n   \n\t\n")
    assert count_words.count_words_file(str(p)) == 0


def test_count_multiline(text_file):
    content = textwrap.dedent(
        """
        first line of text
        second line is here
        third short
        """
    ).strip("\n")
    p = text_file(content)
    # 4 + 4 + 2 = 10
    assert count_words.count_words_file(str(p)) == 10


def test_count_missing_file_raises(tmp_path):
    """Regression test: previously the function silently returned 0 on
    file errors, which would let a missing file pass the budget check.
    The caller now gets an OSError it can detect or surface."""
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(OSError):
        count_words.count_words_file(str(missing))


def test_count_dir_walks_extensions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta\n", encoding="utf-8")
    (tmp_path / "b.json").write_text("gamma delta epsilon\n", encoding="utf-8")
    (tmp_path / "c.bin").write_text("ignored ignored ignored\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("zeta\n", encoding="utf-8")

    total, files = count_words.count_words_dir(str(tmp_path))

    assert total == 2 + 3 + 1
    assert len(files) == 3  # .bin excluded


def test_count_dir_skips_hidden_dirs(tmp_path: Path):
    (tmp_path / "visible.txt").write_text("a b c\n", encoding="utf-8")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "hidden.txt").write_text("should not count\n", encoding="utf-8")

    total, files = count_words.count_words_dir(str(tmp_path))

    assert total == 3
    assert len(files) == 1


def test_count_dir_unreadable_file_skipped_loudly(tmp_path: Path, capsys):
    """A file that cannot be read is reported on stderr and skipped,
    rather than silently contributing 0 to the total.
    """
    good = tmp_path / "good.txt"
    good.write_text("alpha beta\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("hello\n", encoding="utf-8")
    bad.chmod(0o000)
    try:
        total, files = count_words.count_words_dir(str(tmp_path))
        captured = capsys.readouterr()
    finally:
        bad.chmod(0o644)

    # Only the readable file should be in totals; the unreadable one is
    # reported on stderr.
    assert total == 2
    assert len(files) == 1
    if os.geteuid() != 0:  # root can read anything
        assert "SKIP" in captured.err


def test_format_count_thresholds():
    assert count_words.format_count(999) == "999"
    assert count_words.format_count(1_000) == "1.0K"
    assert count_words.format_count(1_500) == "1.5K"
    assert count_words.format_count(1_000_000) == "1.0M"
    assert count_words.format_count(92_500_000) == "92.5M"
