"""Tests for the QFrBLiMP scoring harness (eval/qfrblimp/run.py).

The module under test imports torch/transformers/datasets at module level, so
these tests skip cleanly when the heavy stack is absent. What they cover is the
part that decides the headline number and that no test guarded before:

  - pair orientation against the `label` field (the bug that collapsed the
    reported score to chance),
  - aggregation, including the released rows whose two sentences are identical,
  - the log-likelihood sum recovery and its unscoreable-sentence policy,
  - output naming, which the orchestrator depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "eval" / "qfrblimp" / "run.py"

pytest.importorskip("torch", reason="qfrblimp harness imports torch at module level")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


def _load_run_module():
    spec = importlib.util.spec_from_file_location("qfrblimp_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qfrblimp_run"] = module
    spec.loader.exec_module(module)
    return module


run = _load_run_module()


# --------------------------------------------------------------------------
# orient_pair: label semantics
# --------------------------------------------------------------------------

ROW = {"sentence_a": "je ne peux pas", "sentence_b": "je peux pas", "label": 0.0}


def test_label_zero_keeps_sentence_a_as_grammatical():
    good, bad = run.orient_pair(ROW, "sentence_a", "sentence_b", "label")
    assert good == "je ne peux pas"
    assert bad == "je peux pas"


def test_label_one_swaps_the_pair():
    row = dict(ROW, label=1.0)
    good, bad = run.orient_pair(row, "sentence_a", "sentence_b", "label")
    assert good == "je peux pas"
    assert bad == "je ne peux pas"


@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0, "0", "1"])
def test_integral_labels_of_any_type_are_accepted(label):
    good, _ = run.orient_pair(dict(ROW, label=label), "sentence_a", "sentence_b", "label")
    assert good == ("je ne peux pas" if float(label) == 0.0 else "je peux pas")


@pytest.mark.parametrize("label", [0.4, 0.5, 2, -1, "oui", None, float("nan")])
def test_non_binary_labels_raise_instead_of_truncating(label):
    # int(0.4) == 0 would silently orient the pair as if sentence_a were
    # grammatical. A label outside {0, 1} means the schema changed; fail loudly.
    with pytest.raises(ValueError, match="must be 0 or 1"):
        run.orient_pair(dict(ROW, label=label), "sentence_a", "sentence_b", "label")


def test_boolean_labels_are_rejected():
    # bool is an int subclass; accepting True would hide a schema mismatch.
    with pytest.raises(ValueError, match="must be 0 or 1"):
        run.orient_pair(dict(ROW, label=True), "sentence_a", "sentence_b", "label")


def test_error_message_carries_the_row_index():
    with pytest.raises(ValueError, match="row 17"):
        run.orient_pair(dict(ROW, label=7), "sentence_a", "sentence_b", "label", row_index=17)


def test_ignoring_the_label_field_collapses_competence_to_chance():
    """Reproduces the original bug on a synthetic corpus.

    A model with 100% real competence, scored by a harness that always treats
    sentence_a as grammatical, lands near chance at the released 845/916 label
    split. This is the arithmetic that made the reproduction read ~48.5%.
    """
    # The grammatical side follows `label`, as in the released file.
    rows = [
        {"sentence_a": f"bon {i}", "sentence_b": f"mauvais {i}", "label": 0.0}
        if i < 845 else
        {"sentence_a": f"mauvais {i}", "sentence_b": f"bon {i}", "label": 1.0}
        for i in range(1761)
    ]

    def competent_model(good: str, bad: str) -> bool:
        return good.startswith("bon")

    oriented = [competent_model(*run.orient_pair(r, "sentence_a", "sentence_b", "label"))
                for r in rows]
    naive = [competent_model(r["sentence_a"], r["sentence_b"]) for r in rows]

    assert sum(oriented) / len(oriented) == 1.0
    assert 0.47 < sum(naive) / len(naive) < 0.50


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def _rec(paradigm="syntax", correct=True, degenerate=False, phenomenon=1):
    return {"paradigm": paradigm, "phenomenon": phenomenon,
            "correct": correct, "degenerate": degenerate}


def test_overall_and_buckets_use_the_paradigm_mapping():
    result = run.aggregate([
        _rec("syntax", True), _rec("syntax", False),
        _rec("anglicism", True), _rec("anglicism", True),
    ])
    assert result["overall"] == 0.75
    assert result["buckets"]["syntactic"] == 0.5
    assert result["buckets"]["anglicism_related"] == 1.0
    assert result["buckets"]["semantic"] is None
    assert result["n"] == 4


def test_unmapped_paradigm_is_reported_and_still_counts_in_overall():
    result = run.aggregate([_rec("syntax", True), _rec("neologisme", False)])
    assert result["unknown_paradigms"] == ["neologisme"]
    assert result["overall"] == 0.5
    assert result["buckets"]["syntactic"] == 1.0


def test_degenerate_pairs_are_counted_in_overall_but_broken_out():
    # The released file has 18 rows whose two sentences are byte-identical; no
    # model can score them, so they must not silently vanish nor silently sink
    # the headline number.
    result = run.aggregate([
        _rec("syntax", True), _rec("syntax", True), _rec("syntax", True),
        _rec("syntax", False, degenerate=True),
    ])
    assert result["n_degenerate"] == 1
    assert result["overall"] == 0.75
    assert result["overall_excluding_degenerate"] == 1.0


def test_unscoreable_items_are_excluded_from_the_denominator():
    result = run.aggregate([
        _rec("syntax", True), _rec("syntax", False), _rec("syntax", None),
    ])
    assert result["n_unscoreable"] == 1
    assert result["n"] == 2
    assert result["overall"] == 0.5


def test_per_phenomenon_splits_finer_than_the_four_buckets():
    result = run.aggregate([
        _rec("morphology", True, phenomenon=3), _rec("morphology", False, phenomenon=3),
        _rec("morphology", True, phenomenon=9),
    ])
    assert result["per_phenomenon"]["3"] == {"n": 2, "accuracy": 0.5}
    assert result["per_phenomenon"]["9"] == {"n": 1, "accuracy": 1.0}
    assert result["buckets"]["morphological"] == pytest.approx(2 / 3)


def test_phenomenon_absent_yields_no_breakdown():
    result = run.aggregate([_rec("syntax", True, phenomenon=None)])
    assert result["per_phenomenon"] == {}
    assert result["overall"] == 1.0


def test_empty_input_does_not_divide_by_zero():
    result = run.aggregate([])
    assert result["overall"] == 0.0
    assert result["n"] == 0
    assert all(v is None for v in result["buckets"].values())


# --------------------------------------------------------------------------
# sentence_loglikelihood
# --------------------------------------------------------------------------

class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self, n_tokens: int):
        self.n_tokens = n_tokens

    def __call__(self, sentence, return_tensors=None, add_special_tokens=None):
        import torch
        assert add_special_tokens is False, "special tokens would skew the pair comparison"
        return _FakeEncoding(input_ids=torch.zeros((1, self.n_tokens), dtype=torch.long))


class _FakeModel:
    """Returns a fixed mean cross-entropy, like a HF causal LM would."""

    def __init__(self, mean_loss: float):
        import torch
        self.loss = torch.tensor(mean_loss)

    def __call__(self, input_ids=None, labels=None):
        import types
        return types.SimpleNamespace(loss=self.loss)


def test_loglikelihood_recovers_the_sum_from_the_mean_loss():
    # HF reports mean CE over (n_tokens - 1) targets; the harness compares sums,
    # so the multiplication back must use n-1, not n.
    score = run.sentence_loglikelihood(_FakeModel(2.0), _FakeTokenizer(5), "phrase", "cpu")
    assert score == pytest.approx(-8.0)


def test_single_token_sentence_is_unscoreable_not_zero():
    # A 0.0 sentinel outranks every real (negative) score, handing the pair to
    # whichever side was too short to score.
    assert run.sentence_loglikelihood(_FakeModel(2.0), _FakeTokenizer(1), "a", "cpu") is None


def test_two_token_sentence_is_scoreable():
    score = run.sentence_loglikelihood(_FakeModel(1.5), _FakeTokenizer(2), "ab", "cpu")
    assert score == pytest.approx(-1.5)


# --------------------------------------------------------------------------
# output_filename
# --------------------------------------------------------------------------

def test_default_name_matches_what_the_orchestrator_renames():
    # scripts/run_paper_part1.sh mv's exactly this path after each epoch.
    assert run.output_filename("models/seed42/chck_92M_epoch3", 42) == "seed42_qfrblimp"


def test_tag_keeps_per_epoch_runs_apart():
    assert run.output_filename("models/seed42/chck_92M_epoch3", 42, "epoch3") == \
        "seed42_epoch3_qfrblimp"


def test_without_seed_the_checkpoint_name_is_used():
    assert run.output_filename("models/seed42/chck_92M_epoch3", None) == \
        "chck_92M_epoch3_qfrblimp"


def test_path_separators_in_a_tag_cannot_escape_the_output_dir():
    stem = run.output_filename("ckpt", 42, "../../etc/passwd")
    assert "/" not in stem
    assert stem == "seed42_etc_passwd_qfrblimp"
