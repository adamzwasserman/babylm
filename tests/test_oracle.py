"""Tests for the Haitian-Creole-derived French oracle and the
oracle-density weighting used by build_french_corpus.

The oracle is a *lexical filter*: it tags French sentences by their density
of HC-cognate French lemmas, which feeds into oversampling. Two properties
matter for reproducibility:

1. The seed dictionary KNOWN_HC_FRENCH_COGNATES is well-formed (no duplicate
   keys, no empty values, all values resolve to French lemmas), since
   silent dict overwrites would change the oracle composition without
   changing the file diff.
2. oracle_density() returns a value in [0, 1] for the kinds of sentences
   the corpus actually contains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


# ---- Oracle seed dictionary ------------------------------------------------

def test_oracle_seed_no_duplicate_keys():
    """KNOWN_HC_FRENCH_COGNATES is a dict literal; duplicate keys silently
    overwrite earlier entries. We re-parse the source to detect duplicates
    that the dict itself would hide."""
    import ast

    src_path = ROOT / "scripts" / "build_creole_oracle.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "KNOWN_HC_FRENCH_COGNATES":
                    target = node.value
                    break
    assert target is not None, "KNOWN_HC_FRENCH_COGNATES not found in source"
    assert isinstance(target, ast.Dict)
    keys = [k.value for k in target.keys if isinstance(k, ast.Constant)]
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"duplicate HC keys in seed dict: {duplicates}"


def test_oracle_seed_values_nonempty():
    import build_creole_oracle as bc

    for hc, fr in bc.KNOWN_HC_FRENCH_COGNATES.items():
        assert hc and isinstance(hc, str), f"empty HC key near {hc!r}"
        assert fr and isinstance(fr, str), f"empty French value for {hc!r}"


def test_oracle_seed_french_values_lowercase_and_no_whitespace():
    """Each slash-separated French equivalent should be a single token,
    lowercase, with no leading or trailing whitespace. The downstream
    oracle_french_lemmas.txt iterates over .split('/') and writes raw
    items, so any spacing here would corrupt the lemma list."""
    import build_creole_oracle as bc

    for hc, fr in bc.KNOWN_HC_FRENCH_COGNATES.items():
        for piece in fr.split("/"):
            assert piece == piece.strip(), (
                f"value for {hc!r} has whitespace around {piece!r}"
            )
            assert piece, f"empty piece in value for {hc!r}: {fr!r}"


def test_oracle_save_writes_unique_lemmas(tmp_path, monkeypatch):
    """save_oracle must deduplicate French lemmas across HC entries
    (e.g., 'la' appears in multiple slash-separated values). The oracle
    text file should contain unique lines."""
    import build_creole_oracle as bc

    oracle = [
        {"hc_lemma": "li", "french_equivalent": "il/elle/le/la", "frequency": 100},
        {"hc_lemma": "la", "french_equivalent": "la/le", "frequency": 80},
        {"hc_lemma": "a", "french_equivalent": "à/le/la", "frequency": 70},
    ]
    monkeypatch.setattr(bc, "OUTPUT_DIR", str(tmp_path))
    bc.save_oracle(oracle)
    txt = (tmp_path / "oracle_french_lemmas.txt").read_text(encoding="utf-8")
    lemmas = [line for line in txt.splitlines() if line.strip()]
    assert len(lemmas) == len(set(lemmas)), f"duplicates in {lemmas}"


# ---- oracle_density ---------------------------------------------------------

@pytest.fixture
def density():
    import build_french_corpus as bfc
    return bfc.oracle_density


def test_density_zero_when_oracle_empty(density):
    assert density("le chat dort sur le tapis", set()) == 0.0


def test_density_zero_when_sentence_empty(density):
    assert density("", {"chat", "le"}) == 0.0
    assert density("   ", {"chat", "le"}) == 0.0


def test_density_full_match(density):
    # Every word is in the oracle.
    assert density("le chat dort", {"le", "chat", "dort"}) == pytest.approx(1.0)


def test_density_partial_match(density):
    sent = "le chat dort sur le tapis"
    oracle = {"le", "chat"}
    # words: ["le", "chat", "dort", "sur", "le", "tapis"]; hits: le, chat, le -> 3/6
    assert density(sent, oracle) == pytest.approx(0.5)


def test_density_strips_punctuation_before_lookup(density):
    # "chat," should match "chat" because re.sub strips non-letter chars.
    sent = "le chat, dort"
    oracle = {"chat"}
    # words: ["le", "chat,", "dort"]; "chat," -> "chat" matches -> 1/3
    assert density(sent, oracle) == pytest.approx(1 / 3)


def test_density_handles_french_accents(density):
    sent = "été après hiver"
    oracle = {"été", "hiver"}
    assert density(sent, oracle) == pytest.approx(2 / 3)


def test_density_preserves_oe_ligature(density):
    """Regression test: the punctuation-stripping regex used to drop the
    œ ligature, mapping "cœur" to "cur" and missing any oracle entry for
    the lemma. Same for æ and ÿ."""
    sent = "cœur sœur œuvre"
    oracle = {"cœur", "sœur", "œuvre"}
    assert density(sent, oracle) == pytest.approx(1.0)


def test_density_preserves_ae_ligature(density):
    sent = "tænia"
    oracle = {"tænia"}
    assert density(sent, oracle) == pytest.approx(1.0)


def test_density_lowercases_input(density):
    sent = "Le Chat Dort"
    oracle = {"le", "chat"}
    # All lowercased before split; "le" + "chat" hit, "dort" miss -> 2/3
    assert density(sent, oracle) == pytest.approx(2 / 3)


def test_density_in_unit_interval(density):
    sent = "le chat le chat le chat le chat"
    oracle = {"le", "chat"}
    assert 0.0 <= density(sent, oracle) <= 1.0


def test_density_no_match(density):
    assert density("Hello world", {"chat", "dort"}) == 0.0


def test_density_oversample_weight_in_range(density):
    """The downstream weighting formula
        weight = 1.0 + (OVERSAMPLE_WEIGHT - 1.0) * density
    must yield a value in [1.0, OVERSAMPLE_WEIGHT] for every sentence.
    """
    import build_french_corpus as bfc

    sentences = [
        "le chat dort",
        "the dog runs",
        "le le le le",
        "",
        "alpha beta gamma delta",
    ]
    oracle = {"le", "chat"}
    for s in sentences:
        d = density(s, oracle)
        weight = 1.0 + (bfc.OVERSAMPLE_WEIGHT - 1.0) * d
        assert 1.0 <= weight <= bfc.OVERSAMPLE_WEIGHT, (s, d, weight)
