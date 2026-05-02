"""
Extract high-frequency lemmas from Haitian Creole as a vocabulary oracle.

Strategy:
- Haitian Creole derives ~90% of its lexicon from French
- But only the highest-frequency, most composable French words survived pidginization
- These are load-bearing words: function words, core verbs, basic nouns
- Use their frequency distribution to OVERSAMPLE sentences in our French corpus
  that are rich in these lemmas -- while keeping French morphology intact

Sources for Haitian Creole text:
- CHILDES Haitian Creole corpora
- Kreyol.com corpus
- Universal Dependencies Haitian Creole treebank
- Bible translations (large, available)
- News sources (Le Nouvelliste)

Output: top_lemmas.json -- the oracle vocabulary list
"""

import json
import os
from collections import Counter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "haitian_creole")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known high-frequency Haitian Creole lemmas with their French cognates
# This is a seed list based on linguistic literature -- will be augmented
# from actual corpus frequency analysis
KNOWN_HC_FRENCH_COGNATES = {
    # Pronouns
    "mwen": "je/moi/mon/ma/mes",
    "ou": "tu/vous/ton/ta/tes",
    "li": "il/elle/son/sa/ses",
    "nou": "nous/on/notre/nos",
    "yo": "ils/elles/leur/leurs",
    "sa": "ça/cela/ce/cette/ces",
    # Core verbs
    "ale": "aller",
    "vini": "venir",
    "fe": "faire",
    "di": "dire",
    "wè": "voir",
    "konnen": "connaître/savoir",
    # "manje" intentionally appears once below (Basic nouns) with both
    # the verb and noun mappings: "manger" + "nourriture".
    "bay": "donner",
    "pran": "prendre",
    "rete": "rester",
    "vle": "vouloir",
    "kapab": "pouvoir/capable",
    "dwe": "devoir",
    # Tense/Aspect markers — mapped to French LEMMAS not descriptions
    "te": "être/était/été",
    "ap": "être/en",
    "va": "aller/va",
    "pral": "aller",
    # Function words
    "pa": "pas/ne",
    "nan": "dans/en/au",
    "pou": "pour",
    "ak": "avec/et",
    "sou": "sur",
    "la": "là/le/la",
    "a": "à/le/la",
    "an": "en/dans/un/une",
    "ki": "qui/que",
    "kè": "que/quoi",
    "se": "est/être",
    "gen": "avoir",
    # Basic nouns
    "moun": "personne/gens",
    "bagay": "chose",
    "kote": "endroit/où",
    "tan": "temps",
    "lajan": "argent",
    "travay": "travail",
    "kay": "maison",
    "peyi": "pays",
    "tè": "terre",
    "dlo": "eau",
    "manje": "nourriture/manger",
    # Adjectives/Adverbs
    "bon": "bon/bien",
    "gwo": "grand/gros",
    "piti": "petit",
    "anpil": "beaucoup",
    "toujou": "toujours",
    "deja": "déjà",
    "menm": "même",
    "tout": "tout",
    "okenn": "aucun",
    "lòt": "autre",
    # Caillou-validated supplements: high-frequency French lemmas whose
    # Creole reflexes diverged too far for automatic cognate detection,
    # but confirmed as load-bearing by convergence analysis between
    # pidginization (HC oracle) and child-directed register (Caillou).
    "men": "mais",
    "byen": "bien",
    "trè": "très",
    "plis": "plus",
    "tou": "aussi",
    "tankou": "comme",
    "ankò": "encore",
    "wi": "oui",
    "jou": "jour",
    "isit": "ici",
    "de": "deux",
    "ye": "être",
}

def download_ud_haitian():
    """Download Universal Dependencies Haitian Creole treebank."""
    import subprocess
    ud_url = "https://github.com/UniversalDependencies/UD_Haitian_Creole-Autogramm/archive/refs/heads/main.zip"
    out_zip = os.path.join(OUTPUT_DIR, "ud_haitian.zip")

    print("Downloading UD Haitian Creole treebank...")
    try:
        subprocess.run(["curl", "-L", "-o", out_zip, ud_url], check=True, capture_output=True)
        subprocess.run(["unzip", "-o", out_zip, "-d", OUTPUT_DIR], check=True, capture_output=True)
        print("  Downloaded UD Haitian Creole treebank")
        return True
    except Exception as e:
        print(f"  Could not download UD treebank: {e}")
        return False

def extract_lemmas_from_conllu(conllu_path):
    """Extract lemma frequencies from CoNLL-U format."""
    lemma_counts = Counter()

    with open(conllu_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 3:
                    lemma = parts[2].lower()
                    if lemma not in ("_", ""):
                        lemma_counts[lemma] += 1

    return lemma_counts

def build_oracle_vocabulary():
    """Build the Haitian Creole oracle vocabulary."""

    all_counts = Counter()

    # Load from all UD treebanks
    ud_dirs = [
        os.path.join(OUTPUT_DIR, "UD_Haitian_Creole-Autogramm-main"),
        os.path.join(OUTPUT_DIR, "UD_Haitian_Creole-Adolphe"),
    ]
    for ud_dir in ud_dirs:
        if os.path.exists(ud_dir):
            for fname in os.listdir(ud_dir):
                if fname.endswith(".conllu"):
                    path = os.path.join(ud_dir, fname)
                    counts = extract_lemmas_from_conllu(path)
                    all_counts.update(counts)
                    print(f"  Extracted {len(counts)} lemma types from {fname}")

    # Add seed list regardless
    for hc_lemma in KNOWN_HC_FRENCH_COGNATES:
        # Boost seed items if not already in corpus counts
        if hc_lemma not in all_counts:
            all_counts[hc_lemma] = 100  # synthetic frequency

    # Get top 300
    top_lemmas = all_counts.most_common(300)

    # Map to French equivalents where known
    oracle = []
    for lemma, count in top_lemmas:
        french = KNOWN_HC_FRENCH_COGNATES.get(lemma, lemma)  # fallback to HC lemma itself
        oracle.append({
            "hc_lemma": lemma,
            "french_equivalent": french,
            "frequency": count
        })

    return oracle

def save_oracle(oracle):
    """Save oracle vocabulary as JSON and plain text."""

    # JSON version (full detail)
    json_path = os.path.join(OUTPUT_DIR, "oracle_vocabulary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(oracle, f, ensure_ascii=False, indent=2)
    print(f"\nOracle saved to {json_path}")

    # Plain text version (French equivalents only, for corpus filtering)
    txt_path = os.path.join(OUTPUT_DIR, "oracle_french_lemmas.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        seen = set()
        for item in oracle:
            # Handle multiple French equivalents (e.g., "je/moi")
            for french in item["french_equivalent"].split("/"):
                french = french.strip()
                if french and french not in seen:
                    f.write(french + "\n")
                    seen.add(french)
    print(f"French lemma list saved to {txt_path}")

    # Summary
    print("\n=== Oracle Vocabulary Summary ===")
    print(f"Total HC lemmas: {len(oracle)}")
    print("\nTop 20 by frequency:")
    for item in oracle[:20]:
        print(f"  {item['hc_lemma']:15} -> {item['french_equivalent']:20} (freq: {item['frequency']})")

    return json_path, txt_path

if __name__ == "__main__":
    print("=== Building Haitian Creole Vocabulary Oracle ===\n")

    # Try to get UD treebank data
    download_ud_haitian()

    # Build oracle
    oracle = build_oracle_vocabulary()

    # Save
    json_path, txt_path = save_oracle(oracle)

    print("\n=== Next Step ===")
    print("Run scripts/build_french_corpus.py to use this oracle")
    print("to oversample high-signal sentences in the French training corpus.")
