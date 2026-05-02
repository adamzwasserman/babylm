"""
Assemble the final French training corpus for BabyLM 2026 Strict track.

Strategy:
1. Load French CDS from CHILDES (gold standard, low volume)
2. Load supplementary French sources to reach ~90-100M words
3. Load Haitian Creole oracle lemmas
4. Score each sentence by oracle lemma density
5. Oversample high-density sentences (keep morphology 100% French)
6. Shuffle and output final corpus
7. Verify word count is within BabyLM budget

CRITICAL: All training text must be standard French with full morphology.
The HC oracle is a FILTER only -- it influences sampling weights, never
contributes text to the training corpus.
"""

import argparse
import json
import os
import random
import re

from tqdm import tqdm

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
CHILDES_DIR = os.path.join(CORPUS_DIR, "childes_french")
HC_DIR = os.path.join(CORPUS_DIR, "haitian_creole")
BILINGUAL_DIR = os.path.join(CORPUS_DIR, "bilingual")
FINAL_DIR = os.path.join(CORPUS_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)

# BabyLM word budget
STRICT_BUDGET = 100_000_000
STRICT_SMALL_BUDGET = 10_000_000
TARGET_WORDS = 95_000_000  # leave 5M buffer

# Harness C: bilingual lemma repetitions
BILINGUAL_REPETITIONS = 500  # ~1.3M words at current lemma set size

# Oversampling weight for high-oracle-density sentences
OVERSAMPLE_WEIGHT = 3.0  # oracle-rich sentences appear 3x as often

def load_oracle_lemmas():
    """Load the Haitian Creole oracle lemma list."""
    oracle_path = os.path.join(HC_DIR, "oracle_french_lemmas.txt")
    if not os.path.exists(oracle_path):
        print(f"WARNING: Oracle not found at {oracle_path}")
        print("Run scripts/build_creole_oracle.py first.")
        return set()

    with open(oracle_path, encoding="utf-8") as f:
        lemmas = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded {len(lemmas)} oracle lemmas")
    return lemmas

def oracle_density(sentence, oracle_lemmas):
    """
    Compute what fraction of words in the sentence are oracle lemmas.
    Higher = more load-bearing vocabulary = higher sampling weight.
    """
    if not oracle_lemmas:
        return 0.0
    words = sentence.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if re.sub(r"[^a-zàâäéèêëîïôùûüç]", "", w) in oracle_lemmas)
    return hits / len(words)

def load_childes_sentences():
    """Load all sentences from CHILDES French corpus."""
    sentences = []
    if not os.path.exists(CHILDES_DIR):
        print(f"WARNING: CHILDES directory not found: {CHILDES_DIR}")
        print("Run scripts/download_childes_french.py first.")
        return sentences

    for fname in os.listdir(CHILDES_DIR):
        if fname.endswith(".txt"):
            path = os.path.join(CHILDES_DIR, fname)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and len(line.split()) >= 3:  # min 3 words
                        sentences.append(line)

    print(f"Loaded {len(sentences):,} sentences from CHILDES French")
    return sentences

def load_babylm_fra_cds():
    """
    Load child-directed and child-available content from BabyLM-community/babylm-fra.
    Excludes OpenSubtitles (padding) — keeps only gold CDS-adjacent sources:
      - Caillou (YouTube children's TV subtitles)
      - CHILDES French
      - Ririro/GlotStoryBook (children's stories)
      - Claire-Dialogue-French (spontaneous conversations)
      - Vikidia/Wikimini (children's encyclopedias)
      - fr.wikisource.org (children's literature)
    """
    cache_dir = os.path.join(CORPUS_DIR, "babylm_fra")
    cache_path = os.path.join(cache_dir, "cds_content.txt")

    if os.path.exists(cache_path):
        print(f"Loading cached babylm-fra CDS from {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return [line.strip() for line in f if len(line.strip().split()) >= 3]

    from datasets import load_dataset
    print("Downloading babylm-fra (non-subtitle CDS content)...")
    ds = load_dataset("BabyLM-community/babylm-fra", split="train")

    # Exclude OpenSubtitles — it's bulk padding, not CDS-quality
    exclude_sources = {"opensubtitles"}
    sentences = []
    source_counts = {}

    for row in ds:
        src = str(row.get("data-source", "unknown")).strip("'\"").lower()
        if src in exclude_sources:
            continue
        text = str(row.get("text", ""))
        for line in text.split("\n"):
            line = line.strip()
            if len(line.split()) >= 3:
                sentences.append(line)
                source_counts[src] = source_counts.get(src, 0) + 1

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    word_count = sum(len(s.split()) for s in sentences)
    print(f"  babylm-fra CDS: {len(sentences):,} sentences, {word_count/1e6:.1f}M words")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count:,} sentences")

    return sentences


def load_supplementary_french():
    """
    Load supplementary French text sources.
    Priority: babylm-fra CDS content first, then Wikipedia to fill budget.
    """
    sentences = []
    supp_dir = os.path.join(CORPUS_DIR, "supplementary")

    # 1. Load babylm-fra CDS content (Caillou, CHILDES, stories, etc.)
    cds_sentences = load_babylm_fra_cds()
    sentences.extend(cds_sentences)
    cds_words = sum(len(s.split()) for s in cds_sentences)
    print(f"  babylm-fra CDS total: {cds_words/1e6:.1f}M words")

    # 2. Load cached supplementary files if they exist
    if os.path.exists(supp_dir):
        for fname in sorted(os.listdir(supp_dir)):
            if fname.endswith(".txt"):
                path = os.path.join(supp_dir, fname)
                with open(path, encoding="utf-8") as f:
                    batch = [line.strip() for line in f if len(line.strip().split()) >= 3]
                    sentences.extend(batch)
                    print(f"  Loaded {len(batch):,} sentences from {fname}")

    # 3. If still short, download Wikipedia
    current_words = sum(len(s.split()) for s in sentences)
    if current_words < TARGET_WORDS:
        remaining = TARGET_WORDS - current_words
        print(f"  Have {current_words/1e6:.1f}M words, need {remaining/1e6:.1f}M more from Wikipedia")
        wiki_sentences = download_french_wikipedia(target_words=remaining)
        sentences.extend(wiki_sentences)

    return sentences

def download_french_wikipedia(target_words=95_000_000):
    """Download French Wikipedia from HuggingFace (no auth required)."""
    from datasets import load_dataset

    supp_dir = os.path.join(CORPUS_DIR, "supplementary")
    os.makedirs(supp_dir, exist_ok=True)
    cache_path = os.path.join(supp_dir, "wikipedia_french.txt")

    if os.path.exists(cache_path):
        print(f"Loading cached Wikipedia French from {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return [line.strip() for line in f if len(line.strip().split()) >= 3]

    print("Downloading French Wikipedia from HuggingFace (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)

    sentences = []
    word_count = 0

    for example in tqdm(ds, desc="French Wikipedia"):
        text = example.get("text", "").strip()
        if not text:
            continue
        for line in text.split("\n"):
            line = line.strip()
            words = line.split()
            if len(words) >= 3:
                sentences.append(line)
                word_count += len(words)
        if word_count >= target_words:
            break

    print(f"Downloaded {len(sentences):,} sentences ({word_count/1_000_000:.1f}M words)")

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    print(f"Cached to {cache_path}")

    return sentences

def load_bilingual_lemmas(repetitions=BILINGUAL_REPETITIONS):
    """
    Load bilingual FR->EN lemma pairs for Harness C.
    Repeated `repetitions` times so the model internalizes the mappings.
    Returns list of sentences and their total word count.
    """
    lemma_path = os.path.join(BILINGUAL_DIR, "bilingual_lemmas.txt")
    if not os.path.exists(lemma_path):
        print(f"WARNING: Bilingual lemmas not found at {lemma_path}")
        print("Run scripts/build_bilingual_lemmas.py first.")
        return [], 0

    with open(lemma_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Repeat the lemma lines
    repeated = lines * repetitions
    word_count = sum(len(line.split()) for line in repeated)
    print(f"Bilingual lemmas: {len(lines)} entries x {repetitions} = "
          f"{len(repeated):,} lines, {word_count:,} words "
          f"({word_count/STRICT_BUDGET*100:.2f}% of budget)")
    return repeated, word_count


def count_words(sentences):
    return sum(len(s.split()) for s in sentences)

def build_weighted_corpus(all_sentences, oracle_lemmas, target_words):
    """
    Build the final corpus by sampling sentences with oracle-density weighting.
    High oracle-density sentences are oversampled.
    """
    print(f"\nScoring {len(all_sentences):,} sentences by oracle density...")

    scored = []
    for sent in tqdm(all_sentences, desc="Scoring"):
        density = oracle_density(sent, oracle_lemmas)
        weight = 1.0 + (OVERSAMPLE_WEIGHT - 1.0) * density
        scored.append((sent, weight))

    # Separate high and normal density
    high_density = [(s, w) for s, w in scored if w > 1.5]
    normal = [(s, w) for s, w in scored if w <= 1.5]

    print(f"High oracle-density sentences: {len(high_density):,}")
    print(f"Normal sentences: {len(normal):,}")

    # Build corpus by weighted sampling
    final_sentences = []
    current_words = 0

    # First pass: include all sentences with weights
    pool = scored.copy()
    random.shuffle(pool)

    for sent, weight in tqdm(pool, desc="Building corpus"):
        word_count = len(sent.split())
        if current_words + word_count > target_words:
            break

        # Add sentence weight times (oversampling)
        repeats = max(1, round(weight))
        for _ in range(repeats):
            if current_words + word_count <= target_words:
                final_sentences.append(sent)
                current_words += word_count

    # If we're short, add more from pool
    if current_words < target_words * 0.9:
        print(f"Only {current_words/1_000_000:.1f}M words. Need more data.")
        print("Consider downloading more French text sources.")

    random.shuffle(final_sentences)
    return final_sentences, current_words

def save_corpus(sentences, word_count, harness="a"):
    """Save the final corpus in BabyLM-compatible format."""
    suffix = "" if harness == "a" else f"_{harness}"

    # Full corpus
    full_path = os.path.join(FINAL_DIR, f"train_french{suffix}.txt")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    print(f"\nFull corpus saved: {full_path}")

    # Strict-Small subset (10M words)
    small_sentences = []
    small_words = 0
    for sent in sentences:
        w = len(sent.split())
        if small_words + w > STRICT_SMALL_BUDGET:
            break
        small_sentences.append(sent)
        small_words += w

    small_path = os.path.join(FINAL_DIR, f"train_french{suffix}_10M.txt")
    with open(small_path, "w", encoding="utf-8") as f:
        f.write("\n".join(small_sentences))
    print(f"Strict-Small corpus saved: {small_path}")

    # Metadata
    harness_descriptions = {
        "a": "Harness A: 100% French, evaluated on French-translated benchmarks",
        "b": "Harness B: 100% French, evaluated on English benchmarks (cross-lingual transfer)",
        "c": "Harness C: French + bilingual lemma bridge, evaluated on English benchmarks",
    }
    meta = {
        "harness": harness,
        "harness_description": harness_descriptions.get(harness, ""),
        "total_words": word_count,
        "total_sentences": len(sentences),
        "strict_budget": STRICT_BUDGET,
        "strict_small_budget": STRICT_SMALL_BUDGET,
        "budget_used_pct": round(word_count / STRICT_BUDGET * 100, 1),
        "oracle_oversample_weight": OVERSAMPLE_WEIGHT,
        "language": "French" + (" + bilingual lemmas" if harness == "c" else ""),
        "oracle_source": "Haitian Creole frequency oracle (vocabulary filter only)",
    }
    meta_path = os.path.join(FINAL_DIR, f"corpus_metadata{suffix}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    return full_path, small_path

def parse_args():
    p = argparse.ArgumentParser(description="Build French BabyLM training corpus")
    p.add_argument("--harness", choices=["a", "b", "c", "all"], default="all",
                    help="Which harness variant to build (a=French only, "
                         "b=same as a, c=French+bilingual bridge, all=build a and c)")
    p.add_argument("--bilingual-reps", type=int, default=BILINGUAL_REPETITIONS,
                    help=f"Bilingual lemma repetitions for harness C (default: {BILINGUAL_REPETITIONS})")
    return p.parse_args()


def build_base_corpus(oracle_lemmas):
    """Build the base French corpus (shared by all harnesses)."""
    print("\n--- Loading French text sources ---")
    childes_sents = load_childes_sentences()
    supp_sents = load_supplementary_french()

    all_sentences = childes_sents + supp_sents
    total_available = count_words(all_sentences)

    print(f"\nTotal available: {total_available/1_000_000:.1f}M words "
          f"from {len(all_sentences):,} sentences")

    if total_available < 10_000_000:
        print("\nWARNING: Less than 10M words available.")
        print("Download more French text before proceeding.")
        return None, None, 0

    return all_sentences, oracle_lemmas, total_available


def main():
    args = parse_args()
    harnesses = ["a", "c"] if args.harness == "all" else [args.harness]

    print("=== Building French BabyLM Training Corpus ===")
    print(f"Harness(es): {', '.join(harnesses)}\n")
    random.seed(42)

    # Load oracle
    oracle_lemmas = load_oracle_lemmas()

    # Build base sentence pool (shared across harnesses)
    all_sentences, oracle_lemmas, total_available = build_base_corpus(oracle_lemmas)
    if all_sentences is None:
        return

    # Load bilingual lemmas if needed
    bilingual_lines, bilingual_words = [], 0
    if "c" in harnesses:
        bilingual_lines, bilingual_words = load_bilingual_lemmas(args.bilingual_reps)

    for harness in harnesses:
        print(f"\n{'='*60}")
        print(f"  HARNESS {harness.upper()}")
        print(f"{'='*60}")

        if harness in ("a", "b"):
            # Pure French corpus
            target = TARGET_WORDS
            print(f"Target: {target/1_000_000:.0f}M words (100% French)")

            random.seed(42)  # Reset seed for reproducibility across harnesses
            final_sentences, final_words = build_weighted_corpus(
                all_sentences, oracle_lemmas, target)

        elif harness == "c":
            # French + bilingual bridge
            french_target = TARGET_WORDS - bilingual_words
            print(f"Target: {french_target/1_000_000:.1f}M French + "
                  f"{bilingual_words/1_000_000:.1f}M bilingual = "
                  f"{TARGET_WORDS/1_000_000:.0f}M total")

            random.seed(42)
            final_sentences, final_words = build_weighted_corpus(
                all_sentences, oracle_lemmas, french_target)

            # Interleave bilingual lemmas throughout the corpus
            # Insert them at regular intervals so the model sees them throughout training
            interval = max(1, len(final_sentences) // len(bilingual_lines)) if bilingual_lines else 1
            merged = []
            bi_idx = 0
            for i, sent in enumerate(final_sentences):
                merged.append(sent)
                if bi_idx < len(bilingual_lines) and i % interval == 0:
                    merged.append(bilingual_lines[bi_idx])
                    bi_idx += 1
            # Append any remaining bilingual lines
            while bi_idx < len(bilingual_lines):
                merged.append(bilingual_lines[bi_idx])
                bi_idx += 1

            final_sentences = merged
            final_words += bilingual_words

        print(f"\nFinal corpus: {final_words/1_000_000:.1f}M words "
              f"({len(final_sentences):,} sentences)")
        print(f"Budget usage: {final_words/STRICT_BUDGET*100:.1f}% of Strict budget")

        # Save
        print("\n--- Saving corpus ---")
        save_corpus(final_sentences, final_words, harness=harness)

    print("\n=== Done ===")
    for h in harnesses:
        suffix = "" if h == "a" else f"_{h}"
        print(f"  Harness {h.upper()}: corpus/final/train_french{suffix}.txt")
    print("Verify with: uv run python scripts/count_words.py corpus/final/train_french.txt")


if __name__ == "__main__":
    main()
