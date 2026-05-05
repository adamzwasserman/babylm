"""
Compare Caillou subtitle vocabulary against the Haitian Creole oracle.

Two independent simplification pressures:
  1. Pidginization (French -> Haitian Creole): evolutionary selection for
     highest-frequency, most composable vocabulary
  2. Child-directed register (Caillou): vocabulary simplified for 2-6 year olds

If these converge on the same core lemmas, it validates the oracle strategy
from a completely independent angle — and it's a paper-worthy finding.

Usage:
  uv run python scripts/analyze_caillou_oracle.py
"""

import json
import os
from collections import Counter

import spacy

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
HC_DIR = os.path.join(CORPUS_DIR, "haitian_creole")
BABYLM_FRA_DIR = os.path.join(CORPUS_DIR, "babylm_fra")
os.makedirs(BABYLM_FRA_DIR, exist_ok=True)


def load_oracle_lemmas():
    """Load the French equivalents from the HC oracle."""
    oracle_path = os.path.join(HC_DIR, "oracle_french_lemmas.txt")
    with open(oracle_path, encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def load_oracle_full():
    """Load full oracle with HC lemma -> French mapping."""
    oracle_path = os.path.join(HC_DIR, "oracle_vocabulary.json")
    with open(oracle_path, encoding="utf-8") as f:
        return json.load(f)


def download_babylm_fra():
    """Download and separate babylm-fra by source."""
    from datasets import load_dataset

    cache_dir = os.path.join(BABYLM_FRA_DIR, "raw")
    if os.path.exists(os.path.join(cache_dir, "done")):
        print("babylm-fra already downloaded")
        return

    os.makedirs(cache_dir, exist_ok=True)
    print("Downloading babylm-fra from HuggingFace...")
    ds = load_dataset("BabyLM-community/babylm-fra", split="train")

    # Check what columns exist
    print(f"Columns: {ds.column_names}")
    print(f"Total rows: {len(ds)}")
    print(f"Sample: {ds[0]}")

    # Save full dataset
    sources = Counter()
    by_source = {}

    for row in ds:
        text = row.get("text", "")
        source = row.get("data-source", row.get("source", "unknown"))
        source = str(source).strip("'\"").lower().replace(" ", "_")

        if source not in by_source:
            by_source[source] = []
        by_source[source].append(text)
        sources[source] += 1

    print(f"\nSources found: {len(sources)}")
    for src, count in sources.most_common():
        total_words = sum(len(t.split()) for t in by_source[src])
        print(f"  {src}: {count:,} docs, {total_words/1e6:.1f}M words")

        # Save each source separately
        out_path = os.path.join(cache_dir, f"{src}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(by_source[src]))

    with open(os.path.join(cache_dir, "done"), "w") as f:
        f.write("ok")

    return by_source


def extract_lemma_frequencies(texts, nlp, label=""):
    """Extract lemma frequencies from text using spaCy."""
    lemma_counts = Counter()
    word_count = 0

    # Process in batches for speed
    all_text = "\n".join(texts)
    # Limit to 2M chars for spaCy (otherwise too slow)
    if len(all_text) > 2_000_000:
        all_text = all_text[:2_000_000]
        print(f"  [{label}] Truncated to 2M chars for lemmatization")

    print(f"  [{label}] Lemmatizing {len(all_text)/1e6:.1f}M chars...")
    for doc in nlp.pipe([all_text], batch_size=1):
        for token in doc:
            if token.is_alpha and len(token.text) > 1:
                lemma_counts[token.lemma_.lower()] += 1
                word_count += 1

    print(f"  [{label}] {word_count:,} words, {len(lemma_counts):,} unique lemmas")
    return lemma_counts, word_count


def compare_vocabularies(caillou_lemmas, oracle_french_lemmas, all_fra_lemmas=None):
    """Compare Caillou vocabulary against the HC oracle."""
    # Get top N Caillou lemmas
    top_caillou = {lemma for lemma, _ in caillou_lemmas.most_common(200)}

    # Oracle French lemmas
    oracle_set = oracle_french_lemmas

    # Overlap
    overlap = top_caillou & oracle_set
    caillou_only = top_caillou - oracle_set
    oracle_only = oracle_set - top_caillou

    print(f"\n{'='*60}")
    print("CAILLOU vs CREOLE ORACLE VOCABULARY COMPARISON")
    print(f"{'='*60}")
    print(f"\nTop 200 Caillou lemmas: {len(top_caillou)}")
    print(f"Oracle French lemmas:   {len(oracle_set)}")
    print(f"Overlap:                {len(overlap)} ({len(overlap)/len(top_caillou)*100:.0f}% of Caillou top-200)")
    print(f"Caillou-only:           {len(caillou_only)}")
    print(f"Oracle-only:            {len(oracle_only)}")

    # Jaccard similarity
    jaccard = len(overlap) / len(top_caillou | oracle_set)
    print(f"\nJaccard similarity:     {jaccard:.3f}")

    # Show the overlapping lemmas with their Caillou frequency
    print("\n--- Overlapping lemmas (sorted by Caillou frequency) ---")
    overlap_with_freq = [(lemma, caillou_lemmas[lemma]) for lemma in overlap]
    overlap_with_freq.sort(key=lambda x: -x[1])
    for lemma, freq in overlap_with_freq[:30]:
        print(f"  {lemma:20} freq={freq:>6}")

    # Show Caillou-only top lemmas (what kids hear that didn't survive pidginization)
    print("\n--- Top Caillou lemmas NOT in oracle ---")
    caillou_only_freq = [(lemma, caillou_lemmas[lemma]) for lemma in caillou_only]
    caillou_only_freq.sort(key=lambda x: -x[1])
    for lemma, freq in caillou_only_freq[:20]:
        print(f"  {lemma:20} freq={freq:>6}")

    # Show oracle lemmas NOT in Caillou top-200
    print("\n--- Oracle lemmas NOT in Caillou top-200 ---")
    for lemma in sorted(oracle_only)[:20]:
        print(f"  {lemma}")

    # Now compare at different thresholds
    print("\n--- Overlap at different Caillou thresholds ---")
    for n in [50, 100, 200, 500, 1000]:
        top_n = {lemma for lemma, _ in caillou_lemmas.most_common(n)}
        ov = top_n & oracle_set
        print(f"  Top {n:>4}: {len(ov):>3} overlap ({len(ov)/min(n, len(oracle_set))*100:.0f}%)")

    # If we have all-fra lemmas, compare density
    if all_fra_lemmas:
        print("\n--- Oracle density comparison ---")
        total_caillou = sum(caillou_lemmas.values())
        oracle_hits_caillou = sum(caillou_lemmas[lemma] for lemma in oracle_set if lemma in caillou_lemmas)
        density_caillou = oracle_hits_caillou / total_caillou if total_caillou else 0

        total_all = sum(all_fra_lemmas.values())
        oracle_hits_all = sum(all_fra_lemmas[lemma] for lemma in oracle_set if lemma in all_fra_lemmas)
        density_all = oracle_hits_all / total_all if total_all else 0

        print(f"  Caillou oracle density:  {density_caillou:.3f} ({oracle_hits_caillou:,}/{total_caillou:,})")
        print(f"  All-fra oracle density:  {density_all:.3f} ({oracle_hits_all:,}/{total_all:,})")
        print(f"  Ratio:                   {density_caillou/density_all:.2f}x" if density_all > 0 else "")

    return {
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / len(top_caillou) * 100,
        "jaccard": jaccard,
        "overlapping_lemmas": sorted(overlap),
    }


def main():
    print("=== Caillou vs Creole Oracle Analysis ===\n")

    # Load oracle
    oracle_lemmas = load_oracle_lemmas()
    print(f"Oracle French lemmas: {len(oracle_lemmas)}")

    # Download and separate babylm-fra
    by_source = download_babylm_fra()

    # Load from cache if already downloaded
    cache_dir = os.path.join(BABYLM_FRA_DIR, "raw")
    if by_source is None:
        by_source = {}
        for fname in os.listdir(cache_dir):
            if fname.endswith(".txt"):
                source = fname[:-4]
                with open(os.path.join(cache_dir, fname), encoding="utf-8") as f:
                    by_source[source] = [line.strip() for line in f if line.strip()]
                print(f"  Loaded {source}: {len(by_source[source]):,} docs")

    # Find Caillou source
    caillou_key = None
    for key in by_source:
        if "caillou" in key.lower():
            caillou_key = key
            break

    if not caillou_key:
        print(f"\nAvailable sources: {list(by_source.keys())}")
        # Try to find it by content
        for key in by_source:
            sample = " ".join(by_source[key][:5]).lower()
            if "caillou" in sample or "maman" in sample:
                caillou_key = key
                print(f"Found probable Caillou source: {key}")
                break

    if not caillou_key:
        print("\nERROR: Could not identify Caillou source in babylm-fra")
        print("Available sources:", list(by_source.keys()))
        print("\nFalling back to comparing ALL non-subtitle sources against oracle")
        # Use everything except opensubtitles
        caillou_key = "non_subtitle"
        by_source["non_subtitle"] = []
        for key, texts in by_source.items():
            if "subtitle" not in key.lower() and "opensub" not in key.lower():
                by_source["non_subtitle"].extend(texts)

    print(f"\nUsing source: {caillou_key} ({len(by_source[caillou_key]):,} docs)")

    # Lemmatize with spaCy
    print("\nLoading spaCy French model...")
    nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])
    nlp.max_length = 3_000_000

    caillou_lemmas, caillou_words = extract_lemma_frequencies(
        by_source[caillou_key], nlp, label="Caillou")

    # Also lemmatize a general source for comparison
    general_key = None
    for key in by_source:
        if "subtitle" in key.lower() or "opensub" in key.lower():
            general_key = key
            break
    if general_key is None:
        general_key = [k for k in by_source if k != caillou_key][0] if len(by_source) > 1 else None

    all_fra_lemmas = None
    if general_key and general_key != caillou_key:
        all_fra_lemmas, _ = extract_lemma_frequencies(
            by_source[general_key], nlp, label=general_key)

    # Compare
    results = compare_vocabularies(caillou_lemmas, oracle_lemmas, all_fra_lemmas)

    # Save results
    results_path = os.path.join(BABYLM_FRA_DIR, "oracle_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
