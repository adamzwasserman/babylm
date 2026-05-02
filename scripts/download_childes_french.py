"""
Download French child-directed speech from CHILDES via childes-db (public MySQL).

childes-db provides public read-only access to all CHILDES data. The host and
read-only credentials are documented at https://childes-db.stanford.edu and on
the project README; we read them from environment variables rather than
hardcoding so the source stays free of credentials.

Set before running:
    export CHILDESDB_HOST=...
    export CHILDESDB_USER=...
    export CHILDESDB_PASSWORD=...
    export CHILDESDB_DATABASE=2021.1   # optional, defaults to 2021.1

We extract adult utterances (MOT, FAT, INV, EXP, etc.) from all French corpora.
This gives us ~2.1M words of gold child-directed speech.
"""

import os
import sys
import pymysql

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus", "childes_french")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _db_config():
    required = ("CHILDESDB_HOST", "CHILDESDB_USER", "CHILDESDB_PASSWORD")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        sys.exit(
            f"missing env var(s): {', '.join(missing)}. See module docstring."
        )
    return {
        "host": os.environ["CHILDESDB_HOST"],
        "user": os.environ["CHILDESDB_USER"],
        "password": os.environ["CHILDESDB_PASSWORD"],
        "database": os.environ.get("CHILDESDB_DATABASE", "2021.1"),
        "charset": "utf8mb4",
        "connect_timeout": 30,
        "read_timeout": 120,
    }


DB_CONFIG = None  # populated lazily inside download_childes_french()

# Adult speaker roles (child-directed speech)
ADULT_ROLES = ("Mother", "Father", "Investigator", "Adult", "Experimenter",
               "Grandmother", "Grandfather", "Teacher", "Caretaker", "Other")


def download_childes_french():
    config = _db_config()
    print("=== Downloading CHILDES French via childes-db ===")
    print(f"Connecting to {config['host']}...")

    conn = pymysql.connect(**config)
    cursor = conn.cursor()

    # Get all French corpora
    cursor.execute("""
        SELECT DISTINCT corpus_name
        FROM utterance
        WHERE collection_name = 'French'
        ORDER BY corpus_name
    """)
    corpora = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(corpora)} French corpora\n")

    word_counts = {}
    total_utterances = 0

    for corpus_name in corpora:
        print(f"  Fetching: {corpus_name}...", end=" ", flush=True)

        cursor.execute("""
            SELECT gloss, num_tokens
            FROM utterance
            WHERE corpus_name = %s
              AND collection_name = 'French'
              AND speaker_role IN %s
              AND gloss IS NOT NULL
              AND gloss != ''
              AND num_tokens > 0
            ORDER BY transcript_id, utterance_order
        """, (corpus_name, ADULT_ROLES))

        rows = cursor.fetchall()
        if not rows:
            print("no adult utterances found")
            word_counts[corpus_name] = 0
            continue

        utterances = [row[0] for row in rows]
        word_count = sum(row[1] for row in rows)

        out_path = os.path.join(OUTPUT_DIR, f"{corpus_name.lower()}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(utterances))

        word_counts[corpus_name] = word_count
        total_utterances += len(utterances)
        print(f"{len(utterances):,} utterances, {word_count:,} words")

    cursor.close()
    conn.close()

    total_words = sum(word_counts.values())
    print(f"\n=== CHILDES French Summary ===")
    for corpus, count in sorted(word_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {corpus}: {count:,} words")
    print(f"  TOTAL: {total_words:,} words ({total_utterances:,} utterances)")
    print(f"\n  Target: 90-100M words for BabyLM Strict track")
    print(f"  CHILDES provides ~{total_words/1_000_000:.1f}M words of gold CDS")

    return word_counts


if __name__ == "__main__":
    download_childes_french()
