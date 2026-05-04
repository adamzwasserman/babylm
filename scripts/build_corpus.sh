#!/bin/bash
# Build the MÉTRON-FR training corpus from scratch.
#
# Phases (each skips itself if its output already exists):
#   1. Download official BabyLM corpus samples         -> corpus/babylm_official/
#   2. Download French CHILDES via childes-db (MySQL)  -> corpus/childes_french/
#   3. Build the Haitian Creole vocabulary oracle      -> corpus/haitian_creole/oracle_french_lemmas.txt
#   4. Build the FR/EN bilingual lemma bridge          -> corpus/bilingual/bilingual_lemmas.txt
#   5. Assemble the final 100M-word training corpus    -> corpus/final/train_french.txt
#   6. Sanity-check the BabyLM word budget             -> stdout
#
# Total wall-clock on a fresh machine: roughly 45-90 minutes (CHILDES MySQL
# pulls and HuggingFace dataset cache fills dominate).
#
# Prerequisites:
#   - Python deps installed (pip install -e ".[dev]" or pip install -r requirements.txt)
#   - CHILDES read-only credentials exported (the childes-db service is
#     publicly read-only; ask Adam for the current values if you don't have
#     them, or check https://childes-db.stanford.edu)
#       export CHILDESDB_HOST=...
#       export CHILDESDB_USER=...
#       export CHILDESDB_PASSWORD=...
#       export CHILDESDB_DATABASE=2021.1   # optional, defaults to 2021.1
#
# Usage:
#   bash scripts/build_corpus.sh                # run all phases, skip done ones
#   FORCE=1 bash scripts/build_corpus.sh        # re-run every phase
#   PHASE=5 bash scripts/build_corpus.sh        # only run phase 5
#   SKIP_PHASES="1 2" bash scripts/build_corpus.sh   # skip listed phases
#
# Env knobs:
#   FORCE          re-run a phase even if its expected output already exists
#   PHASE          run only one phase number
#   SKIP_PHASES    space-separated phase numbers to skip
#
# Logs land under logs/build_corpus/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

LOG_DIR="logs/build_corpus"
mkdir -p "$LOG_DIR"

skip_phase() {
    local phase=$1
    if [ -n "${PHASE:-}" ] && [ "$PHASE" != "$phase" ]; then
        return 0
    fi
    if [ -n "${SKIP_PHASES:-}" ]; then
        for s in $SKIP_PHASES; do
            if [ "$s" = "$phase" ]; then
                return 0
            fi
        done
    fi
    return 1
}

phase_header() {
    echo
    echo "================================================"
    echo "  Phase $1: $2"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"
}

skipped() {
    if [ -z "${FORCE:-}" ] && [ -e "$1" ]; then
        return 0
    fi
    return 1
}

require_childes_creds() {
    local missing=()
    for v in CHILDESDB_HOST CHILDESDB_USER CHILDESDB_PASSWORD; do
        if [ -z "${!v:-}" ]; then
            missing+=("$v")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: missing CHILDES env var(s): ${missing[*]}" >&2
        echo "       The childes-db credentials are publicly read-only; see" >&2
        echo "       https://childes-db.stanford.edu or ask Adam for the values." >&2
        echo "       To skip CHILDES (degraded corpus, smaller CDS share), pass" >&2
        echo "       SKIP_PHASES=\"2\" but the final assembly will warn." >&2
        return 1
    fi
    return 0
}

# -------- Phase 1: official BabyLM corpus samples ---------------------------

if ! skip_phase 1; then
    phase_header 1 "Download official BabyLM corpus samples"
    if skipped "corpus/babylm_official/corpus_summary.json"; then
        echo "  corpus/babylm_official/corpus_summary.json exists, skipping"
    else
        log="$LOG_DIR/phase1_babylm.log"
        echo "  log: $log"
        python scripts/download_babylm_corpus.py > "$log" 2>&1
        echo "  done"
    fi
fi

# -------- Phase 2: CHILDES French ------------------------------------------

if ! skip_phase 2; then
    phase_header 2 "Download French CHILDES via childes-db"
    # If anything already downloaded under corpus/childes_french/, treat as
    # done unless FORCE=1.
    if skipped "corpus/childes_french" && \
       [ -n "$(ls -A corpus/childes_french 2>/dev/null)" ]; then
        echo "  corpus/childes_french/ already populated, skipping"
    else
        require_childes_creds
        log="$LOG_DIR/phase2_childes.log"
        echo "  log: $log"
        python scripts/download_childes_french.py > "$log" 2>&1
        echo "  done"
    fi
fi

# -------- Phase 3: HC vocabulary oracle ------------------------------------

if ! skip_phase 3; then
    phase_header 3 "Build the Haitian Creole vocabulary oracle"
    if skipped "corpus/haitian_creole/oracle_french_lemmas.txt"; then
        echo "  corpus/haitian_creole/oracle_french_lemmas.txt exists, skipping"
    else
        log="$LOG_DIR/phase3_oracle.log"
        echo "  log: $log"
        python scripts/build_creole_oracle.py > "$log" 2>&1
        echo "  done"
    fi
fi

# -------- Phase 4: FR/EN bilingual bridge ----------------------------------

if ! skip_phase 4; then
    phase_header 4 "Build the FR/EN bilingual lemma bridge"
    if skipped "corpus/bilingual/bilingual_lemmas.txt"; then
        echo "  corpus/bilingual/bilingual_lemmas.txt exists, skipping"
    else
        log="$LOG_DIR/phase4_bilingual.log"
        echo "  log: $log"
        python scripts/build_bilingual_lemmas.py > "$log" 2>&1
        echo "  done"
    fi
fi

# -------- Phase 5: assemble final corpus -----------------------------------

if ! skip_phase 5; then
    phase_header 5 "Assemble the final 100M-word French training corpus"
    if skipped "corpus/final/train_french.txt"; then
        echo "  corpus/final/train_french.txt exists, skipping"
    else
        log="$LOG_DIR/phase5_final.log"
        echo "  log: $log"
        python scripts/build_french_corpus.py 2>&1 | tee "$log"
        echo "  done"
    fi
fi

# -------- Phase 6: budget sanity -------------------------------------------

if ! skip_phase 6; then
    phase_header 6 "BabyLM word-budget check"
    if [ ! -f "corpus/final/train_french.txt" ]; then
        echo "ERROR: corpus/final/train_french.txt is missing; phase 5 failed?" >&2
        exit 1
    fi
    python scripts/count_words.py corpus/final/train_french.txt
fi

echo
echo "================================================"
echo "Corpus build complete."
echo "  -> $(ls -lh corpus/final/train_french.txt 2>/dev/null | awk '{print $5, $9}')"
echo
echo "Next:"
echo "  python scripts/build_tokenizer.py    # one-shot, then reused by all seeds"
echo "  bash scripts/run_paper_part1.sh 42 43 44 45 46"
echo "================================================"
