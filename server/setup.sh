#!/usr/bin/env bash
#
# One-time, idempotent setup for the full MÉTRON-FR reproduction.
# Prepares the three inputs the orchestrator needs (corpus, tokenizer, BabyLM
# eval-pipeline data) and verifies the two data files that ship in the repo.
# Safe to run more than once: every step checks for its own output first.
#
# Run from the repo root:  bash server/setup.sh
# Requires: an activated Python env with the repo requirements installed, and
# `huggingface-cli login` already done (the EWoK subset is gated).
set -uo pipefail
cd "$(dirname "$0")/.."
FAIL=0

echo "==================== [1/4] training corpus ===================="
mkdir -p corpus/final
if [ ! -f corpus/final/train_french.txt ]; then
  echo "downloading the published French corpus..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("openhonest/babylm-2026-fr-corpus", "train_french.txt", repo_type="dataset")
shutil.copy(p, "corpus/final/train_french.txt")
print("corpus placed at corpus/final/train_french.txt")
PY
else
  echo "corpus already present."
fi
WORDS=$(wc -w < corpus/final/train_french.txt)
echo "corpus word count: $WORDS   (expected ~92.5M; must be under 100M for the Strict track)"
[ "$WORDS" -gt 80000000 ] && [ "$WORDS" -lt 100000000 ] || { echo "WARN: word count outside the expected 80-100M range"; FAIL=1; }

echo "==================== [2/4] shared tokenizer ===================="
if [ ! -f models/tokenizer/tokenizer.json ]; then
  echo "building the shared 50k BPE tokenizer (one tokenizer for all seeds)..."
  python scripts/build_tokenizer.py --vocab_size 50000
else
  echo "tokenizer already present at models/tokenizer/tokenizer.json"
fi
[ -f models/tokenizer/tokenizer.json ] || { echo "ERROR: tokenizer build failed"; FAIL=1; }

echo "==================== [3/4] repo-shipped eval data ===================="
for f in corpus/bilingual/bilingual_lemmas.txt submission/glue_fr/rte.valid.jsonl submission/glue_fr/boolq.train.jsonl; do
  if [ -f "$f" ]; then echo "present: $f"; else echo "MISSING (wrong branch or bad checkout): $f"; FAIL=1; fi
done

echo "==================== [4/4] BabyLM eval-pipeline data (for the suite phase) ===================="
PDIR="eval/evaluation-pipeline-2025"
pip install -q osfclient 2>&1 | tail -1
if [ ! -d "$PDIR" ]; then
  echo "cloning the BabyLM 2025 evaluation pipeline..."
  git clone --depth 1 https://github.com/babylm/evaluation-pipeline-2025.git "$PDIR" || FAIL=1
fi
if [ ! -d "$PDIR/evaluation_data" ]; then
  echo "downloading evaluation_data from OSF (project ryjfm)..."
  ( cd "$PDIR" && osf -p ryjfm clone . && mv osfstorage/evaluation_data . && rmdir osfstorage ) || { echo "WARN: OSF download failed"; FAIL=1; }
fi
if [ -d "$PDIR" ]; then
  pip install -q -r "$PDIR/requirements.txt" 2>&1 | tail -1
  # The pipeline install can leave a torchvision built for a different torch,
  # which makes transformers' AutoProcessor import die on
  # "torchvision::nms does not exist" and crashes every eval at import time.
  # The text eval never uses vision, so remove any mismatched torchvision.
  pip uninstall -y torchvision >/dev/null 2>&1 || true
  python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" >/dev/null 2>&1 || true
  echo "generating the EWoK subset (gated; needs huggingface-cli login for ewok-core)..."
  python "$PDIR/evaluation_pipeline/ewok/dl_and_filter.py" \
    || echo "WARN: EWoK generation failed. Confirm 'huggingface-cli whoami' works and that access to ewok-core/ewok-core-1.0 has been requested/granted. The suite phase will otherwise skip EWoK."
fi

echo "==================== summary ===================="
if [ "$FAIL" = "0" ]; then
  echo "SETUP OK. Next: bash scripts/run_paper_part1.sh 42 43 44 45 46"
else
  echo "SETUP INCOMPLETE. Resolve the WARN/MISSING/ERROR lines above before launching the full run."
  exit 1
fi
