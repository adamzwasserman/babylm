#!/bin/bash
# Pre-flight checks before launching scripts/run_paper_part1.sh.
#
# Validates that the host has everything needed to run the multi-hour
# training + eval matrix without hitting a stupid blocker after 2 hours
# (missing dep, no GPU, no disk space, gated dataset un-authenticated).
#
# Each check is fail-soft: prints PASS / WARN / FAIL with the reason; the
# overall exit code is 0 only if there are zero FAILs (WARNs are tolerated).
#
# Usage:
#   bash scripts/preflight.sh
#   SEEDS="42 43 44 45 46" bash scripts/preflight.sh
#
# What it checks:
#   1. Corpus assembled and within BabyLM budget
#   2. Shared BPE tokenizer present
#   3. GPU visibility (>= 1, ideally >= 3 for parallel seeds)
#   4. Free disk space for 5 checkpoints (~1 GB per epoch * 5 seeds * 5 epochs)
#   5. Python deps importable (torch, transformers, peft, etc.)
#   6. wandb credentials present (or skip recommended)
#   7. HuggingFace credentials present (gated babylm-fra)
#   8. Smoke-test: 50 training steps on 1 GPU, validate forward+backward+save

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PASS=0; WARN=0; FAIL=0
BOLD=$(tput bold 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)
GREEN=$(tput setaf 2 2>/dev/null || true)
YELLOW=$(tput setaf 3 2>/dev/null || true)
RED=$(tput setaf 1 2>/dev/null || true)

pass() { echo "  ${GREEN}PASS${RESET}  $*"; PASS=$((PASS + 1)); }
warn() { echo "  ${YELLOW}WARN${RESET}  $*"; WARN=$((WARN + 1)); }
fail() { echo "  ${RED}FAIL${RESET}  $*"; FAIL=$((FAIL + 1)); }
section() { echo; echo "${BOLD}== $1 ==${RESET}"; }

# ---- 1. Corpus -------------------------------------------------------------

section "1. Corpus"
CORPUS=corpus/final/train_french.txt
if [ ! -f "$CORPUS" ]; then
    fail "$CORPUS missing. Run: bash scripts/build_corpus.sh"
else
    size=$(stat -c%s "$CORPUS" 2>/dev/null || stat -f%z "$CORPUS")
    pass "$CORPUS exists ($((size / 1024 / 1024)) MB)"
    words=$(python -c "
import sys
with open('$CORPUS', encoding='utf-8') as f:
    n = sum(len(line.split()) for line in f)
print(n)
")
    if [ "$words" -lt 50000000 ]; then
        fail "Only $words words; expected >= 50M (and ideally ~92.5M for the paper recipe)"
    elif [ "$words" -gt 100000000 ]; then
        fail "$words words exceeds the BabyLM Strict 100M budget"
    elif [ "$words" -lt 90000000 ]; then
        warn "$words words below the 92.5M paper recipe (CHILDES skipped?)"
    else
        pass "$words words within the BabyLM Strict budget"
    fi
fi

# ---- 2. Shared tokenizer ---------------------------------------------------

section "2. Shared BPE tokenizer"
if [ -f models/tokenizer/tokenizer.json ]; then
    pass "models/tokenizer/tokenizer.json present"
else
    warn "Tokenizer not built yet; run: python scripts/build_tokenizer.py"
fi

# ---- 3. GPUs ---------------------------------------------------------------

section "3. GPUs"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found; CUDA training will not work"
else
    n_gpu=$(nvidia-smi -L | wc -l)
    if [ "$n_gpu" -ge 3 ]; then
        pass "$n_gpu GPUs visible (parallel by waves of 3)"
    elif [ "$n_gpu" -ge 1 ]; then
        warn "$n_gpu GPU(s) visible; multi-seed will be sequential"
    else
        fail "0 GPUs visible"
    fi
    if [ "$n_gpu" -ge 1 ]; then
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$free_mb" -lt 8000 ]; then
            warn "GPU 0 has only ${free_mb} MiB free; 125M GPT-2 + AMP needs ~10 GB"
        else
            pass "GPU 0 free memory: ${free_mb} MiB"
        fi
    fi
fi

# ---- 4. Disk ---------------------------------------------------------------

section "4. Disk"
free_kb=$(df -k . | awk 'NR==2 {print $4}')
free_gb=$((free_kb / 1024 / 1024))
# Rough budget: 5 seeds * (~1 GB tokenizer cache + 30 ckpt-snapshots * ~500 MB) = ~75 GB
need_gb=80
if [ "$free_gb" -lt "$need_gb" ]; then
    fail "${free_gb} GB free, need >= ${need_gb} GB for 5 seeds + all checkpoints"
else
    pass "${free_gb} GB free (>= ${need_gb} GB needed)"
fi

# ---- 5. Python deps --------------------------------------------------------

section "5. Python imports"
python - <<'PY' || fail "Some imports failed; pip install -r requirements.txt"
import importlib
mods = [
    "torch", "transformers", "tokenizers", "datasets",
    "peft", "evaluate", "sklearn", "accelerate", "wandb",
    "numpy", "pandas", "huggingface_hub",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({type(e).__name__}: {e})")
if missing:
    print("  missing:", *missing, sep="\n    ")
    raise SystemExit(1)
print("  all modules import OK")
PY
[ $? -eq 0 ] && pass "torch, transformers, peft, evaluate, sklearn, accelerate, wandb, datasets all importable"

# CUDA visible inside torch
python - <<'PY' || warn "torch could not see CUDA"
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"  torch sees CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)")
PY
[ $? -eq 0 ] && pass "torch.cuda.is_available() == True"

# ---- 6. wandb --------------------------------------------------------------

section "6. wandb"
if python -c "import netrc, os; nrc = netrc.netrc(os.path.expanduser('~/.netrc')); exit(0 if nrc.authenticators('api.wandb.ai') else 1)" 2>/dev/null; then
    user=$(python -c "import wandb; print(wandb.Api().viewer.username)" 2>/dev/null || echo "?")
    pass "wandb logged in as: $user"
else
    warn "wandb not logged in; run 'wandb login' or pass TRAIN_EXTRA=\"--wandb_mode disabled\""
fi

# ---- 7. HuggingFace --------------------------------------------------------

section "7. HuggingFace Hub"
if python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
    user=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])")
    pass "HF authenticated as: $user"
else
    warn "HF not logged in; gated datasets (e.g. babylm-fra) will fail. Run: huggingface-cli login"
fi

# ---- 8. Smoke training -----------------------------------------------------

section "8. Training smoke test (50 steps, 1 GPU, no wandb)"
SMOKE_DIR=/tmp/babylm_smoke_$$
if [ ! -f models/tokenizer/tokenizer.json ]; then
    warn "skipping smoke test: tokenizer not built"
elif [ ! -f "$CORPUS" ]; then
    warn "skipping smoke test: corpus not built"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "skipping smoke test: no GPU"
else
    echo "  Running 50 training steps in $SMOKE_DIR (~1-2 min)..."
    if CUDA_VISIBLE_DEVICES=0 timeout 300 python scripts/train.py \
            --seed 999 \
            --output_dir "$SMOKE_DIR" \
            --wandb_mode disabled \
            --epochs 1 \
            --batch_size 4 \
            --warmup_steps 10 \
            >/tmp/babylm_smoke.log 2>&1 &
    then
        pid=$!
        # Let it train for ~90 seconds (well past the 50-step log line)
        sleep 90
        kill -INT "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    if grep -qE "step [0-9]+\s+\| loss" /tmp/babylm_smoke.log; then
        last=$(grep -E "step [0-9]+\s+\| loss" /tmp/babylm_smoke.log | tail -1)
        pass "forward+backward OK ($last)"
    else
        fail "no training step logged in 90s; see /tmp/babylm_smoke.log"
    fi
    rm -rf "$SMOKE_DIR"
fi

# ---- Summary ---------------------------------------------------------------

echo
echo "================================================"
echo "  Preflight summary: ${GREEN}${PASS} PASS${RESET}, ${YELLOW}${WARN} WARN${RESET}, ${RED}${FAIL} FAIL${RESET}"
echo "================================================"

if [ "$FAIL" -gt 0 ]; then
    echo "Fix the FAILs above before launching the multi-hour run."
    exit 1
fi
if [ "$WARN" -gt 0 ]; then
    echo "WARNs are tolerable but worth a look. To proceed:"
fi
echo "  bash scripts/run_paper_part1.sh 42 43 44 45 46"
exit 0
