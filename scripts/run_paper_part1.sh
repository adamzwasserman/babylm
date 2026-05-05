#!/bin/bash
# Master orchestrator for §4 of the paper (5 seeds, headline results).
#
# Phases:
#   0. Pre-flight checks (corpus, tokenizer, GPU count, wandb)
#   1. Train 5 seeded checkpoints       -> models/seed{42..46}/chck_92M
#   2. QFrBLiMP zero-shot               -> eval_results/seed{S}_qfrblimp.json
#   3. QFrCoLA fine-tune + MCC          -> eval_results/seed{S}_qfrcola.json
#   4. BabyLM 2025 suite                -> eval_results/seed{S}_babylm.json
#   5. BLI Procrustes (vs GPT-2 EN)     -> eval_results/seed{S}_bli_gpt2_en.json
#   6. Cross-lingual GLUE 4-lever grid  -> eval_results/seed{S}_xglue_<lever>_<task>.json
#   7. Aggregate Tables 1-3 + prose     -> paper_tables.tex, paper_tables.md
#
# Each phase skips itself if the expected output already exists, so the
# script is resumable. Set FORCE=1 to re-run even when output is present.
#
# Usage:
#   bash scripts/run_paper_part1.sh                 # default seeds 42..46
#   bash scripts/run_paper_part1.sh 1 2 3 4 5
#   SEEDS="42 43" bash scripts/run_paper_part1.sh   # alternative
#   PHASE=4 bash scripts/run_paper_part1.sh         # only run phase 4
#   SKIP_PHASES="6" bash scripts/run_paper_part1.sh # skip cross-lingual GLUE
#
# Environment knobs:
#   N_GPUS         override GPU count for parallel trainings (default: nvidia-smi -L)
#   TRAIN_EXTRA    extra flags forwarded to train.py (e.g. --wandb_mode disabled)
#   CKPT_NAME      checkpoint name to evaluate (default: chck_92M)
#   FORCE          re-run a phase even if the expected output exists
#   PHASE          run only one phase number; otherwise run all
#   SKIP_PHASES    space-separated phase numbers to skip
#
# Outputs land under eval_results/. Logs land under logs/paper_part1/.

set -euo pipefail

# Stream Python stdout/stderr live (no buffering) so progress prints and
# tqdm bars appear in real time when piped through tee.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Auto-load .env (CHILDESDB_*, wandb knobs, etc.) so the user does not have
# to export them manually. .env is gitignored; copy from .env.example on
# first use: `cp .env.example .env`.
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Seeds: positional args, else SEEDS env var, else 42..46
if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
elif [ -n "${SEEDS:-}" ]; then
    # shellcheck disable=SC2206
    SEEDS=($SEEDS)
else
    SEEDS=(42 43 44 45 46)
fi

# If CKPT_NAME is unset, we auto-detect per seed: pick the highest-numbered
# chck_NM/ directory under models/seed{S}/. This makes the orchestrator work
# whether the training ran 1, 3, or 5 epochs (different word counts -> the
# final checkpoint name varies).
CKPT_NAME="${CKPT_NAME:-}"
LOG_DIR="logs/paper_part1"
mkdir -p "$LOG_DIR" eval_results

# Resolve the checkpoint path for one seed. Resolution order:
#   1. $CKPT_NAME (explicit override, e.g. "chck_92M_epoch3")
#   2. models/seed{S}/best         (symlink set by phase 2's pick)
#   3. highest chck_NM_epoch{E}/   (last epoch saved)
#   4. highest chck_NM/            (BabyLM-cadence checkpoint, legacy)
resolve_ckpt() {
    local seed=$1
    local seed_dir="models/seed${seed}"
    if [ -n "$CKPT_NAME" ]; then
        echo "${seed_dir}/${CKPT_NAME}"
        return
    fi
    if [ -L "${seed_dir}/best" ] || [ -d "${seed_dir}/best" ]; then
        echo "${seed_dir}/best"
        return
    fi
    local best=""
    local best_n=-1
    for d in "${seed_dir}"/chck_*M_epoch*; do
        [ -d "$d" ] || continue
        local epoch
        epoch=$(basename "$d" | sed -E 's/.*_epoch([0-9]+).*/\1/')
        if [[ "$epoch" =~ ^[0-9]+$ ]] && [ "$epoch" -gt "$best_n" ]; then
            best_n=$epoch
            best=$d
        fi
    done
    if [ -z "$best" ]; then
        for d in "${seed_dir}"/chck_*M; do
            [ -d "$d" ] || continue
            local base
            base=$(basename "$d")
            local n=${base#chck_}
            n=${n%M}
            if [[ "$n" =~ ^[0-9]+$ ]] && [ "$n" -gt "$best_n" ]; then
                best_n=$n
                best=$d
            fi
        done
    fi
    if [ -z "$best" ]; then
        echo "${seed_dir}/chck_92M"
    else
        echo "$best"
    fi
}

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
    echo "  Seeds: ${SEEDS[*]}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================"
}

skipped() {
    if [ -z "${FORCE:-}" ] && [ -e "$1" ]; then
        return 0
    fi
    return 1
}

# -------- Phase 0: pre-flight ------------------------------------------------

if ! skip_phase 0; then
    phase_header 0 "Pre-flight checks"
    if [ ! -f "corpus/final/train_french.txt" ]; then
        echo "ERROR: corpus/final/train_french.txt not found." >&2
        echo "       Run scripts/build_french_corpus.py first." >&2
        exit 2
    fi
    if [ ! -f "models/tokenizer/tokenizer.json" ]; then
        echo "Building shared tokenizer..."
        python scripts/build_tokenizer.py
    fi
    # wandb credentials are read from the local wandb config (~/.netrc after
    # `wandb login`). Run `wandb login` once on this host before launching;
    # to skip wandb entirely, pass TRAIN_EXTRA="--wandb_mode disabled".
    if ! python -c "import netrc, os; nrc = netrc.netrc(os.path.expanduser('~/.netrc')); nrc.authenticators('api.wandb.ai') or exit(1)" 2>/dev/null; then
        echo "WARN: no wandb credentials found in ~/.netrc."
        echo "      Run 'wandb login' once, or pass TRAIN_EXTRA=\"--wandb_mode disabled\"."
    fi
    if [ -n "$CKPT_NAME" ]; then
        echo "Pre-flight OK. Seeds: ${SEEDS[*]}, fixed ckpt name: $CKPT_NAME"
    else
        echo "Pre-flight OK. Seeds: ${SEEDS[*]}, auto-detect highest chck_NM/ per seed"
    fi
fi

# -------- Phase 1: training (5 seeded checkpoints) --------------------------

if ! skip_phase 1; then
    phase_header 1 "Training (parallel by N_GPUS waves)"
    need_train=()
    for s in "${SEEDS[@]}"; do
        # Consider a seed already trained if there is at least one chck_*M
        # directory under models/seed{S}/. If you want to force a re-run
        # with more epochs, set FORCE=1 or delete the seed dir.
        if [ -z "${FORCE:-}" ] && \
           ls -d "models/seed${s}"/chck_*M >/dev/null 2>&1; then
            existing=$(resolve_ckpt "$s")
            echo "  seed=$s -> $existing already exists, skipping"
        else
            need_train+=("$s")
        fi
    done
    if [ ${#need_train[@]} -gt 0 ]; then
        echo "Training seeds: ${need_train[*]}"
        bash scripts/run_multi_seed.sh "${need_train[@]}"
    fi
fi

# Helper: per-seed eval that writes a JSON; skip if file present.
eval_seed() {
    local phase=$1 desc=$2 seed=$3 expected=$4
    shift 4
    if skipped "$expected"; then
        echo "  [phase $phase] seed=$seed: $expected exists, skipping"
        return 0
    fi
    local log="$LOG_DIR/phase${phase}_seed${seed}.log"
    echo "  [phase $phase] seed=$seed: $desc -> $expected (log: $log)"
    "$@" 2>&1 | tee "$log"
}

CKPT_PATH() { resolve_ckpt "$1"; }

# -------- Phase 2: QFrBLiMP per epoch + best-epoch selection ---------------

if ! skip_phase 2; then
    phase_header 2 "QFrBLiMP zero-shot (per epoch) + best-epoch selection"
    for s in "${SEEDS[@]}"; do
        epoch_ckpts=()
        for ckpt in models/seed${s}/chck_*M_epoch*; do
            [ -d "$ckpt" ] || continue
            epoch_ckpts+=("$ckpt")
        done
        if [ ${#epoch_ckpts[@]} -eq 0 ]; then
            # Fall back to legacy single-checkpoint behaviour: eval the
            # highest BabyLM-cadence checkpoint and treat it as "epoch 1".
            ckpt=$(resolve_ckpt "$s")
            out="eval_results/seed${s}_qfrblimp_epoch1.json"
            if ! skipped "$out"; then
                log="$LOG_DIR/phase2_seed${s}_epoch1.log"
                echo "  [phase 2] seed=$s epoch=1 (legacy): $ckpt -> $out (log: $log)"
                python eval/qfrblimp/run.py "$ckpt" --seed "$s" 2>&1 | tee "$log"
                mv "eval_results/seed${s}_qfrblimp.json" "$out"
            fi
        else
            for ckpt in "${epoch_ckpts[@]}"; do
                epoch=$(basename "$ckpt" | sed -E 's/.*_epoch([0-9]+).*/\1/')
                out="eval_results/seed${s}_qfrblimp_epoch${epoch}.json"
                if skipped "$out"; then
                    echo "  [phase 2] seed=$s epoch=$epoch: $out exists, skipping"
                    continue
                fi
                log="$LOG_DIR/phase2_seed${s}_epoch${epoch}.log"
                echo "  [phase 2] seed=$s epoch=$epoch: $ckpt -> $out (log: $log)"
                python eval/qfrblimp/run.py "$ckpt" --seed "$s" 2>&1 | tee "$log"
                # The eval script writes seed{S}_qfrblimp.json; rename to
                # the per-epoch filename so all epochs are preserved.
                mv "eval_results/seed${s}_qfrblimp.json" "$out"
            done
        fi
    done

    echo
    echo "-- Picking best epoch per seed (argmax QFrBLiMP overall) --"
    for s in "${SEEDS[@]}"; do
        python scripts/_pick_best_epoch.py --seed "$s"
    done
fi

# -------- Phase 3: QFrCoLA ---------------------------------------------------

if ! skip_phase 3; then
    phase_header 3 "QFrCoLA fine-tune + MCC"
    for s in "${SEEDS[@]}"; do
        eval_seed 3 "QFrCoLA" "$s" "eval_results/seed${s}_qfrcola.json" \
            python eval/qfrcola/run.py "$(CKPT_PATH "$s")" --seed "$s"
    done
fi

# -------- Phase 4: BabyLM 2025 suite -----------------------------------------

if ! skip_phase 4; then
    phase_header 4 "BabyLM 2025 suite (BLiMP, BLiMP-Sup, EWoK, GLUE)"
    for s in "${SEEDS[@]}"; do
        eval_seed 4 "BabyLM suite" "$s" "eval_results/seed${s}_babylm.json" \
            python scripts/eval_babylm_suite.py "$(CKPT_PATH "$s")" --seed "$s"
    done
fi

# -------- Phase 5: BLI Procrustes -------------------------------------------

if ! skip_phase 5; then
    phase_header 5 "BLI Procrustes (vs GPT-2 EN)"
    for s in "${SEEDS[@]}"; do
        eval_seed 5 "BLI vs GPT-2 EN" "$s" "eval_results/seed${s}_bli_gpt2_en.json" \
            python scripts/run_bli_procrustes.py "$(CKPT_PATH "$s")" --seed "$s"
    done
fi

# -------- Phase 6: Cross-lingual GLUE grid ----------------------------------

if ! skip_phase 6; then
    phase_header 6 "Cross-lingual GLUE LoRA grid (5 levers x 5 tasks)"
    for s in "${SEEDS[@]}"; do
        # Each (lever, task) cell writes its own JSON; the script skips B
        # internally (placebo, handled in §5.2).
        log="$LOG_DIR/phase6_seed${s}.log"
        echo "  [phase 6] seed=$s: full grid (log: $log, live output below)"
        if ! python scripts/run_xling_glue.py "$(CKPT_PATH "$s")" --seed "$s" \
                --all_levers --all_tasks 2>&1 | tee "$log"; then
            echo "  [phase 6] seed=$s FAILED, see $log" >&2
            exit 1
        fi
    done
fi

# -------- Phase 7: aggregation ----------------------------------------------

if ! skip_phase 7; then
    phase_header 7 "Aggregating Tables 1-3 + §4.1/§4.2 prose"
    python scripts/aggregate_paper_tables.py \
        --output paper_tables.tex \
        --md_output paper_tables.md
fi

echo
echo "================================================"
echo "Done. Outputs:"
echo "  paper_tables.tex"
echo "  paper_tables.md"
echo "  eval_results/seed{42..46}_*.json"
echo "================================================"
