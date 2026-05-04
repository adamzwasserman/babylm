#!/bin/bash
# Run multiple training seeds in parallel across N GPUs.
#
# Assumes a local server with N visible CUDA GPUs (e.g. 3x RTX 4000 Ada).
# For each wave of N seeds, launches one training process per GPU with
# CUDA_VISIBLE_DEVICES set, then waits for the wave to finish before
# starting the next.
#
# Usage:
#   ./scripts/run_multi_seed.sh                    # default seeds: 1 2 3 4 5
#   ./scripts/run_multi_seed.sh 10 11 12 13 14
#   N_GPUS=2 ./scripts/run_multi_seed.sh 1 2 3 4
#
# Each seed writes:
#   logs/seed{S}.log         training stdout/stderr
#   models/seed{S}/chck_*M/  per-checkpoint HF model dirs
#
# Prereqs:
#   - venv activated (or .venv/ in project root, auto-sourced)
#   - corpus at corpus/final/train_french.txt
#   - wandb logged in locally via `wandb login` (or pass
#     TRAIN_EXTRA="--wandb_mode disabled" to skip wandb entirely)
#
# Optional env vars:
#   N_GPUS         override GPU count (default: nvidia-smi -L | wc -l)
#   TRAIN_EXTRA    extra flags appended to every train.py call
#                  (e.g. TRAIN_EXTRA="--epochs 2 --batch_size 16")

set -eo pipefail

# Kill any background training children if we get interrupted or fail.
cleanup() {
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        echo "Cleaning up background processes: $pids" >&2
        # shellcheck disable=SC2086
        kill $pids 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(1 2 3 4 5)
fi

if [ -z "$N_GPUS" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi -L | wc -l)
    else
        N_GPUS=1
    fi
fi

# nvidia-smi -L can return 0 when no GPUs are visible; treat that as fatal
# rather than entering an inner for-loop that launches nothing and looping
# the outer while forever.
if ! [[ "$N_GPUS" =~ ^[0-9]+$ ]] || [ "$N_GPUS" -lt 1 ]; then
    echo "ERROR: N_GPUS must be a positive integer (got: '$N_GPUS')." >&2
    echo "       nvidia-smi may have failed or no GPUs are visible." >&2
    exit 2
fi

echo "================================================"
echo "Multi-seed training"
echo "  Seeds       : ${SEEDS[*]}"
echo "  Parallel    : $N_GPUS GPU(s) per wave"
echo "  Project dir : $PROJECT_DIR"
echo "  Extra args  : ${TRAIN_EXTRA:-(none)}"
echo "================================================"

mkdir -p logs

# Build the shared tokenizer once if missing (parallel seeds must not race on it).
if [ ! -f "models/tokenizer/tokenizer.json" ]; then
    echo ""
    echo "--- Building shared tokenizer ---"
    python scripts/build_tokenizer.py
fi

i=0
total=${#SEEDS[@]}
wave_num=0
while [ $i -lt "$total" ]; do
    wave_num=$((wave_num + 1))
    wave_pids=()
    wave_seeds=()
    wave_gpus=()

    for ((j=0; j<N_GPUS && (i+j)<total; j++)); do
        seed=${SEEDS[$((i+j))]}
        gpu=$j
        log="logs/seed${seed}.log"

        echo ""
        echo "[$(date '+%H:%M:%S')] Wave $wave_num: launching seed=$seed on GPU $gpu"
        echo "  log : $log"

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu python scripts/train.py \
            --seed "$seed" \
            --output_dir "models/seed${seed}" \
            --wandb_run_name "seed${seed}" \
            $TRAIN_EXTRA \
            > "$log" 2>&1 &

        wave_pids+=($!)
        wave_seeds+=("$seed")
        wave_gpus+=("$gpu")
    done

    echo ""
    echo "[$(date '+%H:%M:%S')] Wave $wave_num running:"
    for k in "${!wave_pids[@]}"; do
        echo "  seed=${wave_seeds[$k]} gpu=${wave_gpus[$k]} pid=${wave_pids[$k]}"
    done

    fail=0
    for k in "${!wave_pids[@]}"; do
        pid=${wave_pids[$k]}
        seed=${wave_seeds[$k]}
        if ! wait "$pid"; then
            echo "[FAIL] seed=$seed (pid $pid) exited non-zero. See logs/seed${seed}.log"
            fail=1
        else
            echo "[ OK ] seed=$seed completed"
        fi
    done

    if [ $fail -ne 0 ]; then
        echo ""
        echo "Wave $wave_num had failures, aborting before next wave."
        exit 1
    fi

    i=$((i + N_GPUS))
done

echo ""
echo "================================================"
echo "All ${total} seed(s) complete."
echo "Aggregate eval results with:"
echo "  python scripts/aggregate_seeds.py 'eval_results/seed*.json'"
echo "================================================"
