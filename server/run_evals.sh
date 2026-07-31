#!/usr/bin/env bash
#
# Run the cross-lingual GLUE grid (paper Table 3) and BLI Procrustes (Table 2)
# for the five French checkpoints, on a single GPU server.
#
# It is safe to stop and restart this script at any time: every result is
# written to disk the instant it is computed, and any cell already on disk is
# skipped on the next run. Nothing is recomputed and nothing is lost.
#
# Usage:
#   bash server/run_evals.sh
#
# Configuration is via the environment variables below; defaults are sensible.
set -uo pipefail

# --- configuration ------------------------------------------------------------
SEEDS="${SEEDS:-42 43 44 45 46}"          # reproduction seeds
CKPT_REPO_TMPL="${CKPT_REPO_TMPL:-openhonest/babylm-2026-fr-92m-seedSEED}"  # SEED is substituted
CKPT_REVISION="${CKPT_REVISION:-chck_92M_epoch3}"   # the epoch-3 (grammatical-peak) checkpoint
RESULTS_DIR="${RESULTS_DIR:-server/results}"        # local output root
CKPT_ROOT="${CKPT_ROOT:-server/checkpoints}"        # where checkpoints are downloaded
LEVERS="${LEVERS:-Baseline C D+C A B}"    # essential contrast (Baseline,C,D+C) first
TASKS="${TASKS:-boolq rte mrpc wsc mnli}"
# ------------------------------------------------------------------------------

cd "$(dirname "$0")/.."   # repo root
mkdir -p "$RESULTS_DIR" "$CKPT_ROOT"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'" \
  || { echo "FATAL: no GPU visible to PyTorch"; exit 1; }

for S in $SEEDS; do
  OUT="$RESULTS_DIR/seed$S"; mkdir -p "$OUT"
  CKPT="$CKPT_ROOT/seed$S"
  REPO="${CKPT_REPO_TMPL/SEED/$S}"

  echo "==================== seed $S ===================="
  # 1) fetch the epoch-3 checkpoint (idempotent; skips if already present)
  if [ ! -f "$CKPT/config.json" ]; then
    echo "[seed $S] downloading $REPO @ $CKPT_REVISION"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', revision='$CKPT_REVISION', local_dir='$CKPT')" \
      || { echo "[seed $S] checkpoint download FAILED, skipping seed"; continue; }
  else
    echo "[seed $S] checkpoint already present at $CKPT"
  fi

  # 2) BLI Procrustes vs GPT-2 (Table 2). Fast. Skips if the output exists.
  if [ ! -f "$OUT/seed${S}_bli_gpt2_en.json" ]; then
    echo "[seed $S] BLI Procrustes"
    python scripts/run_bli_procrustes.py "$CKPT" --seed "$S" --output_dir "$OUT" \
      || echo "[seed $S] WARN: BLI failed"
  else
    echo "[seed $S] BLI already done"
  fi

  # 3) CL-GLUE grid (Table 3), one (lever,task) cell at a time so each is
  #    preserved immediately and skipped on resume. Essential levers first.
  for L in $LEVERS; do
    for T in $TASKS; do
      # run_xling_glue writes seed{S}_xglue_<LEVER>_<TASK>.json and skips it if
      # it already exists (it sanitises "D+C" -> "DC" in the filename).
      echo "[seed $S] xglue lever=$L task=$T"
      python scripts/run_xling_glue.py "$CKPT" --lever "$L" --task "$T" \
        --seed "$S" --output_dir "$OUT" \
        || echo "[seed $S] WARN: xglue $L/$T failed"
    done
  done
  echo "[seed $S] done. results in $OUT"
done

echo "==================== ALL SEEDS COMPLETE ===================="
echo "Results tree: $RESULTS_DIR"
find "$RESULTS_DIR" -name '*.json' | wc -l | xargs echo "total result files:"
echo "Expected at full coverage: 5 BLI + 125 xglue = 130 files."
