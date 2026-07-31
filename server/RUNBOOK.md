# Runbook: cross-lingual GLUE grid (Table 3) and BLI Procrustes (Table 2)

This runbook produces two tables for the MÉTRON-FR paper from the five preserved French checkpoints. It is written to be run end to end by an automated agent on a single-GPU server, with no prior context required.

## Context: this completes an existing run, it does not start one

Most of Section 4 is already done. A clean five-seed run has produced Table 1 (QFrBLiMP), the leaderboard suite, QFrCoLA, the new QFrCoRE/QFrCoRT probes, the training-compute numbers, all five BLI runs (Table 2), and a partial Table 3. Those results, with the actual numbers and where the raw files live on the Hugging Face Hub, are in `server/RESULTS.md`; read it first. The five checkpoints and all per-seed result files are public and browsable. What this runbook adds is the compute-heavy remainder: finishing the Table 3 grid (and re-running BLI is included only for completeness, since it is cheap and idempotent). Two smaller items, translated BLiMP and the LoRA-preservation table, have no harness yet and are out of scope here.

## What you are producing

Five French GPT-2 checkpoints (125M parameters, trained on ~92M words of French) already exist and are published. You will evaluate them, on the epoch-3 checkpoint of each seed, to fill two tables:

- **Table 2 — BLI Procrustes alignment.** A closed-form orthogonal map from the French model's embedding space to GPT-2's, reporting alignment fit and word-translation precision@k on a held-out dictionary split. One run per seed. Fast (seconds to a minute each).
- **Table 3 — cross-lingual GLUE grid.** For each seed, a grid of five adaptation levers x five GLUE tasks, each an independent LoRA fine-tune. This is the compute-heavy part (25 fine-tunes per seed, 125 total).

Both tables are reported as mean +/- standard deviation across the five seeds (42, 43, 44, 45, 46).

## Prerequisites

- A machine with one CUDA GPU. An L4 (24 GB) or larger is sufficient; the model is small.
- Python 3.10 or newer.
- A Hugging Face account token with read access (to download the checkpoints), exported as `HF_TOKEN` or configured via `huggingface-cli login`. The checkpoints are public, so read access is all that is needed.
- Outbound network access to huggingface.co and to raw.githubusercontent.com (one GLUE dependency and the BLI seed dictionary ship with the repo, so no other downloads are required).

## One-time setup

```bash
git clone https://github.com/adamzwasserman/babylm.git
cd babylm
git checkout david-eval-handoff        # the branch this runbook ships on

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.7.0" "transformers==4.51.3" "tokenizers==0.21.1" \
            datasets huggingface_hub "peft>=0.13,<0.16" scikit-learn scipy numpy
# If a GPU-specific torch build is needed for your CUDA version, install that
# torch first, then the remaining packages.
```

## Pre-flight checklist (verify BEFORE launching the long run)

Do not assume any of these. Check each one; each has bitten reproduction runs before.

1. **GPU is visible to PyTorch:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Must print `True` and a device name. If `False`, stop and fix the environment.

2. **The GLUE and BLI input data are present in the checkout** (they ship in the repo; do not download them separately):
   ```bash
   ls submission/glue_fr/rte.valid.jsonl corpus/bilingual/bilingual_lemmas.txt
   ```
   Both must exist. If either is missing, you are on the wrong branch or the checkout is incomplete.

3. **A single checkpoint downloads and loads** before committing to all five:
   ```bash
   python -c "from huggingface_hub import snapshot_download; \
     snapshot_download('openhonest/babylm-2026-fr-92m-seed42', revision='chck_92M_epoch3', local_dir='server/checkpoints/seed42')"
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('server/checkpoints/seed42'); print('checkpoint OK')"
   ```

4. **A single GLUE cell completes end to end** as a smoke test, so a bug is caught in two minutes rather than after an hour:
   ```bash
   python scripts/run_xling_glue.py server/checkpoints/seed42 --lever Baseline --task rte --seed 42 --output_dir server/results/seed42
   ls server/results/seed42/seed42_xglue_Baseline_rte.json    # must exist and be non-empty
   ```

Only after all four pass, launch the full run.

## Running

```bash
bash server/run_evals.sh
```

The script iterates the five seeds; for each it downloads the checkpoint (if not already local), runs BLI, then runs the 25-cell GLUE grid one cell at a time. Configuration is via environment variables with sensible defaults (`SEEDS`, `LEVERS`, `TASKS`, `RESULTS_DIR`); you do not need to set any of them for the standard run.

## Preservation and resume (why it is safe to interrupt)

- **Every result is written to disk the moment it is computed.** BLI writes one JSON per seed; the GLUE grid writes one JSON per (lever, task) cell. There is no end-of-run aggregation step that could lose work.
- **The run is fully resumable.** Both harnesses skip any output file that already exists. If the process is killed, the machine reboots, or the job runs out of time, just run `bash server/run_evals.sh` again: it re-downloads nothing it has, recomputes nothing it has finished, and continues from the first missing cell.
- **The lever order puts the essential contrast first** (`Baseline`, `C`, `D+C`). If you must stop early, stopping after those three levers still yields the paper's headline result for Table 3; `A` and `B` are secondary.

## Verifying completeness

```bash
find server/results -name '*.json' | wc -l
```

Full coverage is **130 files**: 5 BLI (one per seed) + 125 GLUE cells (5 seeds x 5 levers x 5 tasks). To see per-seed coverage:

```bash
for s in 42 43 44 45 46; do echo "seed $s: $(ls server/results/seed$s 2>/dev/null | wc -l) files"; done
```

Each seed should show 26 files (1 BLI + 25 GLUE) at full coverage.

## What each output contains

- `seed{S}_bli_gpt2_en.json`: `fit`, `p@1`, `p@5`, `p@10` for the French-to-GPT-2 alignment, plus reference rows (random-orthogonal and chance) for the table.
- `seed{S}_xglue_{LEVER}_{TASK}.json`: `accuracy` for that lever/task cell. Table 3 is built from these; the key column is the `D+C` gain over `Baseline` per task.

## Returning results

The entire `server/results/` tree is small (all JSON, a few hundred KB). Return it whichever way is convenient:

- **Simplest:** tar it and send it back: `tar czf metronfr_table2_3_results.tgz server/results` and share the archive.
- **Or upload to the shared dataset** (needs write access, so only if that access has been shared with you):
  ```bash
  python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='server/results', repo_id='openhonest/babylm-2026-fr-92m-5seed-results', repo_type='dataset', path_in_repo='.', allow_patterns=['**/*.json'])"
  ```

## Troubleshooting

- **A cell prints a warning and the script continues.** That cell failed; the script does not abort on a single-cell failure. Re-running the script retries only the missing cells.
- **Out-of-memory on a small GPU.** Lower the fine-tune batch size: `run_xling_glue.py` accepts `--batch_size 8` (default 16). Edit `server/run_evals.sh` to add it to the `run_xling_glue.py` invocation.
- **A seed's checkpoint will not download.** The five repositories are `openhonest/babylm-2026-fr-92m-seed{42..46}`, revision `chck_92M_epoch3`. Confirm the token has read access and the network reaches huggingface.co.
