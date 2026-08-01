# Runbook: full MÉTRON-FR §4 reproduction from scratch

This runbook reproduces all of Section 4 of the MÉTRON-FR paper from scratch on a single server: it trains the five seeded French models and runs every evaluation, ending in the paper's Tables 1 to 3, the leaderboard suite, and the compute numbers. It is written to be executed end to end by an automated agent with no prior context. Follow it in order. Do not skip the pre-flight.

It runs the repository's own orchestrator, `scripts/run_paper_part1.sh`, on the `main` branch, so every number uses the current, tested evaluation code (including the corrected QFrBLiMP scorer merged in PR #26).

## What this produces

`scripts/run_paper_part1.sh` runs seven phases, each of which skips itself if its output already exists:

1. **Train** five seeded checkpoints, five epochs each (42, 43, 44, 45, 46) -> `models/seed{S}/chck_92M_epoch{1..5}/`. The paper's reported model is the epoch-3 checkpoint; phase 2 picks it automatically and links it as `models/seed{S}/best`. See Running for the required `--epochs 5`.
2. **QFrBLiMP** zero-shot, every epoch -> `eval_results/seed{S}_qfrblimp_epoch{1..5}.json` (Table 1 + the trajectory figure)
3. **QFrCoLA** fine-tune + MCC -> `eval_results/seed{S}_qfrcola.json`
4. **BabyLM suite** -> `eval_results/seed{S}_babylm.json`. Zero-shot: BLiMP, BLiMP-Supplement, EWoK, entity-tracking, wug adjective-nominalization, wug past-tense, COMPS, and reading (eye-tracking / self-paced). Fine-tune: (Super)GLUE (BoolQ, RTE, MRPC, WSC, MNLI, MultiRC, QQP). (GlobalPIQA is not part of this suite; it was a separate leaderboard-submission task and is not a paper table.)
5. **BLI Procrustes** -> `eval_results/seed{S}_bli_*.json` (Table 2)
6. **Cross-lingual GLUE grid** (5 levers x 5 tasks) -> `eval_results/seed{S}_xglue_*.json` (Table 3). This is frozen-base LoRA: `run_xling_glue.py` wraps the French base in a PEFT adapter and never fine-tunes the base, which is the paper's methodology. (The leaderboard (Super)GLUE inside phase 4 is a different evaluation: it runs the official BabyLM pipeline's full fine-tune, as the leaderboard requires. The two are intentionally not the same procedure.)
7. **Aggregate** -> `paper_tables.tex` and `paper_tables.md` (mean +/- std across the five seeds)

## Hardware and time

- One or more CUDA GPUs. Training and the GLUE grid parallelise across all visible GPUs in waves; more GPUs is faster, one GPU works. The model is 125M parameters, so a 16 to 24 GB card is enough.
- Wall-clock is dominated by training (five models, five epochs each on ~92M words) and by the GLUE grid (125 small fine-tunes). Budget several hours to a day depending on GPU count. The run is fully resumable, so it does not need to finish in one sitting.

## Prerequisites

- Python 3.11 or newer, in a fresh virtual environment. The repo's `pyproject.toml` declares `requires-python >=3.13`; match 3.13 to the authors' environment if you can.
- A working C compiler (`build-essential` on Debian/Ubuntu). `scripts/train.py` calls `torch.compile` on CUDA, which builds Triton kernels and fails without one. A normal GPU dev box has this; a minimal container may not.
- A Hugging Face account with `huggingface-cli login` completed. Needed because the EWoK evaluation subset is gated (request access to `ewok-core/ewok-core-1.0` if you have not before) and, if you choose to upload results, for write access.
- Outbound access to huggingface.co, github.com, osf.io, and raw.githubusercontent.com.
- Optional: a Weights & Biases account. If you do not want it, training runs with `--wandb_mode disabled` (see Running).

## One-time setup

```bash
git clone https://github.com/adamzwasserman/babylm.git
cd babylm
git checkout david-reproduction        # this branch; based on main

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt         # authoritative dependency list (torch, transformers,
                                        # datasets, peft, scikit-learn, accelerate, spacy, etc.)
pip install osfclient nltk              # needed by the eval-pipeline setup; not in requirements.txt

huggingface-cli login                   # paste a token; needed for gated EWoK

bash server/setup.sh                    # corpus + tokenizer + eval-pipeline data
```

`server/setup.sh` is idempotent. It downloads the published French corpus to `corpus/final/train_french.txt`, builds the shared 50k tokenizer at `models/tokenizer/`, verifies the GLUE and BLI data that ship in the repo, and clones the BabyLM eval pipeline plus its OSF data for the suite phase. It prints `SETUP OK` only when all inputs are present. Do not proceed past a non-OK result.

**Environment note (transformers version).** The BabyLM eval pipeline pins `transformers==4.51.3`, `tokenizers==0.21.1`, and `torch==2.7.0` into this same virtual environment, so that is the version you reproduce under. The paper's original runs used a newer transformers via a lock file that is not committed here; the difference that moved the reported QFrBLiMP numbers was the scorer correction in PR #26, not the transformers version, so 4.51.3 is the accepted reproduction environment. Do not upgrade transformers to a 5.x release in this venv; the GPT-2 loader used here can break on it.

## Pre-flight checklist (run BEFORE the long job; each has bitten runs before)

1. **GPU visible to PyTorch:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
   ```
   Must print `True` and a count of at least 1.

2. **Corpus present and within budget:**
   ```bash
   wc -w corpus/final/train_french.txt      # expect ~92.5M, must be < 100M
   ```

3. **Tokenizer present:**
   ```bash
   test -f models/tokenizer/tokenizer.json && echo OK
   ```

4. **Eval-pipeline data present (for phase 4):**
   ```bash
   ls eval/evaluation-pipeline-2025/evaluation_data >/dev/null && echo OK
   ```

5. **Training smoke, one seed, one epoch, to a throwaway directory.** This exercises the whole training path in a few minutes and catches an environment problem before you commit to the full five-model run:
   ```bash
   head -c 5000000 corpus/final/train_french.txt > /tmp/smoke_corpus.txt
   python scripts/train.py --seed 99 --epochs 1 --corpus /tmp/smoke_corpus.txt \
     --tokenizer_dir models/tokenizer --output_dir /tmp/train_smoke --wandb_mode disabled
   ls /tmp/train_smoke/chck_*_epoch1/config.json && echo "TRAIN SMOKE OK"
   rm -rf /tmp/train_smoke /tmp/smoke_corpus.txt
   ```
   Do not reuse seed 99 or `/tmp/train_smoke` for anything real; this only proves the path works. The real seeds are 42 to 46 and write to `models/seed{S}/`.

Only after all five pass, launch the full run.

## Running

The paper trains each seed for five epochs and reports the epoch-3 checkpoint (`chck_92M_epoch3`) as the grammatical-competence peak. `scripts/train.py` defaults to a single epoch and the orchestrator does not override it, so you MUST pass `--epochs 5` through `TRAIN_EXTRA`. This is not optional: without it phase 1 trains one epoch, phase 2 produces only an `epoch1` file per seed, the best-epoch pick collapses to epoch 1, and the epoch-3 headline plus the five-epoch QFrBLiMP trajectory can never be built. The run would appear to succeed while reproducing the wrong model.

```bash
TRAIN_EXTRA="--epochs 5" bash scripts/run_paper_part1.sh 42 43 44 45 46
```

To also run without Weights & Biases:

```bash
TRAIN_EXTRA="--epochs 5 --wandb_mode disabled" bash scripts/run_paper_part1.sh 42 43 44 45 46
```

Useful controls (all optional):
- `N_GPUS=3` forces the training and GLUE-grid parallelism width (default: auto-detected from `nvidia-smi`).
- `PHASE=4` runs only one phase (1 to 7). Handy for re-running a single stage.
- `SKIP_PHASES="6"` skips one or more phases (space-separated).

## Resume and preservation (why interruption is safe)

- **Every phase is idempotent.** Training skips any seed that already has a `models/seed{S}/chck_*M` checkpoint; each eval phase skips any seed whose output JSON already exists; aggregation reads whatever is present. If the process stops for any reason, re-run the exact same command and it continues from the first missing piece. Nothing is recomputed.
- **Results land incrementally.** Per-seed, per-epoch JSONs are written to `eval_results/` as each finishes, and checkpoints to `models/seed{S}/` as each seed trains. There is no single end-of-run step that could lose everything.
- **Back up as you go.** After each phase, or at the end, copy `eval_results/`, `paper_tables.*`, and the checkpoints somewhere durable (see Returning results). Do not rely on the working directory alone.

## Verifying completeness

After the run, confirm the expected outputs exist:

```bash
ls models/seed{42,43,44,45,46}/chck_*M >/dev/null && echo "checkpoints OK"
ls eval_results/seed42_qfrblimp_epoch{1,2,3,4,5}.json >/dev/null && echo "QFrBLiMP epochs OK"
for s in 42 43 44 45 46; do
  echo "seed $s: qfrcola=$(ls eval_results/seed${s}_qfrcola.json 2>/dev/null | wc -l) babylm=$(ls eval_results/seed${s}_babylm.json 2>/dev/null | wc -l) bli=$(ls eval_results/seed${s}_bli_*.json 2>/dev/null | wc -l) xglue=$(ls eval_results/seed${s}_xglue_*.json 2>/dev/null | wc -l)"
done
test -f paper_tables.tex && test -f paper_tables.md && echo "TABLES OK"
```

At full coverage each seed shows: 5 QFrBLiMP epoch files, 1 QFrCoLA, 1 BabyLM suite, at least 1 BLI, and 25 xglue cells. The two final artefacts are `paper_tables.tex` and `paper_tables.md`.

## Returning results

The whole point is the aggregated tables plus the raw per-seed JSONs. Return:

- `paper_tables.tex` and `paper_tables.md` (the tables themselves), and
- the `eval_results/` tree (all JSON, a few hundred KB).

Simplest: `tar czf metronfr_s4_results.tgz paper_tables.* eval_results/` and send the archive. The trained checkpoints are large; keep them on the server unless asked, but note their location.

## Out of scope for this run

Three paper items are deliberately not produced here, to keep the run clean and well-defined:

- **Translated BLiMP** and **Translated BLiMP-Supplement**: a separate zero-shot eval that needs its translated data and harness.
- **The LoRA-preservation table**: a small demonstration script, not yet written.
- **QFrCoRE / QFrCoRT** (the COLE idiom and regional-term probes): a new, exploratory harness whose current results sit below chance and need a scoring review before use.

These are follow-ups, not part of this reproduction.

## Troubleshooting

- **A GPT-2 import or load fails right after install.** `requirements.txt` leaves `transformers` unpinned, so a plain `pip install -r requirements.txt` can pull a much newer major that breaks the GPT-2 loader used here. The BabyLM eval pipeline that `server/setup.sh` installs pins `transformers==4.51.3`, `tokenizers==0.21.1`, `torch==2.7.0` into the same environment, so running `server/setup.sh` before the job usually settles the versions. If a repo script still fails to load the model, pin explicitly: `pip install "transformers==4.51.3" "tokenizers==0.21.1"`.
- **Training dies before the first step with a compiler or Triton error.** `scripts/train.py` calls `torch.compile` on CUDA and needs a working C compiler to build Triton kernels. Install one (`apt-get install -y build-essential`) and re-run. The pre-flight training smoke exercises `torch.compile`, so it catches this before the full run.
- **Phase 4 crashes at import with `torchvision::nms does not exist`.** This is a torch/torchvision version mismatch dragged in alongside the eval-pipeline install; the text eval never uses vision. Remove it and re-run the phase: `pip uninstall -y torchvision`.
- **GLUE fine-tuning fails with "tokenizer does not have a padding token".** This is handled automatically: `scripts/eval_babylm_suite.py` idempotently patches the pipeline's `finetune/trainer.py` to set a `pad_token` (eos, then unk) on first run. If you see this, you invoked a pipeline script directly instead of through the wrapper. Run phase 4 via the orchestrator (or `scripts/eval_babylm_suite.py`) so the patch is applied.
- **`setup.sh` prints WARN on EWoK.** The EWoK subset is gated. Run `huggingface-cli whoami` to confirm you are logged in, and request access to `ewok-core/ewok-core-1.0`. Without it, phase 4 still runs but reports EWoK as unavailable rather than as a null result.
- **OSF download fails.** `pip install osfclient`, then re-run `server/setup.sh`; it only re-fetches what is missing.
- **Out-of-memory during training or GLUE.** Lower batch size: `TRAIN_EXTRA="--batch_size 16"` for training; for the GLUE grid, `run_xling_glue.py` accepts `--batch_size 8` (edit is only needed if OOM occurs).
- **A phase fails partway.** Re-run the same `run_paper_part1.sh` command; completed seeds and cells are skipped, so it resumes. To force a single phase, use `PHASE=N`.
- **Training seems to re-run seeds you already have.** The skip check looks for `models/seed{S}/chck_*M`. Confirm the checkpoints are under exactly that path; a partial or moved checkpoint directory will trigger a retrain.
