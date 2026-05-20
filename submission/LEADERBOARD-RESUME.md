# BabyLM 2026 Leaderboard Submission: Resume Notes

**Status as of 2026-05-20:** First-pass eval complete; 7/9 sections populated; EWoK and AoA missing. Waiting on external disk for trajectory checkpoints before final resubmission.

---

## Where things stand

### Done

1. **Model published to HF Hub:** https://huggingface.co/openhonest/babylm-2026-fr-92m
   - Source files: `submission/model/chck_92M_epoch3/chck_92M/` (epoch 3, 92,469,402 words)
   - Model card written with paper-headline framing, MIT license
   - Tokenizer config patched on HF (`pad_token: <|padding|>` id 1, `eos/bos/unk: <|endoftext|>` id 0, `tokenizer_class: PreTrainedTokenizerFast`)
   - `config.json` patched on HF (`pad_token_id: 1`)
2. **Eval pipeline cloned locally:** `/Users/adam/dev/honest/adamzwasserman/babylm-eval/` (upstream: `github.com/babylm-org/babylm-eval`)
3. **HF Jobs eval run complete** (job `6a0de2ceccb6cd133d158a99`, a100-large, ~3 hours)
   - Results dataset: https://huggingface.co/datasets/openhonest/babylm-2026-fr-92m-strict-results
   - Local mirror: `/tmp/babylm-results/` (re-download with `hf download --repo-type dataset openhonest/babylm-2026-fr-92m-strict-results --local-dir <path>` if cleared)
   - Submission JSON: `/tmp/babylm-results/babylm-2026-fr-92m/all_full_preds_and_fast_scores_causal.json` (~16 MB)
4. **A100 run text log:** `/Users/adam/Downloads/A100 run.txt` (1158 lines)

### Populated in current submission JSON

| Section | Status |
|---|---|
| blimp | full (67 sub-tasks; overall 50.99) |
| blimp_supplement | full (5 sub-tasks; overall 44.40) |
| entity_tracking | full (18 sub-tasks; overall 13.98) |
| comps | full (4 sub-tasks; overall 50.47) |
| reading | full (Eye Tracking 0.69, Self-Paced 0.01) |
| glue | full (all 7 tasks: boolq, mnli, mrpc, multirc, qqp, rte, wsc) |
| ewok | **NULL** (dataset never downloaded; see Known Issues) |
| aoa | **NULL** (trajectory checkpoints missing; see Known Issues) |
| fast_eval_results | partial (same trajectory-checkpoint issue) |

### Headline numbers worth remembering

- **BoolQ fine-tune best epoch: 0.667 accuracy.** Strict baseline (English GPT-2): 0.6746. Gap **−0.7 pp** with a French-only model.
- BLiMP zero-shot 50.99 vs baseline 74.53; expected gap for French-only on English minimal pairs.
- QFrBLiMP 85.97% overall (from prior eval, not part of this leaderboard run; the paper's headline result).

---

## Known issues and fixes

### Bugs already discovered and resolved on disk

| Bug | Root cause | Fix |
|---|---|---|
| AoA script rejected `strict` | Script accepts only `strict-small` / `non-strict-small` | Pass `non-strict-small` for any model trained on more than 10M words |
| Finetune trainer `ModuleNotFoundError: wandb` | `wandb` imported unconditionally even when `--wandb` flag absent; not in `requirements.txt` for Python 3.12 | `pip install wandb` + `--env WANDB_MODE=disabled` in the Job |
| Job timed out at default | HF Jobs default timeout cut us off mid-GLUE | `--timeout 6h` |
| Tokenizer "no pad_token" | Model checkpoint pre-dates babylm/babylm PR #15; tokenizer_config used custom class | Patched `tokenizer_config.json` + `config.json` on HF Hub (already pushed) |
| Auth missing for final upload | HF Jobs doesn't auto-pass token | `--secrets HF_TOKEN` |

### Bugs not yet fixed

| Bug | Root cause | Fix on resume |
|---|---|---|
| **EWoK empty** | `download_evals.py` doesn't pull EWoK; needs separate gated-dataset download | Add `python -m evaluation_pipeline.ewok.dl_and_filter` between `download_evals.py` and `eval_zero_shot.sh` in the Job script. Requires `HF_TOKEN` (already passed). |
| **AoA empty** | AoA needs ~28 checkpoint revisions on HF Hub (`chck_10M`, `chck_20M`, ..., `chck_1000M`) to fit learning curves; only `main` exists | Upload trajectory checkpoints from external disk as HF revisions. See "Resume sequence" below. |
| **fast_eval_results sparse** | Same trajectory-checkpoint dependency as AoA | Same fix |

### `hf jobs logs` CLI is flaky

The CLI's log-streaming endpoint occasionally drops with `httpx.RemoteProtocolError: peer closed connection without sending complete message body`. This is a server-side SSE issue, not your CLI bug. Workaround: use the HF web UI for live log monitoring (https://huggingface.co/jobs/openhonest/<JOB_ID>), or download the log text after the job completes.

---

## Resume sequence (when external disk is back)

### Step 0: Smoke-test job (build first)

A pre-flight job that verifies every eval task before committing to a full 3-hour A100 run. Not yet written. Should:

- Install deps and clone `babylm-org/babylm-eval`
- Run `download_evals.py` and `python -m evaluation_pipeline.ewok.dl_and_filter`
- For each zero-shot task: verify data path exists, model loads with `AutoModelForCausalLM`, tokenizer has `pad_token`, one batch of 2 examples forward-passes
- For AoA: query the HF Hub for revisions matching the expected `chck_*M` schedule, list which are present, exit non-zero if too few
- For each GLUE task: verify data path, imports, one train step + one val step on 2 examples
- Report per-task PASS/FAIL with the specific failure mode

Budget: t4-small for 15 minutes (~$0.50) or l4x1 to reuse cache (~$1).

### Step 1: Mount external disk, locate trajectory checkpoints

External disk should contain checkpoints saved during the original training run at fixed token-count intervals. Look for directory naming like `chck_10M/`, `chck_20M/`, ..., `chck_100M/`, then `chck_200M/`, ..., `chck_1000M/`. Each should contain `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`.

If naming differs from `chck_<N>M`, map to those names before upload.

### Step 2: Upload trajectory as HF revisions on `openhonest/babylm-2026-fr-92m`

For each checkpoint `chck_NM`, push as a named revision (Git branch on HF Hub):

```bash
# For each checkpoint dir on the external disk:
cd /path/to/trajectory/chck_NM
hf upload openhonest/babylm-2026-fr-92m . . \
  --repo-type model \
  --revision chck_NM \
  --create-pr false \
  --commit-message "Trajectory checkpoint chck_NM"
```

Each checkpoint is ~485 MB; 28 of them is ~13.5 GB of upload. Plan for ~30–60 min wall time depending on bandwidth.

**Tokenizer config / pad_token issue:** the original training run's tokenizer_config.json might have the same `TokenizersBackend` class issue. Before uploading, run the same patch as for the main checkpoint:

```json
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "model_max_length": 512,
  "bos_token": "<|endoftext|>",
  "eos_token": "<|endoftext|>",
  "unk_token": "<|endoftext|>",
  "pad_token": "<|padding|>"
}
```

And ensure `config.json` has `"pad_token_id": 1`.

(Optional but cleaner: write a small script that walks the trajectory directory, patches each tokenizer_config.json and config.json in place, then uploads all revisions. Avoids hand-patching 28 times.)

### Step 3: Run smoke test against the now-complete repo

```bash
hf jobs run --detach --secrets HF_TOKEN --env WANDB_MODE=disabled \
  --flavor t4-small --timeout 30m \
  python:3.12 bash -c '<smoke-test-script>'
```

If smoke test passes for all tasks, proceed to Step 4. If it fails on something, fix and retest.

### Step 4: Run the full eval with EWoK fix

Same job command as the successful run, with these additions:

```bash
hf jobs run --detach --secrets HF_TOKEN --env WANDB_MODE=disabled \
  --flavor a100-large --timeout 6h \
  python:3.12 bash -c '
set -euo pipefail

MODEL=openhonest/babylm-2026-fr-92m
RESULTS_REPO=openhonest/babylm-2026-fr-92m-strict-results

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq git

echo "=== Installing Python deps ==="
pip install --quiet --upgrade pip
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet huggingface_hub wandb

echo "=== Cloning babylm-eval ==="
git clone https://github.com/babylm-org/babylm-eval /workspace/babylm-eval
cd /workspace/babylm-eval

pip install --quiet -r requirements.txt
touch .env

cd strict

echo "=== Downloading eval datasets ==="
python scripts/download_evals.py

echo "=== EWoK FIX: separate gated download ==="
python -m evaluation_pipeline.ewok.dl_and_filter

echo "=== Step 3: Zero-shot eval ==="
bash scripts/eval_zero_shot.sh $MODEL causal

echo "=== Step 4: AoA eval ==="
bash scripts/eval_aoa.sh $MODEL causal non-strict-small

echo "=== Step 5: Finetune eval ==="
bash scripts/eval_finetuning.sh --model_path $MODEL

echo "=== Step 6: Collate predictions ==="
bash scripts/collate_preds.sh $MODEL causal strict

echo "=== Uploading results to HF ==="
hf upload $RESULTS_REPO results . --repo-type dataset \
  --commit-message "BabyLM 2026 Strict-track eval results (with EWoK + AoA trajectory)"

echo "=== DONE ==="
'
```

Expected cost: ~$10 on A100.

### Step 5: Fill in the leaderboard form

Open https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026 → Submit tab.

Form values from this model (verified against `config.json`, `training_meta.json`, MASTER_PLAN.md):

| Field | Value | Required for Challenge eligibility? |
|---|---|---|
| Model name | `right-tool-right-job-fr-92m (causal)` | ⚠️ required |
| Revision commit | `main` | optional |
| Main contributions/approaches | `Training language choice (French)` | 👶 required |
| Base architecture | `GPT-2` | 👶 required |
| Learning rate scheduler | `cosine` | 👶 required |
| Number of training epochs | `3` (epoch 3 is the submitted checkpoint of a 5-epoch trajectory) | 👶 required |
| Tokenizer | `BPE 50K (French Wikipedia)` | 👶 required |
| Random seed | (look up in `training_log.csv` or `MASTER_PLAN.md`; if unknown leave blank or note "unrecorded") | 👶 required |
| Number of attention heads | `12` | 👶 required |
| Max sequence length | `512` | 👶 required |
| Approximate GPU hours for development | (your call; if unsure: ~6 for this checkpoint trajectory through epoch 3) | 👶 required |
| Training data | (since custom French corpus, leave dropdown on "not applicable" and fill custom-dataset fields) | 👶 required |
| Approximate number of words for custom dataset | `92469402` | 👶 required |
| Genre of sources | `CHILDES, Orléans corpus, French Wikipedia, OpenSubtitles (oversampled per Haitian Creole oracle)` | 👶 required |
| Preprocessing of custom dataset | `Lemma-frequency oversampling guided by Haitian Creole vocabulary oracle; documents separated with newlines; 100% French morphology preserved` | 👶 required |
| Results file (JSON) | upload `babylm-2026-fr-92m/all_full_preds_and_fast_scores_causal.json` from the results dataset | ⚠️ required |
| HuggingFace repository | `openhonest/babylm-2026-fr-92m` | 👶 required |
| Track | `strict` | ⚠️ required |
| Model type | `Decoder only` | 👶 required |
| Max learning rate | `1.0e-4` | 👶 required |
| Optimizer | `AdamW` (verify from training scripts) | 👶 required |
| Average batch size (in tokens) | `512 × 32 = 16384` (sequence_length × batch) | 👶 required |
| Token set size | `50000` | 👶 required |
| Number of layers | `12` | 👶 required |
| Total number of parameters | `~125,000,000` | 👶 required |
| Approximate number of training FLOPS | (compute from 92M tokens × 6 × params × epochs) | 👶 required |
| Approximate GPU hours for training | `~3` for epoch 3 (or `~5` for full 5-epoch trajectory) | 👶 required |
| Custom data human annotation | `Not applicable` | 👶 required |
| Custom synthetic data | `Not applicable` | 👶 required |
| Brief textual description | `125M GPT-2 causal LM trained from scratch on 92.5M words of French; primary checkpoint for "Right Tool, Right Job: Why Training Language Matters More Than Training Data" (Wasserman & Beauchemin, BabyLM 2026)` | 👶 required |
| Teacher Models | (leave blank; no distillation) | 👶 required |

### Step 6: Submit and verify

After clicking submit, verify the submission appears on the leaderboard Strict tab within a few minutes. If scores look off vs. expected, check the predictions file for the obvious mistake.

---

## Cost ledger so far

| Item | Cost |
|---|---:|
| Failed run 1 (AoA track name) | ~$0.15 |
| Failed run 2 (wandb missing) | ~$0.30 |
| Failed run 3 (job timeout, mid-GLUE) | ~$0.50 |
| Successful A100 run | ~$8 |
| **Total used** | **~$9** (per billing page: $0.93 → ~$10 after the A100 run completed) |
| **Remaining HF credit** | **~$5–6** of the original $14.23 + $0.93-already-used baseline |

If running fresh on a different machine, top up HF credits before starting. Estimate $15 for the smoke test + full re-run with trajectory.

---

## Files to look at on resume

| Path | What |
|---|---|
| `submission/LEADERBOARD-RESUME.md` | This file |
| `submission/README.md` | Submission package description |
| `submission/model/chck_92M_epoch3/chck_92M/` | The main checkpoint files |
| `submission/model/chck_92M_epoch3/eval_results.log` | QFrBLiMP / QFrCoLA results (paper headline) |
| `submission/model/chck_92M_epoch3/training_log.csv` | Training metrics per step |
| `MASTER_PLAN.md` | Project plan and strategic context |
| `CLAUDE.md` | Project Claude config |
| `submission/all_full_preds_and_fast_scores_causal.json` | Current leaderboard-submission JSON (16 MB, persisted locally) |
| `/tmp/babylm-results/` | Full local mirror of the results dataset (re-download if cleared) |
| `/Users/adam/Downloads/A100 run.txt` | Text log of the successful A100 eval run |

External resources:

- HF Hub model: https://huggingface.co/openhonest/babylm-2026-fr-92m
- HF Hub results: https://huggingface.co/datasets/openhonest/babylm-2026-fr-92m-strict-results
- Eval pipeline: https://github.com/babylm-org/babylm-eval
- Leaderboard Space: https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026
- HF Job (completed): https://huggingface.co/jobs/openhonest/6a0de2ceccb6cd133d158a99

---

## Open questions / loose ends

1. **Random seed for the training run.** The MASTER_PLAN says "submitted leaderboard checkpoint was trained with a single (unrecorded) seed." If the form's seed field is required, note "unrecorded" or pick a reasonable placeholder. The paper's 5-seed analysis is separate.
2. **EWoK gated-dataset access.** `evaluation_pipeline.ewok.dl_and_filter` requires HF authentication to a gated dataset. The token already passed via `--secrets HF_TOKEN` should suffice, but if you get a 403 on EWoK download, verify the HF account has accepted the EWoK license at https://huggingface.co/datasets/ewok-core/ewok-core-1.0 (or wherever the upstream lives).
3. **Trajectory naming convention.** Confirm the external-disk checkpoint dirs map cleanly to the `chck_<N>M` schedule the AoA script expects. If they're named differently (e.g., by epoch number or step number), build a name-mapping table before bulk upload.
4. **Pre-tokenization of EWoK?** Some BabyLM teams pre-filter EWoK to remove items containing words outside the model's tokenizer vocabulary. Check whether `dl_and_filter.py` does this automatically or whether a separate step is needed.
