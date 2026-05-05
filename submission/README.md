# Submission — Data Supporting the Paper

This directory contains everything the paper cites. It is intended as the reproducibility package: anyone reading the paper should be able to verify every claim from the contents of this folder plus the upstream `corpus/`, `scripts/`, and `eval/` directories.

## Contents

### `model/`

| Directory | What it is | Where cited |
|-----------|-----------|-------------|
| `chck_92M_epoch1/` | Primary submission after 1 epoch (the reference checkpoint) | §3, §4.1, §5.3 |
| `chck_92M_epoch2/` through `chck_92M_epoch5/` | Multi-epoch trajectory | §4.2 trajectory table |
| `chck_92M_epoch3/` | Grammatical-competence peak; **primary submission model** | §4.2, §5 |
| `tokenizer/` | 50K BPE French Wikipedia tokenizer | §3.2 |
| `v2_ablation/` | Reallocated corpus (v2), regressed on every metric | §6.1 v1 vs v2 |
| `v3d_ablation/` | Minimal-ablation isolating tokenizer as cause of v2 regression | §6.4 |
| `full_ft_comparison/` | Full fine-tune on BoolQ for LoRA-vs-FullFT comparison | §5.3 |

### `adapters/`

Per-task LoRA adapters corresponding to the E3 configuration (rank 16, French-translated training data, 3 epochs) reported in §5.2 experiment grid.

### `eval_logs/`

Raw evaluation outputs referenced by each paper table:
- `FINAL_RESULTS.md` — canonical summary
- `full_eval_grid.txt` — complete experiment grid
- `dict_axioms_placebo_correction.md` — §6.2 placebo correction
- `dict_axioms_summary.md` — §6.2 original (retracted) result
- `qfrblimp_epoch3_full_eval.log` and `qfrblimp_epoch4_full_eval.log` — full eval outputs for epochs 3 and 4

### `tables/`

CSVs backing each paper table (to be populated; the source data is currently in `eval_logs/`).

### `glue_fr/`

French translations of the BabyLM 2025 evaluation pipeline's `glue_filtered/` splits, used in the §5.2 cross-lingual GLUE experiment grid (E3 configuration: French-translated training data plus rank-16 LoRA). One JSONL file per split, original alignment targets preserved:

- `boolq.{train,valid}.jsonl`
- `mnli.{train,valid}.jsonl`
- `mrpc.{train,valid}.jsonl`
- `rte.{train,valid}.jsonl`
- `wsc.{train,valid}.jsonl`

Produced by `../scripts/translate_glue_fr.py` (LLM-pass EN→FR; source script at the repository root).

## Reproducing paper claims

For a specific table or claim in the paper, find the corresponding entry in `eval_logs/` and follow the script path in `scripts/` that produced it. The `scripts/` directory is at the repository root because reproduction also depends on training code, eval code, and corpus processing that are shared across submission and research.
