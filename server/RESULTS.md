# MÉTRON-FR §4 evaluation: what is done, what remains

A clean five-seed run (seeds 42, 43, 44, 45, 46; 125M French GPT-2, ~92M words, epoch-3 grammatical-peak checkpoint) has already been completed for most of Section 4. This document is the state of that run. The remaining items are what `server/run_evals.sh` and two small follow-up harnesses cover.

All raw per-seed result files are public and browsable at:

- **Checkpoints:** `openhonest/babylm-2026-fr-92m-seed{42,43,44,45,46}` on the Hugging Face Hub (branch `chck_92M_epoch3` is the reported checkpoint; `chck_92M_epoch{1..5}` are the per-epoch checkpoints behind the trajectory figure).
- **Eval results:** dataset `openhonest/babylm-2026-fr-92m-5seed-results`, one directory per seed.

All numbers below are mean +/- standard deviation across the five seeds unless noted.

## Done

### Table 1 — QFrBLiMP (native Quebec-French grammar), epoch 3
| Bucket | Score |
|---|---|
| Overall | 80.12 +/- 0.32 |
| Syntactic | 82.26 +/- 0.92 |
| Semantic | 83.22 +/- 0.78 |
| Morphological | 77.77 +/- 0.91 |
| Anglicism-related | 78.80 +/- 1.65 |

Per-epoch trajectory (for the Figure 1 rebuild): 77.41, 79.97, 80.12, 81.24, 81.33 (epochs 1 to 5). Grammatical accuracy reaches ~80% by epoch 2 and rises slowly thereafter while training loss keeps falling.

Note on the headline number: this is measured on the current public QFrBLiMP release. It reads ~5 points below the earlier reported 85.97%, which was computed on a since-withdrawn release of the benchmark; the current release is the reproducible one.

### Table 2 — BLI Procrustes (French embeddings to GPT-2), 5 seeds, DONE
| | Score |
|---|---|
| fit | 0.442 +/- 0.002 |
| p@1 | 66.93 +/- 7.46 |
| p@5 | 91.16 +/- 5.26 |
| p@10 | 97.67 +/- 2.15 |

Consistent with the previously reported single-run figures (fit ~0.40, p@1 ~68.8).

### Official BabyLM leaderboard suite (English zero-shot; near chance for a French model, as expected)
BLiMP 51.18 +/- 0.62, BLiMP-Supplement 46.44 +/- 0.60, EWoK 49.25 +/- 0.75, GlobalPIQA-nonparallel 50.00 +/- 2.55, comps 50.34 +/- 0.28. GlobalPIQA-parallel (19.03) and entity-tracking (18.92) are multi-way tasks with a lower baseline.

GLUE (English, frozen-LoRA rank 16), per task: BoolQ 66.45, RTE 59.71, MRPC 70.49, WSC 65.00, MNLI 49.22, MultiRC 58.26, QQP 71.90; **GLUE average 63.00**.

### QFrCoLA (native acceptability, fine-tuned)
Accuracy 71.91 +/- 0.22, MCC 0.247 +/- 0.007 (in-domain test).

### QFrCoRE / QFrCoRT (COLE idiom and regional-term probes, new)
Accuracy 5.59 +/- 0.32 and 2.57 +/- 0.32, both below the 10% chance level. This is unusual and flagged for validation: the harness is new, and a systematic below-chance result on a 10-way task warrants a scoring check before it is treated as a finding. The prior expectation (from the paper) was near chance.

### Training compute (per seed)
123.8M parameters, 695.7M tokens over 5 epochs, 5.17e17 training FLOPs (6ND), ~40 min at ~215 TFLOP/s on an H200.

## In progress / partial

### Table 3 — cross-lingual GLUE grid
The five-lever (Baseline, A, B, C, D+C) by five-task (BoolQ, RTE, MRPC, WSC, MNLI) grid is partially complete. The essential contrast (Baseline, C, D+C) is finished for three seeds; on those, the D+C-minus-Baseline gradient reproduces the paper's pattern:

| Task | D+C − Baseline (pp) |
|---|---|
| BoolQ | +3.59 +/- 0.80 |
| RTE | +1.36 +/- 3.70 |
| MRPC | +2.86 +/- 0.37 |
| WSC | -2.56 +/- 4.84 |
| MNLI | -10.50 +/- 1.49 |

Relational tasks (BoolQ, MRPC) gain; the world-knowledge task MNLI regresses sharply; WSC is noisy. `server/run_evals.sh` completes this to the full five-seed, five-lever grid (it skips the cells already present).

## Not started (no harness yet; separate, cheaper follow-ups)

- **Translated BLiMP and Translated BLiMP-Supplement** (the French machine-translated BLiMP, and the per-paradigm breakdown). A zero-shot eval; needs its translated data and a harness.
- **LoRA-preservation table** (the demonstration that a downstream LoRA adapter leaves the QFrBLiMP score bit-identical). A small script; needs writing.

## One open question for validation

The QFrCoRE/QFrCoRT below-chance result is the one number here that should be double-checked (scoring direction / length normalization in the new harness) before it goes into the paper. Everything else is either consistent with prior results or is the expected near-chance sanity check.
