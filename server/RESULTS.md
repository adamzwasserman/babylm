# Prior-run numbers, for cross-checking the fresh reproduction

This runbook reproduces Section 4 from scratch. A previous five-seed run already produced most of these numbers; they are listed here only so the fresh run can be sanity-checked against them. They are not inputs to anything. Expect the fresh run to land close to these, with one deliberate exception noted below.

All values are mean +/- standard deviation across seeds 42 to 46.

| Result | Prior-run value | Note |
|---|---|---|
| **Table 1 QFrBLiMP overall (epoch 3)** | 80.12 +/- 0.32 | **Will change.** Computed with the pre-PR-#26 scorer, which counted 18 byte-identical rows as errors and mis-scored short sentences. The corrected scorer on `main` will read differently (higher once the 18 degenerate rows are excluded; it also reports a 98.98% ceiling). Treat the fresh number as authoritative. |
| QFrBLiMP buckets (epoch 3) | syn 82.26, sem 83.22, morph 77.77, angl 78.80 | same caveat as above |
| QFrBLiMP trajectory (epochs 1-5) | 77.41, 79.97, 80.12, 81.24, 81.33 | same caveat; the shape (rise to ~epoch 2, then slow climb) should survive |
| **Table 2 BLI** | fit 0.442, p@1 66.93, p@5 91.16, p@10 97.67 | should reproduce closely |
| **Table 3 CL-GLUE, D+C minus Baseline** | BoolQ +3.6, RTE +1.4, MRPC +2.9, WSC -2.6, MNLI -10.5 | partial prior run (3 seeds); relational gains and the MNLI regression should reproduce |
| Leaderboard (English zero-shot) | BLiMP 51.2, EWoK 49.3, GlobalPIQA-nonpar 50.0 | near chance, as expected for a French model |
| Leaderboard GLUE average | 63.0 | per task in the prior run: BoolQ 66.5, RTE 59.7, MRPC 70.5, WSC 65.0, MNLI 49.2, MultiRC 58.3, QQP 71.9 |
| QFrCoLA | acc 71.9, MCC 0.25 | should reproduce closely |
| Compute | 5.17e17 FLOPs/seed, ~215 TFLOP/s (on H200) | throughput will differ on other hardware; FLOPs/seed is hardware-independent |

The one number to expect to move is Table 1 (QFrBLiMP), and that is by design: the fresh run uses the corrected scorer. Everything else is a consistency check.
