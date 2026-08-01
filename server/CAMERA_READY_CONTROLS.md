# Camera-ready controls and pre-registered criteria

These are the additional controls and decisions agreed for the camera-ready revision, on top of the base reproduction in `RUNBOOK.md`. They run on the same five checkpoints (and, where noted, the English baseline).

## 1. MNLI fine-tuning-scale control (rules out a scale confound in the §4.4 gradient)

**Why.** The CL-GLUE gradient (relational tasks gain, MNLI regresses ~-11pp) is only a *task-type* effect if it is not explained by training-set size. MNLI has ~393k training examples; MRPC ~3.7k, RTE ~2.5k. A frozen-base LoRA can plateau first on the largest task, which would make the MNLI regression a fine-tuning-scale artifact, not a world-knowledge effect.

**What to run.** `run_xling_glue.py` now takes `--max_train N`, which caps and reshuffles the training split (per-seed) and writes to a distinct file (`seed{S}_xglue_{lever}_{task}_max{N}.json`; the `+` in a lever name is stripped, so `D+C` becomes `DC` -> `seed{S}_xglue_DC_{task}_max{N}.json`), so it does not overwrite the full cell. Run the MNLI cell at 3000 (matching the small tasks) for `Baseline` and `D+C`, on both the French checkpoint and the English baseline, across the five seeds. The output filename encodes only seed/lever/task/max, not the model, so the English runs must go to a separate `--output_dir` or they collide with (and are skipped in favour of) the French cells of the same name:

```bash
for S in 42 43 44 45 46; do
  for L in Baseline "D+C"; do
    python scripts/run_xling_glue.py <french_ckpt_seed$S>  --lever "$L" --task mnli --seed $S --max_train 3000 --output_dir eval_results
    python scripts/run_xling_glue.py <english_baseline>    --lever "$L" --task mnli --seed $S --max_train 3000 --output_dir eval_results/english_baseline
  done
done
```

**Pre-specified decision rule (written before looking at the result).** Let `Δ_full` be the D+C-minus-Baseline MNLI delta at full training size and `Δ_3k` the delta at `--max_train 3000`, both as 5-seed means. If `Δ_3k` rises to within the band of the relational-task deltas (that is, the ~-11pp regression shrinks to a magnitude comparable to BoolQ/MRPC/RTE, no longer a distinct large negative), the MNLI regression is a fine-tuning-scale effect and §4.4 must be reframed away from a task-type gradient. If `Δ_3k` stays a large negative comparable to `Δ_full`, the task-type reading survives. We commit to reporting whichever outcome occurs.

## 2. English-side rank 8-vs-16 control, and naming the baseline

**Why.** The French side isolates LoRA rank with lever C (rank 16, English data) vs Baseline (rank 8, English data). The same control must be shown on the English-pretrained model so a rank effect cannot masquerade as the pipeline effect.

**What to run.** The full lever grid on the English baseline gives it directly (Baseline = rank 8, C = rank 16, both English data). As in control 1, the English cells share the `seed{S}_xglue_{lever}_{task}.json` naming with the French phase-6 grid (the filename does not encode the model), so send them to a separate `--output_dir` to avoid colliding with the French run:

```bash
for S in 42 43 44 45 46; do
  python scripts/run_xling_glue.py <english_baseline> --all_levers --all_tasks --seed $S --output_dir eval_results/english_baseline
done
```

**Reporting.** Name the English baseline explicitly and state how it is matched. The only matched-architecture 125M English model available is the one that plateaued near chance on English grammar (~25% BLI alignment) despite ~6.5B training tokens; name it and its budget so the comparison is not read as against a competent model.

## 3. QFrBLiMP saturation criterion (pre-registered, fixed before reading the 5-seed curve)

The 95% binomial CI half-width on n=1761 is about 1.9pp, so an epoch-to-epoch QFrBLiMP-overall change below 1.9pp is within noise. We define, before looking at the five-seed trajectory:

- Let `m(e)` be the 5-seed mean QFrBLiMP-overall at epoch `e`, and `δ = 1.9pp`.
- **Saturation epoch `E*`** is the smallest `E in {1..4}` such that for every `e >= E`, `|m(e+1) - m(e)| < δ`.
- The claim "grammatical competence saturates before perplexity" is supported **iff** `E*` exists **and** the mean training loss is still strictly decreasing at `E*` (loss keeps falling past the point where QFrBLiMP has stopped moving by more than `δ`).
- If no such `E*` exists within five epochs, we report "no saturation detected within five epochs" and drop the saturation claim rather than asserting a plateau.

This criterion is fixed now; the reproduction's per-epoch 5-seed QFrBLiMP numbers are then applied to it as-is.

## 4. Reconstructing the two untraceable numbers

Both are in the abstract/conclusion and currently have no provenance outside the .tex.

**Weighted leaderboard 62.80%.** The prior official run had EWoK and AoA at NULL, so a weighted score could not have been formed from it. Reconstruct by a *complete* leaderboard run: EWoK present (requires `ewok-core/ewok-core-1.0` access, see setup), AoA produced, and the official weighted metric computed. Note that the repo aggregator cannot currently assemble this (see the `aggregate_paper_tables.py` / `eval_babylm_suite.py` key-mismatch flagged separately), so the weighted number should come from an actual leaderboard submission of the collated predictions, or from a fixed aggregator. Until reconstructed, 62.80 does not stand; replace it with the reconstructed value or remove it.

**BLiMP-Supplement "34% below chance" and the medecin/chaise confound.** The translated-Supplement real value is ~96.40% (official English Supplement 44.40%, qa_congruence_easy 100%); the 34% figure and the lexical-frequency confound example have no trace in any run. Reconstruct by re-running the translated BLiMP-Supplement evaluation (this needs the translated-Supplement harness, not yet built) to obtain the real number; replace 34% with it. If the medecin/chaise confound analysis cannot be reproduced from an actual run, remove that specific claim rather than restate it.
