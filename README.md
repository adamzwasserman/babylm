# Right Tool, Right Job (BabyLM 2026)

*Right Tool, Right Job: Why Training Language Matters More Than Training Data.* Code, data pipeline, and paper sources for the BabyLM 2026 submission (model MÉTRON-FR) on French morphological efficiency at child scale.

**About.** This work tests, at child-scale data budgets, whether the structure of the training language drives grammar acquisition more than the volume of data does, a direct probe of the Language-Only Hypothesis. By **Adam Zachary Wasserman** ([ORCID](https://orcid.org/0009-0002-8865-6583), [OSF](https://osf.io/user/8t64r)) and David Beauchemin (Université Laval), part of the research program of the [Open Honest Foundation](https://openhonest.org). See also [`fractal-language`](https://github.com/adamzwasserman/fractal-language).

**Reproducing §4 from scratch.** [`server/RUNBOOK.md`](server/RUNBOOK.md) is the full from-scratch reproduction handoff: run [`server/setup.sh`](server/setup.sh) once, then `TRAIN_EXTRA="--epochs 5 --wandb_mode disabled" bash scripts/run_paper_part1.sh 42 43 44 45 46` to train five seeds and run every evaluation into the paper tables.
