# MEMORIES.md — Persistent Context for Claude Code Sessions

This file carries forward everything a new Claude Code session needs to know
to pick up this project without a lengthy briefing. Update it after each
significant session.

---

## Who Adam Is

Adam Zachary Wasserman. Independent researcher, fractional CTO (Sizzl),
founder of Buckler AI. 30+ years enterprise technology. Former CTO at IATA,
Director of Consulting at CGI. Self-taught, no formal degree. Treat as a
senior peer with deep implementation fluency across the full stack.

Fluent in French. Based in Kansas City.

Active on Hacker News as adamzwasserman. Builds karma through substantive
historical and conceptual commentary. Two style rules for any drafted text:
no dashes or em-dashes, no "X is not Y, it's Z" constructions.

---

## What This Project Is

A submission to the BabyLM Challenge at EMNLP 2026 (Budapest, Oct 24-29).

BabyLM is a shared task where participants train language models on ≤100M words
(approximating the linguistic exposure of a human child by age 13). The core
question: can you close the efficiency gap between humans and LLMs?

Adam's answer: yes, if you train in French instead of English.

---

## The Core Experimental Claim

French morphological redundancy delivers denser grammatical learning signal
per token than English. Prior controlled ablations (fractal-language project)
showed:

- French reaches 100% grammatical competence at 197M tokens
- English never reaches it, stuck at 40% (chance) after 3B tokens
- Perplexity gap: French ~27 vs English ~1340 at same training budget
- These results are pre-registered on OSF (sj48b, pcx2d)

This paper extends those results to child-scale (≤100M words) for BabyLM.

---

## The Haitian Creole Oracle

Haitian Creole derives ~90% of its lexicon from French but is morphologically
simpler (more analytical). The words that survived pidginization are the
load-bearing high-frequency composable core of French vocabulary.

We use HC as a vocabulary oracle: extract top 100-200 HC lemmas, map to French
cognates, oversample French sentences rich in those lemmas.

CRITICAL: HC text does NOT enter the training corpus. Prior ablations showed
analytical language contact drags down morphologically rich languages. HC is
a filter only.

---

## Paper Title

English: "Right Tool, Right Job: Why Training Language Matters More Than Training Data"
French:  "Les bons outils font les bons ouvriers"

Submitted model name: MÉTRON-FR.

The earlier working title was "Born Speaking French: Why the Crib Beats the
Cluster When the Language is Right" with French subtitle "La langue de Molière,
quatre cents ans plus tard : toujours redoutable". The reframing during writing
was about the framing, not the experiment.

"Morphology Eats Scale for Breakfast" is the title of a DIFFERENT paper
(the 7-language ablation from fractal-language). Do not confuse them.

---

## Deadlines

- Early April 2026: BabyLM eval pipeline + baselines released -- grab immediately
- May 25, 2026: ARR submission deadline (preferred)
- Mid-July 2026: Direct submission deadline (fallback)

---

## Current State of Work

### Done
- MASTER_PLAN.md, CLAUDE.md, MEMORIES.md written
- Directory structure, requirements.txt, pyproject.toml, .gitignore
- Data pipeline scripts:
  - scripts/setup.sh, cloud_setup.sh, deploy_to_vast.sh, sync_checkpoints.sh
  - scripts/download_babylm_corpus.py
  - scripts/download_childes_french.py (creds via env vars)
  - scripts/build_creole_oracle.py (top-300 HC lemmas)
  - scripts/build_bilingual_lemmas.py (73-lemma FR/EN bridge)
  - scripts/build_french_corpus.py (oracle-weighted, harnesses A/B/C)
  - scripts/analyze_caillou_oracle.py (convergence check)
  - scripts/count_words.py
- Training pipeline:
  - scripts/build_tokenizer.py        <-- pre-trains shared BPE tokenizer
  - scripts/train.py                  <-- 125M GPT-2 trainer (single seed,
                                          wandb-aware, linear warmup +
                                          cosine decay, BabyLM ckpt cadence)
  - scripts/_checkpoint_schedule.py
  - scripts/run_multi_seed.sh         <-- N-GPU parallel seeds
  - scripts/aggregate_seeds.py        <-- mean +/- std across seeds
- Tests under tests/ (pytest, ruff). Pure helpers covered without GPU stack.
- Submitted model: chck_92M_epoch3 (grammatical-competence peak), 5 epochs
  on 92.5M French words (single seed, pre-seeding fix; not bit-reproducible
  from current train.py)
- paper/prior_work/README.md written (points to fractal-language)

### Not Yet Done (post-submission cleanup)
- Tokenizer-ablation harness (the +7.7pp finding in §6.4 of the paper) is
  not in this repo: 16K CDS / 50K Wiki / 50K CDS / 64K CDS sweep was run
  out-of-tree and needs to be reproduced as scripts/run_tokenizer_swap.py
- §6 negative-result ablations: dict-axioms placebo, EWoK 4 interventions,
  v3 minimal ablation

### Done (Session 3, 2026-05-05/06)
- Phase 1-3 evals: QFrBLiMP per-epoch + best-epoch selection, QFrCoLA
- Phase 4: BabyLM 2025 suite via shell wrapper that auto-clones the pipeline,
  patches sentence_zero_shot/dataset.py and finetune/trainer.py for our
  tokenizer-only checkpoints, harvests results.txt + best_temperature_report.txt
  by snapshot diff (path, mtime).
- Phase 5: BLI Procrustes from local corpus/bilingual/bilingual_lemmas.txt.
- Phase 6: Cross-lingual GLUE LoRA grid (5 levers x 5 tasks). Now parallel
  across GPUs in waves of N_GPUS, resumable via per-cell skip-if-exists.
- Phase 7: aggregate_paper_tables.py emits Tables 1-3 + trajectory subtable.

### Operational gotchas captured
- QFrBLiMP HF dataset card declares qfrblimp.jsonl but doesn't ship it;
  default fetched from raw GitHub at davebulaval/QFrBLiMP/datastore/.../release/.
- QFrBLiMP schema: sentence_a / sentence_b / category (4-bucket aligned with
  the paper). 1761 pairs total.
- QFrCoLA on graalul/qfrcola: sentence / label / category.
- BabyLM eval pipeline does not ship eval_blimp.sh etc.; the real entry
  points are eval_zero_shot.sh and eval_finetuning.sh. Pipeline expects
  evaluation_data/ from OSF ryjfm and EWoK generated locally via the
  pipeline's dl_and_filter.py.
- Pipeline's AutoProcessor.from_pretrained fails on our tokenizer-only
  checkpoints; we wrap it in a try/except fallback to AutoTokenizer (idempotent
  post-clone patch in scripts/eval_babylm_suite.py).
- Pipeline's finetune trainer needs pad_token; same patcher sets it from
  eos -> bos -> unk -> registers [PAD].
- train.py now wraps the BPE tokenizer in GPT2TokenizerFast directly so
  AutoTokenizer can reload it; older checkpoints need their
  tokenizer_config.json patched (one-shot script in conversation log).
- Cross-lingual GLUE FR shipped under submission/glue_fr/{task}.{train,valid}.jsonl.
  RTE schema diverges from super_glue (sentence1/sentence2 vs premise/hypothesis);
  TaskSpec carries fr_text_a_field/fr_text_b_field overrides for that case.

---

## Architecture Decisions (Locked)

- 125M GPT-2 style (matches fractal-language ablations for comparability)
- Joint 50k BPE tokenizer, French only
- Batch=32, seq=512
- Same hyperparameters as fractal-language for direct result comparison

---

## Key Files in Related Project

/Users/adam/dev/fractal-language/
  MASTER_PLAN.md       -- full multi-paper research program
  CLAUDE.md            -- training infrastructure docs
  RESULTS_WIKI.md      -- experimental results log
  paper/               -- draft of "Morphology Eats Scale for Breakfast"
  scripts/             -- training, eval, probe scripts (reuse where possible)

---

## Corpus Sources (Planned)

1. CHILDES French (via pylangacq) -- child-directed speech gold standard
2. CC-100 French / OSCAR French -- filtered web text for volume
3. French OpenSubtitles -- conversational register
4. French children's books (Gutenberg) -- CDS-adjacent
5. French Wikipedia (filtered) -- accessible prose

Target: 90-100M words, 100% French, morphologically dense.
Use scripts/count_words.py to track budget at every step.

---

## Session Log

### Session 1 (2026-03-30, via claude.ai)
- Recovered project context from scattered conversation history
- Confirmed BabyLM is the right venue (Budapest, mid-July deadline)
- Locked title and anti-hegemony framing
- Created directory structure and initial scripts
- Wrote CLAUDE.md and MEMORIES.md

### Session 2 (2026-05-04, branch feature/multi-seed-wandb)
- Added multi-seed training infrastructure on top of audit/lint-and-tests:
  - train.py: per-seed output_dir, --tokenizer_dir decoupled,
    wandb logging (loss/ppl/lr/tokens-per-sec/words/checkpoint events),
    try/finally around the loop so wandb.finish() runs on crashes
  - extracted compute_next_ckpt_idx + CHECKPOINT_WORDS to
    scripts/_checkpoint_schedule.py so the regression test runs without torch
  - scripts/build_tokenizer.py: one-shot tokenizer pre-train to avoid
    parallel-seed races on tokenizer.json
  - scripts/run_multi_seed.sh: waves of N seeds (auto-detected from
    nvidia-smi -L) with CUDA_VISIBLE_DEVICES, EXIT/INT/TERM trap to kill
    children, fail-fast on N_GPUS=0
  - scripts/aggregate_seeds.py: mean +/- std across per-seed eval JSONs
    (markdown + optional LaTeX), UTF-8 safe
- Audit-fix on the new code: 5 root-cause bugs fixed and 23 new tests added
  (51 pass + 1 skip when torch missing locally)
- CLAUDE.md and MEMORIES.md aligned to current code state

### Session 3 (2026-05-05/06, on caribou — 5 seeds full §4 reproduction)
- Trained 5 seeds (42, 43, 44, 45, 46) x 5 epochs on 91M French words.
- Wired all eval harnesses to real datasets (graalul/qfrcola, davebulaval
  GitHub for QFrBLiMP, local files for BLI seed-dict and FR-translated GLUE).
- Wrote the BabyLM 2025 pipeline wrapper (scripts/eval_babylm_suite.py)
  that auto-clones, auto-patches, and harvests by snapshot mtime diff.
- Added phase 6 parallelisation across GPUs and per-cell skip-if-exists for
  resumable multi-hour runs.
- Per-language column overrides on TaskSpec (fr_text_a_field/fr_text_b_field)
  for RTE, where the FR translation uses sentence1/sentence2.
- One-shot patches for already-saved checkpoints documented in the
  conversation log (config.json tokenizer_class, tokenizer_config.json fields).

---

## What to Update After Each Session

- Add a new entry to Session Log above
- Update "Current State of Work" section
- Note any architectural decisions made
- Note any results obtained
