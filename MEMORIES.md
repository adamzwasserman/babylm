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

English: "Born Speaking French: Why the Crib Beats the Cluster When the Language is Right"
French:  "La langue de Molière, quatre cents ans plus tard : toujours redoutable"

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
- MASTER_PLAN.md written
- CLAUDE.md and MEMORIES.md written
- Directory structure created
- requirements.txt + pyproject.toml written
- .gitignore written
- Initial data pipeline scripts:
  - scripts/setup.sh
  - scripts/download_babylm_corpus.py
  - scripts/download_childes_french.py
  - scripts/build_creole_oracle.py
  - scripts/build_bilingual_lemmas.py
  - scripts/analyze_caillou_oracle.py
  - scripts/build_french_corpus.py  <-- CRITICAL: assembles final corpus
  - scripts/count_words.py
- Training pipeline:
  - scripts/build_tokenizer.py     <-- pre-trains shared BPE tokenizer
  - scripts/train.py               <-- single-seed training (wandb-aware)
  - scripts/_checkpoint_schedule.py
  - scripts/run_multi_seed.sh      <-- N-GPU parallel seeds
  - scripts/aggregate_seeds.py     <-- mean +/- std across seeds
- Cloud / remote helpers:
  - scripts/cloud_setup.sh
  - scripts/deploy_to_vast.sh
  - scripts/sync_checkpoints.sh
- Tests under tests/ (pytest, lint via ruff). Pure helpers covered without
  requiring the GPU stack.
- Submitted leaderboard checkpoint chck_92M_epoch3 trained (single seed,
  pre-seeding fix; not bit-reproducible from current train.py)
- paper/prior_work/README.md written (points to fractal-language)

### Not Yet Done
- 5-seed run on a 3xRTX-4000-Ada server (waves of 3 then 2)
- Aggregated paper table from per-seed eval JSONs
- Eval pipeline not yet cloned (waiting on April 2026 release)

### Next Immediate Actions
1. On the 3-GPU server: `bash scripts/cloud_setup.sh` (or local venv install)
2. `export WANDB_API_KEY=...` and `wandb login`
3. `bash scripts/run_multi_seed.sh 1 2 3 4 5`  (waves of 3 across GPU 0/1/2)
4. Clone BabyLM eval pipeline once released
5. Run eval against each `models/seed{S}/chck_*M/` checkpoint
6. `python scripts/aggregate_seeds.py 'eval_results/seed*.json' --latex`

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

---

## What to Update After Each Session

- Add a new entry to Session Log above
- Update "Current State of Work" section
- Note any architectural decisions made
- Note any results obtained
