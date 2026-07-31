# CLAUDE.md — Right Tool, Right Job (BabyLM 2026)

This file provides full context for Claude Code sessions in this repository.

## Project Identity

**Paper title:**
"Right Tool, Right Job: Why Training Language Matters More Than Training Data"

**French subtitle:**
"Les bons outils font les bons ouvriers"

**Submitted model:** MÉTRON-FR (125M GPT-2, French-only, 92.5M words). The earlier project name was "Born Speaking French"; the manuscript was reframed during the writing phase, but the corpus and training pipeline below are unchanged.

**Target venue:** BabyLM Workshop at EMNLP 2026, Budapest, Hungary (Oct 24-29)

**Submission track:** Strict (≤100M words, custom corpus fully allowed)

**Pre-registrations:**
- OSF sj48b: https://osf.io/sj48b (French/English cross-linguistic training dynamics)
- OSF pcx2d: https://osf.io/pcx2d (morphological complexity gradient)

---

## Core Thesis

Training exclusively on French at child-scale (≤100M words) outperforms English
baselines by 4-12x on grammatical competence benchmarks. French morphological
redundancy delivers denser learning signal per token. This falsifies the assumption
that scaling laws are language-independent.

This paper is a child-scale extension of prior work ("Morphology Eats Scale for
Breakfast") which demonstrated the same effect at 780M tokens/language across 7
languages. BabyLM forces the experiment into the 100M-word regime where the
effect should be even more pronounced.

---

## Key Prior Results (from /Users/adam/dev/fractal-language)

| Metric                  | French  | English       | Ratio |
|-------------------------|---------|---------------|-------|
| Tokens to 100% grammar  | 197M    | >3B (never)   | >15x  |
| Perplexity at 3B tokens | ~27     | ~1340         | 50x   |
| Grammar accuracy at 3B  | 100%    | 40% (chance)  | --    |

The English model was not broken. It performed exactly as Pythia and other
English-trained 125M models predict. French is the anomaly.

---

## The Haitian Creole Oracle Strategy

**What it is:** Pidginization acts as a selection filter, preserving only the
highest-frequency, most composable vocabulary. Words that survived from French
into Haitian Creole are load-bearing by evolutionary pressure.

**How we use it:** Extract the top 100-200 high-frequency lemmas from Haitian
Creole. These map back to French cognates. Oversample sentences in the French
training corpus that are rich in these lemmas.

**CRITICAL CONSTRAINT:** Do NOT mix Creole sentences or grammar into the training
data. Prior ablations in fractal-language conclusively showed that adding
analytical languages (like English) to morphologically rich training data drags
down the rich language rather than lifting the analytical one. Haitian Creole
is more analytical than French. It is an oracle only, not a training source.

---

## BabyLM 2026 Deadlines

- Feb 25, 2026: CfP + training data released (DONE)
- Early April 2026: Eval pipeline + baselines released (WATCH FOR THIS)
- May 25, 2026: ARR submission deadline (preferred route)
- Mid-July 2026: Direct submission deadline (fallback)
- Oct 24-29, 2026: EMNLP Budapest

---

## Directory Structure

```
babylm/
  CLAUDE.md                   -- this file
  MEMORIES.md                 -- project context for Claude Code
  MASTER_PLAN.md              -- full strategic plan
  corpus/
    childes_french/           -- child-directed speech from CHILDES
    babylm_official/          -- official BabyLM corpus (reference only)
    haitian_creole/           -- HC oracle vocabulary
    bilingual/                -- FR->EN lemma bridge for Harness C
    final/                    -- assembled training corpus
  scripts/
    setup.sh                  -- local environment setup
    cloud_setup.sh            -- remote (Vast.ai) environment setup
    deploy_to_vast.sh         -- ship corpus + train.py to a Vast.ai box
    sync_checkpoints.sh       -- pull checkpoints back from a remote box
    run_multi_seed.sh         -- launch N seeds in parallel across N GPUs
    download_babylm_corpus.py -- fetch official corpus
    download_childes_french.py-- fetch CHILDES French CDS (env-var creds)
    build_creole_oracle.py    -- build HC vocabulary oracle (top-300 lemmas)
    build_bilingual_lemmas.py -- 73-lemma FR/EN bridge for Harness C
    analyze_caillou_oracle.py -- Caillou-vs-HC convergence check
    build_french_corpus.py    -- assemble the final 100M-word corpus, oracle-weighted
    count_words.py            -- BabyLM word budget tracker
    build_tokenizer.py        -- pre-train the shared 50k BPE tokenizer
    train.py                  -- 125M GPT-2 trainer (single seed)
    aggregate_seeds.py        -- compute mean +/- std across seed eval JSONs (generic)
    aggregate_paper_tables.py -- emit Tables 1-3 LaTeX + §4.1/§4.2 prose from per-seed JSONs
    _checkpoint_schedule.py   -- shared CHECKPOINT_WORDS + ckpt-index helper
    eval_babylm_suite.py      -- wrap the BabyLM 2025 pipeline (BLiMP, BLiMP-Sup, EWoK, GLUE)
    run_bli_procrustes.py     -- §4.3 closed-form orthogonal Procrustes vs GPT-2 EN
    run_xling_glue.py         -- §4.4 LoRA grid (5 levers x 5 tasks) for cross-lingual GLUE
    run_paper_part1.sh        -- §4 master orchestrator: train 5 seeds + all evals + tables
  eval/
    qfrblimp/run.py           -- §4.1 zero-shot QFrBLiMP harness with bucket aggregation
    qfrcola/run.py            -- §4.1 QFrCoLA fine-tune + accuracy + MCC
    evaluation-pipeline-2025/ -- cloned BabyLM 2025 pipeline (auto-cloned by eval_babylm_suite.py)
  models/
    tokenizer/                -- shared BPE tokenizer (one for all seeds)
    seed{S}/chck_*M/          -- per-seed HuggingFace checkpoints
  eval_results/               -- per-seed JSONs (one file per benchmark)
  paper/                      -- LaTeX source
  tests/                      -- pytest suite covering pure helpers
```

---

## Model Architecture

125M GPT-2 style transformer (matching prior fractal-language ablations for
direct comparability):
- 12 layers, d_model=768, 12 heads
- Batch=32, seq=512
- Joint 50k BPE tokenizer (French only for this project)

This matches the architecture in MASTER_PLAN.md and fractal-language/CLAUDE.md.

---

## Reproducing §4 of the paper (5 seeds, headline results)

One-time setup on the host:

1. `wandb login` (stores the API key in `~/.netrc`; `wandb.init()` reads it).
2. `huggingface-cli login` (gated datasets: BabyLM corpus, EWoK).
3. Pre-download the BabyLM eval pipeline data, one-shot:
   ```
   cd eval/evaluation-pipeline-2025          # auto-cloned on first phase 4 run
   osf -p ryjfm clone . && mv osfstorage/evaluation_data . && rmdir osfstorage
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   python evaluation_pipeline/ewok/dl_and_filter.py   # gated; needs HF auth
   ```
   The orchestrator wrapper auto-patches the pipeline's
   `sentence_zero_shot/dataset.py` and `finetune/trainer.py` (AutoProcessor
   fallback to AutoTokenizer + pad_token resolution) on first run; the patch
   is idempotent and survives re-clones.

Then:

```
bash scripts/run_paper_part1.sh 42 43 44 45 46
```

To skip wandb entirely, prepend `TRAIN_EXTRA="--wandb_mode disabled"`.

The orchestrator runs 7 phases (train 5 seeds, then QFrBLiMP, QFrCoLA, BabyLM
suite, BLI Procrustes, cross-lingual GLUE, aggregate). Each phase skips
itself if its expected output already exists, so a partial run is resumable.
`PHASE=N` runs a single phase; `SKIP_PHASES="6"` skips one. Final output:
`paper_tables.tex` and `paper_tables.md` (Tables 1-3 + §4.1/§4.2 prose).

Phase 6 (cross-lingual GLUE, the heaviest) is parallelised across GPUs in
waves of `N_GPUS` (auto-detected via `nvidia-smi -L`); each
`(lever, task)` cell skips if its JSON already exists, so killing and
restarting is safe.

Per-script entry points (for ad-hoc runs):

| Script | Section | Output |
|---|---|---|
| `eval/qfrblimp/run.py` | §4.1 Table 1 | `seed{S}_qfrblimp.json` |
| `eval/qfrcola/run.py` | §4.1 prose | `seed{S}_qfrcola.json` |
| `scripts/eval_babylm_suite.py` | §4.2 prose | `seed{S}_babylm.json` |
| `scripts/run_bli_procrustes.py` | §4.3 Table 2 | `seed{S}_bli_<target>.json` |
| `scripts/run_xling_glue.py` | §4.4 Table 3 | `seed{S}_xglue_<lever>_<task>.json` |
| `scripts/aggregate_paper_tables.py` | tables out | `paper_tables.tex` + `.md` |

External dataset sources (override via CLI flags if needed):

- QFrBLiMP: raw GitHub URL of `davebulaval/QFrBLiMP/datastore/QFrBLiMP/release/qfrblimp.jsonl`
  (the `graalul/qfrblimp` HF card declares but does not publish the data).
  Schema: `sentence_a`, `sentence_b`, `label`, `category`, `type`, `subcat`,
  plus per-annotator columns. 1761 pairs.
  **`label` decides which side is grammatical** (0.0 -> `sentence_a`,
  1.0 -> `sentence_b`; released split 845/916). A scorer that assumes
  `sentence_a` is always the good sentence is right on the label=0 items and
  inverted on the label=1 items, which pins any real competence at ~48.5%.
  `category` (4 values, counts 380/398/716/267) is the paper's bucket level;
  `type` (20 values) is the fine-grained phenomenon.
  18 released rows have byte-identical `sentence_a`/`sentence_b`, so the
  benchmark ceiling is 1743/1761 = 98.98%, not 100%.
- QFrCoLA: `graalul/qfrcola` (HF). `sentence`, `label`, `category`.
- BabyLM eval pipeline: cloned from `babylm/evaluation-pipeline-2025`,
  data from OSF `ryjfm` + EWoK generated locally via the pipeline's
  `dl_and_filter.py`.
- Cross-lingual GLUE FR: local files under `submission/glue_fr/{task}.{train,valid}.jsonl`,
  shipped with the submission. RTE schema diverges (`sentence1/sentence2`
  vs super_glue's `premise/hypothesis`); the task spec carries
  `fr_text_a_field`/`fr_text_b_field` overrides.
- BLI seed dictionary: local `corpus/bilingual/bilingual_lemmas.txt`
  (73 lemmas → ~242 single-token pairs after Procrustes filtering).

---

## Multi-seed Training

The submitted leaderboard checkpoint was trained with a single (unrecorded)
seed. For the paper we report mean +/- std across 5 seeds. Workflow:

1. `python scripts/build_tokenizer.py` once to materialise the shared BPE
   tokenizer at `models/tokenizer/`. All seeds reuse it; this avoids both a
   race on `tokenizer.json` between parallel seeds and an extra source of
   variance.
2. Run `wandb login` once on this host. The key is stored in `~/.netrc` and
   `wandb.init()` picks it up automatically; no env var to export. Training
   metrics (loss, ppl, lr, tokens/sec, words processed, checkpoint events)
   stream to the `babylm-2026` wandb project under run name `seed{S}`.
3. `bash scripts/run_multi_seed.sh 1 2 3 4 5` on a multi-GPU host.
   `nvidia-smi -L | wc -l` auto-detects N; the script launches in waves of N
   seeds, one per GPU via `CUDA_VISIBLE_DEVICES`. Per-seed logs go to
   `logs/seed{S}.log`, checkpoints to `models/seed{S}/chck_*M/`.
4. After eval, `python scripts/aggregate_seeds.py 'eval_results/seed*.json'`
   prints a markdown (and optional `--latex`) table with mean, std, n,
   min, max for every metric.

`--wandb_mode disabled` short-circuits wandb entirely (offline / no key).

---

## Word Budget Rules

BabyLM counts words using simple whitespace splitting, not subword tokens.
Always use scripts/count_words.py to verify budget compliance.

- Strict track: ≤100M words total exposure
- Strict-Small track: ≤10M words (use as ablation)
- Max 10 epochs (each repeat counts toward exposure)

---

## Corpus Strategy

Target: 90-100M words, 100% French, morphologically dense

Sources (in priority order):
1. CHILDES French (Paris, Lyon, York, Geneva corpora) -- gold CDS, low volume
2. OSCAR French / CC-100 French -- filtered web text
3. French OpenSubtitles -- conversational register
4. French children's books (Project Gutenberg)
5. French Wikipedia (filtered for accessible prose)

Oversample sentences containing oracle lemmas (from Haitian Creole analysis)
while keeping all training text 100% standard French with full morphology intact.

---

## Evaluation Plan

1. Official BabyLM eval pipeline (BLiMP, EWoK, GLUE, BEAR)
2. Custom French BLiMP-style probes (adapted from fractal-language grammar probes)
3. Cross-lingual zero-shot: French-trained model tested on Haitian Creole sentences
4. Piagetian emergence curves: plot tokens-to-threshold (60%, 70%, 80%, 100%)
5. Perplexity vs. competence orthogonality check

---

## Paper Anti-Hegemony Framing (Budapest audience)

The paper makes no apology for its political subtext: English is the expensive
outlier, not the default. The morphological poverty of English is a structural
disadvantage that costs 12x in compute. French (and other morphologically rich
languages) are not "foreign" alternatives -- they are the efficient baseline.

EU sovereignty angle: morphologically rich languages produce better-value AI
systems. This is relevant to European AI policy.

---

## Related Projects

- /Users/adam/dev/fractal-language -- prior French/English ablation experiments
  (source of all prior results cited in this paper)
- fractal-language/paper/ -- draft of "Morphology Eats Scale for Breakfast"
  (DO NOT reuse title or claim it as the same paper)
- fractal-language/MASTER_PLAN.md -- full multi-paper research program context

---

## Author

Adam Zachary Wasserman
Independent researcher
wassermana@gmail.com
@adamzwasserman (X/HN)
OSF: https://osf.io/sj48b
