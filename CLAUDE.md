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
    setup.sh                  -- environment setup
    cloud_setup.sh            -- vast.ai bootstrap
    deploy_to_vast.sh         -- push code+data to vast.ai instance
    sync_checkpoints.sh       -- pull checkpoints back from vast.ai
    download_babylm_corpus.py -- fetch official corpus
    download_childes_french.py-- fetch CHILDES French CDS (env-var creds)
    build_creole_oracle.py    -- build HC vocabulary oracle (top-300 lemmas)
    build_bilingual_lemmas.py -- 73-lemma FR/EN bridge for Harness C
    build_french_corpus.py    -- assemble final corpus, oracle-weighted
    analyze_caillou_oracle.py -- Caillou-vs-HC convergence check
    count_words.py            -- BabyLM word budget tracker
    train.py                  -- 125M GPT-2 trainer + BPE tokenizer
  eval/
    evaluation-pipeline/      -- cloned BabyLM eval repo (TODO: integrate)
  models/                     -- trained checkpoints (HF Hub)
  paper/                      -- LaTeX source
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
