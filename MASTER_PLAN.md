# Right Tool, Right Job: BabyLM 2026 Submission

## Title
**Right Tool, Right Job: Why Training Language Matters More Than Training Data**
*Les bons outils font les bons ouvriers*

(Earlier working title: "Born Speaking French". Reframed during writing; corpus
strategy and architecture below are unchanged.)

## Core Claim
Training exclusively on French at child-scale (≤100M words) produces a model that
outperforms English baselines by 4-12x on grammatical competence benchmarks,
because French morphological redundancy delivers denser learning signal per token.
Haitian Creole is used as a vocabulary oracle to identify the highest load-bearing
lemmas in French, which are then oversampled in the training corpus.

## Pre-registrations
- OSF sj48b: https://osf.io/sj48b (French/English cross-linguistic training dynamics)
- OSF pcx2d: https://osf.io/pcx2d (morphological complexity gradient)

## Key Prior Results (from fractal-language experiments)
| Metric                   | French  | English       | Ratio |
|--------------------------|---------|---------------|-------|
| Tokens to 100% grammar   | 197M    | >3B (never)   | >15x  |
| Perplexity at 3B tokens  | ~27     | ~1340         | 50x   |
| Grammar accuracy at 3B   | 100%    | 40% (chance)  | -     |

## BabyLM 2026 Deadlines
- Feb 25, 2026: CfP + training data released (DONE)
- Early April 2026: Eval pipeline + baselines released
- May 25, 2026: ARR submission deadline
- Mid-July 2026: Direct submission deadline
- Oct 24-29, 2026: EMNLP Budapest (workshop likely 28-29)

## Tracks
- Primary: Strict (≤100M words, custom corpus allowed)
- Possible: Strict-Small (≤10M words) as ablation

## Three-Track Parallel Workstream

### Track 1: Corpus
- French CDS base: CHILDES French subsets, Orléans corpus
- Haitian Creole oracle: extract top 100-200 high-frequency lemmas
- Oversample French sentences rich in those lemmas (morphology intact)
- Target: 90-100M words

### Track 2: Eval Pipeline
- Clone babylm/evaluation-pipeline (update to 2025 version when released)
- Get running against official baselines
- Add French BLiMP-style probes from fractal-language experiments
- Cross-lingual zero-shot probe: French model -> Haitian Creole test sentences

### Track 3: Baseline Model
- Architecture: 125M GPT-2 style (matches prior fractal-language ablations)
- Get official BabyLM baseline training and evaluating cleanly
- Then swap in French corpus

## Haitian Creole Oracle Strategy
The key insight: pidginization preserves only the highest-frequency,
highest-composability vocabulary. Words that survive into Haitian Creole from
French are load-bearing by selection pressure. Use their frequency distribution
to prioritize lemmas in the French corpus -- but the training data remains 100%
morphologically rich French.

DO NOT mix Creole sentences into training data. Prior ablations showed analytical
languages drag down morphologically rich ones, not the reverse.

## Paper Sections (Draft Outline)
1. Introduction: The anglophone tax -- why English costs 12x more to learn
2. Related Work: BabyLM prior submissions, morphological efficiency lit
3. Methods: Corpus curation + Creole oracle + training setup
4. Results: Emergence curves, French BLiMP, cross-lingual zero-shot
5. Discussion: Poka-yoke redundancy, Piagetian stage emergence, EU sovereignty
6. Conclusion: Language-contingent scaling as the default assumption

## Directory Structure
- corpus/   -- raw and processed training data
- scripts/  -- data pipeline, word counting, lemma extraction
- eval/     -- evaluation pipeline and probes
- models/   -- trained checkpoints
- paper/    -- LaTeX source
