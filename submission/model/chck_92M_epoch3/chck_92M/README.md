---
license: mit
language:
- fr
tags:
- babylm
- babylm-2026
- gpt2
- causal-lm
- french
- cross-linguistic
- low-resource
- right-tool-right-job
library_name: transformers
pipeline_tag: text-generation
---

# BabyLM 2026 Strict, French (92M words)

A 125M-parameter GPT-2 trained from scratch on 92,469,402 words of French text. Submitted to the BabyLM 2026 Strict track and the primary checkpoint reported in *Right Tool, Right Job: Why Training Language Matters More Than Training Data* (Wasserman & Beauchemin, BabyLM 2026 / ACL Rolling Review submission).

## Headline result

QFrBLiMP (Quebec French native minimal-pair benchmark, 1761 pairs): **85.97% overall**.

| Subset | Pairs | Accuracy |
|---|---:|---:|
| Anglicism | 267 | 80.15% |
| Morphology | 716 | 85.47% |
| Semantic | 398 | 87.19% |
| Syntax | 380 | 89.74% |
| **Overall** | **1761** | **85.97%** |

QFrCoLA (Quebec French acceptability classification, fine-tuned with LoRA rank 16): test accuracy ~72%, MCC ~0.24 (epoch 3 of fine-tune).

## Argument supported by this model

The companion paper develops the cross-linguistic argument that training-language morphological richness, not neural architecture or pretraining scale, is the load-bearing variable for grammar acquisition. This checkpoint is the child-scale (under 100M words) French anchor; the broader argument is also supported by the *Scaling Hypothesis Is Language-Contingent* and *English Considered Harmful* deposits cited below, which test the same claim at different scales and with different ablations.

## Model details

- Architecture: GPT-2 decoder-only, causal LM (`GPT2LMHeadModel`)
- Parameters: ~125M
- Layers: 12
- Attention heads: 12
- Hidden size: 768
- Max sequence length: 512
- Vocabulary: 50,000 BPE, French Wikipedia source
- Precision: float32

## Training data

- 92,469,402 words of French (under the BabyLM 2026 Strict 100M-word cap)
- Custom corpus assembled from CHILDES French subsets and the Orléans corpus as a developmental base, with lemma-frequency oversampling guided by a Haitian Creole vocabulary oracle (high-frequency, high-composability lemmas surviving pidginization)
- Training data is 100% morphologically rich French; Haitian Creole sentences are not mixed in
- See *Right Tool, Right Job* §3 for full corpus curation methodology

## Training procedure

- Peak learning rate: 1.0e-4
- LR schedule: cosine decay to ~1.9e-7
- Epoch: 3 (of a 5-epoch trajectory; epoch 3 is the grammatical-competence peak reported in §4.2 of the paper)
- Tokens/sec: ~94,000 (CUDA)
- Approximate GPU hours through epoch 3: ~3
- Final training loss: 3.19, perplexity 24.4

## Intended use

Suitable for:
- Replicating *Right Tool, Right Job* results
- Cross-linguistic emergence research
- Quebec French native-benchmark development
- Studies of morphological redundancy and training-data efficiency at child scale

Not suitable for:
- General-purpose French text generation at production quality (corpus is developmental, not web-scale)
- Any English-language task (the model has zero English training exposure)

## Limitations

- French-only training; zero exposure to English or other non-French data
- Child-scale corpus (92M words) is far below typical web-scale pretraining
- BPE tokenizer trained on French Wikipedia, which differs in register from the CHILDES / Orléans developmental sources
- LoRA fine-tuning was used in downstream evaluation grids (see *Right Tool, Right Job* §5)

## Citation

```bibtex
@inproceedings{wasserman_beauchemin_2026_right_tool,
  title     = {Right Tool, Right Job: Why Training Language Matters More Than Training Data},
  author    = {Wasserman, Adam Z. and Beauchemin, David},
  booktitle = {BabyLM 2026 Workshop / ACL Rolling Review submission},
  year      = {2026}
}
```

Companion deposits supporting the broader cross-linguistic argument:

- Wasserman, Adam Z. (2026). *The Scaling Hypothesis Is Language-Contingent.* Zenodo DOI [10.5281/zenodo.19423151](https://doi.org/10.5281/zenodo.19423151).
- Wasserman, Adam Z. (2026). *English Considered Harmful: How Morphological Poverty Pollutes Language Model Training.* Zenodo DOI [10.5281/zenodo.19443357](https://doi.org/10.5281/zenodo.19443357).

Pre-registrations on OSF: [SJ48B](https://osf.io/sj48b) (Language-Only Hypothesis), [PCX2D](https://osf.io/pcx2d) (morphological complexity gradient).

## Acknowledgments

The QFrBLiMP and QFrCoLA evaluation benchmarks are by David Beauchemin and collaborators (Université Laval, Institut Intelligence et Données).
