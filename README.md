# The Grokking Boundary

Code and data for the preprint **"The Grokking Boundary: Discrete Logarithm Hardness and Group Structure in Transformers"** by David Vesterlund.

## Overview

This repository contains the experimental code, results, and figure-generation scripts for a systematic study of when and why neural networks can learn the discrete logarithm problem (DLP) through gradient-based training.

Key findings:
- MLPs can learn individual DLP parity bits for small primes (b ≤ 16) through direct generalization, but fail for b ≥ 20, confirming the predicted hardness threshold.
- Bit-tokenizer Transformers fail to grok the prime-order subgroup DLP at any tested bit size (b = 7–26), consistent with the hardness prediction.
- The same Transformer architecture **does** grok the additive group (modular addition) at b = 16, but fails at b ≥ 18, revealing that the grokking boundary is group-structure-dependent.
- Across a 2×2 design varying base visibility and target granularity, no configuration produces grokking on prime-order DLP for b = 7–14 at 200k epochs.

These results establish a sharp **grokking boundary** determined by the interaction of group structure and representation, not by model capacity alone.

## Repository Structure

```
dlp-grokking-boundary/
├── README.md
├── LICENSE
├── paper/
│   ├── paper.tex              # Manuscript (LaTeX)
│   ├── paper.pdf              # Compiled manuscript
│   ├── references.bib         # Bibliography
│   └── figures/               # All manuscript figures (PDF + PNG)
├── scripts/
│   └── generate_figures.py    # Figure generation from results
└── results/
    ├── stage00/               # Stage 0: MLP replication (90 runs)
    ├── stage00b/              # Stage 0b: Theorem uniform (9 runs)
    ├── stage01/               # Stage 1: Architecture bridge (18 runs)
    ├── stage02/               # Stage 2: Grokking bridge (18 runs)
    └── stage03/               # Stage 3: 2×2 discovery (64 runs)
```

## Experimental Design

The study uses a five-stage design with 199 total runs on LUMI-G (AMD MI250X):

| Stage | Description | Runs | Architecture | Task |
|-------|-------------|------|--------------|------|
| 0 | MLP replication | 90 | 2-layer MLP (1M params) | DLP parity bit, b=16-26 |
| 0b | Theorem uniform | 9 | 2-layer MLP (1M params) | DLP parity bit, uniform dist. |
| 1 | Architecture bridge | 18 | Bit-tokenizer Transformer (425k) | Prime-order DLP, b=16-26 |
| 2 | Grokking bridge | 18 | Bit-tokenizer Transformer (425k) | Additive group, b=16-26 |
| 3 | 2×2 discovery | 64 | Bit-tokenizer Transformer (425k) | Prime-order DLP, b=7-14 |

## Regenerate Figures

```bash
pip install matplotlib numpy
python3 scripts/generate_figures.py
```

## Citation

```bibtex
@misc{vesterlund2026grokkingboundary,
  title={The Grokking Boundary: Discrete Logarithm Hardness and Group Structure in Transformers},
  author={Vesterlund, David},
  year={2026},
  url={https://github.com/VesterlundCoder/dlp-grokking-boundary}
}
```

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

Compute resources provided by LUMI-G (EuroHPC JU), project `project_465003364`.
