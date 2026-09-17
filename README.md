# The Discrete-Log Learnability Boundary

Code and data for the preprint **"The Discrete-Log Learnability Boundary: Representation, Scale, and Gradient Hardness in Isomorphic Cyclic Groups"** by David Vesterlund.

## Overview

This repository contains the experimental code, results, and analysis scripts for a systematic study of when and why neural networks can learn the discrete logarithm problem (DLP) through gradient-based training.

**Central question:** Is neural learnability invariant under isomorphic representations of the same abstract cyclic group?

For a prime $q$, the additive group $(\mathbb{Z}_q, +)$ and a multiplicative subgroup $\langle g_0 \rangle \subset \mathbb{F}_p^*$ of order $q$ are abstractly isomorphic. We test whether two tasks that are mathematically isomorphic at the group level can have dramatically different learning dynamics when presented to a neural network through different concrete representations.

## Key Findings (so far)

1. **Stage 2 of the prior version was NOT modular addition.** A forensic audit revealed that the prior paper's claim about "group-structure-dependent grokking" was based on a misidentified experiment. See `AUDIT_REPORT.md`.

2. **MLP replication confirms gradient-concentration hardness.** Direct generalization at $b=16$ transitions to memorization-only at $b \geq 20$, consistent with the asymptotic theorem of Takhanov et al.

3. **Matched isomorphic experiment is running.** 40 runs at $q=113$ testing additive vs. multiplicative × integer vs. bit tokenizer × Paper 1 vs. Paper 2 optimization × 5 seeds.

## Repository Structure

```
dlp-grokking-boundary/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── AUDIT_REPORT.md           # Forensic audit of 199 prior runs
├── THEOREM_MAPPING.md        # Maps Takhanov theorem to experiments
│
├── paper/
│   ├── paper.tex             # Manuscript (LaTeX)
│   ├── paper.pdf             # Compiled manuscript
│   ├── references.bib        # Bibliography
│   └── figures/              # All manuscript figures
│
├── src/
│   ├── groups.py             # Cyclic group abstractions (additive + multiplicative)
│   ├── datasets.py           # Matched isomorphic datasets from same latent manifest
│   ├── representations.py    # Integer, bit, scalar tokenizers
│   ├── models.py             # GrokkingTransformer (from Paper 1)
│   ├── training.py           # Unified training loop (Paper 1 + Paper 2 optimization)
│   └── phase_classifier.py   # Canonical phase definitions with sustained thresholds
│
├── experiments/
│   └── matched_isomorphic/
│       └── run_core_40.py    # Core 40-run experiment (q=113)
│
├── scripts/
│   ├── generate_figures.py   # Figure generation from results
│   └── generate_audit_report.py  # Generate AUDIT_REPORT.md
│
├── manifests/
│   └── latent_pairs/
│       └── q113_manifest.csv # Canonical q=113 latent manifest
│
└── results/
    ├── stage00/              # Stage 0: MLP replication (90 runs)
    ├── stage00b/             # Stage 0b: Theorem uniform (9 runs)
    ├── stage01/              # Stage 1: Architecture bridge (18 runs)
    ├── stage02/              # Stage 2: Grokking bridge (18 runs, re-interpreted)
    ├── stage03/              # Stage 3: 2×2 discovery (64 runs)
    └── matched_isomorphic/   # New: 40-run core experiment (in progress)
```

## Experimental Design

### Prior Experiments (199 runs, LUMI-G)

| Stage | Description | Runs | Architecture | Task |
|-------|-------------|------|--------------|------|
| 0 | MLP replication | 90 | 2-layer MLP (1M params) | DLP parity bit, b=16-26 |
| 0b | Theorem uniform | 9 | 2-layer MLP (1M params) | DLP parity bit, uniform dist. |
| 1 | Architecture bridge | 18 | Bit-tokenizer Transformer (425k) | DLP parity, b=16-26 |
| 2 | Grokking bridge (re-interpreted) | 18 | Bit-tokenizer Transformer (425k) | DLP parity with WD, b=16-26 |
| 3 | 2×2 discovery | 64 | Bit-tokenizer Transformer (425k) | DLP, b=7-14 |

### New: Matched Isomorphic Experiment (40 runs, local + LUMI-G)

2 × 2 × 2 × 5 factorial at q=113:
- Group: {additive (Z_113, +), multiplicative (subgroup of F_227*)}
- Tokenizer: {atomic integer, binary bit}
- Optimization: {Paper 1 (progressive WD), Paper 2 (fixed WD=0.3)}
- Seeds: {42, 123, 456, 789, 2026}

All runs use the same latent manifest, ensuring identical train/test membership and labels across both group realizations.

## Reproduce

```bash
pip install -r requirements.txt

# Generate figures from prior results
python3 scripts/generate_figures.py

# Generate audit report
python3 scripts/generate_audit_report.py

# Run the 40-run core experiment
python3 experiments/matched_isomorphic/run_core_40.py
```

## Citation

```bibtex
@misc{vesterlund2026learnability,
  title={The Discrete-Log Learnability Boundary: Representation, Scale, and Gradient Hardness in Isomorphic Cyclic Groups},
  author={Vesterlund, David},
  year={2026},
  url={https://github.com/VesterlundCoder/dlp-grokking-boundary}
}
```

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

Compute resources provided by LUMI-G (EuroHPC JU), project `project_465003364`.
