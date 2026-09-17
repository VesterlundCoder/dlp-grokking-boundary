# Predicting the Discrete-Log Learnability Boundary

Code and data for the preprint **"Predicting the Discrete-Log Learnability Boundary: Representation, Spectral Accessibility, and Grokking"** by David Vesterlund.

## Overview

**Central question:** Can representation-level and architecture-conditioned spectral measurements obtained *before training* prospectively predict whether a DLP formulation will directly generalize, grok after memorization, memorize without generalizing, or fail to fit?

This is a **prospective prediction study** with strict locking protocols. Predictions must precede the training they predict.

## Key Design Principles

1. **Prediction precedes training.** No confirmatory run may start without a valid prediction lock.
2. **Two predictor classes:** P0 (strict pre-training, t=0 only) and P1 (early-training, 1% probe window).
3. **Primary causal intervention: GF(2) affine bijections.** Same bits, same length, same alphabet, same information, same labels, different coordinate geometry.
4. **Literature positioning:** We do NOT claim NTK-target alignment is novel (Kumar et al.). We test whether spectral accessibility can *prospectively predict* learning phase.
5. **Falsifiable:** If the spectral predictor doesn't outperform simple baselines, we report that.

## Repository Structure

```
dlp-grokking-boundary/
├── README.md
├── LICENSE, CITATION.cff, requirements.txt
├── AUDIT_REPORT.md           # Forensic audit of legacy runs
├── THEOREM_MAPPING.md        # Maps Takhanov theorem to experiments
│
├── paper/                     # Manuscript (LaTeX + PDF)
│
├── src/
│   ├── groups.py              # Cyclic group abstractions
│   ├── latent_manifests.py    # Canonical latent manifests
│   ├── datasets.py            # Matched isomorphic datasets
│   ├── models.py              # GrokkingTransformer
│   ├── training.py            # Unified training loop
│   ├── outcomes.py            # Canonical phase definitions (K=10 sustained)
│   ├── checkpoints.py         # Checkpoint management
│   ├── phase_classifier.py    # Legacy phase classifier
│   │
│   ├── encodings/             # Information-preserving encodings
│   │   ├── identity_binary.py
│   │   ├── gray.py
│   │   ├── gf2_affine.py       # PRIMARY causal intervention
│   │   ├── feistel.py          # OOD holdout
│   │   └── atomic.py
│   │
│   ├── tasks/                 # Task definitions
│   │   ├── fixed_base_parity.py
│   │   ├── variable_base_full_log.py
│   │   ├── additive_inverse.py
│   │   └── modular_addition_control.py
│   │
│   ├── predictors/            # Prediction modules (to be built)
│   │   ├── fourier.py
│   │   ├── ntk.py
│   │   ├── kernel_spectrum.py
│   │   ├── gradients.py
│   │   ├── pretraining_features.py
│   │   ├── early_probe.py
│   │   ├── timescale_models.py
│   │   ├── phase_predictor.py
│   │   └── baselines.py
│   │
│   ├── statistics/            # Statistical analysis (to be built)
│   │   ├── survival.py
│   │   ├── grouped_cv.py
│   │   ├── bootstrap.py
│   │   └── metrics.py
│   │
│   └── locking/               # Prediction lock infrastructure
│       ├── create_lock.py
│       ├── verify_lock.py
│       └── hash_manifest.py
│
├── theory/                    # Theory track (to be built)
│
├── manifests/
│   ├── groups.csv             # 62 frozen groups with roles
│   └── latent/                # Canonical latent manifests
│
├── predictions/               # Prediction files (locked before training)
├── locks/                     # Prediction locks (append-only)
│
├── experiments/
│   ├── CAL_A_clean_pilot/     # q=113, 4 encodings × 2 opt × 5 seeds
│   ├── CAL_B_boundary_grid/   # 3 groups × 2 tasks × 4 enc × 3 frac × 3 seeds
│   ├── CAL_C_capacity/        # 6 configs × 4 capacities × 5 seeds
│   ├── CAL_D_optimization/    # 6 configs × 4 opt conditions × 5 seeds
│   ├── CAL_R_random_labels/   # 6 configs × 4 capacities × 3 seeds
│   ├── CONF_A_encoding_prediction/  # 3 q × 18 enc × 5 seeds (PRIMARY)
│   ├── CONF_B_encoding_design/     # Search + construct encodings
│   ├── CONF_C_ood_encoding/         # Feistel holdout
│   └── CONF_D_isomorphic_demo/     # Additive vs multiplicative
│
├── legacy/                    # LEGACY_EXPLORATORY_V0 data
│   ├── README.md
│   └── legacy_40_status_*.csv
│
├── results/
│   ├── stage00/ ... stage03/  # Legacy exploratory runs
│   └── matched_isomorphic/    # Legacy 40-run experiment (quarantined)
│
└── reproduce/                # Reproduction scripts (to be built)
```

## Evidence Classes

| Class | Description | Use |
|-------|-------------|-----|
| LEGACY_EXPLORATORY | Pre-protocol runs | Debugging, calibration only |
| CALIBRATION | Post-protocol calibration | Develop and fit predictor |
| CONFIRMATORY | Blind prospective evaluation | Final proof, no refitting |

## Pre-Registered Hypotheses

- **H1:** Pre-training spectral accessibility predicts time-to-generalization
- **H2:** GF(2) bijections cause systematic learning differences
- **H3:** High-accessibility encodings generalize faster on unseen groups
- **H4:** Predictor can construct high-accessibility encodings
- **H5:** If t=0 fails, early probe provides additional predictive information

## Citation

```bibtex
@misc{vesterlund2026predicting,
  title={Predicting the Discrete-Log Learnability Boundary: Representation, Spectral Accessibility, and Grokking},
  author={Vesterlund, David},
  year={2026},
  url={https://github.com/VesterlundCoder/dlp-grokking-boundary}
}
```

## License

MIT License. See `LICENSE` for details.
