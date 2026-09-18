# Research Process Flowchart
## Predicting the Discrete-Log Learnability Boundary

---

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 0: SETUP                                  │
│                                                                         │
│  Freeze groups.csv (62 groups) → Generate latent manifests              │
│  → Select encodings (identity, Gray, GF2-A, GF2-B, Feistel, ...)       │
│  → Define configurations (q × encoding × N × P × optimizer × task)      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PRE-TRAINING (t=0)                          │
│                                                                         │
│  For each configuration, compute BEFORE any training:                   │
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐      │
│  │ Fourier Features │  │ Kernel Features  │  │ Gradient Features  │      │
│  │                 │  │                  │  │                    │      │
│  │ • Entropy H_F   │  │ • NTK K = UΛU^T │  │ • ||E[g]||         │      │
│  │ • Eff support   │  │ • Target align   │  │ • E[||g||²]       │      │
│  │ • Participation │  │ • Eigenvalue     │  │ • Var(g)           │      │
│  │   ratio PR_F    │  │   decay          │  │ • GSNR             │      │
│  │ • Top-k energy  │  │ • Kernel eff     │  │ • Pairwise cosine  │      │
│  │ • Spectral rank │  │   rank           │  │ • Grad eff rank   │      │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬───────────┘      │
│           │                    │                     │                  │
│           │    ┌───────────────┘                     │                  │
│           │    │  ┌──────────────────┐               │                  │
│           │    │  │ T_spec(ε)        │               │                  │
│           │    │  │ KSA_τ            │◄──────────────┘                  │
│           │    │  │ (kernel spectral │                                   │
│           │    │  │  accessibility)  │                                   │
│           │    │  └────────┬────────┘                                   │
│           ▼    ▼           ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │           P0 FEATURE TABLE (one row per configuration)      │        │
│  │                                                             │        │
│  │  config_id, q, N, P, encoding_id, encoding_family,         │        │
│  │  fourier_entropy, fourier_PR, fourier_top1, ...,            │        │
│  │  kernel_target_alignment, kernel_eff_rank, ...,            │        │
│  │  T_spec_0.10, T_spec_0.05, KSA_tau0, KSA_0.5tau0, ...,     │        │
│  │  GSNR, grad_eff_rank, grad_cosine_mean, ...                │        │
│  └──────────────────────────────┬──────────────────────────────┘        │
└─────────────────────────────────┤───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PHASE 2: P0 PREDICTION LOCK                          │
│                                                                         │
│  Fit predictor on CALIBRATION data (or compute for pilot):              │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  BASELINE 0: Empirical class frequency (chance)              │       │
│  │  BASELINE 1: log(q) only                                     │       │
│  │  BASELINE 2: log(q) + log(N) + log(P) + target + opt + WD   │       │
│  │  BASELINE 3: Fourier descriptors only                        │       │
│  │  BASELINE 4: Kernel-target alignment only                   │       │
│  │  BASELINE 5: T_mem/capacity timescale features (Song & Ye)  │       │
│  │  BASELINE 6: SR-simple (Teddy's PySR on simple features)    │       │
│  │  FULL P0: Baseline2 + spectral + kernel + T_spec + grad     │       │
│  │  FULL P0+SR: FULL P0 features → SR discovers formula        │       │
│  └──────────────────────────────┬───────────────────────────────┘       │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  PREDICTIONS CSV: predicted_log_T_gen, P(direct),          │       │
│  │  P(grok), P(memorized), P(underfit) for each config         │       │
│  └──────────────────────────────┬───────────────────────────────┘       │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  LOCK: SHA256(predictions) + SHA256(features) + SHA256(code)│       │
│  │  + git SHA + timestamp → locks/<phase>_P0_lock.json         │       │
│  │  APPEND-ONLY. Never overwrite.                               │       │
│  └──────────────────────────────┬───────────────────────────────┘       │
└─────────────────────────────────┤───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: TRAINING — 1% PROBE WINDOW                         │
│                                                                         │
│  Train each model for exactly 1% of budget (e.g., 2000 of 200k steps)   │
│  Then PAUSE. Do NOT resume until P1 lock is created.                    │
│                                                                         │
│  Record during probe:                                                   │
│  • Train loss curve (every eval)                                        │
│  • Train accuracy curve                                                 │
│  • Weight norm, gradient norm, update norm                              │
│  • Kernel matrix at probe endpoint K_probe                              │
│  • Gradient statistics at probe endpoint                                │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                PHASE 4: P1 EARLY-PROBE FEATURES                         │
│                                                                         │
│  Compute drift features (t=0 → probe):                                  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │  Δ kernel-target alignment     │  Δ T_spec proxy           │        │
│  │  d(alignment)/dt               │  Δ KSA                    │        │
│  │  ||K_probe - K_0||_F / ||K_0||  │  GSNR drift               │        │
│  │  Eigenspace angle drift         │  Gradient-rank drift     │        │
│  │  Effective-rank drift           │  Training-loss slope    │        │
│  └──────────────────────────────┬──────────────────────────────┘        │
│                                 │                                       │
│                                 ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │  P1 PREDICTIONS: Updated predictions using P0 + drift       │       │
│  │  LOCK: locks/<phase>_P1_lock.json (append-only)              │       │
│  │  Verify: no checkpoint beyond probe_step exists              │       │
│  └──────────────────────────────┬───────────────────────────────┘       │
└─────────────────────────────────┤───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 5: RESUME TRAINING — 99%                             │
│                                                                         │
│  Train remaining 99% of budget. Record at every eval:                   │
│  • Optimizer step, epoch, example presentations                         │
│  • Train/test loss + accuracy                                           │
│  • Weight norm, gradient norm, update norm                              │
│                                                                         │
│  Stop conditions:                                                       │
│  • Success: test ≥ 0.99 sustained for 10 evals + stability tail        │
│  • Failure: full budget (do NOT early-stop boundary failures)          │
│                                                                         │
│  Record checkpoints at: init, probe, T_mem, mid-grok, T_gen, final     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                PHASE 6: OUTCOME CLASSIFICATION                           │
│                                                                         │
│  For each run, compute using K=10 sustained thresholds:                │
│                                                                         │
│  T_mem = first step where train_acc ≥ 0.99 for 10 evals               │
│  T_gen = first step where test_acc ≥ 0.90 for 10 evals                │
│                                                                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐                │
│  │  T_gen ≤ T_mem  │  T_mem < T_gen  │  T_mem exists   │                │
│  │  → DIRECT GEN  │  → GROKKING     │  T_gen missing  │                │
│  │                 │                 │  → MEMORIZED    │                │
│  └─────────────────┴─────────────────┴─────────────────┘                │
│  ┌─────────────────┬─────────────────────────────────────┐              │
│  │  T_mem missing  │  0.55 ≤ test < 0.90                │              │
│  │  → UNDERFIT    │  → PARTIAL                         │              │
│  └─────────────────┴─────────────────────────────────────┘              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PHASE 7: PREDICTOR EVALUATION                               │
│                                                                         │
│  Compare P0 and P1 predictions against actual outcomes:                 │
│                                                                         │
│  PRIMARY: Δ concordance-index (FULL P0 vs Baseline 2)                   │
│           95% cluster-bootstrap CI must exclude 0                       │
│           Bootstrap at CONFIGURATION level, not seed level              │
│                                                                         │
│  SECONDARY: macro-F1, Brier score, calibration error,                  │
│             Spearman ρ with log(T_gen), MAE on log(T_gen)              │
│                                                                         │
│  ┌──────────────────┬──────────────────────────────────────────┐        │
│  │  Case A: P0 wins  │ "Pre-training spectral accessibility    │        │
│  │  vs Baseline 2    │  prospectively predicts DLP learnability"│       │
│  ├──────────────────┼──────────────────────────────────────────┤        │
│  │  Case B: P0 fails │ "Early kernel/feature dynamics          │        │
│  │  P1 wins          │  prospectively predict DLP learnability" │       │
│  ├──────────────────┼──────────────────────────────────────────┤        │
│  │  Case C: Both fail│ "Initial kernel accessibility does not  │        │
│  │                   │  contain sufficient information"          │       │
│  └──────────────────┴──────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The 40-Run Clean Pilot (CAL_A) in Detail

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CAL_A: CLEAN q=113 PILOT                         │
│                                                                     │
│  Evidence class: CALIBRATION (not confirmatory)                     │
│  Purpose: Debug predictor, check if features vary across encodings │
│                                                                     │
│  Group: q=113, p=227, g0=4 (from groups.csv, role=confirmatory)     │
│  Task: Variable-base full-log DLP                                   │
│  Training fraction: 30% (3796 train, 8860 test)                    │
│  Budget: 200,000 optimizer steps                                     │
│  Model: 2-layer Transformer, d_model=128, ~425k params             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  4 ENCODINGS (binary, 8-bit for p=227):                     │    │
│  │                                                             │    │
│  │  R1: Identity binary     r(z) = z                          │    │
│  │  R2: Gray code            r(z) = z XOR (z >> 1)            │    │
│  │  R3: GF2 affine seed A    r(z) = A₁z + c₁ mod 2            │    │
│  │  R4: GF2 affine seed B    r(z) = A₂z + c₂ mod 2            │    │
│  │                                                             │    │
│  │  All 4 preserve: same bits, same length, same info,         │    │
│  │  same labels, same latent manifest. Only geometry differs.  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────────┬──────────────────┐                             │
│  │  O1: Standard    │  O2: Grokking    │                             │
│  │  Adam, WD=0      │  AdamW, WD=0.3   │                             │
│  │  (no WD)         │  (fixed)         │                             │
│  └──────────────────┴──────────────────┘                             │
│                                                                     │
│  Seeds: {42, 123, 456, 789, 2026}                                   │
│                                                                     │
│  TOTAL: 4 × 2 × 5 = 40 runs                                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH OF THE 40 RUNS:                                   │    │
│  │                                                             │    │
│  │  Step 1: Compute P0 features (t=0)                         │    │
│  │    → Fourier entropy, PR, top-k energy of target function  │    │
│  │    → Kernel K at init, target alignment, eigenvalue decay   │    │
│  │    → T_spec(0.10), KSA_tau0                               │    │
│  │    → GSNR, ||E[g]||, grad effective rank at init           │    │
│  │                                                             │    │
│  │  Step 2: Record P0 features in feature table               │    │
│  │    (NO prediction lock needed for CALIBRATION)              │    │
│  │                                                             │    │
│  │  Step 3: Train 1% of budget (2000 steps)                   │    │
│  │    → Record loss, accuracy, weight/grad norms              │    │
│  │    → Compute K_probe, gradient stats at probe              │    │
│  │                                                             │    │
│  │  Step 4: Compute P1 drift features                          │    │
│  │    → Δ alignment, ||K_probe - K_0||/||K_0||               │    │
│  │    → Δ T_spec, Δ KSA, GSNR drift, loss slope              │    │
│  │                                                             │    │
│  │  Step 5: Resume training (remaining 99%, 198k steps)       │    │
│  │    → Eval every 100 steps                                  │    │
│  │    → Record T_mem, T_gen, phase                             │    │
│  │    → Checkpoint at: init, probe, T_mem, T_gen, final       │    │
│  │                                                             │    │
│  │  Step 6: Classify outcome                                   │    │
│  │    → DIRECT / GROKKING / MEMORIZED / UNDERFIT / PARTIAL     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  AFTER ALL 40 RUNS:                                                 │
│  → Check: Do P0 features differ across the 4 encodings?             │
│  → Check: Do outcomes differ across the 4 encodings?                │
│  → Check: Does T_spec correlate with T_gen across encodings?        │
│  → Check: Does KSA_τ rank encodings correctly?                      │
│  → Check: Does P1 drift add information beyond P0?                  │
│  → Fit initial predictor (linear + SR) on these 40 data points      │
│  → This is CALIBRATION — predictor may be refined, not confirmed    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What Predictors Start With

```
                    ┌─────────────────────────────────┐
                    │     P0: PRE-TRAINING PREDICTOR   │
                    │     (t=0 only, no training)      │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ INTRINSIC       │     │ KERNEL           │     │ GRADIENT          │
│ (task + repr)   │     │ (model + repr)   │     │ (model + task)    │
│                 │     │                  │     │                   │
│ Does NOT depend │     │ Depends on model │     │ Depends on model  │
│ on model init   │     │ architecture +   │     │ + task + init     │
│                 │     │ encoding         │     │                   │
│ • Fourier       │     │ • NTK K₀         │     │ • ||E[g]||       │
│   entropy       │     │ • Target align   │     │ • GSNR            │
│ • Participation │     │ • Eigendecay     │     │ • Grad eff rank   │
│   ratio         │     │ • T_spec(ε)      │     │ • Pairwise cos    │
│ • Top-k energy  │     │ • KSA_τ          │     │ • Class-conditional│
│ • Spectral rank │     │ • Kernel rank    │     │   separation      │
│                 │     │                  │     │                   │
│ Answers:        │     │ Answers:         │     │ Answers:          │
│ "Is the target  │     │ "Can the kernel │     │ "Does the gradient│
│  function       │     │  reach the      │     │  signal point in  │
│  spectrally     │     │  target quickly?"│     │  the right        │
│  concentrated?" │     │                  │     │  direction?"      │
└─────────────────┘     └──────────────────┘     └───────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │   P0 FEATURE VECTOR per config   │
                    │                                  │
                    │   ~20-30 features total          │
                    │   (exact set frozen before CAL)  │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │     PREDICTOR MODELS             │
                    │                                  │
                    │  Primary: Regularized Cox/AFT    │
                    │           survival model         │
                    │                                  │
                    │  Teddy's: PySR discovers         │
                    │           explicit formula       │
                    │           T_gen ≈ f(features)    │
                    │                                  │
                    │  Secondary: Multinomial logistic │
                    │             for phase prediction │
                    └─────────────────────────────────┘


                    ┌─────────────────────────────────┐
                    │  P1: EARLY-PROBE PREDICTOR       │
                    │  (after 1% training)             │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ KERNEL DRIFT    │     │ GRADIENT DRIFT   │     │ TRAINING DYNAMICS│
│                 │     │                  │     │                   │
│ • Δ alignment   │     │ • Δ GSNR         │     │ • Loss slope      │
│ • ||K_p - K_0|| │     │ • Δ grad rank    │     │ • Train acc slope │
│ • Eigenspace    │     │ • Δ pairwise cos │     │ • Weight norm     │
│   angle drift   │     │                  │     │   growth rate     │
│ • Δ T_spec      │     │                  │     │                   │
│ • Δ KSA         │     │                  │     │                   │
│                 │     │                  │     │                   │
│ Answers:        │     │ Answers:         │     │ Answers:          │
│ "Is the kernel  │     │ "Is the gradient │     │ "How fast is the  │
│  escaping the   │     │  signal growing  │     │  model fitting    │
│  initial regime?"│    │  or shrinking?"  │     │  training data?"  │
└─────────────────┘     └──────────────────┘     └───────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │   P1 FEATURE VECTOR per config   │
                    │                                  │
                    │   P0 features + ~10 drift       │
                    │   features                       │
                    └──────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │   P1 PREDICTOR (updated)         │
                    │                                  │
                    │   Same model class as P0,         │
                    │   but now with drift features    │
                    │                                  │
                    │   Tests: Does lazy→rich movement │
                    │   add info beyond t=0 kernel?     │
                    └─────────────────────────────────┘
```

---

## What Curves We Predict On

```
For each run, we record these curves and extract:

Training accuracy:    A_train(t)  ──→  T_mem = first sustained ≥ 0.99
Test accuracy:         A_test(t)   ──→  T_gen = first sustained ≥ 0.90
Training loss:         L_train(t)  ──→  loss slope (P1 feature)
Weight norm:           ||w||(t)    ──→  WD effect
Gradient norm:         ||g||(t)    ──→  gradient diagnostics

    A_train ──────────────────────────────────── ════════
                       ╲                         (memorized)
                        ╲                    ╱
                         ╲──────────────╱
                          T_mem
    A_test  ────────────────────────────────────── ════════
                                                   ╱ (generalized)
                                              ╱───
                                         ╱
                                    T_gen
    ──────────┬──────────┬──────────────────┬──────────
    0       probe     T_mem             T_gen      budget
             (1%)                                 (100%)

    Phase = GROKKING (T_mem < T_gen)
    grok_delay = T_gen - T_mem
    grok_ratio = T_gen / T_mem

PREDICTOR TARGETS:
    • log(T_gen)         → survival regression (primary)
    • phase class         → multinomial logistic (secondary)
    • log(T_mem)         → regression (for Song & Ye connection)
    • grok_delay          → regression (if grokking occurs)
```

---

## Full Experiment Timeline

```
CAL_A (40 runs)    CAL_B (216)    CAL_C (120)    CAL_D (120)    CAL_R (72)
    │                  │              │              │              │
    │  Local MPS        │  LUMI-G      │  LUMI-G      │  LUMI-G      │  LUMI-G
    │  ~2-3 days        │  ~50 GPU-h   │  ~40 GPU-h   │  ~40 GPU-h   │  ~25 GPU-h
    │                  │              │              │              │
    └──────┬───────────┴──────────────┴──────────────┴──────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────┐
    │  CALIBRATION COMPLETE: ~568 runs                  │
    │                                                    │
    │  → Compute P0 + P1 features for all 568 runs      │
    │  → Fit predictor (Cox + SR) on calibration data   │
    │  → FREEZE predictor-v1                             │
    │  → Git tag: predictor-v1-frozen                   │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  CONFIRMATORY (blind, locked)                     │
    │                                                    │
    │  CONF_A: 270 runs (3 new q × 18 enc × 5 seeds)   │
    │    → P0 lock BEFORE training                       │
    │    → P1 lock at 1% probe                          │
    │    → Train to completion                           │
    │    → Evaluate WITHOUT refitting                    │
    │                                                    │
    │  CONF_B: 100 runs (steering experiment)           │
    │  CONF_C: 120 runs (Feistel OOD holdout)           │
    │  CONF_D: 20 runs (additive vs multiplicative)     │
    │                                                    │
    │  Total: ~510 confirmatory runs                    │
    │  LUMI-G: ~150 GPU-h                               │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  EVALUATION + PAPER COMPLETION                    │
    │                                                    │
    │  → Compare P0/P1 vs baselines on confirmatory     │
    │  → Survival analysis (Kaplan-Meier, Cox)          │
    │  → Phase diagrams                                  │
    │  → SR distillation of grokked models (Teddy)      │
    │  → Mechanistic analysis (CKA, spectral, probes)  │
    │  → Complete paper with real results               │
    │  → Zenodo archive + GitHub release                 │
    └──────────────────────────────────────────────────┘
```
