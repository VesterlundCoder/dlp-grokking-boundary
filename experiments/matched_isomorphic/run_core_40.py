#!/usr/bin/env python3
"""Core 40-run experiment: q=113, 2×2×2×5 factorial design.

Factors:
  1. Group realization: {additive (Z_113, +), multiplicative (subgroup of F_227*)}
  2. Tokenizer: {atomic integer, binary bit}
  3. Optimization: {Paper 1 (progressive WD ramp), Paper 2 (fixed WD=0.3)}
  4. Seeds: {42, 123, 456, 789, 2026}

Total: 2 × 2 × 2 × 5 = 40 runs.

All runs use the SAME latent manifest, ensuring identical train/test membership
and labels across both group realizations.
"""
import sys
import os
import json
import time
from pathlib import Path

# Add repo root to path so we can import src package
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.groups import matched_groups, AdditiveGroup, MultiplicativeSubgroup
from src.datasets import (
    generate_latent_manifest, load_latent_manifest, manifest_hash,
    realize_additive, realize_multiplicative, get_split,
)
from src.representations import IntegerTokenizer, BitTokenizer
from src.training import TrainConfig, train, TokenDataset
from src.phase_classifier import classify_from_metrics_file

# ============================================================================
# Configuration
# ============================================================================

Q = 113
SEEDS = [42, 123, 456, 789, 2026]
TRAIN_FRAC = 0.30
EPOCHS = 100_000
EVAL_INTERVAL = 100
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "matched_isomorphic"
MANIFEST_DIR = Path(__file__).parent.parent.parent / "manifests" / "latent_pairs"

# Model parameters (matched to Paper 1 M04)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2

# ============================================================================
# Generate manifest
# ============================================================================

def ensure_manifest():
    """Generate or load the canonical q=113 latent manifest."""
    manifest_path = MANIFEST_DIR / f"q{Q}_manifest.csv"
    if manifest_path.exists():
        print(f"Loading existing manifest from {manifest_path}")
        manifest = load_latent_manifest(manifest_path)
    else:
        print(f"Generating manifest for q={Q}...")
        manifest = generate_latent_manifest(Q, seed=42, train_frac=TRAIN_FRAC, output_path=manifest_path)
    
    mhash = manifest_hash(manifest)
    n_train = sum(1 for p in manifest if p.split == "train")
    n_test = sum(1 for p in manifest if p.split == "test")
    print(f"Manifest: {len(manifest)} pairs, train={n_train}, test={n_test}, hash={mhash[:16]}...")
    return manifest, mhash

# ============================================================================
# Realization + tokenizer setup
# ============================================================================

def prepare_data(manifest, group_type: str, tokenizer_type: str):
    """Prepare train/test datasets for a given group realization and tokenizer."""
    Ga, Gm = matched_groups(Q)
    
    if group_type == "additive":
        group = Ga
        samples = realize_additive(manifest, group)
        modulus = group.q
        n_bits = None
    elif group_type == "multiplicative":
        group = Gm
        samples = realize_multiplicative(manifest, group)
        modulus = group.p
        n_bits = Q.bit_length()
    else:
        raise ValueError(f"Unknown group_type: {group_type}")
    
    train_samples = get_split(samples, "train")
    test_samples = get_split(samples, "test")
    
    # Create tokenizer
    if tokenizer_type == "integer":
        tokenizer = IntegerTokenizer(modulus)
    elif tokenizer_type == "bit":
        if n_bits is None:
            n_bits = modulus.bit_length()
        tokenizer = BitTokenizer(n_bits)
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")
    
    # Create datasets
    train_ds = TokenDataset(
        bases=[s.base for s in train_samples],
        targets=[s.target for s in train_samples],
        labels=[s.x for s in train_samples],
        tokenizer=tokenizer,
    )
    test_ds = TokenDataset(
        bases=[s.base for s in test_samples],
        targets=[s.target for s in test_samples],
        labels=[s.x for s in test_samples],
        tokenizer=tokenizer,
    )
    
    return tokenizer, modulus, train_ds, test_ds, group

# ============================================================================
# Optimization regimes
# ============================================================================

def make_config(run_id, seed, opt_regime):
    """Create a TrainConfig for a given optimization regime."""
    if opt_regime == "paper1":
        return TrainConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            epochs=EPOCHS, lr=1e-3, eval_interval=EVAL_INTERVAL,
            optimizer="adamw", weight_decay=0.05,
            progressive_wd=True, wd_start=0.05, wd_step=0.05,
            wd_ramp_interval=1000, lr_drop_factor=0.1,
            early_stop_patience=200, early_stop_threshold=0.99,
            seed=seed, run_id=run_id,
            output_dir=str(RESULTS_DIR),
        )
    elif opt_regime == "paper2":
        return TrainConfig(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            epochs=EPOCHS, lr=1e-3, eval_interval=EVAL_INTERVAL,
            optimizer="adamw", weight_decay=0.3,
            progressive_wd=False,
            early_stop_patience=200, early_stop_threshold=0.99,
            seed=seed, run_id=run_id,
            output_dir=str(RESULTS_DIR),
        )
    else:
        raise ValueError(f"Unknown opt_regime: {opt_regime}")

# ============================================================================
# Main
# ============================================================================

def main():
    manifest, mhash = ensure_manifest()
    
    group_types = ["additive", "multiplicative"]
    tokenizer_types = ["integer", "bit"]
    opt_regimes = ["paper1", "paper2"]
    
    total = len(group_types) * len(tokenizer_types) * len(opt_regimes) * len(SEEDS)
    print(f"\n{'='*60}")
    print(f"Core 40-run experiment: q={Q}")
    print(f"  Groups: {group_types}")
    print(f"  Tokenizers: {tokenizer_types}")
    print(f"  Optimization: {opt_regimes}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {total}")
    print(f"  Epochs per run: {EPOCHS:,}")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"{'='*60}\n")
    
    run_idx = 0
    for group_type in group_types:
        for tokenizer_type in tokenizer_types:
            for opt_regime in opt_regimes:
                for seed in SEEDS:
                    run_idx += 1
                    run_id = f"q{Q}_{group_type}_{tokenizer_type}_{opt_regime}_s{seed}"
                    
                    # Skip if already done
                    summary_path = RESULTS_DIR / run_id / "summary.json"
                    if summary_path.exists():
                        print(f"[{run_idx}/{total}] SKIP {run_id} (already done)")
                        continue
                    
                    print(f"\n[{run_idx}/{total}] RUN {run_id}")
                    print(f"  group={group_type} tokenizer={tokenizer_type} opt={opt_regime} seed={seed}")
                    
                    try:
                        tokenizer, modulus, train_ds, test_ds, group = prepare_data(
                            manifest, group_type, tokenizer_type
                        )
                        config = make_config(run_id, seed, opt_regime)
                        
                        t0 = time.time()
                        summary = train(config, train_ds, test_ds, tokenizer, modulus)
                        elapsed = time.time() - t0
                        
                        print(f"  DONE: best_test={summary['best_test']:.4f} "
                              f"final_train={summary['final_train']:.4f} "
                              f"final_test={summary['final_test']:.4f} "
                              f"epochs={summary['total_epochs']} "
                              f"time={elapsed/60:.1f}min")
                        
                        # Classify phase
                        metrics_path = RESULTS_DIR / run_id / "metrics.jsonl"
                        if metrics_path.exists():
                            phase_result = classify_from_metrics_file(metrics_path)
                            print(f"  Phase: {phase_result.phase} "
                                  f"Tmem={phase_result.t_mem} T90={phase_result.t90} "
                                  f"deltaT={phase_result.delta_t}")
                        
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        import traceback
                        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"All {total} runs complete.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
