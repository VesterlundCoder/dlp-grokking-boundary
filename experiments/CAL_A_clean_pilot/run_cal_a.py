#!/usr/bin/env python3
"""CAL_A: Clean q=113 pilot experiment.

40 runs: 4 encodings × 2 optimizers × 5 seeds.

Evidence class: CALIBRATION (not confirmatory).
Purpose: Debug predictor, check if features vary across encodings.

For each run:
    1. Compute P0 features at t=0
    2. Train 1% of budget (probe), compute P1 drift features
    3. Resume training to completion
    4. Classify outcome (DIRECT/GROKKING/MEMORIZED/UNDERFIT/PARTIAL)

Usage:
    cd /Users/davidsvensson/Desktop/dlp_grokking_boundary
    python3 experiments/CAL_A_clean_pilot/run_cal_a.py [--probe-only] [--resume-only] [--analyze-only]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Setup path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.groups import MultiplicativeSubgroup, subgroup_from_qp
from src.latent_manifests import generate_latent_manifest, load_latent_manifest, get_split
from src.encodings.identity_binary import IdentityBinaryEncoding
from src.encodings.gray import GrayEncoding
from src.encodings.gf2_affine import generate_gf2_encoding
from src.representations import BitTokenizer
from src.models import GrokkingTransformer, estimate_params
from src.training import TrainConfig, TokenDataset, train, compute_accuracy
from src.outcomes import classify_outcome, Phase


# ============================================================================
# Configuration
# ============================================================================

Q = 113
P = 227
G0 = 4
N_BITS = 8  # ceil(log2(227)) = 8
TRAIN_FRAC = 0.30
BUDGET = 10_000  # optimizer steps (pilot budget; full study uses 200k)
PROBE_FRAC = 0.01  # 1% of budget
PROBE_STEPS = int(BUDGET * PROBE_FRAC)  # 100 steps
EVAL_INTERVAL = 50
SEEDS = [42, 123, 456, 789, 2026]

# Model: 2-layer Transformer, d_model=128, ~425k params
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2

# 4 encodings
ENCODINGS = [
    ("identity", IdentityBinaryEncoding(N_BITS)),
    ("gray", GrayEncoding(N_BITS)),
    ("gf2_a", generate_gf2_encoding(N_BITS, seed=1001)),
    ("gf2_b", generate_gf2_encoding(N_BITS, seed=2002)),
]

# 2 optimizers
OPTIMIZERS = [
    ("standard", {"optimizer": "adam", "weight_decay": 0.0}),
    ("grokking", {"optimizer": "adamw", "weight_decay": 0.3}),
]

OUTPUT_DIR = REPO_ROOT / "results" / "CAL_A"


# ============================================================================
# Data preparation
# ============================================================================

def prepare_data(encoding) -> Tuple[TokenDataset, TokenDataset, BitTokenizer, int]:
    """Prepare train/test datasets for q=113 with the given encoding."""
    # Load or generate latent manifest
    manifest_path = REPO_ROOT / "manifests" / "latent" / "q113.csv"
    if manifest_path.exists():
        manifest = load_latent_manifest(manifest_path)
    else:
        manifest = generate_latent_manifest(q=Q, seed=42, train_frac=TRAIN_FRAC,
                                             output_path=manifest_path)

    # Realize in multiplicative subgroup
    group = MultiplicativeSubgroup(q=Q, p=P, g0=G0)
    group.validate()

    train_samples = []
    test_samples = []
    for pair in manifest:
        base = group.element(pair.a)       # g0^a mod p
        target = group.element(pair.a * pair.x)  # g0^(a*x) mod p
        # Apply encoding to base and target
        encoded_base = encoding.encode_int(base)
        encoded_target = encoding.encode_int(target)
        sample = (encoded_base, encoded_target, pair.x, pair.latent_id)
        if pair.split == "train":
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    # Create tokenizer
    tokenizer = BitTokenizer(N_BITS)

    # Create datasets
    train_ds = TokenDataset(
        bases=[s[0] for s in train_samples],
        targets=[s[1] for s in train_samples],
        labels=[s[2] for s in train_samples],
        tokenizer=tokenizer,
    )
    test_ds = TokenDataset(
        bases=[s[0] for s in test_samples],
        targets=[s[1] for s in test_samples],
        labels=[s[2] for s in test_samples],
        tokenizer=tokenizer,
    )

    return train_ds, test_ds, tokenizer, Q


# ============================================================================
# P0 feature computation
# ============================================================================

def compute_p0_features_for_run(
    encoding_name: str,
    encoding,
    optimizer_name: str,
    seed: int,
) -> Dict:
    """Compute P0 features for a single run configuration."""
    from src.predictors.fourier import compute_fourier_features
    from src.predictors.gradients import compute_gradient_features

    train_ds, test_ds, tokenizer, q = prepare_data(encoding)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create model at initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GrokkingTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=tokenizer.max_len,
        dropout=0.0,
    ).to(device)
    model.eval()

    n_params = model.count_parameters()

    # Fourier features (intrinsic)
    targets_np = np.array(train_ds.labels, dtype=np.int64)
    fourier_feats = compute_fourier_features(targets_np, q, n_classes=q)

    # Gradient features (simplified for speed)
    # Use a subset for gradient computation
    max_grad_samples = min(256, len(train_ds))
    grad_feats = compute_gradient_features(
        model,
        tokenizer.encode_batch(train_ds.bases[:max_grad_samples],
                                 train_ds.targets[:max_grad_samples]),
        torch.tensor(train_ds.labels[:max_grad_samples], dtype=torch.long),
        n_classes=q,
        device=device,
        max_samples=max_grad_samples,
    )

    # Assemble feature row
    config_id = f"cal_a_{encoding_name}_{optimizer_name}_s{seed}"
    row = {
        "config_id": config_id,
        "q": q,
        "n_classes": q,
        "encoding_id": encoding.encoding_id(),
        "encoding_name": encoding_name,
        "encoding_family": encoding.family(),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "params": n_params,
        "optimizer": optimizer_name,
        "weight_decay": 0.3 if optimizer_name == "grokking" else 0.0,
        "target_type": "full_log",
        "seed": seed,
    }
    row.update(fourier_feats)
    row.update(grad_feats)

    return row


# ============================================================================
# Training
# ============================================================================

def run_single_experiment(
    encoding_name: str,
    encoding,
    optimizer_name: str,
    optimizer_config: Dict,
    seed: int,
    probe_only: bool = False,
    resume_only: bool = False,
) -> Dict:
    """Run a single experiment with probe/resume protocol."""
    config_id = f"cal_a_{encoding_name}_{optimizer_name}_s{seed}"
    print(f"\n{'='*60}")
    print(f"  {config_id}")
    print(f"{'='*60}")

    train_ds, test_ds, tokenizer, q = prepare_data(encoding)

    output_dir = OUTPUT_DIR / config_id
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_ckpt_path = output_dir / "probe_checkpoint.pt"

    if probe_only:
        # Phase 1: Probe (1% of budget)
        print(f"[PROBE] Training for {PROBE_STEPS} steps (1% of {BUDGET})...")
        config = TrainConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            epochs=BUDGET,
            lr=1e-3,
            eval_interval=EVAL_INTERVAL,
            early_stop_patience=10**9,  # disable early stopping in probe
            optimizer=optimizer_config["optimizer"],
            weight_decay=optimizer_config["weight_decay"],
            seed=seed,
            run_id=config_id,
            output_dir=str(OUTPUT_DIR),
            probe_steps=PROBE_STEPS,
        )
        summary = train(config, train_ds, test_ds, tokenizer, q)
        print(f"[PROBE] Done. Final train={summary['final_train']:.4f} test={summary['final_test']:.4f}")
        return summary

    if resume_only:
        # Phase 2: Resume from probe and train to completion
        if not probe_ckpt_path.exists():
            print(f"[RESUME] No probe checkpoint found at {probe_ckpt_path}, skipping")
            return {"run_id": config_id, "error": "no_probe_checkpoint"}

        print(f"[RESUME] Training from probe to {BUDGET} steps...")
        config = TrainConfig(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            epochs=BUDGET,
            lr=1e-3,
            eval_interval=EVAL_INTERVAL,
            early_stop_patience=50,
            early_stop_threshold=0.99,
            optimizer=optimizer_config["optimizer"],
            weight_decay=optimizer_config["weight_decay"],
            seed=seed,
            run_id=config_id,
            output_dir=str(OUTPUT_DIR),
            resume_from_probe=True,
            probe_checkpoint_path=str(probe_ckpt_path),
        )
        summary = train(config, train_ds, test_ds, tokenizer, q)
        print(f"[RESUME] Done. Best test={summary['best_test']:.4f} at epoch {summary['best_epoch']}")
        return summary

    # Full run (no probe/resume split, just train to completion)
    print(f"[FULL] Training for {BUDGET} steps...")
    config = TrainConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        epochs=BUDGET,
        lr=1e-3,
        eval_interval=EVAL_INTERVAL,
        early_stop_patience=50,
        early_stop_threshold=0.99,
        optimizer=optimizer_config["optimizer"],
        weight_decay=optimizer_config["weight_decay"],
        seed=seed,
        run_id=config_id,
        output_dir=str(OUTPUT_DIR),
    )
    summary = train(config, train_ds, test_ds, tokenizer, q)
    print(f"[FULL] Done. Best test={summary['best_test']:.4f} at epoch {summary['best_epoch']}")
    return summary


# ============================================================================
# Outcome classification
# ============================================================================

def classify_run_outcome(config_id: str) -> Dict:
    """Classify the outcome of a completed run using K=10 sustained thresholds."""
    metrics_path = OUTPUT_DIR / config_id / "metrics.jsonl"
    if not metrics_path.exists():
        return {"config_id": config_id, "error": "no_metrics"}

    metrics = []
    with open(metrics_path) as f:
        for line in f:
            metrics.append(json.loads(line))

    if not metrics:
        return {"config_id": config_id, "error": "empty_metrics"}

    # Extract curves
    steps = [m["epoch"] for m in metrics]
    train_accs = [m["train_acc"] for m in metrics]
    test_accs = [m["test_acc"] for m in metrics]

    # Classify using outcomes.py
    outcome = classify_outcome(
        steps=steps,
        train_accs=train_accs,
        test_accs=test_accs,
        budget=BUDGET,
        k=10,
        mem_threshold=0.99,
        gen_threshold=0.90,
    )

    return {
        "config_id": config_id,
        "phase": outcome.phase.value,
        "T_mem": outcome.t_mem,
        "T_gen": outcome.t_gen,
        "best_test": outcome.best_test,
        "final_test": outcome.final_test,
        "total_epochs": outcome.total_steps,
        "censored": outcome.t_gen is None and outcome.t_mem is not None,
    }


# ============================================================================
# Main
# ============================================================================

def run_all(probe_only=False, resume_only=False, full_only=False):
    """Run all 40 experiments."""
    # First compute P0 features for all configs
    print("\n" + "="*60)
    print("  Computing P0 features for all 40 configurations")
    print("="*60)

    p0_features = []
    for enc_name, encoding in ENCODINGS:
        for opt_name, opt_config in OPTIMIZERS:
            for seed in SEEDS:
                print(f"  P0: {enc_name} / {opt_name} / s{seed}")
                feats = compute_p0_features_for_run(enc_name, encoding, opt_name, seed)
                p0_features.append(feats)

    # Save P0 feature table
    p0_path = OUTPUT_DIR / "p0_features.csv"
    p0_path.parent.mkdir(parents=True, exist_ok=True)
    if p0_features:
        with open(p0_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=p0_features[0].keys())
            writer.writeheader()
            writer.writerows(p0_features)
        print(f"\nSaved P0 features: {p0_path} ({len(p0_features)} rows)")

    # Run experiments
    if not full_only:
        print("\n" + "="*60)
        print("  Running experiments")
        print("="*60)

    summaries = []
    for enc_name, encoding in ENCODINGS:
        for opt_name, opt_config in OPTIMIZERS:
            for seed in SEEDS:
                config_id = f"cal_a_{enc_name}_{opt_name}_s{seed}"

                # Check if already completed
                summary_path = OUTPUT_DIR / config_id / "summary.json"
                if summary_path.exists() and not probe_only and not resume_only:
                    with open(summary_path) as f:
                        existing = json.load(f)
                    if existing.get("total_epochs", 0) >= BUDGET * 0.99 or existing.get("best_test", 0) >= 0.99:
                        print(f"  [SKIP] {config_id} already completed")
                        summaries.append(existing)
                        continue

                if probe_only:
                    summary = run_single_experiment(
                        enc_name, encoding, opt_name, opt_config, seed,
                        probe_only=True,
                    )
                elif resume_only:
                    summary = run_single_experiment(
                        enc_name, encoding, opt_name, opt_config, seed,
                        resume_only=True,
                    )
                else:
                    summary = run_single_experiment(
                        enc_name, encoding, opt_name, opt_config, seed,
                    )
                summaries.append(summary)

    return summaries


def analyze_results():
    """Analyze CAL_A results and classify outcomes."""
    print("\n" + "="*60)
    print("  Analyzing CAL_A results")
    print("="*60)

    outcomes = []
    for enc_name, _ in ENCODINGS:
        for opt_name, _ in OPTIMIZERS:
            for seed in SEEDS:
                config_id = f"cal_a_{enc_name}_{opt_name}_s{seed}"
                outcome = classify_run_outcome(config_id)
                outcomes.append(outcome)
                print(f"  {config_id}: {outcome.get('phase', 'ERROR')} "
                      f"T_mem={outcome.get('T_mem', 'N/A')} "
                      f"T_gen={outcome.get('T_gen', 'N/A')} "
                      f"best_test={outcome.get('best_test', 0):.4f}")

    # Save outcomes
    outcomes_path = OUTPUT_DIR / "outcomes.csv"
    if outcomes:
        with open(outcomes_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=outcomes[0].keys())
            writer.writeheader()
            writer.writerows(outcomes)
        print(f"\nSaved outcomes: {outcomes_path}")

    # Summary statistics
    phases = [o.get("phase", "ERROR") for o in outcomes]
    unique, counts = np.unique(phases, return_counts=True)
    print("\nPhase distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    # Check if features vary across encodings
    p0_path = OUTPUT_DIR / "p0_features.csv"
    if p0_path.exists():
        import pandas as pd
        df = pd.read_csv(p0_path)
        print("\nP0 feature variation across encodings:")
        for feat in ["fourier_entropy", "fourier_PR", "fourier_top1",
                      "grad_mean_norm", "GSNR", "grad_eff_rank"]:
            if feat in df.columns:
                grouped = df.groupby("encoding_name")[feat].agg(["mean", "std"])
                print(f"\n  {feat}:")
                print(grouped.to_string())

    return outcomes


def main():
    parser = argparse.ArgumentParser(description="CAL_A: Clean q=113 pilot")
    parser.add_argument("--probe-only", action="store_true",
                        help="Only run the 1-percent probe phase")
    parser.add_argument("--resume-only", action="store_true",
                        help="Only resume from probe to completion")
    parser.add_argument("--full-only", action="store_true",
                        help="Skip probe/resume, run full training directly")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results")
    parser.add_argument("--p0-only", action="store_true",
                        help="Only compute P0 features")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        analyze_results()
        return

    if args.p0_only:
        run_all(full_only=True)
        return

    if args.probe_only:
        run_all(probe_only=True)
        return

    if args.resume_only:
        run_all(resume_only=True)
        analyze_results()
        return

    # Default: full run
    run_all(full_only=not args.probe_only and not args.resume_only)
    analyze_results()


if __name__ == "__main__":
    main()
