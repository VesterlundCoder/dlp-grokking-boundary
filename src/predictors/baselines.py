"""Baseline predictors for the prediction hierarchy.

Baseline 0: Empirical class frequency (chance)
Baseline 1: log(q) only
Baseline 2: log(q) + log(N) + log(P) + target + opt + WD
Baseline 3: Fourier descriptors only
Baseline 4: Kernel-target alignment only
Baseline 5: T_mem/capacity timescale (Song & Ye)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


def compute_baselines(
    features: List[Dict],
    outcomes: Optional[List[Dict]] = None,
) -> Dict[str, Dict]:
    """Compute baseline predictions for a set of configurations.

    Args:
        features: List of P0 feature dicts.
        outcomes: Optional list of outcome dicts (for empirical frequency baseline).

    Returns:
        Dict of baseline_name -> {predictions, feature_names}
    """
    n = len(features)
    baselines = {}

    # --- Baseline 0: Empirical class frequency ---
    if outcomes is not None:
        phases = [o.get("phase", "UNKNOWN") for o in outcomes]
        unique, counts = np.unique(phases, return_counts=True)
        freq = dict(zip(unique, counts / n))
        baselines["baseline_0_empirical"] = {
            "predictions": [{"phase_proba": freq, "log_T_gen": np.mean([o.get("log_T_gen", 0) for o in outcomes])}] * n,
            "description": "Empirical class frequency",
        }
    else:
        baselines["baseline_0_empirical"] = {
            "predictions": [{"phase_proba": {}, "log_T_gen": 0.0}] * n,
            "description": "Empirical class frequency (no outcomes available)",
        }

    # --- Baseline 1: log(q) only ---
    log_q = np.array([np.log(max(f.get("q", 2), 2)) for f in features])
    baselines["baseline_1_log_q"] = {
        "predictions": [{"log_T_gen": float(lq)} for lq in log_q],
        "feature_names": ["log_q"],
        "description": "log(q) only",
    }

    # --- Baseline 2: log(q) + log(N) + log(P) + target + opt + WD ---
    log_N = np.array([np.log(max(f.get("n_train", 1), 1)) for f in features])
    log_P = np.array([np.log(max(f.get("params", 1), 1)) for f in features])
    # Simple linear: log_T_gen ~ a*log_q + b*log_N + c*log_P + d
    X_b2 = np.column_stack([log_q, log_N, log_P, np.ones(n)])
    baselines["baseline_2_config"] = {
        "predictions": [{"log_T_gen": float(v)} for v in X_b2 @ np.array([1.0, -0.5, -0.5, 0.0])],
        "feature_names": ["log_q", "log_N", "log_P"],
        "description": "log(q) + log(N) + log(P) configuration features",
    }

    # --- Baseline 3: Fourier descriptors only ---
    fourier_cols = ["fourier_entropy", "fourier_PR", "fourier_top1", "fourier_top4", "fourier_top8"]
    fourier_vals = np.array([[f.get(c, 0.0) for c in fourier_cols] for f in features])
    baselines["baseline_3_fourier"] = {
        "predictions": [{"log_T_gen": float(-v[0])} for v in fourier_vals],  # higher entropy -> lower T_gen
        "feature_names": fourier_cols,
        "description": "Fourier descriptors only",
    }

    # --- Baseline 4: Kernel-target alignment only ---
    kta = np.array([f.get("kernel_target_alignment", 0.0) for f in features])
    baselines["baseline_4_kta"] = {
        "predictions": [{"log_T_gen": float(-k)} for k in kta],  # higher alignment -> lower T_gen
        "feature_names": ["kernel_target_alignment"],
        "description": "Kernel-target alignment only",
    }

    # --- Baseline 5: Song & Ye timescale ---
    T_gen_spec = np.array([f.get("T_spec_primary", 0.0) for f in features])
    baselines["baseline_5_song_ye"] = {
        "predictions": [{"log_T_gen": float(np.log(max(t, 1.0)))} for t in T_gen_spec],
        "feature_names": ["T_spec_primary"],
        "description": "Song & Ye T_mem/capacity timescale",
    }

    return baselines
