"""Timescale models: T_spec and Song & Ye connection.

Implements:
1. Fixed-kernel timescale T_spec as a baseline predictor
2. Song & Ye's T_mem/capacity timescale as Baseline 5
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional


def fit_timescale_model(
    features: Dict,
    q: int,
    N: int,
    P: int,
) -> Dict[str, float]:
    """Compute timescale-based baseline predictions.

    Baseline 5 (Song & Ye): uses T_mem and capacity ratios.
    T_spec: uses the fixed-kernel spectral accessibility time.

    Args:
        features: P0 feature dict (must contain T_spec and KSA).
        q: Group order.
        N: Number of training samples.
        P: Number of parameters.

    Returns:
        Dict of timescale predictions.
    """
    T_spec = features.get("T_spec_primary", float('inf'))
    KSA = features.get("KSA_primary", 0.0)

    # Song & Ye baseline: T_gen ~ T_mem * f(capacity)
    # T_mem ~ q / N (memorization time scales with group size / samples)
    # Capacity factor: more params -> faster memorization
    T_mem_proxy = q / max(N, 1)
    capacity_factor = np.log(max(P, 2)) / np.log(max(q, 2))
    T_gen_song_ye = T_mem_proxy * capacity_factor

    # T_spec scaling: T_spec is in normalized gradient-flow time
    # To convert to optimizer steps, multiply by learning rate factor
    # For now, use T_spec directly as the prediction
    T_gen_spec = T_spec

    return {
        "T_gen_T_spec": float(T_gen_spec),
        "T_gen_song_ye": float(T_gen_song_ye),
        "T_mem_proxy": float(T_mem_proxy),
        "capacity_factor": float(capacity_factor),
        "KSA_primary": float(KSA),
    }
