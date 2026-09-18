"""Kernel spectral accessibility features.

Given the NTK eigendecomposition K = U Λ U^T and target projections
C_j = U_j^T Y, we compute:

1. T_spec(ε) = inf{s : E(s) ≤ ε}
   where E(s) = Σ_j ||C_j||² exp(-2λ_j s) / Σ_j ||C_j||²

2. KSA_τ = Σ_j [λ_j / (λ_j + τ)] ||C_j||² / ||Y||²

These are the PRIMARY spectral accessibility predictors.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional


def compute_T_spec(
    eigvals: np.ndarray,   # (n,) eigenvalues of kernel K
    target_projections: np.ndarray,  # (n, n_classes) projections C_j = U_j^T Y
    epsilon: float = 0.10,
) -> float:
    """Compute spectral accessibility time T_spec(epsilon).

    Under fixed-kernel gradient flow, the residual energy decays as:
        E(s) = Σ_j ||C_j||² exp(-2λ_j s) / Σ_j ||C_j||²

    T_spec is the first s where E(s) ≤ epsilon.

    Args:
        eigvals: Kernel eigenvalues (sorted descending).
        target_projections: Target energy in each eigenmode.
        epsilon: Energy threshold.

    Returns:
        T_spec (in normalized gradient-flow time units).
    """
    # Energy per mode
    energy_per_mode = np.sum(target_projections ** 2, axis=1)  # (n,)
    total_energy = energy_per_mode.sum()

    if total_energy <= 0:
        return float('inf')

    # Filter to positive eigenvalues
    mask = eigvals > 1e-10
    if not mask.any():
        return float('inf')

    lam = eigvals[mask]
    energy = energy_per_mode[mask]

    # Binary search for T_spec
    # E(s) = Σ energy_j * exp(-2*lam_j*s) / total_energy ≤ epsilon

    s_low, s_high = 0.0, 1e6
    for _ in range(100):
        s_mid = (s_low + s_high) / 2
        E = np.sum(energy * np.exp(-2 * lam * s_mid)) / total_energy
        if E <= epsilon:
            s_high = s_mid
        else:
            s_low = s_mid

    return float(s_high)


def compute_KSA(
    eigvals: np.ndarray,
    target_projections: np.ndarray,
    tau: float,
) -> float:
    """Compute Kernel Spectral Accessibility KSA_tau.

    KSA_τ = Σ_j [λ_j / (λ_j + τ)] ||C_j||² / ||Y||²

    Higher KSA = more target energy in high-eigenvalue modes = more accessible.

    Args:
        eigvals: Kernel eigenvalues.
        target_projections: Target energy per eigenmode.
        tau: Regularization scale.

    Returns:
        KSA value in [0, 1].
    """
    energy_per_mode = np.sum(target_projections ** 2, axis=1)
    total_energy = energy_per_mode.sum()

    if total_energy <= 0:
        return 0.0

    mask = eigvals > 1e-10
    if not mask.any():
        return 0.0

    lam = eigvals[mask]
    energy = energy_per_mode[mask]

    ksa = np.sum((lam / (lam + tau)) * energy) / total_energy
    return float(ksa)


def compute_kernel_spectral_features(
    K: np.ndarray,           # (N, N) kernel matrix
    Y: np.ndarray,           # (N, n_classes) target one-hot (centered)
    tau_0: Optional[float] = None,
    epsilons: list = [0.20, 0.10, 0.05],
    tau_ratios: list = [0.25, 0.5, 1.0, 2.0, 4.0],
) -> Dict[str, float]:
    """Compute all kernel spectral accessibility features.

    Args:
        K: Kernel matrix (N, N).
        Y: Centered one-hot target (N, n_classes).
        tau_0: Primary tau scale. If None, set to trace(K)/N.
        epsilons: List of epsilon values for T_spec.
        tau_ratios: List of tau/tau_0 ratios for KSA sensitivity.

    Returns:
        Dict of spectral features.
    """
    N = K.shape[0]

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(K)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvals = np.maximum(eigvals, 0)  # numerical stability

    # Target projections: C_j = U_j^T Y
    target_projections = eigvecs.T @ Y  # (N, n_classes)

    # Primary tau
    if tau_0 is None:
        tau_0 = np.trace(K) / N if N > 0 else 1.0
    if tau_0 <= 0:
        tau_0 = 1.0

    features = {}

    # T_spec for each epsilon
    for eps in epsilons:
        t_spec = compute_T_spec(eigvals, target_projections, epsilon=eps)
        features[f"T_spec_{eps:.2f}"] = t_spec

    # KSA for each tau ratio
    for ratio in tau_ratios:
        tau = tau_0 * ratio
        ksa = compute_KSA(eigvals, target_projections, tau)
        features[f"KSA_{ratio:.2f}tau0"] = ksa

    # Primary values
    features["T_spec_primary"] = features.get("T_spec_0.10", float('inf'))
    features["KSA_primary"] = features.get("KSA_1.00tau0", 0.0)
    features["tau_0"] = float(tau_0)

    return features
