"""Fourier features of the target function on a cyclic group.

These are INTRINSIC features: they depend on the task + encoding
but NOT on model initialization. They answer:
"Is the target function spectrally concentrated?"

For a cyclic group Z_q, the characters are:
    chi_k(x) = exp(2*pi*i*k*x/q), k = 0, 1, ..., q-1

We compute the DFT of the target function and measure:
    - Fourier entropy H_F
    - Effective support exp(H_F)
    - Participation ratio PR_F
    - Top-1, top-4, top-8, top-16 energy
    - Spectral effective rank
"""
from __future__ import annotations

import numpy as np
from typing import Dict


def compute_fourier_features(
    targets: np.ndarray,  # (N,) target values for N samples
    q: int,               # group order
    n_classes: int = None, # number of classes (q for full-log, 2 for parity)
) -> Dict[str, float]:
    """Compute Fourier features of the target function.

    For full-log DLP: targets are integers in [0, q-1], we compute
    the DFT of the one-hot encoded target function.

    For parity: targets are {0, 1}, we compute DFT of the ±1 valued function.

    Args:
        targets: Array of target values.
        q: Group order.
        n_classes: Number of classes. If None, inferred from targets.

    Returns:
        Dict of Fourier feature values.
    """
    if n_classes is None:
        n_classes = int(targets.max()) + 1

    if n_classes == 2:
        # Parity: convert to ±1 and compute DFT
        y = 2.0 * targets.astype(np.float64) - 1.0  # {0,1} -> {-1,+1}
        # Compute DFT over the group
        # For parity, the function is defined on Z_q
        # We need the full function, not just samples
        # Use the empirical distribution as approximation
        spectrum = np.fft.fft(y)
        power = np.abs(spectrum) ** 2
    else:
        # Full-log: compute DFT of one-hot target
        # For each class k, compute the indicator function
        # Then take the DFT
        # For simplicity, compute the DFT of the centered one-hot
        N = len(targets)
        # Centered one-hot: Y[i, k] = 1(targets[i]==k) - 1/n_classes
        Y = np.zeros((N, n_classes), dtype=np.float64)
        for k in range(n_classes):
            Y[:, k] = (targets == k).astype(np.float64) - 1.0 / n_classes

        # Compute DFT for each class dimension
        # The "target spectrum" is the set of |DFT(Y[:,k])|^2
        power = np.zeros(N, dtype=np.float64)
        for k in range(n_classes):
            spectrum_k = np.fft.fft(Y[:, k])
            power += np.abs(spectrum_k) ** 2

    # Normalize
    total_power = power.sum()
    if total_power == 0:
        total_power = 1.0
    normalized_power = power / total_power

    # Fourier entropy
    p = normalized_power[normalized_power > 0]
    H_F = -np.sum(p * np.log2(p))

    # Effective support
    eff_support = np.exp(H_F * np.log(2))  # convert from log2 to natural

    # Participation ratio
    PR_F = np.sum(normalized_power ** 2) ** 2 / np.sum(normalized_power ** 4) if np.sum(normalized_power ** 4) > 0 else 0.0

    # Top-k energy
    sorted_power = np.sort(normalized_power)[::-1]
    top1 = sorted_power[:1].sum()
    top4 = sorted_power[:4].sum()
    top8 = sorted_power[:8].sum()
    top16 = sorted_power[:16].sum()

    # Spectral effective rank (participation ratio variant)
    p2 = normalized_power[normalized_power > 0]
    spectral_eff_rank = (p2.sum() ** 2) / (p2 ** 2).sum() if (p2 ** 2).sum() > 0 else 0.0

    return {
        "fourier_entropy": float(H_F),
        "fourier_eff_support": float(eff_support),
        "fourier_PR": float(PR_F),
        "fourier_top1": float(top1),
        "fourier_top4": float(top4),
        "fourier_top8": float(top8),
        "fourier_top16": float(top16),
        "fourier_spectral_eff_rank": float(spectral_eff_rank),
    }
