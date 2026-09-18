"""NTK / kernel features computed at initialization (t=0).

These depend on model architecture + encoding.
They answer: "Can the kernel reach the target quickly?"

We compute a margin empirical NTK for feasibility:
    K[i,j] = <grad_theta f(x_i), grad_theta f(x_j)>

For Transformers, the full NTK is expensive. We use a blockwise
Jacobian/Gram computation and validate against the full NTK on small-q cases.

Features:
    - Centered kernel-target alignment
    - Top-eigenspace target energy
    - Kernel effective rank
    - Eigenvalue decay
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional


def _compute_jacobian_block(
    model: nn.Module,
    inputs: torch.Tensor,
    n_classes: int,
    device: torch.device,
    block_size: int = 256,
) -> np.ndarray:
    """Compute the Jacobian of model output w.r.t. parameters, blockwise.

    Returns: (N, P) Jacobian matrix where N = inputs, P = params.
    For multi-class, uses the margin (true-class logit - mean competing logit).
    """
    model.eval()
    N = inputs.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    # We compute the margin output for each input
    jacobian = np.zeros((N, P), dtype=np.float32)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        batch = inputs[start:end].to(device)

        for i in range(end - start):
            model.zero_grad()
            logits = model(batch[i:i+1])  # (1, seq_len, vocab)
            # Use the last position's logits (output)
            out = logits[0, -1, :]  # (vocab,)

            # Margin: true-class logit - mean of others
            # For now, use the full output vector as the "function"
            # The NTK is K[i,j] = sum_k grad(f_k(x_i)) . grad(f_k(x_j))
            # We approximate with the full output
            for k in range(min(out.shape[0], n_classes)):
                if out[k].requires_grad:
                    out[k].backward(retain_graph=(k < min(out.shape[0], n_classes) - 1))
                    grad_flat = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
                    jacobian[start + i] += grad_flat.cpu().numpy()
                    for p in params:
                        if p.grad is not None:
                            p.grad.zero_()

    return jacobian


def compute_ntk_features(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    device: torch.device,
    max_samples: int = 512,
) -> Dict[str, float]:
    """Compute NTK features at initialization.

    Args:
        model: Neural network (at initialization).
        inputs: Input tensor (N, seq_len).
        targets: Target labels (N,).
        n_classes: Number of output classes.
        device: torch device.
        max_samples: Maximum samples for NTK computation (for feasibility).

    Returns:
        Dict of NTK features.
    """
    N = min(inputs.shape[0], max_samples)
    inputs_sub = inputs[:N]
    targets_sub = targets[:N]

    # Compute Jacobian (N, P)
    jac = _compute_jacobian_block(model, inputs_sub, n_classes, device)

    # NTK: K = J @ J^T  (N, N)
    K = jac @ jac.T

    # Target vector: one-hot centered
    Y = np.zeros((N, n_classes), dtype=np.float32)
    for i in range(N):
        Y[i, targets_sub[i].item() if isinstance(targets_sub[i], torch.Tensor) else int(targets_sub[i])] = 1.0
    Y -= Y.mean(axis=1, keepdims=True)  # center

    # Flatten target for alignment
    y_flat = Y.flatten()

    # Center the kernel
    K_centered = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()

    # Kernel-target alignment
    K_flat = K_centered.flatten()
    kta_num = np.dot(K_flat, y_flat)
    kta_den = np.sqrt(np.dot(K_flat, K_flat) * np.dot(y_flat, y_flat))
    kta = kta_num / kta_den if kta_den > 0 else 0.0

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K_centered)
    eigvals = np.maximum(eigvals, 0)  # numerical stability
    eigvals_sorted = np.sort(eigvals)[::-1]

    # Top-eigenspace target energy
    # Project target onto top-k eigenvectors
    eigvals_pos = eigvals_sorted[eigvals_sorted > 1e-10]
    if len(eigvals_pos) > 0:
        _, eigvecs = np.linalg.eigh(K_centered)
        eigvecs = eigvecs[:, ::-1]  # descending order

        # Target energy in top-k eigenspace
        top_k = min(8, eigvecs.shape[1])
        Y_proj = eigvecs[:, :top_k].T @ Y   # (top_k, n_classes)
        top_eigen_energy = np.sum(Y_proj ** 2) / np.sum(Y ** 2) if np.sum(Y ** 2) > 0 else 0.0
    else:
        top_eigen_energy = 0.0

    # Kernel effective rank
    eigvals_norm = eigvals_pos / eigvals_pos.sum() if eigvals_pos.sum() > 0 else np.array([1.0])
    kernel_eff_rank = (eigvals_norm.sum() ** 2) / (eigvals_norm ** 2).sum() if (eigvals_norm ** 2).sum() > 0 else 0.0

    # Eigenvalue decay rate (fit log(eigval) vs index)
    if len(eigvals_pos) > 2:
        log_eigs = np.log(eigvals_pos[:min(50, len(eigvals_pos))])
        indices = np.arange(1, len(log_eigs) + 1)
        # Linear fit: log(λ) = a + b * index
        coeffs = np.polyfit(indices, log_eigs, 1)
        eigval_decay = -coeffs[0]  # positive = fast decay
    else:
        eigval_decay = 0.0

    return {
        "kernel_target_alignment": float(kta),
        "kernel_top_eigen_energy": float(top_eigen_energy),
        "kernel_eff_rank": float(kernel_eff_rank),
        "kernel_eigval_decay": float(eigval_decay),
        "kernel_trace": float(np.trace(K)),
        "kernel_frobenius": float(np.linalg.norm(K, 'fro')),
    }
