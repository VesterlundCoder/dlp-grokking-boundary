"""Gradient accessibility features computed at initialization (t=0).

These depend on model + task + init. They answer:
"Does the gradient signal point in the right direction?"

Features:
    - ||E[g]||: Mean gradient norm (the theorem's bounded quantity)
    - E[||g||²]: Second moment of gradient
    - Var(g): Gradient variance
    - GSNR = ||E[g]||² / E[||g - E[g]||²]: Gradient signal-to-noise ratio
    - Pairwise gradient cosine similarity
    - Gradient covariance effective rank
    - Class-conditional gradient separation
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional


def compute_gradient_features(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    device: torch.device,
    criterion: nn.Module = None,
    max_samples: int = 512,
) -> Dict[str, float]:
    """Compute gradient features at initialization.

    Args:
        model: Neural network (at initialization).
        inputs: Input tensor (N, seq_len).
        targets: Target labels (N,).
        n_classes: Number of output classes.
        device: torch device.
        criterion: Loss function. Default: CrossEntropyLoss.
        max_samples: Maximum samples for gradient computation.

    Returns:
        Dict of gradient features.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    N = min(inputs.shape[0], max_samples)

    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    # Compute per-sample gradients
    gradients = np.zeros((N, P), dtype=np.float32)

    for i in range(N):
        model.zero_grad()
        x = inputs[i:i+1].to(device)
        logits = model(x)
        out = logits[0, -1, :]  # last position
        target = targets[i]
        if isinstance(target, torch.Tensor):
            target = target.item()
        target = torch.tensor([int(target)], device=device, dtype=torch.long)

        loss = criterion(out.unsqueeze(0), target)
        loss.backward()

        grad_flat = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        gradients[i] = grad_flat.cpu().numpy()

    # Mean gradient
    mean_grad = gradients.mean(axis=0)  # (P,)
    mean_grad_norm = np.linalg.norm(mean_grad)

    # Second moment
    second_moment = np.mean(np.sum(gradients ** 2, axis=1))

    # Variance
    grad_var = np.mean(np.sum((gradients - mean_grad) ** 2, axis=1))

    # GSNR
    gsnr = mean_grad_norm ** 2 / grad_var if grad_var > 0 else 0.0

    # Pairwise cosine similarity (sample if N is large)
    n_pairs = min(N, 100)
    sample_idx = np.random.RandomState(42).choice(N, n_pairs, replace=False)
    grad_sample = gradients[sample_idx]
    norms = np.linalg.norm(grad_sample, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = grad_sample / norms
    cos_sim_matrix = normalized @ normalized.T
    # Mean off-diagonal
    mask = ~np.eye(n_pairs, dtype=bool)
    mean_cosine = cos_sim_matrix[mask].mean()

    # Gradient effective rank
    # Use SVD of the gradient matrix
    try:
        svd_vals = np.linalg.svd(gradients, compute_uv=False)
        svd_norm = svd_vals / svd_vals.sum() if svd_vals.sum() > 0 else np.array([1.0])
        grad_eff_rank = (svd_norm.sum() ** 2) / (svd_norm ** 2).sum() if (svd_norm ** 2).sum() > 0 else 0.0
    except np.linalg.LinAlgError:
        grad_eff_rank = 0.0

    # Class-conditional gradient separation
    targets_np = targets[:N].cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets[:N])
    class_means = {}
    for c in range(min(n_classes, n_classes)):
        mask_c = targets_np == c
        if mask_c.sum() > 0:
            class_means[c] = gradients[mask_c].mean(axis=0)

    if len(class_means) >= 2:
        # Mean pairwise distance between class-conditional means
        class_keys = list(class_means.keys())
        separations = []
        for i in range(len(class_keys)):
            for j in range(i+1, len(class_keys)):
                sep = np.linalg.norm(class_means[class_keys[i]] - class_means[class_keys[j]])
                separations.append(sep)
        class_sep = np.mean(separations) if separations else 0.0
    else:
        class_sep = 0.0

    return {
        "grad_mean_norm": float(mean_grad_norm),
        "grad_second_moment": float(second_moment),
        "grad_variance": float(grad_var),
        "GSNR": float(gsnr),
        "grad_mean_cosine": float(mean_cosine),
        "grad_eff_rank": float(grad_eff_rank),
        "grad_class_separation": float(class_sep),
    }
