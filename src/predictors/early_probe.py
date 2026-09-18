"""Early-probe (P1) features: kernel/gradient drift during 1% of training.

Computed after training for exactly probe_steps (1% of budget).
Measures how the kernel/gradient is MOVING away from the initial regime.

Features:
    - Δ kernel-target alignment
    - d(alignment)/dt
    - ||K_probe - K_0||_F / ||K_0||_F (kernel drift)
    - Eigenspace principal-angle drift
    - Effective-rank drift
    - Δ T_spec proxy
    - Δ KSA
    - GSNR drift
    - Gradient-rank drift
    - Training-loss slope
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional

from .ntk import _compute_jacobian_block, compute_ntk_features
from .kernel_spectrum import compute_kernel_spectral_features
from .gradients import compute_gradient_features


def build_p1_features(
    model_init: torch.nn.Module,
    model_probe: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    device: torch.device,
    probe_loss_curve: List[float] = None,
    probe_steps: int = None,
    max_samples: int = 256,
) -> Dict[str, float]:
    """Compute P1 early-probe drift features.

    Args:
        model_init: Model at initialization (t=0).
        model_probe: Model after 1% of training.
        inputs: Input tensor.
        targets: Target labels.
        n_classes: Number of output classes.
        device: torch device.
        probe_loss_curve: Loss values during probe window.
        probe_steps: Number of steps in probe window.

    Returns:
        Dict of P1 drift features.
    """
    N = min(inputs.shape[0], max_samples)
    inputs_sub = inputs[:N]
    targets_sub = targets[:N]
    targets_np = targets_sub.cpu().numpy() if isinstance(targets_sub, torch.Tensor) else np.array(targets_sub[:N])

    # Compute kernel at init and at probe
    jac_init = _compute_jacobian_block(model_init, inputs_sub, n_classes, device)
    K_init = jac_init @ jac_init.T

    jac_probe = _compute_jacobian_block(model_probe, inputs_sub, n_classes, device)
    K_probe = jac_probe @ jac_probe.T

    # Centered one-hot target
    Y = np.zeros((N, n_classes), dtype=np.float32)
    for j in range(N):
        Y[j, int(targets_np[j])] = 1.0
    Y -= Y.mean(axis=1, keepdims=True)

    # Kernel features at init and probe
    feats_init = compute_kernel_spectral_features(K_init, Y)
    feats_probe = compute_kernel_spectral_features(K_probe, Y)

    # NTK features at init and probe
    ntk_init = compute_ntk_features(model_init, inputs_sub, targets_sub, n_classes, device, max_samples=N)
    ntk_probe = compute_ntk_features(model_probe, inputs_sub, targets_sub, n_classes, device, max_samples=N)

    # Gradient features at init and probe
    grad_init = compute_gradient_features(model_init, inputs, targets, n_classes, device, max_samples=N)
    grad_probe = compute_gradient_features(model_probe, inputs, targets, n_classes, device, max_samples=N)

    # Compute drift features
    features = {}

    # Kernel drift
    K_diff = K_probe - K_init
    K_drift = np.linalg.norm(K_diff, 'fro') / max(np.linalg.norm(K_init, 'fro'), 1e-10)
    features["kernel_drift_frobenius"] = float(K_drift)

    # Alignment drift
    features["delta_kernel_target_alignment"] = float(
        ntk_probe["kernel_target_alignment"] - ntk_init["kernel_target_alignment"]
    )

    # d(alignment)/dt
    if probe_steps and probe_steps > 0:
        features["alignment_rate"] = features["delta_kernel_target_alignment"] / probe_steps
    else:
        features["alignment_rate"] = 0.0

    # Eigenspace angle drift (principal angles between eigenvector bases)
    try:
        _, eigvecs_init = np.linalg.eigh(K_init)
        _, eigvecs_probe = np.linalg.eigh(K_probe)
        # Subspace angle: ||U_init U_init^T - U_probe U_probe^T||_F
        proj_diff = eigvecs_init @ eigvecs_init.T - eigvecs_probe @ eigvecs_probe.T
        features["eigenspace_angle_drift"] = float(np.linalg.norm(proj_diff, 'fro'))
    except np.linalg.LinAlgError:
        features["eigenspace_angle_drift"] = 0.0

    # Effective rank drift
    features["delta_kernel_eff_rank"] = float(
        feats_probe["kernel_eff_rank"] - feats_init["kernel_eff_rank"]
    )

    # T_spec drift
    features["delta_T_spec"] = float(
        feats_probe["T_spec_primary"] - feats_init["T_spec_primary"]
    )

    # KSA drift
    features["delta_KSA"] = float(
        feats_probe["KSA_primary"] - feats_init["KSA_primary"]
    )

    # GSNR drift
    features["delta_GSNR"] = float(
        grad_probe["GSNR"] - grad_init["GSNR"]
    )

    # Gradient rank drift
    features["delta_grad_eff_rank"] = float(
        grad_probe["grad_eff_rank"] - grad_init["grad_eff_rank"]
    )

    # Training loss slope
    if probe_loss_curve and len(probe_loss_curve) > 1:
        losses = np.array(probe_loss_curve)
        # Fit linear slope
        x = np.arange(len(losses))
        slope = np.polyfit(x, losses, 1)[0]
        features["loss_slope"] = float(slope)
        # Relative loss decrease
        if losses[0] > 0:
            features["loss_relative_decrease"] = float((losses[0] - losses[-1]) / losses[0])
        else:
            features["loss_relative_decrease"] = 0.0
    else:
        features["loss_slope"] = 0.0
        features["loss_relative_decrease"] = 0.0

    return features
