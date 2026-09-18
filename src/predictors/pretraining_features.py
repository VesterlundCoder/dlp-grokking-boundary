"""Build the P0 pre-training feature table.

Assembles all t=0 features (Fourier, kernel, gradient) into a single
table with one row per experiment configuration.
"""
from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

from .fourier import compute_fourier_features
from .ntk import compute_ntk_features
from .kernel_spectrum import compute_kernel_spectral_features
from .gradients import compute_gradient_features


def build_p0_feature_table(
    configs: List[Dict],
    model_factory,           # callable: (config) -> nn.Module
    data_factory,            # callable: (config) -> (inputs, targets, n_classes, q)
    device: torch.device,
    max_kernel_samples: int = 256,
    max_gradient_samples: int = 256,
) -> List[Dict]:
    """Build the P0 feature table for a list of configurations.

    Args:
        configs: List of configuration dicts.
        model_factory: Function that creates a model from a config.
        data_factory: Function that returns (inputs, targets, n_classes, q) from a config.
        device: torch device.
        max_kernel_samples: Max samples for NTK computation.
        max_gradient_samples: Max samples for gradient computation.

    Returns:
        List of feature dicts, one per configuration.
    """
    results = []

    for i, config in enumerate(configs):
        config_id = config.get("config_id", f"config_{i}")
        print(f"[{i+1}/{len(configs)}] Computing P0 features for {config_id}...")

        # Create model at initialization
        model = model_factory(config)
        model.to(device)
        model.eval()

        # Get data
        inputs, targets, n_classes, q = data_factory(config)

        # Fourier features (intrinsic, no model needed)
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
        fourier_feats = compute_fourier_features(targets_np, q, n_classes)

        # NTK / kernel features
        ntk_feats = compute_ntk_features(
            model, inputs, targets, n_classes, device, max_samples=max_kernel_samples,
        )

        # Kernel spectral features (T_spec, KSA)
        # Reconstruct kernel from NTK Jacobian
        # For efficiency, we reuse the NTK computation
        # The ntk module already computed K = J @ J^T
        # We need to recompute it here for the spectral features
        from .ntk import _compute_jacobian_block
        N = min(inputs.shape[0], max_kernel_samples)
        inputs_sub = inputs[:N]
        targets_sub = targets[:N]
        jac = _compute_jacobian_block(model, inputs_sub, n_classes, device)
        K = jac @ jac.T

        # Centered one-hot target
        Y = np.zeros((N, n_classes), dtype=np.float32)
        targets_np_sub = targets_sub.cpu().numpy() if isinstance(targets_sub, torch.Tensor) else np.array(targets_sub[:N])
        for j in range(N):
            Y[j, int(targets_np_sub[j])] = 1.0
        Y -= Y.mean(axis=1, keepdims=True)

        spectral_feats = compute_kernel_spectral_features(K, Y)

        # Gradient features
        grad_feats = compute_gradient_features(
            model, inputs, targets, n_classes, device, max_samples=max_gradient_samples,
        )

        # Assemble feature row
        row = {
            "config_id": config_id,
            "q": q,
            "n_classes": n_classes,
            "encoding_id": config.get("encoding_id", "unknown"),
            "encoding_family": config.get("encoding_family", "unknown"),
            "n_train": config.get("n_train", 0),
            "params": config.get("params", 0),
            "optimizer": config.get("optimizer", "unknown"),
            "weight_decay": config.get("weight_decay", 0),
            "target_type": config.get("target_type", "full_log"),
        }
        row.update(fourier_feats)
        row.update(ntk_feats)
        row.update(spectral_feats)
        row.update(grad_feats)

        results.append(row)

    return results


def save_feature_table(features: List[Dict], path: Path):
    """Save feature table as CSV."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not features:
        return
    keys = features[0].keys()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features)
    print(f"Saved feature table: {path} ({len(features)} rows)")
