"""Cluster bootstrap: resample at the configuration level, not seed level.

This accounts for within-configuration correlation.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Tuple, Any


def cluster_bootstrap(
    data: np.ndarray,
    groups: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, np.ndarray]:
    """Cluster bootstrap: resample groups with replacement.

    Args:
        data: Data array (n_samples, ...).
        groups: Group ID for each sample.
        statistic: Function(data_subset) -> float.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        (point_estimate, bootstrap_samples)
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    rng = np.random.RandomState(seed)

    # Point estimate
    point_estimate = statistic(data)

    # Bootstrap
    bootstrap_samples = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample groups with replacement
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        # Build resampled dataset
        indices = []
        for g in sampled_groups:
            indices.extend(np.where(groups == g)[0])
        indices = np.array(indices)
        bootstrap_samples[b] = statistic(data[indices])

    return point_estimate, bootstrap_samples


def bootstrap_ci(
    point_estimate: float,
    bootstrap_samples: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute percentile bootstrap confidence interval.

    Args:
        point_estimate: The point estimate.
        bootstrap_samples: Bootstrap distribution.
        confidence: Confidence level (0.95 = 95% CI).

    Returns:
        (lower, upper) bounds.
    """
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    return lower, upper
