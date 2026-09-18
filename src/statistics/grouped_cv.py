"""Grouped cross-validation: split by configuration, not by seed.

This prevents information leakage from seed-level correlations.
All seeds of the same configuration go in the same fold.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Iterator
from collections import defaultdict


def grouped_kfold(
    groups: np.ndarray,
    k: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """K-fold CV grouped by configuration.

    Args:
        groups: Array of group IDs for each sample.
        k: Number of folds.
        seed: Random seed for shuffling.

    Yields:
        (train_indices, test_indices) for each fold.
    """
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    # Assign groups to folds
    fold_assignments = {}
    for i, g in enumerate(unique_groups):
        fold_assignments[g] = i % k

    for fold in range(k):
        test_groups = [g for g, f in fold_assignments.items() if f == fold]
        train_groups = [g for g, f in fold_assignments.items() if f != fold]

        test_idx = np.array([i for i, g in enumerate(groups) if g in test_groups])
        train_idx = np.array([i for i, g in enumerate(groups) if g in train_groups])

        yield train_idx, test_idx


def grouped_train_test_split(
    groups: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test split grouped by configuration.

    Args:
        groups: Array of group IDs.
        test_size: Fraction of groups for test.
        seed: Random seed.

    Returns:
        (train_indices, test_indices)
    """
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    n_test = max(1, int(len(unique_groups) * test_size))
    test_groups = set(unique_groups[:n_test])

    test_idx = np.array([i for i, g in enumerate(groups) if g in test_groups])
    train_idx = np.array([i for i, g in enumerate(groups) if g not in test_groups])

    return train_idx, test_idx
