"""Prediction metrics: macro-F1, Brier score, calibration error, Spearman, MAE.

All metrics support per-class and aggregate evaluation.
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List = None,
) -> float:
    """Compute macro-averaged F1 score.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: List of class labels.

    Returns:
        Macro F1 score.
    """
    if classes is None:
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    f1_scores = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def brier_score(
    y_true_proba: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Compute multi-class Brier score.

    Args:
        y_true_proba: One-hot true labels (n, K).
        y_pred_proba: Predicted probabilities (n, K).

    Returns:
        Mean Brier score (lower is better).
    """
    return float(np.mean(np.sum((y_true_proba - y_pred_proba) ** 2, axis=1)))


def calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: List = None,
    n_bins: int = 10,
) -> float:
    """Compute expected calibration error (ECE).

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities (n, K).
        classes: List of class labels.
        n_bins: Number of bins for calibration.

    Returns:
        ECE (lower is better).
    """
    if classes is None:
        classes = sorted(np.unique(y_true))

    n = len(y_true)
    ece = 0.0

    for k, c in enumerate(classes):
        proba_k = y_pred_proba[:, k]
        is_class_k = (y_true == c).astype(float)

        # Bin by predicted probability
        bins = np.linspace(0, 1, n_bins + 1)
        for b in range(n_bins):
            mask = (proba_k >= bins[b]) & (proba_k < bins[b + 1])
            if mask.sum() > 0:
                avg_conf = proba_k[mask].mean()
                avg_acc = is_class_k[mask].mean()
                ece += mask.sum() / n * abs(avg_conf - avg_acc)

    return float(ece)


def spearman_rho(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute Spearman rank correlation.

    Args:
        x: First variable.
        y: Second variable.

    Returns:
        Spearman rho in [-1, 1].
    """
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho)


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute mean absolute error.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        MAE.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
