"""Survival analysis: Kaplan-Meier estimator and concordance index.

For grokking prediction, the "event" is generalization (test_acc >= 0.90).
Censoring occurs when a run reaches the budget without generalizing.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional


def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Kaplan-Meier survival curve.

    S(t) = P(T_gen > t) = probability of NOT having generalized by time t.

    Args:
        times: Observed times (T_gen for uncensored, budget for censored).
        events: 1 if generalized (uncensored), 0 if censored.

    Returns:
        (sorted_times, survival_probabilities)
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # Sort by time
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times)
    n_at_risk = len(times)
    km_curve = []

    for t in unique_times:
        mask_t = times == t
        d = events[mask_t].sum()  # events at this time
        n = (times >= t).sum()    # at risk
        if n > 0:
            survival = 1.0 - d / n
        else:
            survival = 1.0
        km_curve.append((t, survival))

    # Cumulative product
    km_times = np.array([c[0] for c in km_curve])
    km_surv = np.array([c[1] for c in km_curve])
    km_surv = np.cumprod(km_surv)

    return km_times, km_surv


def concordance_index(
    predicted: np.ndarray,
    observed: np.ndarray,
    events: np.ndarray,
) -> float:
    """Compute Harrell's concordance index.

    c-index = P(predicted_i > predicted_j | observed_i > observed_j, both uncensored)

    Args:
        predicted: Predicted risk scores (higher = sooner event).
        observed: Observed times.
        events: Event indicators (1 = uncensored).

    Returns:
        c-index in [0, 1]. 0.5 = random.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    events = np.asarray(events, dtype=int)

    n = len(observed)
    concordant = 0
    permissible = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i must have an event and be observed earlier
            if events[i] == 1 and observed[i] < observed[j]:
                permissible += 1
                if predicted[i] > predicted[j]:
                    concordant += 1
                elif predicted[i] == predicted[j]:
                    concordant += 0.5

    return concordant / permissible if permissible > 0 else 0.5


def compute_cindex(
    predicted_log_T_gen: np.ndarray,
    actual_T_gen: np.ndarray,
    censored: np.ndarray,
) -> float:
    """Compute concordance index for log(T_gen) predictions.

    Higher predicted log_T_gen = predicted later generalization.
    Higher actual T_gen = actual later generalization.

    Args:
        predicted_log_T_gen: Predicted log(T_gen).
        actual_T_gen: Actual T_gen (or budget if censored).
        censored: 1 if censored, 0 if observed.

    Returns:
        c-index.
    """
    events = 1 - np.asarray(censored, dtype=int)
    return concordance_index(
        predicted=-np.asarray(predicted_log_T_gen, dtype=float),  # negate: higher pred = sooner
        observed=np.asarray(actual_T_gen, dtype=float),
        events=events,
    )
