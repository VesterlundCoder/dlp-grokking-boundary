"""Canonical outcome definitions and phase classification.

Uses optimizer updates as the primary training clock.
Also logs epochs, optimizer_steps, and example_presentations.

Sustained thresholds (K=10 consecutive evaluations):
    T_mem = first step at which train accuracy >= 0.99 for K evaluations
    T_gen = first step at which test accuracy >= 0.90 for K evaluations

Phase taxonomy:
    DIRECT_GENERALIZATION: T_gen <= T_mem (or no meaningful memorization plateau)
    GROKKING: T_mem < T_gen <= T_budget
    MEMORIZED_CENSORED: T_mem exists but T_gen not observed before T_budget
    UNDERFIT: T_mem not observed before T_budget
    PARTIAL: intermediate sustained generalization that does not reach 0.90
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List


class Phase(str, Enum):
    DIRECT_GENERALIZATION = "direct_generalization"
    GROKKING = "grokking"
    MEMORIZED_CENSORED = "memorized_censored"
    UNDERFIT = "underfit"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class OutcomeResult:
    phase: Phase
    t_mem: Optional[int]       # optimizer step
    t_gen: Optional[int]       # optimizer step
    t_95: Optional[int]
    t_99: Optional[int]
    grok_delay: Optional[int]  # T_gen - T_mem
    grok_ratio: Optional[float] # T_gen / T_mem
    final_train: float
    final_test: float
    best_test: float
    best_test_step: Optional[int]
    total_steps: int


def _sustained_threshold(
    steps: List[int],
    accs: List[float],
    threshold: float,
    k: int = 10,
) -> Optional[int]:
    """Find first step where accuracy >= threshold for K consecutive evaluations."""
    if len(accs) < k:
        if all(a >= threshold for a in accs) and accs:
            return steps[0]
        return None

    count = 0
    for i, a in enumerate(accs):
        if a >= threshold:
            count += 1
            if count >= k:
                return steps[i - k + 1]
        else:
            count = 0
    return None


def classify_outcome(
    steps: List[int],
    train_accs: List[float],
    test_accs: List[float],
    budget: int,
    k: int = 10,
    mem_threshold: float = 0.99,
    gen_threshold: float = 0.90,
    partial_threshold: float = 0.55,
) -> OutcomeResult:
    """Classify a training run into one of the canonical phases.

    Args:
        steps: Optimizer step at each evaluation point.
        train_accs: Training accuracy at each evaluation point.
        test_accs: Test accuracy at each evaluation point.
        budget: Maximum training budget (optimizer steps).
        k: Number of consecutive evaluations for sustained threshold.
        mem_threshold: Accuracy threshold for memorization (0.99).
        gen_threshold: Accuracy threshold for generalization (0.90).
        partial_threshold: Minimum test accuracy for partial generalization (0.55).
    """
    if not steps or not train_accs or not test_accs:
        return OutcomeResult(
            phase=Phase.UNKNOWN, t_mem=None, t_gen=None, t_95=None, t_99=None,
            grok_delay=None, grok_ratio=None,
            final_train=0.0, final_test=0.0, best_test=0.0,
            best_test_step=None, total_steps=0,
        )

    t_mem = _sustained_threshold(steps, train_accs, mem_threshold, k)
    t_gen = _sustained_threshold(steps, test_accs, gen_threshold, k)
    t_95 = _sustained_threshold(steps, test_accs, 0.95, k)
    t_99 = _sustained_threshold(steps, test_accs, 0.99, k)

    final_train = train_accs[-1]
    final_test = test_accs[-1]
    best_test = max(test_accs)
    best_test_step = steps[test_accs.index(best_test)] if test_accs else None

    grok_delay = (t_gen - t_mem) if (t_gen is not None and t_mem is not None) else None
    grok_ratio = (t_gen / t_mem) if (t_gen is not None and t_mem is not None and t_mem > 0) else None

    # Classification
    if t_mem is None:
        phase = Phase.UNDERFIT
    elif t_gen is not None and t_gen <= t_mem:
        phase = Phase.DIRECT_GENERALIZATION
    elif t_gen is not None and t_mem < t_gen:
        phase = Phase.GROKKING
    elif final_test >= partial_threshold:
        phase = Phase.PARTIAL
    else:
        phase = Phase.MEMORIZED_CENSORED

    return OutcomeResult(
        phase=phase,
        t_mem=t_mem, t_gen=t_gen, t_95=t_95, t_99=t_99,
        grok_delay=grok_delay, grok_ratio=grok_ratio,
        final_train=final_train, final_test=final_test,
        best_test=best_test, best_test_step=best_test_step,
        total_steps=steps[-1] if steps else 0,
    )


def classify_from_metrics(
    metrics_path: Path,
    budget: int,
    k: int = 10,
    step_key: str = "optimizer_step",
    train_key: str = "train_acc",
    test_key: str = "test_acc",
) -> OutcomeResult:
    """Classify a run from its metrics file."""
    steps, train_accs, test_accs = [], [], []
    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d.get(step_key, d.get("epoch", 0)))
            train_accs.append(d.get(train_key, d.get("train_exact", 0)))
            test_accs.append(d.get(test_key, d.get("test_exact", 0)))
    return classify_outcome(steps, train_accs, test_accs, budget, k=k)
