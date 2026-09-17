"""Canonical phase classifier for grokking/learnability experiments.

Defines:
  Tmem = min{t : A_train(t) >= 0.99}  (sustained over K consecutive evals)
  T90  = min{t : A_test(t) >= 0.90}   (sustained over K consecutive evals)

Phase taxonomy:
  direct_generalization: T90 <= Tmem (or no distinct memorization plateau)
  grokking:              Tmem < T90 (delayed generalization after memorization)
  memorization_only:     Tmem exists, T90 does not
  underfit:               Tmem does not exist
  partial:               intermediate test accuracy without reaching threshold
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PhaseResult:
    phase: str
    t_mem: Optional[int]
    t90: Optional[int]
    t95: Optional[int]
    t99: Optional[int]
    delta_t: Optional[int]       # T90 - Tmem (for grokking)
    grok_ratio: Optional[float]   # T90 / Tmem (for grokking)
    final_train: float
    final_test: float
    best_test: float
    best_test_epoch: Optional[int]


def _sustained_threshold(
    epochs: list[int],
    accs: list[float],
    threshold: float,
    k: int = 5,
) -> Optional[int]:
    """Find first epoch where accuracy >= threshold for K consecutive evaluations."""
    if len(accs) < k:
        # If all evaluations are above threshold, return first epoch
        if all(a >= threshold for a in accs):
            return epochs[0] if epochs else None
        return None

    count = 0
    for i, a in enumerate(accs):
        if a >= threshold:
            count += 1
            if count >= k:
                return epochs[i - k + 1]
        else:
            count = 0
    return None


def classify_phase(
    epochs: list[int],
    train_accs: list[float],
    test_accs: list[float],
    k: int = 5,
    mem_threshold: float = 0.99,
    gen_threshold: float = 0.90,
    partial_threshold: float = 0.55,
) -> PhaseResult:
    """Classify a training run into one of the canonical phases.

    Args:
        epochs: List of epoch numbers at each evaluation point.
        train_accs: Training accuracy at each evaluation point.
        test_accs: Test accuracy at each evaluation point.
        k: Number of consecutive evaluations required for sustained threshold.
        mem_threshold: Accuracy threshold for memorization (default 0.99).
        gen_threshold: Accuracy threshold for generalization (default 0.90).
        partial_threshold: Minimum test accuracy for partial generalization (default 0.55).
    """
    if not epochs or not train_accs or not test_accs:
        return PhaseResult(
            phase="unknown", t_mem=None, t90=None, t95=None, t99=None,
            delta_t=None, grok_ratio=None,
            final_train=0.0, final_test=0.0, best_test=0.0, best_test_epoch=None,
        )

    t_mem = _sustained_threshold(epochs, train_accs, mem_threshold, k)
    t90 = _sustained_threshold(epochs, test_accs, gen_threshold, k)
    t95 = _sustained_threshold(epochs, test_accs, 0.95, k)
    t99 = _sustained_threshold(epochs, test_accs, 0.99, k)

    final_train = train_accs[-1]
    final_test = test_accs[-1]
    best_test = max(test_accs)
    best_test_epoch = epochs[test_accs.index(best_test)] if test_accs else None

    delta_t = (t90 - t_mem) if (t90 is not None and t_mem is not None) else None
    grok_ratio = (t90 / t_mem) if (t90 is not None and t_mem is not None and t_mem > 0) else None

    # Classification
    if t_mem is None:
        phase = "underfit"
    elif t90 is not None and t90 <= t_mem:
        phase = "direct_generalization"
    elif t90 is not None and t_mem < t90:
        phase = "grokking"
    elif final_test >= partial_threshold:
        phase = "partial"
    else:
        phase = "memorization_only"

    return PhaseResult(
        phase=phase,
        t_mem=t_mem, t90=t90, t95=t95, t99=t99,
        delta_t=delta_t, grok_ratio=grok_ratio,
        final_train=final_train, final_test=final_test,
        best_test=best_test, best_test_epoch=best_test_epoch,
    )


def classify_from_metrics_file(
    metrics_path: Path,
    k: int = 5,
    train_key: str = "train_acc",
    test_key: str = "test_acc",
    epoch_key: str = "epoch",
) -> PhaseResult:
    """Classify a run from its metrics.jsonl file."""
    epochs, train_accs, test_accs = [], [], []
    with open(metrics_path) as f:
        for line in f:
            d = json.loads(line)
            epochs.append(d[epoch_key])
            train_accs.append(d.get(train_key, d.get("train_exact", 0)))
            test_accs.append(d.get(test_key, d.get("test_exact", 0)))
    return classify_phase(epochs, train_accs, test_accs, k=k)
