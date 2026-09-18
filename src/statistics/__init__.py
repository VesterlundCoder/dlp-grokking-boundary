"""Statistics modules for the prospective prediction study.

Implements:
    - Survival analysis (Kaplan-Meier, concordance index)
    - Grouped cross-validation (by configuration, not seed)
    - Cluster bootstrap
    - Metrics (c-index, macro-F1, Brier, calibration, Spearman, MAE)
"""
from .survival import kaplan_meier, concordance_index, compute_cindex
from .grouped_cv import grouped_kfold, grouped_train_test_split
from .bootstrap import cluster_bootstrap, bootstrap_ci
from .metrics import macro_f1, brier_score, calibration_error, spearman_rho, mae

__all__ = [
    "kaplan_meier",
    "concordance_index",
    "compute_cindex",
    "grouped_kfold",
    "grouped_train_test_split",
    "cluster_bootstrap",
    "bootstrap_ci",
    "macro_f1",
    "brier_score",
    "calibration_error",
    "spearman_rho",
    "mae",
]
