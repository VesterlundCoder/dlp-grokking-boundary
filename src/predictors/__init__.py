"""Predictor modules for the prospective prediction study.

Two predictor classes:
  P0: Strict pre-training (t=0 only)
  P1: Early-probe (1% training window)
"""
from .fourier import compute_fourier_features
from .ntk import compute_ntk_features
from .kernel_spectrum import compute_kernel_spectral_features, compute_T_spec, compute_KSA
from .gradients import compute_gradient_features
from .pretraining_features import build_p0_feature_table
from .early_probe import build_p1_features
from .timescale_models import fit_timescale_model
from .phase_predictor import PhasePredictor
from .baselines import compute_baselines

__all__ = [
    "compute_fourier_features",
    "compute_ntk_features",
    "compute_kernel_spectral_features",
    "compute_T_spec",
    "compute_KSA",
    "compute_gradient_features",
    "build_p0_feature_table",
    "build_p1_features",
    "fit_timescale_model",
    "PhasePredictor",
    "compute_baselines",
]
