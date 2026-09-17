"""Task modules for DLP learning experiments.

Each task defines:
- How latent pairs (a, x) map to (input, target) pairs
- The target type (parity bit, full log, etc.)
- The group structure
"""
from .fixed_base_parity import FixedBaseParityTask
from .variable_base_full_log import VariableBaseFullLogTask
from .additive_inverse import AdditiveInverseTask
from .modular_addition_control import ModularAdditionControlTask

__all__ = [
    "FixedBaseParityTask",
    "VariableBaseFullLogTask",
    "AdditiveInverseTask",
    "ModularAdditionControlTask",
]
