"""Additive inverse task: given a, predict -a mod q.

A simple control task to verify the architecture can learn basic
modular arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AdditiveInverseTask:
    """Additive inverse: given a in Z_q, predict (-a) mod q."""
    q: int

    def task_id(self) -> str:
        return f"additive_inverse_q{self.q}"

    def target_type(self) -> str:
        return "full_log"

    def realize(self, a: int, x: int) -> Tuple[int, int]:
        return a % self.q, (-a) % self.q

    def n_classes(self) -> int:
        return self.q
