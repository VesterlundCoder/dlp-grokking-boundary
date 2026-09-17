"""Variable-base full-log DLP task.

Given base g and target h = g^x, predict x.
This is the main task in the prediction study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class VariableBaseFullLogTask:
    """Variable-base DLP: predict x where g^x = h.

    Both g (base) and h (target) are inputs.
    The target is the full discrete log x.
    """
    q: int
    p: int
    g0: int
    group_type: str = "multiplicative"

    def task_id(self) -> str:
        return f"var_fulllog_q{self.q}_{self.group_type}"

    def target_type(self) -> str:
        return "full_log"

    def realize(self, a: int, x: int) -> Tuple[Tuple[int, int], int]:
        """Given latent (a, x), return ((base, target), label).

        For variable-base: base = g0^a, target = g0^(a*x), label = x
        """
        if self.group_type == "multiplicative":
            base = pow(self.g0, a, self.p)
            target = pow(self.g0, a * x, self.p)
        else:
            base = a % self.q
            target = (a * x) % self.q
        return (base, target), x

    def n_classes(self) -> int:
        return self.q
