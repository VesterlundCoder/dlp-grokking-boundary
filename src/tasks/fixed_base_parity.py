"""Fixed-base parity concept task.

Given a fixed base g and target h = g^x, predict bit_i(x).
This corresponds to the concept class in Takhanov et al.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class FixedBaseParityTask:
    """Fixed-base DLP parity: predict bit_i(x) where g^x = h.

    The base g is fixed for the entire concept class.
    Only h is input. The target is bit_i(x).
    """
    q: int
    p: int
    g0: int
    fixed_a: int
    bit_position: int = 0
    group_type: str = "multiplicative"  # or "additive"

    def task_id(self) -> str:
        return f"fixed_parity_q{self.q}_a{self.fixed_a}_bit{self.bit_position}"

    def target_type(self) -> str:
        return "parity"

    def realize(self, a: int, x: int) -> Tuple[int, int]:
        """Given latent (a, x), return (input, target).

        For fixed-base: input = h = g0^(a*x), target = bit_i(x)
        Note: a is fixed (self.fixed_a), so we use it regardless of
        the latent a (which is still stored in the manifest for consistency).
        """
        if self.group_type == "multiplicative":
            h = pow(self.g0, self.fixed_a * x, self.p)
        else:
            h = (self.fixed_a * x) % self.q
        target = (x >> self.bit_position) & 1
        return h, target

    def n_classes(self) -> int:
        return 2
