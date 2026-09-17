"""Modular addition control task: (u, v) -> (u + v) mod q.

Architecture/training sanity control. NOT the isomorphic counterpart of DLP.
Used to answer: can the architecture and tokenizer grok a known modular
task at this scale?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModularAdditionControlTask:
    """Modular addition: given (u, v), predict (u + v) mod q."""
    q: int

    def task_id(self) -> str:
        return f"modular_addition_q{self.q}"

    def target_type(self) -> str:
        return "full_log"

    def realize(self, a: int, x: int) -> Tuple[Tuple[int, int], int]:
        """Use latent (a, x) as (u, v) inputs."""
        return (a % self.q, x % self.q), (a + x) % self.q

    def n_classes(self) -> int:
        return self.q
