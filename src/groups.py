"""Cyclic group abstractions supporting additive and multiplicative realizations.

A cyclic group of order q is abstractly isomorphic to Z_q regardless of whether
it is realized as (Z_q, +) or as a multiplicative subgroup of F_p*.
This module provides both realizations and a canonical isomorphism between them.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True


def _primitive_root(p: int) -> int:
    if p < 2:
        raise ValueError
    if p == 2:
        return 1
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    raise ValueError(f"No primitive root found for p={p}")


@dataclass(frozen=True)
class MultiplicativeSubgroup:
    """Prime-order multiplicative subgroup <g0> of F_p* with |<g0>| = q."""
    q: int
    p: int
    g0: int

    def validate(self) -> None:
        if not is_prime(self.q):
            raise ValueError(f"q={self.q} must be prime")
        if not is_prime(self.p):
            raise ValueError(f"p={self.p} must be prime")
        if (self.p - 1) % self.q != 0:
            raise ValueError(f"q={self.q} must divide p-1={self.p - 1}")
        if pow(self.g0, self.q, self.p) != 1 or self.g0 == 1:
            raise ValueError(f"g0={self.g0} does not have order q={self.q}")

    def element(self, exponent: int) -> int:
        """Map exponent e in Z_q to group element g0^e mod p."""
        return pow(self.g0, exponent % self.q, self.p)

    def log(self, element: int) -> int:
        """Discrete log base g0 in the subgroup (brute force for small q)."""
        element = element % self.p
        for e in range(self.q):
            if pow(self.g0, e, self.p) == element:
                return e
        raise ValueError(f"Element {element} not in subgroup")


@dataclass(frozen=True)
class AdditiveGroup:
    """Additive group (Z_q, +) of order q."""
    q: int

    def element(self, exponent: int) -> int:
        """Map exponent e in Z_q to group element e mod q."""
        return exponent % self.q

    def log(self, element: int) -> int:
        """Discrete log in additive group: log(e) = e mod q."""
        return element % self.q


def subgroup_from_qp(q: int, p: int) -> MultiplicativeSubgroup:
    """Construct a multiplicative subgroup of order q in F_p*."""
    if not is_prime(q) or not is_prime(p) or (p - 1) % q:
        raise ValueError(f"Need primes q,p with q | p-1, got q={q}, p={p}")
    h = _primitive_root(p)
    g0 = pow(h, (p - 1) // q, p)
    G = MultiplicativeSubgroup(q=q, p=p, g0=g0)
    G.validate()
    return G


# Safe-prime registry: (q_bits, q, p=2q+1) where both q and p are prime
SAFE_PRIME_REGISTRY = [
    (5, 23, 47), (6, 53, 107), (7, 113, 227), (8, 173, 347),
    (9, 281, 563), (10, 593, 1187), (11, 1031, 2063), (12, 2063, 4127),
    (13, 4211, 8423), (14, 8243, 16487), (15, 16421, 32843),
    (16, 32771, 65543), (17, 65633, 131267), (18, 131321, 262643),
]


def group_for_bits(bits: int) -> MultiplicativeSubgroup:
    for b, q, p in SAFE_PRIME_REGISTRY:
        if b == bits:
            return subgroup_from_qp(q, p)
    raise KeyError(f"No frozen group for q_bits={bits}")


def matched_groups(q: int) -> tuple[AdditiveGroup, MultiplicativeSubgroup]:
    """Return (additive, multiplicative) realizations of the same abstract Z_q.

    For the multiplicative realization, finds a safe prime p=2q+1 if q is prime,
    otherwise finds the smallest prime p such that q | p-1.
    """
    if not is_prime(q):
        raise ValueError(f"q={q} must be prime for matched groups")

    # Try safe prime first (p = 2q+1)
    p = 2 * q + 1
    if is_prime(p):
        Gm = subgroup_from_qp(q, p)
    else:
        # Find smallest prime p with q | p-1
        p = q + 1
        while True:
            if is_prime(p) and (p - 1) % q == 0:
                break
            p += q
        Gm = subgroup_from_qp(q, p)

    Ga = AdditiveGroup(q=q)
    return Ga, Gm
