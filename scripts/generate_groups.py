"""Generate the frozen group manifest.

For prime-order multiplicative groups, prefer Sophie Germain pairs:
    q prime, p = 2q + 1 prime
so that q defines a prime-order subgroup of F_p*.

For every requested bit size, generate several deterministic candidate groups
using ascending q. Store group_id, bit_size, q, p, subgroup_generator,
discovery_order.

Reserve separate groups for:
    calibration
    development
    confirmatory holdout
    steering experiment
    OOD experiment
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Tuple


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


def find_sophie_germain_pair(min_bits: int, max_bits: int, n_per_bit: int = 3) -> List[Tuple[int, int, int, int]]:
    """Find Sophie Germain pairs (q, p=2q+1) where both q and p are prime.

    Returns list of (bit_size, q, p, discovery_order).
    """
    results = []
    for bit_size in range(min_bits, max_bits + 1):
        q_min = 1 << (bit_size - 1)
        q_max = (1 << bit_size) - 1
        count = 0
        q = q_min
        if q < 3:
            q = 3
        if q % 2 == 0:
            q += 1
        while q <= q_max and count < n_per_bit:
            p = 2 * q + 1
            if is_prime(q) and is_prime(p):
                results.append((bit_size, q, p, count))
                count += 1
            q += 2
    return results


def generate_groups_csv(
    output_path: Path,
    min_bits: int = 5,
    max_bits: int = 18,
    n_per_bit: int = 5,
    roles: List[str] = None,
):
    """Generate and freeze the group manifest.

    Args:
        output_path: Path to write groups.csv
        min_bits: Minimum bit size
        max_bits: Maximum bit size
        n_per_bit: Number of candidate groups per bit size
        roles: List of roles to assign (calibration, development, confirmatory, steering, ood)
    """
    if roles is None:
        roles = ["calibration", "development", "confirmatory", "steering", "ood"]

    pairs = find_sophie_germain_pair(min_bits, max_bits, n_per_bit)

    rows = []
    group_id = 0
    for bit_size, q, p, discovery_order in pairs:
        h = _primitive_root(p)
        g0 = pow(h, (p - 1) // q, p)
        # Verify
        assert pow(g0, q, p) == 1, f"g0 does not have order q for q={q}, p={p}"
        assert g0 != 1

        # Assign role by discovery_order
        role = roles[discovery_order % len(roles)] if discovery_order < len(roles) else "spare"

        rows.append({
            "group_id": group_id,
            "bit_size": bit_size,
            "q": q,
            "p": p,
            "subgroup_generator": g0,
            "discovery_order": discovery_order,
            "role": role,
        })
        group_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} groups at {output_path}")
    for r in rows:
        print(f"  id={r['group_id']} bits={r['bit_size']} q={r['q']} p={r['p']} g0={r['subgroup_generator']} role={r['role']}")

    return rows


if __name__ == "__main__":
    import sys
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("manifests/groups.csv")
    generate_groups_csv(output)
