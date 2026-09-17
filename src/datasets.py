"""Matched isomorphic datasets: same latent samples, different group realizations.

For a prime q, we sample latent pairs (a, x) from Z_q* x Z_q.
Both additive and multiplicative realizations are generated from the SAME
latent manifest, ensuring identical train/test membership and labels.
"""
from __future__ import annotations

import csv
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .groups import AdditiveGroup, MultiplicativeSubgroup, matched_groups


@dataclass
class LatentPair:
    """A single latent sample: (a, x) drawn from Z_q* x Z_q."""
    latent_id: int
    a: int          # base exponent in Z_q*
    x: int          # target exponent in Z_q
    split: str      # "train", "val", or "test"


@dataclass
class RealizedSample:
    """A sample realized in a specific group representation."""
    base: int       # group element representing the base
    target: int     # group element representing the target (base^x)
    x: int          # the discrete log (label)
    latent_id: int
    split: str


def generate_latent_manifest(
    q: int,
    seed: int = 42,
    train_frac: float = 0.30,
    val_frac: float = 0.0,
    output_path: Optional[Path] = None,
) -> list[LatentPair]:
    """Generate the canonical latent manifest for a group of order q.

    Samples all (a, x) pairs with a in Z_q* and x in Z_q.
    Total domain size: (q-1) * q.
    Split: train_frac of pairs for training, rest for test.
    """
    rng = random.Random(seed)

    # Full domain: a in {1, ..., q-1}, x in {0, ..., q-1}
    pairs = []
    latent_id = 0
    for a in range(1, q):
        for x in range(q):
            pairs.append(LatentPair(latent_id=latent_id, a=a, x=x, split=""))
            latent_id += 1

    # Shuffle and split
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    for i, p in enumerate(pairs):
        if i < n_train:
            p.split = "train"
        elif i < n_train + n_val:
            p.split = "val"
        else:
            p.split = "test"

    # Sort by latent_id for reproducibility
    pairs.sort(key=lambda p: p.latent_id)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latent_id", "a", "x", "split"])
            for p in pairs:
                writer.writerow([p.latent_id, p.a, p.x, p.split])

    return pairs


def load_latent_manifest(path: Path) -> list[LatentPair]:
    """Load a latent manifest from CSV."""
    pairs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(LatentPair(
                latent_id=int(row["latent_id"]),
                a=int(row["a"]),
                x=int(row["x"]),
                split=row["split"],
            ))
    return pairs


def realize_additive(
    manifest: list[LatentPair],
    group: AdditiveGroup,
) -> list[RealizedSample]:
    """Realize latent pairs in the additive group (Z_q, +).

    base = a mod q
    target = (a * x) mod q
    task: given base, predict x such that base * x ≡ target (mod q)
    (This is division in Z_q, equivalent to DLP in additive notation)
    """
    samples = []
    for p in manifest:
        base = group.element(p.a)       # a mod q
        target = (p.a * p.x) % group.q   # ax mod q
        samples.append(RealizedSample(
            base=base, target=target, x=p.x,
            latent_id=p.latent_id, split=p.split,
        ))
    return samples


def realize_multiplicative(
    manifest: list[LatentPair],
    group: MultiplicativeSubgroup,
) -> list[RealizedSample]:
    """Realize latent pairs in the multiplicative subgroup of F_p*.

    base = g0^a mod p
    target = g0^(a*x) mod p = base^x mod p
    task: given base, predict x such that base^x ≡ target (mod p)
    """
    samples = []
    for p in manifest:
        base = group.element(p.a)           # g0^a mod p
        target = group.element(p.a * p.x)   # g0^(ax) mod p
        samples.append(RealizedSample(
            base=base, target=target, x=p.x,
            latent_id=p.latent_id, split=p.split,
        ))
    return samples


def realize_variable_base_additive(
    manifest: list[LatentPair],
    group: AdditiveGroup,
) -> list[RealizedSample]:
    """Variable-base additive: inputs are (base, target), predict x.

    base = a mod q
    target = (a * x) mod q
    Model sees both base and target, predicts x.
    """
    # Same as realize_additive but we'll handle both inputs in the tokenizer
    return realize_additive(manifest, group)


def realize_variable_base_multiplicative(
    manifest: list[LatentPair],
    group: MultiplicativeSubgroup,
) -> list[RealizedSample]:
    """Variable-base multiplicative: inputs are (base, target), predict x.

    base = g0^a mod p
    target = g0^(a*x) mod p
    Model sees both base and target, predicts x.
    """
    return realize_multiplicative(manifest, group)


def manifest_hash(manifest: list[LatentPair]) -> str:
    """Compute SHA256 hash of the manifest for reproducibility."""
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["latent_id", "a", "x", "split"])
    for p in sorted(manifest, key=lambda x: x.latent_id):
        writer.writerow([p.latent_id, p.a, p.x, p.split])
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def get_split(samples: list[RealizedSample], split: str) -> list[RealizedSample]:
    """Filter samples by split."""
    return [s for s in samples if s.split == split]
