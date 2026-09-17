"""Canonical latent manifests.

All representation comparisons must originate from the SAME latent samples.
For each group q, create manifests/latent/q_<q>.csv containing:
    latent_id, a, x, split

No representation is allowed to independently sample examples.
All encoded datasets must derive deterministically from the latent manifest.
The same latent_id must always belong to the same split.
"""
from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class LatentPair:
    """A single latent sample: (a, x) drawn from Z_q* x Z_q."""
    latent_id: int
    a: int          # base exponent in Z_q*
    x: int          # target exponent in Z_q
    split: str      # "train", "val", or "test"


def generate_latent_manifest(
    q: int,
    seed: int = 42,
    train_frac: float = 0.30,
    val_frac: float = 0.0,
    output_path: Optional[Path] = None,
) -> List[LatentPair]:
    """Generate the canonical latent manifest for a group of order q.

    Samples all (a, x) pairs with a in Z_q* and x in Z_q.
    Total domain size: (q-1) * q.
    Split: train_frac of pairs for training, rest for test.
    """
    rng = random.Random(seed)

    pairs = []
    latent_id = 0
    for a in range(1, q):
        for x in range(q):
            pairs.append(LatentPair(latent_id=latent_id, a=a, x=x, split=""))
            latent_id += 1

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

    pairs.sort(key=lambda p: p.latent_id)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latent_id", "a", "x", "split"])
            for p in pairs:
                writer.writerow([p.latent_id, p.a, p.x, p.split])

    return pairs


def load_latent_manifest(path: Path) -> List[LatentPair]:
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


def manifest_hash(manifest: List[LatentPair]) -> str:
    """Compute SHA256 hash of the manifest for reproducibility."""
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["latent_id", "a", "x", "split"])
    for p in sorted(manifest, key=lambda x: x.latent_id):
        writer.writerow([p.latent_id, p.a, p.x, p.split])
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


def get_split(manifest: List[LatentPair], split: str) -> List[LatentPair]:
    """Filter manifest by split."""
    return [p for p in manifest if p.split == split]


def verify_no_split_leakage(manifest: List[LatentPair]) -> bool:
    """Verify that each latent_id belongs to exactly one split."""
    ids = {}
    for p in manifest:
        if p.latent_id in ids and ids[p.latent_id] != p.split:
            return False
        ids[p.latent_id] = p.split
    return True
