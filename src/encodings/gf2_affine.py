"""GF(2) affine bijections: the primary causal representation intervention.

For a B-bit vector z:
    r_{A,c}(z) = A z + c mod 2
where A in GL(B,2) (invertible B×B matrix over GF(2)) and c in {0,1}^B.

This is an information-preserving bijection:
- Same bits, same length, same alphabet, same information, same labels.
- Only the coordinate geometry changes.

The encoding ID is derived from SHA256(A || c) so that every distinct
(A, c) pair gets a unique, reproducible ID.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
import random
from typing import List, Tuple, Optional
from .identity_binary import Encoding


def _gf2_mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices over GF(2)."""
    return (A.astype(np.int64) @ B.astype(np.int64)) % 2


def _gf2_mat_inv(A: np.ndarray) -> Optional[np.ndarray]:
    """Invert a matrix over GF(2) using Gaussian elimination.

    Returns None if the matrix is singular.
    """
    n = A.shape[0]
    # Augment [A | I]
    M = np.zeros((n, 2 * n), dtype=np.int64)
    M[:, :n] = A.astype(np.int64) % 2
    M[:, n:] = np.eye(n, dtype=np.int64)

    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if M[row, col] == 1:
                pivot = row
                break
        if pivot is None:
            return None  # Singular

        # Swap rows
        if pivot != col:
            M[[col, pivot]] = M[[pivot, col]]

        # Eliminate
        for row in range(n):
            if row != col and M[row, col] == 1:
                M[row] = (M[row] ^ M[col]) % 2

    return M[:, n:].astype(np.int64) % 2


def _gf2_mat_vec(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Matrix-vector multiply over GF(2)."""
    return (A.astype(np.int64) @ v.astype(np.int64)) % 2


def _random_invertible_gf2(n: int, rng: random.Random) -> np.ndarray:
    """Generate a random invertible n×n matrix over GF(2).

    Uses rejection sampling: generate random matrices until one is invertible.
    For n >= 4, almost all random GF(2) matrices are invertible.
    """
    for _ in range(1000):
        A = np.array(
            [[rng.randint(0, 1) for _ in range(n)] for _ in range(n)],
            dtype=np.int64,
        )
        A_inv = _gf2_mat_inv(A)
        if A_inv is not None:
            return A
    raise RuntimeError(f"Failed to generate invertible {n}x{n} GF(2) matrix after 1000 attempts")


def _mat_to_bytes(A: np.ndarray) -> bytes:
    """Serialize a GF(2) matrix to bytes for hashing."""
    return A.tobytes()


class GF2AffineEncoding(Encoding):
    """GF(2) affine bijection: r(z) = A z + c mod 2.

    This is the primary causal representation intervention.
    Same bits, same length, same alphabet, same information, same labels.
    Different coordinate geometry.
    """

    def __init__(self, n_bits: int, A: np.ndarray, c: np.ndarray):
        super().__init__(n_bits)
        self.A = A.astype(np.int64) % 2
        self.c = c.astype(np.int64).flatten() % 2
        assert self.A.shape == (n_bits, n_bits), f"A must be {n_bits}x{n_bits}"
        assert len(self.c) == n_bits, f"c must have {n_bits} elements"

        # Precompute inverse
        self.A_inv = _gf2_mat_inv(self.A)
        if self.A_inv is None:
            raise ValueError("Matrix A is not invertible over GF(2)")

    def encode_int(self, val: int) -> int:
        """Apply r(z) = A z + c mod 2."""
        z = np.array(self.int_to_bits(val), dtype=np.int64)
        result = (_gf2_mat_vec(self.A, z) + self.c) % 2
        return self.bits_to_int(result.tolist())

    def decode_int(self, val: int) -> int:
        """Apply inverse: r^{-1}(y) = A^{-1} (y - c) mod 2 = A^{-1} (y + c) mod 2."""
        y = np.array(self.int_to_bits(val), dtype=np.int64)
        result = _gf2_mat_vec(self.A_inv, (y + self.c) % 2)
        return self.bits_to_int(result.tolist())

    def encoding_id(self) -> str:
        """SHA256 of A || c for unique identification."""
        data = _mat_to_bytes(self.A) + self.c.tobytes()
        h = hashlib.sha256(data).hexdigest()
        return f"gf2_affine_{self.n_bits}b_{h[:12]}"

    def family(self) -> str:
        return "gf2_affine"

    def matrix(self) -> np.ndarray:
        return self.A.copy()

    def offset(self) -> np.ndarray:
        return self.c.copy()


def generate_gf2_encoding(
    n_bits: int,
    seed: int,
) -> GF2AffineEncoding:
    """Generate a deterministic GF(2) affine encoding from a seed.

    The seed determines both A (invertible matrix) and c (offset vector).
    """
    rng = random.Random(seed)
    A = _random_invertible_gf2(n_bits, rng)
    c = np.array([rng.randint(0, 1) for _ in range(n_bits)], dtype=np.int64)
    return GF2AffineEncoding(n_bits, A, c)


def generate_gf2_encodings(
    n_bits: int,
    n_encodings: int,
    base_seed: int = 0,
) -> List[GF2AffineEncoding]:
    """Generate multiple deterministic GF(2) affine encodings.

    Each encoding gets a unique seed: base_seed, base_seed+1, ...
    """
    return [generate_gf2_encoding(n_bits, base_seed + i) for i in range(n_encodings)]
