"""Tokenizers and representations for cyclic group elements.

Supports:
- R1: Atomic integer tokens (Paper 1 style — one token per residue)
- R2: Binary bit tokenizer (Paper 2 style — each integer as bit sequence)
- R3: Scalar normalized (for MLP controls)
"""
from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple


class IntegerTokenizer:
    """Atomic integer tokenizer: each residue is one token.

    Vocab layout: [PAD=0, SEP=1, BOS=2, EOS=3, 0, 1, ..., M-1]
    Vocab size = M + 4 where M is the modulus (q for additive, p for multiplicative).
    """

    def __init__(self, modulus: int):
        self.modulus = modulus
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.int_offset = 4
        self.vocab_size = modulus + 4
        self.max_len = 5  # BOS base SEP target EOS

    def encode(self, base: int, target: int) -> List[int]:
        """Encode (base, target) pair as token sequence."""
        return [
            self.bos_id,
            self.int_offset + (base % self.modulus),
            self.sep_id,
            self.int_offset + (target % self.modulus),
            self.eos_id,
        ]

    def encode_batch(self, bases: List[int], targets: List[int]) -> torch.Tensor:
        """Encode a batch of (base, target) pairs."""
        return torch.tensor(
            [self.encode(b, t) for b, t in zip(bases, targets)],
            dtype=torch.long,
        )

    def label_to_token(self, x: int) -> int:
        """Convert a label x to a token id."""
        return self.int_offset + (x % self.modulus)

    def token_to_label(self, token: int) -> int:
        """Convert a token id back to a label."""
        return (token - self.int_offset) % self.modulus


class BitTokenizer:
    """Binary bit tokenizer: each integer is represented by its bit sequence.

    Vocab layout: [PAD=0, SEP=1, BOS=2, EOS=3, BIT_0=4, BIT_1=5]
    Vocab size = 6 (fixed, independent of modulus).
    """

    def __init__(self, n_bits: int):
        self.n_bits = n_bits
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.bit_0_id = 4
        self.bit_1_id = 5
        self.vocab_size = 6
        # BOS + n_bits (base) + SEP + n_bits (target) + EOS
        self.max_len = 1 + n_bits + 1 + n_bits + 1

    def _int_to_bits(self, val: int) -> List[int]:
        """Convert integer to list of bit tokens (MSB first)."""
        bits = []
        for i in range(self.n_bits - 1, -1, -1):
            b = (val >> i) & 1
            bits.append(self.bit_1_id if b else self.bit_0_id)
        return bits

    def _bits_to_int(self, bits: List[int]) -> int:
        """Convert list of bit tokens back to integer."""
        val = 0
        for b in bits:
            val = (val << 1) | (1 if b == self.bit_1_id else 0)
        return val

    def encode(self, base: int, target: int) -> List[int]:
        """Encode (base, target) pair as token sequence."""
        return [
            self.bos_id,
            *self._int_to_bits(base),
            self.sep_id,
            *self._int_to_bits(target),
            self.eos_id,
        ]

    def encode_batch(self, bases: List[int], targets: List[int]) -> torch.Tensor:
        """Encode a batch of (base, target) pairs."""
        return torch.tensor(
            [self.encode(b, t) for b, t in zip(bases, targets)],
            dtype=torch.long,
        )


class ScalarNormalizer:
    """Scalar normalized representation for MLP controls: z -> z/(M-1)."""

    def __init__(self, modulus: int):
        self.modulus = modulus
        self.scale = max(modulus - 1, 1)

    def encode(self, base: int, target: int) -> np.ndarray:
        """Encode (base, target) as a 2D normalized vector."""
        return np.array([base / self.scale, target / self.scale], dtype=np.float32)

    def encode_batch(self, bases: List[int], targets: List[int]) -> np.ndarray:
        """Encode a batch of (base, target) pairs."""
        return np.array(
            [[b / self.scale, t / self.scale] for b, t in zip(bases, targets)],
            dtype=np.float32,
        )
