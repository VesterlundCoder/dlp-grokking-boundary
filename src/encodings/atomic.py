"""Atomic integer encoding: each residue is one token (Paper 1 style).

This is a secondary encoding used for comparison. The primary
representation intervention uses binary encodings (identity, Gray, GF2).
"""
from __future__ import annotations

import hashlib
from .identity_binary import Encoding


class AtomicEncoding(Encoding):
    """Atomic integer encoding: each residue maps to a single token.

    Vocab layout: [PAD=0, SEP=1, BOS=2, EOS=3, 0, 1, ..., M-1]
    Vocab size = M + 4 where M is the modulus.

    Note: This encoding does NOT preserve vocabulary size across
    different moduli, so it is secondary to the binary encodings
    for the primary causal comparison.
    """

    def __init__(self, n_bits: int, modulus: int):
        super().__init__(n_bits)
        self.modulus = modulus
        self.vocab_size = modulus + 4
        self.pad_id = 0
        self.sep_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.int_offset = 4

    def encode_int(self, val: int) -> int:
        return val % self.modulus

    def decode_int(self, val: int) -> int:
        return val % self.modulus

    def label_to_token(self, x: int) -> int:
        return self.int_offset + (x % self.modulus)

    def token_to_label(self, token: int) -> int:
        return (token - self.int_offset) % self.modulus

    def encoding_id(self) -> str:
        h = hashlib.sha256(f"atomic_{self.modulus}".encode()).hexdigest()
        return f"atomic_{self.modulus}_{h[:8]}"

    def family(self) -> str:
        return "atomic"
