"""Feistel encoding: a fixed-length nonlinear bijection on B-bit strings.

Used for CONF_C (out-of-family encoding holdout) to test whether the
predictor transfers beyond linear GF(2) transformations.

Uses a balanced Feistel network: splits B bits into two equal halves
of B//2 bits each. If B is odd, the middle bit passes through unchanged.
The round function is hash-based and keyed by (round_index, key_seed).
"""
from __future__ import annotations

import hashlib
from .identity_binary import Encoding


class FeistelEncoding(Encoding):
    """Keyed Feistel permutation on B-bit strings.

    Guaranteed bijection by construction. Invertible by running rounds
    in reverse order.
    """

    def __init__(self, n_bits: int, key_seed: int, n_rounds: int = 4):
        super().__init__(n_bits)
        self.key_seed = key_seed
        self.n_rounds = n_rounds
        self.half_bits = n_bits // 2
        self.has_middle_bit = (n_bits % 2 == 1)

    def _round_function(self, round_idx: int, r_val: int) -> int:
        """Hash-based round function: f(round_idx, R) -> half_bits-sized output."""
        h = hashlib.sha256(
            f"feistel_r{round_idx}_k{self.key_seed}_v{r_val}".encode()
        ).digest()
        result = 0
        for i in range((self.half_bits + 7) // 8):
            result = (result << 8) | h[i % len(h)]
        return result & ((1 << self.half_bits) - 1)

    def encode_int(self, val: int) -> int:
        """Apply Feistel network."""
        half_mask = (1 << self.half_bits) - 1

        # Extract middle bit if odd width
        middle_bit = 0
        if self.has_middle_bit:
            middle_bit = (val >> self.half_bits) & 1
            L = (val >> (self.half_bits + 1)) & half_mask
        else:
            L = (val >> self.half_bits) & half_mask
        R = val & half_mask

        for round_idx in range(self.n_rounds):
            f_out = self._round_function(round_idx, R)
            new_R = L ^ f_out
            L = R
            R = new_R

        if self.has_middle_bit:
            return (L << (self.half_bits + 1)) | (middle_bit << self.half_bits) | R
        else:
            return (L << self.half_bits) | R

    def decode_int(self, val: int) -> int:
        """Apply inverse Feistel network (reverse rounds)."""
        half_mask = (1 << self.half_bits) - 1

        if self.has_middle_bit:
            middle_bit = (val >> self.half_bits) & 1
            L = (val >> (self.half_bits + 1)) & half_mask
        else:
            middle_bit = 0
            L = (val >> self.half_bits) & half_mask
        R = val & half_mask

        for round_idx in range(self.n_rounds - 1, -1, -1):
            f_out = self._round_function(round_idx, L)
            old_R = L
            old_L = R ^ f_out
            L = old_L
            R = old_R

        if self.has_middle_bit:
            return (L << (self.half_bits + 1)) | (middle_bit << self.half_bits) | R
        else:
            return (L << self.half_bits) | R

    def encoding_id(self) -> str:
        h = hashlib.sha256(
            f"feistel_{self.n_bits}b_k{self.key_seed}_r{self.n_rounds}".encode()
        ).hexdigest()
        return f"feistel_{self.n_bits}b_k{self.key_seed}_r{self.n_rounds}_{h[:8]}"

    def family(self) -> str:
        return "feistel"
