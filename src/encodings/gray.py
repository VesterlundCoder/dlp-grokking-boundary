"""Gray code encoding: a bijection on B-bit strings with single-bit transitions."""
from __future__ import annotations

import hashlib
from .identity_binary import Encoding


class GrayEncoding(Encoding):
    """Gray code encoding.

    Gray code maps integer n to the Gray code representation:
        gray(n) = n XOR (n >> 1)

    This is a bijection on B-bit strings (it is its own inverse for the
    binary-to-Gray direction; the inverse is the Gray-to-binary decode).
    """

    def __init__(self, n_bits: int):
        super().__init__(n_bits)

    def encode_int(self, val: int) -> int:
        """Binary to Gray code."""
        return (val ^ (val >> 1)) & ((1 << self.n_bits) - 1)

    def decode_int(self, val: int) -> int:
        """Gray code to binary."""
        mask = val
        result = val
        while mask > 0:
            mask >>= 1
            result ^= mask
        return result & ((1 << self.n_bits) - 1)

    def encoding_id(self) -> str:
        h = hashlib.sha256(f"gray_{self.n_bits}".encode()).hexdigest()
        return f"gray_{self.n_bits}_{h[:8]}"

    def family(self) -> str:
        return "gray"
