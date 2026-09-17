"""Base encoding interface and identity binary encoding."""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import hashlib


class Encoding(ABC):
    """Base class for all encodings.

    An encoding is a bijection on B-bit representations of integers.
    It must be invertible and information-preserving.
    """

    def __init__(self, n_bits: int):
        self.n_bits = n_bits

    @abstractmethod
    def encode_int(self, val: int) -> int:
        """Encode an integer as another integer (same bit width)."""
        pass

    @abstractmethod
    def decode_int(self, val: int) -> int:
        """Decode an integer back to the original."""
        pass

    @abstractmethod
    def encoding_id(self) -> str:
        """Return a unique identifier for this encoding."""
        pass

    @abstractmethod
    def family(self) -> str:
        """Return the encoding family name."""
        pass

    def int_to_bits(self, val: int) -> List[int]:
        """Convert integer to bit list (MSB first)."""
        bits = []
        for i in range(self.n_bits - 1, -1, -1):
            bits.append((val >> i) & 1)
        return bits

    def bits_to_int(self, bits: List[int]) -> int:
        """Convert bit list (MSB first) to integer."""
        val = 0
        for b in bits:
            val = (val << 1) | (b & 1)
        return val

    def encode_bits(self, bits: List[int]) -> List[int]:
        """Encode a bit sequence."""
        val = self.bits_to_int(bits)
        encoded = self.encode_int(val)
        return self.int_to_bits(encoded)

    def decode_bits(self, bits: List[int]) -> List[int]:
        """Decode a bit sequence."""
        val = self.bits_to_int(bits)
        decoded = self.decode_int(val)
        return self.int_to_bits(decoded)

    def verify_roundtrip(self, n_tests: int = 100, rng_seed: int = 42) -> bool:
        """Verify that encode/decode is a perfect roundtrip."""
        import random
        rng = random.Random(rng_seed)
        for _ in range(n_tests):
            val = rng.randint(0, (1 << self.n_bits) - 1)
            encoded = self.encode_int(val)
            decoded = self.decode_int(encoded)
            if decoded != val:
                return False
        return True


class IdentityBinaryEncoding(Encoding):
    """Identity encoding: r(z) = z. No transformation."""

    def __init__(self, n_bits: int):
        super().__init__(n_bits)

    def encode_int(self, val: int) -> int:
        return val & ((1 << self.n_bits) - 1)

    def decode_int(self, val: int) -> int:
        return val & ((1 << self.n_bits) - 1)

    def encoding_id(self) -> str:
        h = hashlib.sha256(f"identity_binary_{self.n_bits}".encode()).hexdigest()
        return f"identity_binary_{self.n_bits}_{h[:8]}"

    def family(self) -> str:
        return "identity"
