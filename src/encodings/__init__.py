"""Encoding modules for information-preserving representations.

Every encoding is a bijection on B-bit strings:
    r_{A,c}(z) = A z + c mod 2
where A in GL(B,2) and c in {0,1}^B.

Requirements enforced by tests:
- A must be invertible over GF(2)
- Input sequence length is unchanged
- Alphabet is unchanged
- Information content is unchanged
- Task labels are unchanged
- Latent examples are unchanged
"""
from .identity_binary import IdentityBinaryEncoding
from .gray import GrayEncoding
from .gf2_affine import GF2AffineEncoding, generate_gf2_encoding
from .feistel import FeistelEncoding
from .atomic import AtomicEncoding

__all__ = [
    "IdentityBinaryEncoding",
    "GrayEncoding",
    "GF2AffineEncoding",
    "generate_gf2_encoding",
    "FeistelEncoding",
    "AtomicEncoding",
]
