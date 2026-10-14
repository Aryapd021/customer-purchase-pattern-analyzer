"""Linear Algebra Engine Package.

Provides custom numerical linear algebra routines implemented from mathematical
first principles without relying on high-level library abstractions:
- Gaussian Elimination with Partial Pivoting (RREF)
- Matrix Rank and Nullity (Rank-Nullity Theorem)
- Fundamental Subspace Bases (Row Space, Column Space, Null Space)
- Vector Inner Products, Gram Matrix G = A * A^T, and Cosine Similarity Matrix S = D^-1 * G * D^-1
"""

from .engine import (
    LinearAlgebraEngine,
    RREFResult,
    SubspaceBases,
    SimilarityResult,
)

__all__ = [
    "LinearAlgebraEngine",
    "RREFResult",
    "SubspaceBases",
    "SimilarityResult",
]
