"""Linear Algebra Engine Package.

Provides custom numerical linear algebra routines implemented from mathematical
first principles without relying on high-level library abstractions:
- Gaussian Elimination with Partial Pivoting (RREF)
- Matrix Rank and Nullity (Rank-Nullity Theorem)
- Fundamental Subspace Bases (Row Space, Column Space, Null Space)
- Vector Inner Products, Gram Matrix G = A * A^T, and Cosine Similarity Matrix S = D^-1 * G * D^-1
- Covariance Matrix C = (1/(m-1)) * X^T * X and Mean Centering
- Eigen-Decomposition & PCA (QR algorithm & Jacobi method)
- Cumulative Variance Ratios R(k)
- Orthogonal Subspace Projection Operator P = V_k * V_k^T and Coordinates Y = X * V_k
"""

from .engine import (
    LinearAlgebraEngine,
    RREFResult,
    SubspaceBases,
    SimilarityResult,
    SpectralResult,
    ProjectionResult,
)

__all__ = [
    "LinearAlgebraEngine",
    "RREFResult",
    "SubspaceBases",
    "SimilarityResult",
    "SpectralResult",
    "ProjectionResult",
]
