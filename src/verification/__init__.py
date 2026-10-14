"""Verification Engine Package.

Provides mathematical validation and cross-verification against analytical benchmarks
and reference scipy.linalg implementations:
- Hardcoded Benchmark Test Matrix A_test (Masterplan Section 4)
- Automated theoretical identities checklist (Trace, Determinant, Orthogonality, Eigenpair)
- Numerical stability comparison against scipy.linalg routines
"""

from .engine import (
    VerificationEngine,
    BenchmarkReport,
    IdentityCheckResult,
    ScipyComparisonResult,
)

__all__ = [
    "VerificationEngine",
    "BenchmarkReport",
    "IdentityCheckResult",
    "ScipyComparisonResult",
]
